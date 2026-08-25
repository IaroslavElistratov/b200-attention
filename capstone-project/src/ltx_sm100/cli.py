from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_CHECKPOINT = Path(
    "models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors"
)
DEFAULT_SPATIAL_UPSAMPLER = Path(
    "models/checkpoints/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)
DEFAULT_GEMMA_ROOT = Path("models/gemma")
DEFAULT_OUTPUT_DIRECTORY = Path("outputs")
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1536
DEFAULT_FRAMES = 249
DEFAULT_STEPS = 8
DEFAULT_STAGE_2_STEPS = 3
DEFAULT_SEED = 4242
DEFAULT_FPS = 24
ATTENTION_CALLS_PER_STEP = 96
VIDEO_ONLY_SD_OPS_NAME = "LTX_VIDEO_ONLY_COMFY_PREFIX_MAP"
LTX_SOURCE_COMMIT = "780984275fd47128b02bef9b5c085404276866ee"
LTX_MODEL_REVISION = "6f3520585aa27248020550da2f453aa0c572398c"
GEMMA_REVISION = "68f7ee4fbd59087436ada77ed2d62f373fdd4482"


@dataclass(frozen=True)
class RunConfig:
    prompt: str
    checkpoint_path: Path
    spatial_upsampler_path: Path
    gemma_root: Path
    output_path: Path
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    frames: int = DEFAULT_FRAMES
    seed: int = DEFAULT_SEED
    fps: int = DEFAULT_FPS
    library_path: Path | None = None

    @property
    def report_path(self) -> Path:
        return self.output_path.with_suffix(".json")


def project_root() -> Path:
    """Return the checkout root without depending on the caller's working directory."""

    return Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a user path, interpreting relative paths from the checkout root."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def expected_attention_calls(*stage_steps: int) -> int:
    if not stage_steps or any(steps <= 0 for steps in stage_steps):
        raise ValueError(f"stage steps must all be positive, got {stage_steps}.")
    return sum(stage_steps) * ATTENTION_CALLS_PER_STEP


def shape_report(
    height: int,
    width: int,
    frames: int,
    *,
    context_length: int | None = None,
) -> dict[str, int]:
    """Validate a diffusion stage's geometry and return its SM100 sequence lengths."""

    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}.")
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            f"height and width must be divisible by 32, got {height}x{width}."
        )
    if frames <= 0 or (frames - 1) % 8 != 0:
        raise ValueError(
            f"frame count must satisfy frames = 8*k + 1, got {frames}."
        )

    latent_frames = (frames - 1) // 8 + 1
    latent_height = height // 32
    latent_width = width // 32
    video_tokens = latent_frames * latent_height * latent_width
    report = {
        "latent_frames": latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "video_tokens": video_tokens,
        "video_tokens_mod_256": video_tokens % 256,
    }
    if video_tokens % 256 != 0:
        raise ValueError(
            "the SM100 kernel requires the video query length to be divisible by 256; "
            f"got {video_tokens} tokens for {height}x{width}, {frames} frames."
        )

    if context_length is not None:
        if context_length <= 0 or context_length % 128 != 0:
            raise ValueError(
                "the SM100 kernel requires the prompt context length to be a positive "
                f"multiple of 128, got {context_length}."
            )
        report["context_length"] = context_length
        report["context_length_mod_128"] = context_length % 128
    return report


def two_stage_shape_report(
    height: int,
    width: int,
    frames: int,
    *,
    context_length: int | None = None,
) -> dict[str, dict[str, int]]:
    """Validate the two-stage geometry and report both attention token shapes."""

    if height <= 0 or width <= 0 or height % 64 != 0 or width % 64 != 0:
        raise ValueError(
            "two-stage height and width must be positive multiples of 64, "
            f"got {height}x{width}."
        )
    return {
        "stage_1": shape_report(
            height // 2,
            width // 2,
            frames,
            context_length=context_length,
        ),
        "stage_2": shape_report(
            height,
            width,
            frames,
            context_length=context_length,
        ),
    }


def pin_sm100_attention(stage: Any, attention: Any) -> Any:
    """Pin the custom callable through LTX's public, functional stage interface."""

    pinned = stage.with_attention(attention)
    if pinned is stage:
        raise RuntimeError("DiffusionStage.with_attention returned the unmodified stage.")
    return pinned


def expected_attention_signature_counts(
    stage_token_steps: Sequence[tuple[int, int]],
    context_length: int,
    *,
    device: str,
) -> dict[str, int]:
    """Return the exact self/cross signature counts for each diffusion stage."""

    calls_per_kind_per_step = ATTENTION_CALLS_PER_STEP // 2
    signatures: dict[str, int] = {}
    for query_length, steps in stage_token_steps:
        for key_value_length in (context_length, query_length):
            signature = (
                f"B=1,Lq={query_length},Lkv={key_value_length},H=32,D=128,"
                f"dtype=torch.bfloat16,device={device}"
            )
            signatures[signature] = (
                signatures.get(signature, 0) + steps * calls_per_kind_per_step
            )
    return signatures


def validate_attention_stats(
    stats: dict[str, Any],
    expected_calls: int,
    expected_signatures: dict[str, int] | None = None,
) -> None:
    """Fail closed if any expected LTX attention call missed the custom backend."""

    actual = stats.get("total_calls")
    if actual != expected_calls:
        raise RuntimeError(
            "strict SM100 routing check failed: "
            f"expected {expected_calls} custom attention calls, observed {actual!r}. "
            "No fallback result will be emitted."
        )
    if expected_signatures is not None and stats.get("signatures") != expected_signatures:
        raise RuntimeError(
            "strict SM100 signature check failed: "
            f"expected {expected_signatures}, observed {stats.get('signatures')!r}. "
            "No fallback result will be emitted."
        )


def _validate_model_paths(
    checkpoint_path: Path,
    spatial_upsampler_path: Path,
    gemma_root: Path,
) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"LTX checkpoint not found: {checkpoint_path}. Run scripts/download_models.sh."
        )
    if not spatial_upsampler_path.is_file():
        raise FileNotFoundError(
            "LTX spatial upscaler not found: "
            f"{spatial_upsampler_path}. Run scripts/download_models.sh."
        )
    if not gemma_root.is_dir():
        raise FileNotFoundError(
            f"Gemma model directory not found: {gemma_root}. Run scripts/download_models.sh."
        )
    if not (gemma_root / "config.json").is_file():
        raise FileNotFoundError(f"Gemma config.json not found under {gemma_root}.")
    if next(gemma_root.rglob("*.safetensors"), None) is None:
        raise FileNotFoundError(f"No Gemma safetensors weights found under {gemma_root}.")


def _validate_run_config(config: RunConfig) -> None:
    if not config.prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text.")
    if config.fps <= 0:
        raise ValueError(f"--fps must be positive, got {config.fps}.")
    if config.output_path.suffix.lower() != ".mp4":
        raise ValueError(f"--output must end in .mp4, got {config.output_path}.")
    two_stage_shape_report(config.height, config.width, config.frames)
    _validate_model_paths(
        config.checkpoint_path,
        config.spatial_upsampler_path,
        config.gemma_root,
    )


def _video_only_sd_ops() -> Any:
    from ltx_core.loader.sd_ops import SDOps

    return SDOps(VIDEO_ONLY_SD_OPS_NAME).with_matching(
        prefix="model.diffusion_model."
    ).with_replacement("model.diffusion_model.", "")


def _run_generation(config: RunConfig) -> dict[str, Any]:  # noqa: PLR0915
    """Run the default two-stage distilled video-only pipeline."""

    _validate_run_config(config)

    import torch

    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.blocks import (
        DiffusionStage,
        PromptEncoder,
        VideoDecoder,
        VideoUpsampler,
    )
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

    from .backend import SM100Attention, get_stats, reset_stats

    if not torch.cuda.is_available():
        raise RuntimeError("generation requires CUDA and an NVIDIA B200 GPU.")
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    device_name = torch.cuda.get_device_name(device)
    if capability != (10, 0) or "B200" not in device_name.upper():
        raise RuntimeError(
            "generation requires an NVIDIA B200 with compute capability 10.0; "
            f"got {device_name} with {capability[0]}.{capability[1]}."
        )

    stage_1_sigmas = DISTILLED_SIGMAS.to(device=device, dtype=torch.float32)
    stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(device=device, dtype=torch.float32)
    stage_1_steps = int(stage_1_sigmas.numel() - 1)
    stage_2_steps = int(stage_2_sigmas.numel() - 1)
    if stage_1_steps != DEFAULT_STEPS:
        raise RuntimeError(
            "pinned LTX stage-1 distilled schedule has "
            f"{stage_1_steps} steps, expected {DEFAULT_STEPS}."
        )
    if stage_2_steps != DEFAULT_STAGE_2_STEPS:
        raise RuntimeError(
            "pinned LTX stage-2 distilled schedule has "
            f"{stage_2_steps} steps, expected {DEFAULT_STAGE_2_STEPS}."
        )

    started = time.perf_counter()
    prompt_started = time.perf_counter()
    encoder = PromptEncoder(
        checkpoint_path=str(config.checkpoint_path),
        gemma_root=str(config.gemma_root),
        dtype=torch.bfloat16,
        device=device,
        registry=None,
        offload_mode=OffloadMode.NONE,
    )
    (context,) = encoder(
        [config.prompt],
        enhance_first_prompt=False,
        enhance_prompt_image=None,
    )
    video_context = context.video_encoding.to(device=device, dtype=torch.bfloat16)
    prompt_elapsed = time.perf_counter() - prompt_started
    if video_context.ndim != 3 or video_context.shape[0] != 1:
        raise RuntimeError(
            "Gemma produced an unexpected video context shape: "
            f"{tuple(video_context.shape)}."
        )
    geometry = two_stage_shape_report(
        config.height,
        config.width,
        config.frames,
        context_length=int(video_context.shape[1]),
    )

    stage = DiffusionStage(
        checkpoint_path=str(config.checkpoint_path),
        dtype=torch.bfloat16,
        device=device,
        loras=(),
        quantization=None,
        registry=None,
        compilation_config=None,
        offload_mode=OffloadMode.NONE,
        model_configurator=LTXVideoOnlyModelConfigurator,
        model_sd_ops=_video_only_sd_ops(),
    )
    stage = pin_sm100_attention(
        stage,
        SM100Attention(config.library_path, build_if_missing=True),
    )

    generator = torch.Generator(device=device).manual_seed(config.seed)
    noiser = GaussianNoiser(generator=generator)
    denoiser = SimpleDenoiser(v_context=video_context, a_context=None)

    def run_stage(
        *,
        sigmas: Any,
        width: int,
        height: int,
        initial_latent: Any | None = None,
    ) -> tuple[Any, float]:
        video_spec: dict[str, Any] = {"context": video_context, "conditionings": []}
        if initial_latent is not None:
            video_spec["noise_scale"] = float(sigmas[0].item())
            video_spec["initial_latent"] = initial_latent
        stage_started = time.perf_counter()
        video_state, audio_state = stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=config.frames,
            fps=float(config.fps),
            video=ModalitySpec(**video_spec),
            audio=None,
            max_batch_size=1,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - stage_started
        if audio_state is not None:
            raise RuntimeError("video-only generation unexpectedly returned an audio state.")
        if video_state is None:
            raise RuntimeError("video-only generation returned no video state.")
        return video_state, elapsed

    def latent_report(latent: Any) -> dict[str, Any]:
        return {
            "shape": tuple(latent.shape),
            "dtype": str(latent.dtype),
            "finite": bool(torch.isfinite(latent).all().item()),
            "nan_count": int(torch.isnan(latent).sum().item()),
            "inf_count": int(torch.isinf(latent).sum().item()),
        }

    torch.cuda.reset_peak_memory_stats(device)
    reset_stats()
    stage_1_state, stage_1_elapsed = run_stage(
        sigmas=stage_1_sigmas,
        width=config.width // 2,
        height=config.height // 2,
    )
    stage_1_latent = latent_report(stage_1_state.latent)
    if not stage_1_latent["finite"]:
        raise RuntimeError(
            "stage-1 video latent contains non-finite values: "
            f"nan={stage_1_latent['nan_count']}, inf={stage_1_latent['inf_count']}."
        )
    stage_1_required_calls = expected_attention_calls(stage_1_steps)
    stage_1_expected_signatures = expected_attention_signature_counts(
        ((geometry["stage_1"]["video_tokens"], stage_1_steps),),
        int(video_context.shape[1]),
        device=str(device),
    )
    stage_1_stats = get_stats()
    validate_attention_stats(
        stage_1_stats,
        stage_1_required_calls,
        stage_1_expected_signatures,
    )

    upsample_started = time.perf_counter()
    upscaled_latent = VideoUpsampler(
        str(config.checkpoint_path),
        str(config.spatial_upsampler_path),
        torch.bfloat16,
        device,
    )(stage_1_state.latent[:1])
    torch.cuda.synchronize(device)
    upsample_elapsed = time.perf_counter() - upsample_started
    upsampled_latent = latent_report(upscaled_latent)
    if not upsampled_latent["finite"]:
        raise RuntimeError(
            "spatially upsampled latent contains non-finite values: "
            f"nan={upsampled_latent['nan_count']}, inf={upsampled_latent['inf_count']}."
        )

    stage_2_state, stage_2_elapsed = run_stage(
        sigmas=stage_2_sigmas,
        width=config.width,
        height=config.height,
        initial_latent=upscaled_latent,
    )

    attention_stats = get_stats()
    required_calls = expected_attention_calls(stage_1_steps, stage_2_steps)
    expected_signatures = expected_attention_signature_counts(
        (
            (geometry["stage_1"]["video_tokens"], stage_1_steps),
            (geometry["stage_2"]["video_tokens"], stage_2_steps),
        ),
        int(video_context.shape[1]),
        device=str(device),
    )
    validate_attention_stats(attention_stats, required_calls, expected_signatures)
    stage_2_required_calls = expected_attention_calls(stage_2_steps)
    stage_2_expected_signatures = expected_attention_signature_counts(
        ((geometry["stage_2"]["video_tokens"], stage_2_steps),),
        int(video_context.shape[1]),
        device=str(device),
    )
    stage_2_stats = {
        "total_calls": attention_stats["total_calls"] - stage_1_stats["total_calls"],
        "signatures": {
            signature: count - stage_1_stats["signatures"].get(signature, 0)
            for signature, count in attention_stats["signatures"].items()
            if count - stage_1_stats["signatures"].get(signature, 0) > 0
        },
    }
    validate_attention_stats(
        stage_2_stats,
        stage_2_required_calls,
        stage_2_expected_signatures,
    )

    latent = stage_2_state.latent
    final_latent = latent_report(latent)
    if not final_latent["finite"]:
        raise RuntimeError(
            "video latent contains non-finite values: "
            f"nan={final_latent['nan_count']}, inf={final_latent['inf_count']}."
        )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    encode_started = time.perf_counter()
    tiling_config = TilingConfig.default()
    decoded_video = VideoDecoder(
        str(config.checkpoint_path),
        torch.bfloat16,
        device,
    )(latent, tiling_config=tiling_config, generator=generator)
    encode_video(
        video=decoded_video,
        fps=config.fps,
        audio=None,
        output_path=str(config.output_path),
        video_chunks_number=get_video_chunks_number(config.frames, tiling_config),
    )
    torch.cuda.synchronize(device)
    encode_elapsed = time.perf_counter() - encode_started
    if not config.output_path.is_file() or config.output_path.stat().st_size <= 0:
        raise RuntimeError(
            f"video encoder did not produce a non-empty MP4: {config.output_path}."
        )

    report: dict[str, Any] = {
        "schema_version": 2,
        "backend": "SM100Attention",
        "pipeline": "two_stage_video_only",
        "output_path": str(config.output_path),
        "checkpoint_path": str(config.checkpoint_path),
        "spatial_upsampler_path": str(config.spatial_upsampler_path),
        "gemma_root": str(config.gemma_root),
        "prompt": config.prompt,
        "provenance": {
            "ltx_source_commit": LTX_SOURCE_COMMIT,
            "ltx_model_revision": LTX_MODEL_REVISION,
            "gemma_revision": GEMMA_REVISION,
            "torch_version": str(torch.__version__),
        },
        "settings": {
            "dtype": "torch.bfloat16",
            "height": config.height,
            "width": config.width,
            "frames": config.frames,
            "stage_1_steps": stage_1_steps,
            "stage_2_steps": stage_2_steps,
            "seed": config.seed,
            "fps": config.fps,
            "schedule": "two_stage_distilled",
        },
        "shape": geometry,
        "stage_1_latent": stage_1_latent,
        "upsampled_latent": upsampled_latent,
        "latent": final_latent,
        "attention": {
            "expected_calls": required_calls,
            "expected_signatures": expected_signatures,
            "stage_1": {
                "expected_calls": stage_1_required_calls,
                "expected_signatures": stage_1_expected_signatures,
                **stage_1_stats,
            },
            "stage_2": {
                "expected_calls": stage_2_required_calls,
                "expected_signatures": stage_2_expected_signatures,
                **stage_2_stats,
            },
            **attention_stats,
        },
        "cuda": {
            "device_name": device_name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "torch_cuda": torch.version.cuda,
            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        },
        "timing_seconds": {
            "prompt_encode": prompt_elapsed,
            "stage_1_denoise": stage_1_elapsed,
            "spatial_upsample": upsample_elapsed,
            "stage_2_denoise": stage_2_elapsed,
            "decode_and_encode": encode_elapsed,
            "total": time.perf_counter() - started,
        },
    }
    config.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _record_check(
    checks: dict[str, dict[str, Any]],
    name: str,
    ok: bool,
    detail: str,
) -> None:
    checks[name] = {"ok": ok, "detail": detail}


def _run_checks(
    checkpoint_path: Path,
    spatial_upsampler_path: Path,
    gemma_root: Path,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    python_ok = sys.version_info[:2] == (3, 12)
    _record_check(
        checks,
        "python",
        python_ok,
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    nvcc = shutil.which("nvcc")
    nvcc_detail = "not found"
    nvcc_ok = False
    if nvcc is not None:
        completed = subprocess.run(
            [nvcc, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        nvcc_output = (completed.stdout or completed.stderr).strip()
        nvcc_detail = nvcc_output.splitlines()[-1]
        nvcc_ok = (
            completed.returncode == 0
            and re.search(r"release\s+12\.8(?:\D|$)", nvcc_output) is not None
        )
    _record_check(checks, "cuda_toolkit", nvcc_ok, nvcc_detail)

    torch_ok = False
    runtime_ok = False
    hardware_ok = False
    try:
        import torch
    except Exception as exc:
        _record_check(checks, "torch", False, f"import failed: {exc}")
        _record_check(checks, "cuda_runtime", False, "torch unavailable")
        _record_check(checks, "gpu", False, "torch unavailable")
    else:
        torch_version = str(torch.__version__).split("+")[0]
        torch_ok = torch_version == "2.8.0"
        _record_check(checks, "torch", torch_ok, str(torch.__version__))
        runtime_ok = torch.version.cuda == "12.8"
        _record_check(checks, "cuda_runtime", runtime_ok, str(torch.version.cuda))
        if not torch.cuda.is_available():
            _record_check(checks, "gpu", False, "CUDA unavailable")
        else:
            capability = torch.cuda.get_device_capability()
            name = torch.cuda.get_device_name()
            hardware_ok = capability == (10, 0) and "B200" in name.upper()
            _record_check(
                checks,
                "gpu",
                hardware_ok,
                f"{name}, compute capability {capability[0]}.{capability[1]}",
            )

    ltx_core_ok = importlib.util.find_spec("ltx_core") is not None
    ltx_pipelines_ok = importlib.util.find_spec("ltx_pipelines") is not None
    _record_check(
        checks,
        "ltx_core",
        ltx_core_ok,
        "importable" if ltx_core_ok else "not importable",
    )
    _record_check(
        checks,
        "ltx_pipelines",
        ltx_pipelines_ok,
        "importable" if ltx_pipelines_ok else "not importable",
    )

    checkpoint_ok = checkpoint_path.is_file()
    _record_check(checks, "checkpoint", checkpoint_ok, str(checkpoint_path))
    spatial_upsampler_ok = spatial_upsampler_path.is_file()
    _record_check(
        checks,
        "spatial_upsampler",
        spatial_upsampler_ok,
        str(spatial_upsampler_path),
    )
    gemma_ok = (
        gemma_root.is_dir()
        and (gemma_root / "config.json").is_file()
        and next(gemma_root.rglob("*.safetensors"), None) is not None
    )
    _record_check(checks, "gemma", gemma_ok, str(gemma_root))

    if python_ok and nvcc_ok and torch_ok and runtime_ok and hardware_ok:
        try:
            from .backend import ensure_loaded

            library = ensure_loaded(build_if_missing=True, verbose=False)
        except Exception as exc:
            _record_check(checks, "extension", False, str(exc))
        else:
            _record_check(checks, "extension", True, str(library or "operator already loaded"))
    else:
        _record_check(checks, "extension", False, "skipped because runtime checks failed")

    return {"ok": all(check["ok"] for check in checks.values()), "checks": checks}


def _default_output_name(args: argparse.Namespace) -> str:
    return (
        f"ltx23_sm100_two_stage_{args.height}x{args.width}_f{args.frames}"
        f"_s{DEFAULT_STEPS}x{DEFAULT_STAGE_2_STEPS}_seed{args.seed}.mp4"
    )


def _command_build(args: argparse.Namespace) -> int:
    from .build import build_extension

    build_directory = (
        resolve_project_path(args.build_dir) if args.build_dir is not None else None
    )
    library = build_extension(build_directory=build_directory, verbose=args.verbose)
    print(json.dumps({"ok": True, "library_path": str(library)}, indent=2))
    return 0


def _command_check(args: argparse.Namespace) -> int:
    report = _run_checks(
        resolve_project_path(DEFAULT_CHECKPOINT),
        resolve_project_path(DEFAULT_SPATIAL_UPSAMPLER),
        resolve_project_path(DEFAULT_GEMMA_ROOT),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def _command_run(args: argparse.Namespace) -> int:
    output_path = (
        resolve_project_path(args.output)
        if args.output is not None
        else resolve_project_path(DEFAULT_OUTPUT_DIRECTORY / _default_output_name(args))
    )
    config = RunConfig(
        prompt=args.prompt,
        checkpoint_path=resolve_project_path(DEFAULT_CHECKPOINT),
        spatial_upsampler_path=resolve_project_path(DEFAULT_SPATIAL_UPSAMPLER),
        gemma_root=resolve_project_path(DEFAULT_GEMMA_ROOT),
        output_path=output_path,
        height=args.height,
        width=args.width,
        frames=args.frames,
        seed=args.seed,
        fps=args.fps,
        library_path=None,
    )
    import torch

    with torch.inference_mode():
        report = _run_generation(config)
    print(
        json.dumps(
            {
                "ok": True,
                "output_path": report["output_path"],
                "report_path": str(config.report_path),
                "attention_calls": report["attention"]["total_calls"],
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ltx-sm100",
        description="Build, verify, and run NVIDIA B200 attention in LTX-2.3.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build and load the SM100 extension.")
    build.add_argument(
        "--build-dir",
        default=None,
        help="Build directory; relative paths use the repository root.",
    )
    build.add_argument("--verbose", action="store_true", help="Show compiler output.")
    build.set_defaults(handler=_command_build)

    check = subparsers.add_parser("check", help="Check the pinned local runtime.")
    check.set_defaults(handler=_command_check)

    run = subparsers.add_parser(
        "run",
        help="Generate one distilled video without audio.",
    )
    run.add_argument("--prompt", required=True, help="Text prompt for the generated video.")
    run.add_argument(
        "--output",
        default=None,
        help="MP4 path; defaults under outputs/. Relative paths use the repository root.",
    )
    run.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    run.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    run.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--fps", type=int, default=DEFAULT_FPS)
    run.set_defaults(handler=_command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ltx-sm100: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
