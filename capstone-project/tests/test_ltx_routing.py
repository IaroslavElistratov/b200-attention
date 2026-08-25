from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from ltx_sm100 import cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _isolated_python(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-S", *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_import_needs_no_torch_or_ltx_installation() -> None:
    completed = _isolated_python(
        "-c",
        (
            "import sys; import ltx_sm100.cli; "
            "assert 'torch' not in sys.modules; "
            "assert 'ltx_core' not in sys.modules; "
            "assert 'ltx_pipelines' not in sys.modules"
        ),
    )
    assert completed.returncode == 0, completed.stderr


def test_cli_help_needs_no_torch_or_ltx_installation() -> None:
    completed = _isolated_python("-m", "ltx_sm100.cli", "--help")
    assert completed.returncode == 0, completed.stderr
    assert "{build,check,run}" in completed.stdout
    assert "NVIDIA B200" in completed.stdout


def test_download_models_help_is_clean() -> None:
    completed = subprocess.run(
        ["bash", "scripts/download_models.sh", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert ".cache/huggingface" in completed.stdout


def test_run_parser_exposes_only_video_settings() -> None:
    args = cli.build_parser().parse_args(["run", "--prompt", "a glass airship"])
    assert (args.height, args.width) == (1024, 1536)
    assert args.frames == 249
    assert not hasattr(args, "steps")
    assert cli.DEFAULT_STAGE_2_STEPS == 3
    assert args.seed == 4242
    assert args.fps == 24
    assert not hasattr(args, "checkpoint")
    assert not hasattr(args, "spatial_upsampler")
    assert not hasattr(args, "gemma_root")
    assert not hasattr(args, "library")


def test_default_two_stage_shapes_satisfy_kernel_alignment() -> None:
    report = cli.two_stage_shape_report(1024, 1536, 249, context_length=1024)
    assert report == {
        "stage_1": {
            "latent_frames": 32,
            "latent_height": 16,
            "latent_width": 24,
            "video_tokens": 12288,
            "video_tokens_mod_256": 0,
            "context_length": 1024,
            "context_length_mod_128": 0,
        },
        "stage_2": {
            "latent_frames": 32,
            "latent_height": 32,
            "latent_width": 48,
            "video_tokens": 49152,
            "video_tokens_mod_256": 0,
            "context_length": 1024,
            "context_length_mod_128": 0,
        },
    }


def test_two_stage_shape_requires_target_multiples_of_64() -> None:
    with pytest.raises(ValueError, match="multiples of 64"):
        cli.two_stage_shape_report(992, 1536, 249)


@pytest.mark.parametrize(
    ("height", "width", "frames", "context_length", "message"),
    [
        (513, 768, 249, None, "divisible by 32"),
        (512, 768, 248, None, "8\\*k \\+ 1"),
        (32, 32, 249, None, "query length"),
        (512, 768, 249, 1000, "context length"),
    ],
)
def test_shape_contract_rejects_unsupported_inputs(
    height: int,
    width: int,
    frames: int,
    context_length: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        cli.shape_report(
            height,
            width,
            frames,
            context_length=context_length,
        )


def test_relative_paths_are_anchored_to_project_not_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.resolve_project_path("models/checkpoint.safetensors") == (
        PROJECT_ROOT / "models/checkpoint.safetensors"
    )


def test_video_only_state_dict_mapping_name() -> None:
    assert cli.VIDEO_ONLY_SD_OPS_NAME == "LTX_VIDEO_ONLY_COMFY_PREFIX_MAP"


def test_public_stage_hook_receives_the_exact_custom_callable() -> None:
    attention = object()
    replacement = object()

    class FakeStage:
        received: object | None = None

        def with_attention(self, value: object) -> object:
            self.received = value
            return replacement

    stage = FakeStage()
    assert cli.pin_sm100_attention(stage, attention) is replacement
    assert stage.received is attention


def test_stage_hook_must_be_functional() -> None:
    class BrokenStage:
        def with_attention(self, _value: object) -> "BrokenStage":
            return self

    stage = BrokenStage()
    with pytest.raises(RuntimeError, match="unmodified stage"):
        cli.pin_sm100_attention(stage, object())


def test_default_run_requires_exactly_1056_custom_calls() -> None:
    assert cli.expected_attention_calls(8, 3) == 1056
    expected_signatures = cli.expected_attention_signature_counts(
        ((12288, 8), (49152, 3)),
        1024,
        device="cuda:0",
    )
    assert sorted(expected_signatures.values()) == [144, 144, 384, 384]
    stats = {"total_calls": 1056, "signatures": expected_signatures}
    cli.validate_attention_stats(stats, 1056, expected_signatures)
    with pytest.raises(RuntimeError, match="No fallback result"):
        cli.validate_attention_stats({"total_calls": 1055}, 1056)
    with pytest.raises(RuntimeError, match="signature check"):
        cli.validate_attention_stats(
            {"total_calls": 1056, "signatures": {}},
            1056,
            expected_signatures,
        )


def test_run_dispatch_uses_inference_mode_and_keeps_runtime_lazy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[cli.RunConfig] = []
    inference_active = False

    @contextmanager
    def fake_inference_mode():
        nonlocal inference_active
        inference_active = True
        try:
            yield
        finally:
            inference_active = False

    def fake_run(config: cli.RunConfig) -> dict[str, object]:
        assert inference_active
        seen.append(config)
        return {
            "output_path": str(config.output_path),
            "attention": {"total_calls": 1056},
        }

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=fake_inference_mode),
    )
    monkeypatch.setattr(cli, "_run_generation", fake_run)
    output = tmp_path / "result.mp4"
    result = cli.main(
        [
            "run",
            "--prompt",
            "a glass airship",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert inference_active is False
    assert len(seen) == 1
    assert seen[0].output_path == output
    assert seen[0].spatial_upsampler_path == PROJECT_ROOT / cli.DEFAULT_SPATIAL_UPSAMPLER
    assert seen[0].report_path == output.with_suffix(".json")
    assert json.loads(capsys.readouterr().out)["attention_calls"] == 1056


def test_runtime_error_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    @contextmanager
    def fake_inference_mode():
        yield

    def fail(_config: cli.RunConfig) -> dict[str, object]:
        raise RuntimeError("routing failure")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=fake_inference_mode),
    )
    monkeypatch.setattr(cli, "_run_generation", fail)
    result = cli.main(["run", "--prompt", "test"])
    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "routing failure" in captured.err
    assert "Traceback" not in captured.err
