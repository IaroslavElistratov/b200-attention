from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

import ltx_sm100.backend as backend  # noqa: E402
from ltx_sm100.backend import SM100Attention  # noqa: E402
from ltx_sm100.cli import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_GEMMA_ROOT,
    DEFAULT_SPATIAL_UPSAMPLER,
    project_root,
)


def _require_b200() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("test requires compute capability 10.0")
    if "B200" not in torch.cuda.get_device_name().upper():
        pytest.skip("test requires an NVIDIA B200")
    if torch.version.cuda != "12.8":
        pytest.skip("test requires a CUDA 12.8 PyTorch build")


def test_constructor_does_not_load_or_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("constructor loaded the extension")

    monkeypatch.setattr(backend, "ensure_loaded", unexpected_load)
    attention = SM100Attention()
    assert attention.label == "SM100Attention"


def test_mask_is_rejected_before_any_cuda_or_build_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid input attempted to load the extension")

    monkeypatch.setattr(backend, "ensure_loaded", unexpected_load)
    query = torch.zeros((1, 256, 128), dtype=torch.bfloat16)
    key = torch.zeros((1, 128, 128), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    with pytest.raises(RuntimeError, match="unmasked attention only"):
        SM100Attention()(query, key, value, 1, mask=torch.ones(1))


def test_cpu_tensors_fail_without_loading_or_counting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = False

    def unexpected_load(*_args: object, **_kwargs: object) -> None:
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(backend, "ensure_loaded", unexpected_load)
    backend.reset_stats()
    query = torch.zeros((1, 256, 128), dtype=torch.bfloat16)
    key = torch.zeros((1, 128, 128), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    with pytest.raises(RuntimeError, match="CUDA"):
        SM100Attention()(query, key, value, 1)
    assert loaded is False
    assert backend.get_stats()["total_calls"] == 0


@pytest.mark.b200
@pytest.mark.parametrize(
    ("query_length", "key_length"),
    [
        (6144, 1024),
        (6144, 6144),
        (12288, 1024),
        (12288, 12288),
        (49152, 1024),
        (49152, 49152),
    ],
    ids=[
        "article-6144-cross",
        "article-6144-self",
        "default-stage1-cross",
        "default-stage1-self",
        "default-stage2-cross",
        "default-stage2-self",
    ],
)
def test_b200_kernel_matches_fp32_sdpa(query_length: int, key_length: int) -> None:
    _require_b200()
    torch.manual_seed(7)
    heads = 32
    hidden = heads * 128
    query = torch.randn(
        (1, query_length, hidden),
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.25)
    key = torch.randn(
        (1, key_length, hidden),
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.25)
    value = torch.randn(
        (1, key_length, hidden),
        device="cuda",
        dtype=torch.bfloat16,
    )

    backend.reset_stats()
    actual = SM100Attention()(query, key, value, heads).float()
    query_heads = query.reshape(1, query_length, heads, 128).transpose(1, 2).float()
    key_heads = key.reshape(1, key_length, heads, 128).transpose(1, 2).float()
    value_heads = value.reshape(1, key_length, heads, 128).transpose(1, 2).float()
    reference = torch.nn.functional.scaled_dot_product_attention(
        query_heads,
        key_heads,
        value_heads,
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2).reshape_as(actual)

    difference = actual - reference
    relative_l2 = difference.norm() / reference.norm().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(),
        reference.flatten(),
        dim=0,
    )
    assert torch.isfinite(actual).all()
    assert float(difference.abs().max()) <= 0.15
    assert float(relative_l2) < 0.005
    assert float(cosine) > 0.9999
    assert backend.get_stats()["total_calls"] == 1


@pytest.mark.b200
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("dtype", "BF16"),
        ("head_dimension", "head dimension 128"),
        ("query_alignment", "query length divisible by 256"),
        ("key_alignment", "key/value length divisible by 128"),
    ],
)
def test_b200_contract_rejects_unsupported_inputs(
    case: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_b200()

    def unexpected_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid input attempted to load the extension")

    monkeypatch.setattr(backend, "ensure_loaded", unexpected_load)
    heads = 2
    query_length = 128 if case == "query_alignment" else 256
    key_length = 64 if case == "key_alignment" else 128
    head_dimension = 64 if case == "head_dimension" else 128
    dtype = torch.float32 if case == "dtype" else torch.bfloat16
    hidden = heads * head_dimension
    query = torch.zeros((1, query_length, hidden), device="cuda", dtype=dtype)
    key = torch.zeros((1, key_length, hidden), device="cuda", dtype=dtype)
    value = torch.zeros_like(key)
    with pytest.raises(RuntimeError, match=message):
        SM100Attention()(query, key, value, heads)


@pytest.mark.b200
@pytest.mark.e2e
@pytest.mark.models
def test_default_real_generation_routes_all_attention(tmp_path: Path) -> None:
    _require_b200()
    if os.environ.get("LTX_SM100_RUN_E2E") != "1":
        pytest.skip("set LTX_SM100_RUN_E2E=1 to run the full model smoke test")

    root = project_root()
    checkpoint = root / DEFAULT_CHECKPOINT
    spatial_upsampler = root / DEFAULT_SPATIAL_UPSAMPLER
    gemma_root = root / DEFAULT_GEMMA_ROOT
    if not checkpoint.is_file() or not spatial_upsampler.is_file() or not gemma_root.is_dir():
        pytest.fail("downloaded LTX-2.3, spatial upscaler, and Gemma models are unavailable")

    output = tmp_path / "smoke.mp4"
    from ltx_sm100.cli import main

    result = main(
        [
            "run",
            "--prompt",
            "A tiny paper sailboat glides across a quiet blue pond.",
            "--output",
            str(output),
        ]
    )
    assert result == 0
    assert output.is_file() and output.stat().st_size > 0
    report_path = output.with_suffix(".json")
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pipeline"] == "two_stage_video_only"
    assert report["settings"]["stage_1_steps"] == 8
    assert report["settings"]["stage_2_steps"] == 3
    assert report["shape"]["stage_1"]["video_tokens"] == 12288
    assert report["shape"]["stage_2"]["video_tokens"] == 49152
    assert report["attention"]["expected_calls"] == 1056
    assert report["attention"]["total_calls"] == 1056
    assert report["attention"]["stage_1"]["total_calls"] == 768
    assert report["attention"]["stage_2"]["total_calls"] == 288
    assert report["latent"]["finite"] is True
    assert report["latent"]["nan_count"] == 0
    assert report["latent"]["inf_count"] == 0
