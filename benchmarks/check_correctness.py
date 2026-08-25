"""Compile one public kernel and compare it with PyTorch SDPA on a B200."""

from __future__ import annotations

from pathlib import Path
import sys

import modal

REMOTE_BENCHMARK_DIR = Path("/root/b200-attention/benchmarks")
if REMOTE_BENCHMARK_DIR.is_dir():
    sys.path.insert(0, str(REMOTE_BENCHMARK_DIR))

from registry import (
    image,
    load_kernel,
    remote_source,
    resolve_kernel,
    resolve_shape,
    sha256_file,
    source_closure_sha256,
)


app = modal.App("b200-attention-correctness", image=image)

RTOL = 3e-2
ATOL = 3e-2
HEAD_DIM = 128


def _make_inputs(batch: int, heads: int, len_q: int, len_kv: int):
    import torch

    # Match Dao's benchmark_attn.py input sequence.
    torch.manual_seed(0)
    q = torch.randn(
        batch,
        len_q,
        heads,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        batch,
        len_kv,
        heads,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        batch,
        len_kv,
        heads,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, v = [
        tensor.detach().to(torch.bfloat16).requires_grad_(False)
        for tensor in (q, k, v)
    ]
    return tuple(tensor.transpose(1, 2).contiguous() for tensor in (q, k, v))


@app.function(gpu="B200", timeout=60 * 10)
def check_remote(kernel: str, shape: str):
    import torch
    import torch.nn.functional as F

    spec = resolve_kernel(kernel)
    shape_name, batch, heads, len_q, len_kv = resolve_shape(shape)
    kernel_path = remote_source(spec)
    kernel_sha256 = sha256_file(kernel_path)
    source_fingerprint = source_closure_sha256(spec)
    print(
        "CORRECTNESS_CONFIG "
        f"kernel={spec.label} source={spec.source} sha256={kernel_sha256} "
        f"source_closure_sha256={source_fingerprint} "
        f"shape={shape_name} batch={batch} heads={heads} "
        f"len_q={len_q} len_kv={len_kv} rtol={RTOL} atol={ATOL}",
        flush=True,
    )

    module = load_kernel(spec)
    q, k, v = _make_inputs(batch, heads, len_q, len_kv)
    reference = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
    )
    torch.cuda.synchronize()
    actual = module.sdpa(q, k, v)
    torch.cuda.synchronize()

    if not torch.isfinite(actual).all():
        raise RuntimeError("kernel output contains NaN or Inf values")
    torch.testing.assert_close(actual, reference, rtol=RTOL, atol=ATOL)

    difference = (actual - reference).float().abs()
    print(
        "CORRECTNESS_RESULT "
        f"kernel={spec.label} shape={shape_name} status=ok "
        f"max_abs={difference.max().item():.8g} "
        f"mean_abs={difference.mean().item():.8g}",
        flush=True,
    )


@app.local_entrypoint()
def main(kernel: str = "", shape: str = "paper4k"):
    resolve_kernel(kernel)
    resolve_shape(shape)
    check_remote.remote(kernel=kernel, shape=shape)
