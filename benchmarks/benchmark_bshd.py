"""Compare Kernel 18 on native BHLD and direct BSHD with stock FA4."""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import modal


REMOTE_BENCHMARK_DIR = Path("/root/b200-attention/benchmarks")

if REMOTE_BENCHMARK_DIR.is_dir():
    sys.path.insert(0, str(REMOTE_BENCHMARK_DIR))
from registry import image as public_image  # noqa: E402
from registry import (  # noqa: E402
    PAPER_FA4_TFLOPS,
    load_kernel,
    remote_source,
    resolve_kernel,
    resolve_shape,
)


REMOTE_KERNEL_DIR = Path("/root/b200-attention/kernels")
REMOTE_CANDIDATE = REMOTE_KERNEL_DIR / "variants" / "18_tma_l2_promotion_bshd.cu"

image = public_image
app = modal.App("b200-attention-k18-bshd-benchmark", image=image)

HEAD_DIM = 128
INPUT_SEED = 0
WARMUP_MS = 5
REPETITION_MS = 10
RTOL = 3e-2
ATOL = 3e-2
ARM_ORDERS = {
    "fa4_first": ("fa4", "direct_bshd", "native_bhld"),
    "bshd_first": ("direct_bshd", "native_bhld", "fa4"),
    "bhld_first": ("native_bhld", "fa4", "direct_bshd"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _full_precision(value: float) -> str:
    return format(float(value), ".17g")


def _gpu_telemetry() -> str:
    fields = (
        "pstate",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.draw",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
    )
    names = (
        "pstate",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_draw_w",
        "temperature_c",
        "gpu_util_pct",
        "memory_util_pct",
    )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        values = [value.strip() for value in output.splitlines()[0].split(",")]
        return ",".join(f"{name}={value}" for name, value in zip(names, values))
    except Exception:
        return "<unavailable>"


def _check_stock_fa4_mode() -> None:
    value = os.environ.get("FA_DISABLE_2CTA", "").strip().lower()
    if value not in ("", "0", "false", "no", "off"):
        raise RuntimeError("FA_DISABLE_2CTA must be unset for stock FA4")


def _make_fa4_fn(q, k, v):
    from flash_attn.cute import flash_attn_func

    parameters = inspect.signature(flash_attn_func).parameters
    kwargs = {}
    if "qv" in parameters:
        kwargs["qv"] = None
    if "causal" in parameters:
        kwargs["causal"] = False
    if "window_size" in parameters:
        kwargs["window_size"] = (None, None)
    if "learnable_sink" in parameters:
        kwargs["learnable_sink"] = None
    if "softcap" in parameters:
        kwargs["softcap"] = 0.0
    if "num_splits" in parameters:
        kwargs["num_splits"] = 0
    if "pack_gqa" in parameters:
        kwargs["pack_gqa"] = None
    if "gather_kv_indices" in parameters:
        kwargs["gather_kv_indices"] = None
    if "dropout_p" in parameters:
        kwargs["dropout_p"] = 0.0
    elif "dropout" in parameters:
        kwargs["dropout"] = 0.0
    if "return_lse" in parameters:
        kwargs["return_lse"] = False

    def run():
        output = flash_attn_func(q, k, v, **kwargs)
        return output[0] if isinstance(output, tuple) else output

    return run


def _write_direct_glue() -> Path:
    path = Path("/tmp/b200_attention_k18_direct_bshd_benchmark_binding.cpp")
    path.write_text(
        textwrap.dedent(
            r"""
            #include <torch/extension.h>
            #include <ATen/cuda/CUDAContext.h>
            #include <c10/cuda/CUDAGuard.h>

            #include <cuda_bf16.h>
            #include <cuda_runtime.h>

            void attention_tcgen05_v18_bshd_launch(
                const nv_bfloat16* Q_ptr,
                const nv_bfloat16* K_ptr,
                const nv_bfloat16* V_ptr,
                nv_bfloat16* O_ptr,
                int batch,
                int heads,
                int len_q,
                int len_kv,
                cudaStream_t stream);

            at::Tensor sdpa(
                const at::Tensor& Q,
                const at::Tensor& K,
                const at::Tensor& V) {
              TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                          "Q/K/V must be CUDA tensors");
              TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
                          "Q/K/V must be on the same CUDA device");
              TORCH_CHECK(Q.scalar_type() == at::kBFloat16 &&
                              K.scalar_type() == at::kBFloat16 &&
                              V.scalar_type() == at::kBFloat16,
                          "Q/K/V must be BF16");
              TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                          "Q/K/V must have shape [B,L,H,D]");
              TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                          "direct-BSHD benchmark performs no input copy");
              TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0),
                          "Q/K/V batch sizes must match");
              TORCH_CHECK(Q.size(2) == K.size(2) && Q.size(2) == V.size(2),
                          "Q/K/V head counts must match");
              TORCH_CHECK(K.size(1) == V.size(1), "K/V lengths must match");
              TORCH_CHECK(Q.size(3) == 128 && K.size(3) == 128 && V.size(3) == 128,
                          "head dimension must be 128");

              const int batch = static_cast<int>(Q.size(0));
              const int len_q = static_cast<int>(Q.size(1));
              const int heads = static_cast<int>(Q.size(2));
              const int len_kv = static_cast<int>(K.size(1));

              c10::cuda::CUDAGuard guard(Q.device());
              auto O = at::empty_like(Q);
              attention_tcgen05_v18_bshd_launch(
                  reinterpret_cast<const nv_bfloat16*>(Q.data_ptr()),
                  reinterpret_cast<const nv_bfloat16*>(K.data_ptr()),
                  reinterpret_cast<const nv_bfloat16*>(V.data_ptr()),
                  reinterpret_cast<nv_bfloat16*>(O.data_ptr()),
                  batch,
                  heads,
                  len_q,
                  len_kv,
                  at::cuda::getCurrentCUDAStream());
              return O;
            }

            TORCH_LIBRARY(b200_attention_k18_direct_bshd_benchmark, library) {
              library.def("sdpa(Tensor Q, Tensor K, Tensor V) -> Tensor");
              library.impl("sdpa", &sdpa);
            }
            """
        ),
        encoding="utf-8",
    )
    return path


def _load_direct_kernel():
    import torch
    import torch.utils.cpp_extension

    torch.utils.cpp_extension.load(
        "module_b200_attention_k18_direct_bshd_benchmark",
        sources=[str(_write_direct_glue()), str(REMOTE_CANDIDATE)],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        extra_include_paths=[str(REMOTE_KERNEL_DIR)],
        extra_ldflags=["-lcuda"],
        verbose=True,
        is_python_module=False,
    )
    return torch.ops.b200_attention_k18_direct_bshd_benchmark


# Modal only provides the remote B200; the same benchmark can run on any B200 provider.
@app.function(gpu="B200", timeout=60 * 10)
def benchmark_remote(arm_order: str, shape: str):
    import torch
    from triton.testing import do_bench

    try:
        sequence = ARM_ORDERS[arm_order]
    except KeyError as exc:
        raise ValueError(f"unknown arm order: {arm_order}") from exc
    _check_stock_fa4_mode()

    original_spec = resolve_kernel("18")
    original_module = load_kernel(original_spec)
    direct_module = _load_direct_kernel()

    shape_name, batch, heads, len_q, len_kv = resolve_shape(shape)
    print(
        "K18_BSHD_CONFIG "
        f"device={torch.cuda.get_device_name()} "
        f"capability={torch.cuda.get_device_capability()} "
        f"shape={shape_name} batch={batch} heads={heads} "
        f"len_q={len_q} len_kv={len_kv} "
        f"arm_order={arm_order} warmup_ms={WARMUP_MS} rep_ms={REPETITION_MS} "
        f"original_sha256={_sha256(remote_source(original_spec))} "
        f"direct_sha256={_sha256(REMOTE_CANDIDATE)}",
        flush=True,
    )

    # Compile both kernels before reproducing the publication input sequence.
    torch.manual_seed(INPUT_SEED)
    q_bshd = torch.randn(
        batch, len_q, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    k_bshd = torch.randn(
        batch, len_kv, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    v_bshd = torch.randn(
        batch, len_kv, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    q_bhld, k_bhld, v_bhld = (
        tensor.transpose(1, 2).contiguous()
        for tensor in (q_bshd, k_bshd, v_bshd)
    )
    assert q_bshd.is_contiguous() and k_bshd.is_contiguous() and v_bshd.is_contiguous()
    assert q_bhld.is_contiguous() and k_bhld.is_contiguous() and v_bhld.is_contiguous()

    fa4_fn = _make_fa4_fn(q_bshd, k_bshd, v_bshd)
    direct_fn = lambda: direct_module.sdpa(q_bshd, k_bshd, v_bshd)
    native_fn = lambda: original_module.sdpa(q_bhld, k_bhld, v_bhld)

    fa4_output = fa4_fn()
    direct_output = direct_fn()
    native_output_bshd = native_fn().transpose(1, 2)
    torch.cuda.synchronize()
    torch.testing.assert_close(direct_output, fa4_output, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(direct_output, native_output_bshd, rtol=RTOL, atol=ATOL)
    difference = (direct_output - fa4_output).float().abs()
    print(
        "K18_BSHD_CORRECTNESS "
        f"shape={shape_name} status=ok max_abs={difference.max().item():.8g} "
        f"mean_abs={difference.mean().item():.8g}",
        flush=True,
    )

    functions = {
        "fa4": fa4_fn,
        "direct_bshd": direct_fn,
        "native_bhld": native_fn,
    }
    elapsed_ms = {}
    for arm in sequence:
        time.sleep(1.0)
        elapsed_ms[arm] = float(
            do_bench(
                functions[arm],
                warmup=WARMUP_MS,
                rep=REPETITION_MS,
                return_mode="mean",
            )
        )
        print(
            "K18_BSHD_ARM "
            f"shape={shape_name} arm={arm} "
            f"elapsed_ms={_full_precision(elapsed_ms[arm])} "
            f"post_arm_gpu={_gpu_telemetry()}",
            flush=True,
        )

    direct_vs_native = elapsed_ms["native_bhld"] / elapsed_ms["direct_bshd"] * 100.0
    direct_vs_fa4 = elapsed_ms["fa4"] / elapsed_ms["direct_bshd"] * 100.0
    native_vs_fa4 = elapsed_ms["fa4"] / elapsed_ms["native_bhld"] * 100.0
    operations = 4 * batch * heads * len_q * len_kv * HEAD_DIM
    direct_tflops = operations / (elapsed_ms["direct_bshd"] * 1e-3) * 1e-12
    native_tflops = operations / (elapsed_ms["native_bhld"] * 1e-3) * 1e-12
    print(
        "K18_BSHD_RESULT "
        f"shape={shape_name} fa4_ms={_full_precision(elapsed_ms['fa4'])} "
        f"direct_bshd_ms={_full_precision(elapsed_ms['direct_bshd'])} "
        f"native_bhld_ms={_full_precision(elapsed_ms['native_bhld'])} "
        f"direct_pct_native={_full_precision(direct_vs_native)} "
        f"direct_pct_fa4={_full_precision(direct_vs_fa4)} "
        f"native_pct_fa4={_full_precision(native_vs_fa4)} "
        f"direct_pct_paper={_full_precision(100.0 * direct_tflops / PAPER_FA4_TFLOPS[shape_name])} "
        f"native_pct_paper={_full_precision(100.0 * native_tflops / PAPER_FA4_TFLOPS[shape_name])}",
        flush=True,
    )


@app.local_entrypoint()
def main(shape: str = "paper4k", arm_order: str = "fa4_first"):
    if arm_order not in ARM_ORDERS:
        raise ValueError(f"--arm-order must be one of: {', '.join(ARM_ORDERS)}")
    resolve_shape(shape)
    benchmark_remote.remote(arm_order=arm_order, shape=shape)
