"""Pair one public kernel with stock FlashAttention-4 on a Modal B200."""

from __future__ import annotations

import importlib.metadata
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

REMOTE_BENCHMARK_DIR = Path("/root/b200-attention/benchmarks")
if REMOTE_BENCHMARK_DIR.is_dir():
    sys.path.insert(0, str(REMOTE_BENCHMARK_DIR))

from registry import (
    PAPER_FA4_TFLOPS,
    image,
    load_kernel,
    remote_source,
    resolve_kernel,
    resolve_shape,
    sha256_file,
    source_closure_sha256,
)


app = modal.App("b200-attention-benchmark", image=image)

HEAD_DIM = 128
INPUT_SEED = 0
WARMUP_MS = 5
REPETITION_MS = 10
ARM_ORDERS = ("fa4_first", "local_first")


def _resolve_arm_order(value: str) -> str:
    arm_order = value.strip().lower()
    if arm_order not in ARM_ORDERS:
        raise ValueError(f"--arm-order must be one of: {', '.join(ARM_ORDERS)}")
    return arm_order


def _full_precision(value: float) -> str:
    return format(float(value), ".17g")


def _driver_version() -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return output.splitlines()[0].strip()
    except Exception:
        return "<unavailable>"


def _tool_version(name: str) -> str:
    try:
        output = subprocess.check_output(
            [f"/usr/local/cuda/bin/{name}", "--version"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
        return output.splitlines()[-1].strip().replace(" ", "_")
    except Exception:
        return "<unavailable>"


def _gpu_telemetry() -> str:
    fields = (
        "pstate",
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks.max.sm",
        "power.draw",
        "power.limit",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
    )
    names = (
        "pstate",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "max_sm_clock_mhz",
        "power_draw_w",
        "power_limit_w",
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
        if len(values) != len(names):
            return "<unavailable>"
        return ",".join(f"{name}={value}" for name, value in zip(names, values))
    except Exception:
        return "<unavailable>"


def _check_stock_fa4_mode() -> None:
    value = os.environ.get("FA_DISABLE_2CTA", "").strip().lower()
    if value not in ("", "0", "false", "no", "off"):
        raise RuntimeError(
            "FA_DISABLE_2CTA must be unset or false for the stock FA4 denominator"
        )


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


def _dense_forward_flops(
    batch: int,
    heads: int,
    len_q: int,
    len_kv: int,
) -> int:
    return 4 * batch * heads * len_q * len_kv * HEAD_DIM


@app.function(gpu="B200", timeout=60 * 10)
def benchmark_remote(kernel: str, shape: str, arm_order: str):
    import flash_attn
    import torch
    from flash_attn import cute as fa4_cute
    from triton.testing import do_bench

    _check_stock_fa4_mode()
    arm_order = _resolve_arm_order(arm_order)
    spec = resolve_kernel(kernel)
    shape_name, batch, heads, len_q, len_kv = resolve_shape(shape)
    kernel_sha256 = sha256_file(remote_source(spec))
    source_fingerprint = source_closure_sha256(spec)

    print(
        "BENCHMARK_ENV "
        f"device={torch.cuda.get_device_name()} "
        f"capability={torch.cuda.get_device_capability()} "
        f"driver={_driver_version()} torch={torch.__version__} "
        f"torch_cuda={torch.version.cuda} "
        f"flash_attn={importlib.metadata.version('flash-attn-4')} "
        f"flash_attn_path={flash_attn.__file__} fa4_cute_path={fa4_cute.__file__} "
        f"nvcc={_tool_version('nvcc')} ptxas={_tool_version('ptxas')}",
        flush=True,
    )
    print(
        "BENCHMARK_CONFIG "
        f"kernel={spec.label} source={spec.source} sha256={kernel_sha256} "
        f"source_closure_sha256={source_fingerprint} "
        f"shape={shape_name} batch={batch} heads={heads} "
        f"len_q={len_q} len_kv={len_kv} arm_order={arm_order} "
        f"warmup_ms={WARMUP_MS} repetition_ms={REPETITION_MS}",
        flush=True,
    )

    module = load_kernel(spec)

    torch.manual_seed(INPUT_SEED)
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
    # The numbered lineage materializes native BHLD here. Direct-BSHD Kernel 18
    # skips these copies; see benchmarks/README.md#input-layout-and-direct-bshd.
    q_local, k_local, v_local = tuple(
        tensor.transpose(1, 2).contiguous() for tensor in (q, k, v)
    )

    fa4_fn = _make_fa4_fn(q, k, v)

    def local_fn():
        return module.sdpa(q_local, k_local, v_local)

    functions = {"fa4": fa4_fn, "local": local_fn}
    sequence = (
        ("fa4", "local") if arm_order == "fa4_first" else ("local", "fa4")
    )
    elapsed_ms = {}
    for arm in sequence:
        # Match Dao: do_bench owns all warmup and timed launches.
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
            "BENCHMARK_ARM "
            f"kernel={spec.label} shape={shape_name} arm={arm} "
            f"elapsed_ms={_full_precision(elapsed_ms[arm])} "
            f"post_arm_gpu={_gpu_telemetry()}",
            flush=True,
        )

    operations = _dense_forward_flops(batch, heads, len_q, len_kv)
    local_tflops = operations / (elapsed_ms["local"] * 1e-3) * 1e-12
    fa4_tflops = operations / (elapsed_ms["fa4"] * 1e-3) * 1e-12
    percent_stock = local_tflops / fa4_tflops * 100.0
    paper_tflops = PAPER_FA4_TFLOPS[shape_name]
    percent_paper = local_tflops / paper_tflops * 100.0

    print(
        "BENCHMARK_RESULT "
        f"kernel={spec.label} shape={shape_name} arm_order={arm_order} status=ok "
        f"local_ms={_full_precision(elapsed_ms['local'])} "
        f"local_tflops={_full_precision(local_tflops)} "
        f"fa4_ms={_full_precision(elapsed_ms['fa4'])} "
        f"fa4_tflops={_full_precision(fa4_tflops)} "
        f"percent_stock_fa4={_full_precision(percent_stock)} "
        f"percent_paper_fa4={_full_precision(percent_paper)}",
        flush=True,
    )
    print(
        "\n| Kernel | Shape | Local ms | Local TFLOPS | % Stock FA4 | % Paper FA4 |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| {spec.label} | {shape_name} | {elapsed_ms['local']:.4f} | "
        f"{local_tflops:.2f} | {percent_stock:.2f}% | {percent_paper:.2f}% |"
    )


@app.local_entrypoint()
def main(
    kernel: str = "",
    shape: str = "paper4k",
    arm_order: str = "fa4_first",
):
    resolve_kernel(kernel)
    resolve_shape(shape)
    _resolve_arm_order(arm_order)
    benchmark_remote.remote(kernel=kernel, shape=shape, arm_order=arm_order)
