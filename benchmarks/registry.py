"""Public kernel registry and Modal CUDA-extension loader."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import textwrap

import modal


LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_BENCHMARK_DIR = LOCAL_REPO_ROOT / "benchmarks"
LOCAL_KERNEL_DIR = LOCAL_REPO_ROOT / "kernels"
REMOTE_REPO_ROOT = Path("/root/b200-attention")
REMOTE_BENCHMARK_DIR = REMOTE_REPO_ROOT / "benchmarks"
REMOTE_KERNEL_DIR = REMOTE_REPO_ROOT / "kernels"


# Keep the benchmark software environment pinned. Installing flash-attn-4
# without dependencies avoids resolver drift in its CuTe stack.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "torch==2.9.1",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .uv_pip_install(
        "ninja",
        "numpy",
        "packaging",
        "psutil",
        "setuptools",
        "wheel",
        "nvidia-cutlass-dsl[cu13]==4.4.2",
    )
    .uv_pip_install(
        "quack-kernels==0.2.10",
        "apache-tvm-ffi",
        "einops",
        "torch-c-dlpack-ext",
    )
    .uv_pip_install("triton==3.5.1")
    .run_commands("python -m pip install --upgrade setuptools wheel")
    .run_commands(
        "python -m pip install --no-build-isolation --no-deps "
        "flash-attn-4==4.0.0b4"
    )
    .add_local_dir(
        LOCAL_BENCHMARK_DIR,
        remote_path=REMOTE_BENCHMARK_DIR,
        ignore=["__pycache__/**", "*.pyc"],
    )
    .add_local_dir(LOCAL_KERNEL_DIR, remote_path=REMOTE_KERNEL_DIR)
)


@dataclass(frozen=True)
class KernelSpec:
    number: int
    label: str
    source: str
    launch_symbol: str
    len_q_multiple: int

    @property
    def op_namespace(self) -> str:
        return f"b200_attention_k{self.number}"


KERNELS = (
    KernelSpec(1, "1_baseline", "1_basic/1_baseline.cu", "attention_tcgen05_v1_launch", 128),
    KernelSpec(2, "2_p_to_tmem", "1_basic/2_p_to_tmem.cu", "attention_tcgen05_v2_launch", 128),
    KernelSpec(3, "3_swizzle", "1_basic/3_swizzle.cu", "attention_tcgen05_v3_launch", 128),
    KernelSpec(
        4,
        "4_warp_specialization",
        "1_basic/4_warp_specialization.cu",
        "attention_tcgen05_v4_launch",
        128,
    ),
    KernelSpec(
        5,
        "5_two_q_tiles",
        "2_coarse-schedule/5_two_q_tiles.cu",
        "attention_tcgen05_v5_launch",
        256,
    ),
    KernelSpec(
        6,
        "6_load_pipeline",
        "2_coarse-schedule/6_load_pipeline.cu",
        "attention_tcgen05_v6_launch",
        256,
    ),
    KernelSpec(
        7,
        "7_compute_pipeline",
        "2_coarse-schedule/7_compute_pipeline.cu",
        "attention_tcgen05_v7_launch",
        256,
    ),
    KernelSpec(
        8,
        "8_hardware_approx_exp2",
        "3_hotloop_optimizations/1_approximations/8_hardware_approx_exp2.cu",
        "attention_tcgen05_v8_launch",
        256,
    ),
    KernelSpec(
        9,
        "9_selective_software_approx_exp2",
        "3_hotloop_optimizations/1_approximations/9_selective_software_approx_exp2.cu",
        "attention_tcgen05_v9_launch",
        256,
    ),
    KernelSpec(
        10,
        "10_cache96_reread32",
        "3_hotloop_optimizations/2_avoid_tmem_reread/10_cache96_reread32.cu",
        "attention_tcgen05_v10_launch",
        256,
    ),
    KernelSpec(
        11,
        "11_skip_o_rescale",
        "3_hotloop_optimizations/3_skip_o_rescale/11_skip_o_rescale.cu",
        "attention_tcgen05_v11_launch",
        256,
    ),
    KernelSpec(
        12,
        "12_split_correction_from_softmax",
        "3_hotloop_optimizations/4_early_pv/12_split_correction_from_softmax.cu",
        "attention_tcgen05_v12_launch",
        256,
    ),
    KernelSpec(
        13,
        "13_early_pv_96_32",
        "3_hotloop_optimizations/4_early_pv/13_early_pv_96_32.cu",
        "attention_tcgen05_v13_launch",
        256,
    ),
    KernelSpec(
        14,
        "14_persistent",
        "4_persistent/14_persistent.cu",
        "attention_tcgen05_v14_launch",
        256,
    ),
    KernelSpec(
        15,
        "15_wide_deferred_score_loads",
        "5_minor/15_wide_deferred_score_loads.cu",
        "attention_tcgen05_v15_launch",
        256,
    ),
    KernelSpec(
        16,
        "16_split_rowsum_accumulation",
        "5_minor/16_split_rowsum_accumulation.cu",
        "attention_tcgen05_v16_launch",
        256,
    ),
    KernelSpec(
        17,
        "17_phasebits",
        "5_minor/17_phasebits.cu",
        "attention_tcgen05_v17_launch",
        256,
    ),
    KernelSpec(
        18,
        "18_tma_l2_promotion",
        "5_minor/18_tma_l2_promotion.cu",
        "attention_tcgen05_v18_launch",
        256,
    ),
)


SHAPES = {
    "paper4k": (8, 16, 4096, 4096),
    "paper8k": (4, 16, 8192, 8192),
    "paper16k": (2, 16, 16384, 16384),
}


PAPER_FA4_TFLOPS = {
    "paper4k": 1532.0,
    "paper8k": 1579.0,
    "paper16k": 1601.0,
}


def resolve_kernel(value: str) -> KernelSpec:
    key = value.strip()
    if not key:
        raise ValueError("Pass --kernel with a number from 1 through 18")

    for spec in KERNELS:
        aliases = {str(spec.number), spec.label, spec.source, Path(spec.source).stem}
        if key in aliases:
            return spec

    raise ValueError(f"Unknown kernel {value!r}; choose a number from 1 through 18")


def resolve_shape(value: str) -> tuple[str, int, int, int, int]:
    name = value.strip().lower()
    try:
        batch, heads, len_q, len_kv = SHAPES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown shape {value!r}; choose one of: {', '.join(SHAPES)}"
        ) from exc
    return name, batch, heads, len_q, len_kv


def remote_source(spec: KernelSpec) -> Path:
    return REMOTE_KERNEL_DIR / spec.source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_closure_sha256(spec: KernelSpec) -> str:
    files = [spec.source, "common.cuh"]
    files.append(
        "smem_layout_noswizzle.cuh"
        if spec.number <= 2
        else "smem_layout_swizzle128.cuh"
    )
    if spec.number >= 8:
        files.append("approximations.cuh")

    digest = hashlib.sha256()
    for relative_path in files:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with (REMOTE_KERNEL_DIR / relative_path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_glue(spec: KernelSpec) -> Path:
    glue_path = Path("/tmp") / f"b200_attention_k{spec.number}_binding.cpp"
    glue_path.write_text(
        textwrap.dedent(
            f"""
            #include <torch/extension.h>
            #include <ATen/cuda/CUDAContext.h>
            #include <c10/cuda/CUDAGuard.h>

            #include <cuda_bf16.h>
            #include <cuda_runtime.h>

            #include <climits>

            using AttentionFn = void(
                const nv_bfloat16* Q_ptr,
                const nv_bfloat16* K_ptr,
                const nv_bfloat16* V_ptr,
                nv_bfloat16* O_ptr,
                int batch_heads,
                int len_q,
                int len_kv,
                cudaStream_t stream);

            AttentionFn {spec.launch_symbol};

            at::Tensor sdpa(
                const at::Tensor& Q,
                const at::Tensor& K,
                const at::Tensor& V) {{
              TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                          "Q/K/V must be CUDA tensors");
              TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
                          "Q/K/V must be on the same CUDA device");
              TORCH_CHECK(Q.scalar_type() == at::kBFloat16 &&
                              K.scalar_type() == at::kBFloat16 &&
                              V.scalar_type() == at::kBFloat16,
                          "Q/K/V must be BF16");
              TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                          "Q/K/V must have shape [batch, heads, length, dim]");
              TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0),
                          "Q/K/V batch sizes must match");
              TORCH_CHECK(Q.size(1) == K.size(1) && Q.size(1) == V.size(1),
                          "Q/K/V head counts must match");
              TORCH_CHECK(K.size(2) == V.size(2),
                          "K/V sequence lengths must match");
              TORCH_CHECK(Q.size(3) == 128 && K.size(3) == 128 && V.size(3) == 128,
                          "Q/K/V head dimension must be 128");

              const int64_t batch = Q.size(0);
              const int64_t heads = Q.size(1);
              const int64_t len_q = Q.size(2);
              const int64_t len_kv = K.size(2);
              TORCH_CHECK(batch > 0 && heads > 0,
                          "batch and head counts must be positive");
              TORCH_CHECK(batch <= INT_MAX / heads,
                          "batch * heads exceeds 32-bit launcher limits");
              const int64_t batch_heads = batch * heads;
              TORCH_CHECK(len_q >= {spec.len_q_multiple},
                          "query length is smaller than this kernel's tile");
              TORCH_CHECK(len_kv >= 128,
                          "key/value length must be at least 128");
              TORCH_CHECK(len_q % {spec.len_q_multiple} == 0,
                          "kernel {spec.number} requires aligned query length");
              TORCH_CHECK(len_kv % 128 == 0,
                          "kernels require len_kv divisible by 128");
              TORCH_CHECK(batch_heads <= INT_MAX && len_q <= INT_MAX && len_kv <= INT_MAX,
                          "kernel dimensions exceed 32-bit launcher limits");
              TORCH_CHECK(batch_heads <= INT_MAX / len_q &&
                              batch_heads <= INT_MAX / len_kv,
                          "flattened tensor heights exceed 32-bit limits");

              c10::cuda::CUDAGuard guard(Q.device());
              auto Qc = Q.contiguous();
              auto Kc = K.contiguous();
              auto Vc = V.contiguous();
              auto O = at::empty_like(Qc);

              {spec.launch_symbol}(
                  reinterpret_cast<const nv_bfloat16*>(Qc.data_ptr()),
                  reinterpret_cast<const nv_bfloat16*>(Kc.data_ptr()),
                  reinterpret_cast<const nv_bfloat16*>(Vc.data_ptr()),
                  reinterpret_cast<nv_bfloat16*>(O.data_ptr()),
                  static_cast<int>(batch_heads),
                  static_cast<int>(len_q),
                  static_cast<int>(len_kv),
                  at::cuda::getCurrentCUDAStream());
              return O;
            }}

            TORCH_LIBRARY({spec.op_namespace}, library) {{
              library.def("sdpa(Tensor Q, Tensor K, Tensor V) -> Tensor");
              library.impl("sdpa", &sdpa);
            }}
            """
        ),
        encoding="utf-8",
    )
    return glue_path


def load_kernel(spec: KernelSpec):
    import torch
    import torch.utils.cpp_extension

    torch.utils.cpp_extension.load(
        f"module_{spec.op_namespace}",
        sources=[str(_write_glue(spec)), str(remote_source(spec))],
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
    return getattr(torch.ops, spec.op_namespace)
