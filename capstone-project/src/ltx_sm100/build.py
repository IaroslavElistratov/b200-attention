# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path


_EXTENSION_NAME = "ltx_sm100_attention"
_OP_NAMESPACE = "ltx_sm100"
_OP_NAME = "attention"
_LOAD_LOCK = threading.RLock()
_LOADED_LIBRARY: Path | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_kernel_directory() -> Path:
    return _project_root().parent / "kernels"


def _kernel_sources() -> tuple[Path, Path]:
    return (
        _project_root() / "kernel" / "binding.cpp",
        _canonical_kernel_directory() / "5_minor" / "18_tma_l2_promotion.cu",
    )


def default_build_directory() -> Path:
    """Return the build directory without creating it."""

    configured = os.environ.get("LTX_SM100_BUILD_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (_project_root() / "build" / "sm100_attention").resolve()


def _operator_available() -> bool:
    import torch

    namespace = getattr(torch.ops, _OP_NAMESPACE)
    return hasattr(namespace, _OP_NAME)


def _verify_operator(source: str) -> None:
    if not _operator_available():
        raise RuntimeError(
            f"Loaded {source}, but it did not register torch.ops.{_OP_NAMESPACE}.{_OP_NAME}. "
            "The library may be stale or built from a different binding."
        )


def _built_library(build_directory: Path) -> Path:
    candidates: list[Path] = []
    for pattern in (f"{_EXTENSION_NAME}*.so", f"{_EXTENSION_NAME}*.dylib", f"{_EXTENSION_NAME}*.dll"):
        candidates.extend(build_directory.glob(pattern))
    if not candidates:
        raise RuntimeError(f"Torch built the SM100 extension but no shared library was found in {build_directory}.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).resolve()


def build_extension(
    build_directory: str | os.PathLike[str] | None = None,
    *,
    verbose: bool = False,
) -> Path:
    """Build and load the custom op for compute_100a/sm_100a.

    Compilation uses the currently imported PyTorch installation and its C++
    extension ABI. A CUDA toolkit new enough to understand ``sm_100a`` is
    required. The pinned environment uses CUDA 12.8.
    """

    global _LOADED_LIBRARY
    with _LOAD_LOCK:
        if _LOADED_LIBRARY is not None and _operator_available():
            return _LOADED_LIBRARY

        sources = _kernel_sources()
        kernel_directory = _canonical_kernel_directory()
        required_files = (
            *sources,
            kernel_directory / "common.cuh",
            kernel_directory / "approximations.cuh",
            kernel_directory / "smem_layout_swizzle128.cuh",
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing SM100 extension source files: {', '.join(missing)}")

        output_dir = (
            Path(build_directory).expanduser().resolve()
            if build_directory is not None
            else default_build_directory()
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        previous_cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
        previous_path = os.environ.get("PATH")
        environment_bin = Path(sys.executable).parent
        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
        if (environment_bin / "ninja").is_file():
            os.environ["PATH"] = os.pathsep.join(
                part for part in (str(environment_bin), previous_path) if part
            )
        try:
            from torch.utils.cpp_extension import load

            load(
                name=_EXTENSION_NAME,
                sources=[str(source) for source in sources],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3", "-std=c++17"],
                extra_include_paths=[str(kernel_directory)],
                extra_ldflags=["-lcuda"],
                build_directory=str(output_dir),
                verbose=verbose,
                with_cuda=True,
                is_python_module=False,
                keep_intermediates=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to build the SM100 attention extension. Use an NVIDIA B200 environment "
                "with a CUDA toolkit that supports compute_100a/sm_100a (tested with CUDA 12.8), "
                "Ninja, and a CUDA-enabled PyTorch installation."
            ) from exc
        finally:
            if previous_cuda_arch_list is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous_cuda_arch_list
            if previous_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = previous_path

        _verify_operator("the freshly built SM100 extension")
        _LOADED_LIBRARY = _built_library(output_dir)
        return _LOADED_LIBRARY


def load_extension(
    library_path: str | os.PathLike[str] | None = None,
    *,
    build_if_missing: bool = True,
    verbose: bool = False,
) -> Path | None:
    """Load a canonical shared library, or build one when no path is supplied."""

    global _LOADED_LIBRARY
    configured = library_path or os.environ.get("LTX_SM100_ATTENTION_LIB")
    with _LOAD_LOCK:
        if configured is not None:
            requested = Path(configured).expanduser().resolve()
            if not requested.is_file():
                raise FileNotFoundError(f"SM100 attention library not found: {requested}")
            if _operator_available():
                if _LOADED_LIBRARY is None or _LOADED_LIBRARY == requested:
                    return _LOADED_LIBRARY
                raise RuntimeError(
                    f"SM100 attention is already loaded from {_LOADED_LIBRARY}; "
                    f"cannot also load {requested} in the same process."
                )

            import torch

            torch.ops.load_library(str(requested))
            _verify_operator(str(requested))
            _LOADED_LIBRARY = requested
            return requested

        if _operator_available():
            return _LOADED_LIBRARY
        if not build_if_missing:
            raise RuntimeError(
                "The SM100 attention operator is not loaded. Supply a canonical library path, "
                "set LTX_SM100_ATTENTION_LIB, or enable build_if_missing."
            )
        return build_extension(verbose=verbose)


__all__ = ["build_extension", "default_build_directory", "load_extension"]
