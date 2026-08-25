# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


_STATS_LOCK = threading.Lock()
_TOTAL_CALLS = 0
_SIGNATURES: Counter[str] = Counter()
_UNIQUE_INPUTS: set[str] = set()


def reset_stats() -> None:
    """Reset process-wide successful custom-attention call statistics."""

    global _TOTAL_CALLS
    with _STATS_LOCK:
        _TOTAL_CALLS = 0
        _SIGNATURES.clear()
        _UNIQUE_INPUTS.clear()


def get_stats() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of custom-attention call statistics."""

    with _STATS_LOCK:
        return {
            "total_calls": _TOTAL_CALLS,
            "signatures": dict(_SIGNATURES),
            "unique_inputs": sorted(_UNIQUE_INPUTS),
        }


def ensure_loaded(
    library_path: str | os.PathLike[str] | None = None,
    *,
    build_if_missing: bool = True,
    verbose: bool = False,
) -> Path | None:
    """Load an existing custom-op library or build it on demand.

    Resolution order is an explicit ``library_path``, then
    ``LTX_SM100_ATTENTION_LIB``, then a local JIT build. Merely importing this
    module never loads or compiles CUDA code.
    """

    from .build import load_extension

    return load_extension(
        library_path,
        build_if_missing=build_if_missing,
        verbose=verbose,
    )


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    import torch

    if mask is not None:
        raise RuntimeError("SM100Attention supports dense, unmasked attention only.")
    if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("SM100Attention requires query, key, and value to be torch.Tensor instances.")
    if not isinstance(heads, int) or isinstance(heads, bool) or heads <= 0:
        raise ValueError(f"SM100Attention requires a positive integer head count, got {heads!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("SM100Attention requires CUDA and an NVIDIA B200 GPU.")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("SM100Attention requires CUDA query/key/value tensors.")
    if query.device != key.device or query.device != value.device:
        raise RuntimeError("SM100Attention requires query/key/value on the same CUDA device.")

    capability = torch.cuda.get_device_capability(query.device)
    if capability != (10, 0):
        raise RuntimeError(
            "SM100Attention requires an NVIDIA B200-class SM100 GPU "
            f"(compute capability 10.0), got {capability[0]}.{capability[1]}."
        )
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise RuntimeError(
            "SM100Attention requires BF16 query/key/value tensors; "
            f"got query={query.dtype}, key={key.dtype}, value={value.dtype}."
        )
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise RuntimeError(
            "SM100Attention expects LTX query/key/value shaped [B, T, H*D]; "
            f"got query={tuple(query.shape)}, key={tuple(key.shape)}, value={tuple(value.shape)}."
        )

    batch, query_length, hidden_dim = query.shape
    key_batch, key_value_length, key_hidden_dim = key.shape
    value_batch, value_length, value_hidden_dim = value.shape
    if batch <= 0 or query_length <= 0 or key_value_length <= 0:
        raise RuntimeError("SM100Attention requires non-empty batch and sequence dimensions.")
    if key_batch != batch or value_batch != batch:
        raise RuntimeError(
            "SM100Attention requires matching query/key/value batch dimensions; "
            f"got query={batch}, key={key_batch}, value={value_batch}."
        )
    if value_length != key_value_length:
        raise RuntimeError(
            "SM100Attention requires matching key/value sequence lengths; "
            f"got key={key_value_length}, value={value_length}."
        )
    if key_hidden_dim != hidden_dim or value_hidden_dim != hidden_dim:
        raise RuntimeError(
            "SM100Attention requires matching query/key/value hidden dimensions; "
            f"got query={hidden_dim}, key={key_hidden_dim}, value={value_hidden_dim}."
        )
    if hidden_dim % heads != 0:
        raise RuntimeError(f"SM100Attention cannot split hidden dimension {hidden_dim} across {heads} heads.")

    head_dim = hidden_dim // heads
    if head_dim != 128:
        raise RuntimeError(f"SM100Attention supports head dimension 128 only, got {head_dim}.")
    if query_length % 256 != 0:
        raise RuntimeError(
            f"SM100Attention requires query length divisible by 256, got {query_length}."
        )
    if key_value_length % 128 != 0:
        raise RuntimeError(
            "SM100Attention requires key/value length divisible by 128, "
            f"got {key_value_length}."
        )
    if batch * heads > 2**31 - 1 or query_length > 2**31 - 1 or key_value_length > 2**31 - 1:
        raise RuntimeError("SM100Attention dimensions exceed the kernel launcher's 32-bit integer limits.")

    return batch, query_length, key_value_length, head_dim


def _head_major(tensor: torch.Tensor, batch: int, length: int, heads: int, head_dim: int) -> torch.Tensor:
    return (
        tensor.reshape(batch, length, heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(batch * heads, length, head_dim)
    )


def _record_success(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    batch: int,
    query_length: int,
    key_value_length: int,
    heads: int,
    head_dim: int,
) -> None:
    global _TOTAL_CALLS
    signature = (
        f"B={batch},Lq={query_length},Lkv={key_value_length},H={heads},D={head_dim},"
        f"dtype={query.dtype},device={query.device}"
    )
    input_record = (
        f"q_shape={tuple(query.shape)} q_stride={tuple(query.stride())}; "
        f"k_shape={tuple(key.shape)} k_stride={tuple(key.stride())}; "
        f"v_shape={tuple(value.shape)} v_stride={tuple(value.stride())}"
    )
    with _STATS_LOCK:
        _TOTAL_CALLS += 1
        _SIGNATURES[signature] += 1
        _UNIQUE_INPUTS.add(input_record)


class SM100Attention:
    """Strict LTX unmasked-attention callable backed by the custom SM100 kernel.

    Instances are directly accepted by ``DiffusionStage.with_attention``.
    The shared library is loaded or built lazily on the first valid call.
    """

    label = "SM100Attention"

    def __init__(
        self,
        library_path: str | os.PathLike[str] | None = None,
        *,
        build_if_missing: bool = True,
        verbose_build: bool = False,
    ) -> None:
        self._library_path = Path(library_path).expanduser() if library_path is not None else None
        self._build_if_missing = build_if_missing
        self._verbose_build = verbose_build
        self._loaded = False

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch

        batch, query_length, key_value_length, head_dim = _validate_inputs(
            query,
            key,
            value,
            heads,
            mask,
        )
        if not self._loaded:
            ensure_loaded(
                self._library_path,
                build_if_missing=self._build_if_missing,
                verbose=self._verbose_build,
            )
            self._loaded = True

        query_head_major = _head_major(query, batch, query_length, heads, head_dim)
        key_head_major = _head_major(key, batch, key_value_length, heads, head_dim)
        value_head_major = _head_major(value, batch, key_value_length, heads, head_dim)
        output_head_major = torch.ops.ltx_sm100.attention(
            query_head_major,
            key_head_major,
            value_head_major,
        )
        expected_shape = (batch * heads, query_length, head_dim)
        if tuple(output_head_major.shape) != expected_shape:
            raise RuntimeError(
                "ltx_sm100.attention returned an unexpected shape: "
                f"expected {expected_shape}, got {tuple(output_head_major.shape)}."
            )
        if output_head_major.dtype != torch.bfloat16 or output_head_major.device != query.device:
            raise RuntimeError(
                "ltx_sm100.attention returned an unexpected dtype/device: "
                f"expected {torch.bfloat16} on {query.device}, got "
                f"{output_head_major.dtype} on {output_head_major.device}."
            )

        output = (
            output_head_major.reshape(batch, heads, query_length, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch, query_length, heads * head_dim)
        )
        _record_success(
            query,
            key,
            value,
            batch=batch,
            query_length=query_length,
            key_value_length=key_value_length,
            heads=heads,
            head_dim=head_dim,
        )
        return output


__all__ = ["SM100Attention", "ensure_loaded", "get_stats", "reset_stats"]
