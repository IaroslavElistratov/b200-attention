# SPDX-License-Identifier: Apache-2.0

"""Custom SM100 attention backend for LTX-2.3 video inference.

Public objects are imported on first access so package discovery and CLI help
do not import PyTorch or initialize CUDA.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backend import SM100Attention, ensure_loaded, get_stats, reset_stats
    from .build import build_extension, default_build_directory, load_extension

_BACKEND_EXPORTS = frozenset({"SM100Attention", "ensure_loaded", "get_stats", "reset_stats"})
_BUILD_EXPORTS = frozenset({"build_extension", "default_build_directory", "load_extension"})

__all__ = [
    "SM100Attention",
    "build_extension",
    "default_build_directory",
    "ensure_loaded",
    "get_stats",
    "load_extension",
    "reset_stats",
]


def __getattr__(name: str) -> Any:
    if name in _BACKEND_EXPORTS:
        module = import_module(f"{__name__}.backend")
    elif name in _BUILD_EXPORTS:
        module = import_module(f"{__name__}.build")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
