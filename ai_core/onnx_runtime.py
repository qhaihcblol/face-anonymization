"""Shared ONNX Runtime loader.

Importing ``onnxruntime`` is centralised here so the CUDA / cuDNN shared
libraries shipped in the ``nvidia-*-cu12`` pip wheels are preloaded exactly
once, *before* any ``InferenceSession`` is created.

Why this matters: those wheels install their ``.so`` files under
``site-packages/nvidia/<pkg>/lib/``, which is **not** on the dynamic linker
search path. Without preloading, onnxruntime's ``libonnxruntime_providers_cuda.so``
cannot resolve its dependency (e.g. ``libcublasLt.so.12``), the CUDA execution
provider silently fails to register, and onnxruntime falls back to the CPU — so
a GPU box ends up running every model on the CPU.

``onnxruntime.preload_dlls()`` dlopens those libs from the wheel directories so
the provider loads. Every ai_core module that builds an ``InferenceSession``
imports onnxruntime through :func:`import_onnxruntime` so the preload happens no
matter which model loads first (web app, CLI, or tests).
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, cast

logger = logging.getLogger(__name__)

# Cache the module so the (idempotent) preload runs only on the first import.
_ort: Any | None = None


def import_onnxruntime() -> Any:
    """Return the ``onnxruntime`` module, preloading CUDA deps on first call.

    Raises:
        ImportError: if onnxruntime is not installed.
    """
    global _ort
    if _ort is not None:
        return _ort

    try:
        module = cast(Any, importlib.import_module("onnxruntime"))
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required. Install `onnxruntime` or `onnxruntime-gpu`."
        ) from exc

    # Preload the CUDA/cuDNN libs from the nvidia-*-cu12 wheels so the CUDA
    # execution provider can load. A no-op on CPU-only installs and on older
    # onnxruntime that predates this helper; never block the CPU fallback.
    preload = getattr(module, "preload_dlls", None)
    if callable(preload):
        try:
            preload()
        except Exception as exc:  # noqa: BLE001 - preload must never be fatal
            logger.warning("onnxruntime.preload_dlls() failed: %s", exc)

    _ort = module
    return module
