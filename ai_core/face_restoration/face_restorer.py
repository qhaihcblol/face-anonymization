from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ai_core.onnx_runtime import import_onnxruntime

__all__ = ["FaceRestorer", "DEFAULT_RESTORER_ONNX"]

# gfpgan_1.4.onnx (FaceFusion) ships in onnx/ next to this module.
DEFAULT_RESTORER_ONNX: Path = (
    Path(__file__).resolve().parent / "onnx" / "gfpgan_1.4.onnx"
)


class FaceRestorer:
    """GFPGAN face restorer that sharpens a swapped face crop.

    BlendSwap outputs a soft, low-detail 256x256 face. This runs GFPGAN on the aligned
    crop to regenerate realistic skin texture/detail, then blends the restored result
    back so the swapped face matches the sharpness of its surroundings.

    Operates on an aligned RGB face crop and returns a restored crop at the *same*
    resolution, so it drops into :meth:`FaceSwapper.swap_aligned` without touching the
    paste-back geometry. Images are handled in RGB to match the rest of the pipeline.
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = DEFAULT_RESTORER_ONNX,
        blend: float = 0.8,
        providers: Sequence[str] | None = None,
        intra_op_num_threads: int | None = None,
    ) -> None:
        # blend: how much of the restored face to mix back (1 = full restore,
        # 0 = original). Keeping a little of the original avoids over-processing.
        self.blend = float(np.clip(blend, 0.0, 1.0))

        self._ort = self._import_onnxruntime()
        self.model_path = self._resolve_model_path(model_path)
        self.session = self._create_session(providers, intra_op_num_threads)
        self._input_name = str(self.session.get_inputs()[0].name)
        self.model_size = self._resolve_model_size()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def restore(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Restore an aligned RGB face crop, returning a crop of the same size."""
        crop_rgb = self._ensure_rgb_uint8(crop_rgb)
        h, w = crop_rgb.shape[:2]

        restored = self._run(crop_rgb)  # RGB uint8 at model size
        restored = cv2.resize(restored, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.blend < 1.0:
            restored = cv2.addWeighted(
                crop_rgb, 1.0 - self.blend, restored, self.blend, 0.0
            )
        return restored

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _run(self, crop_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop_rgb, self.model_size, interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - 0.5) / 0.5  # -> [-1, 1]
        blob = blob.transpose(2, 0, 1)[None, ...]
        blob = np.ascontiguousarray(blob, dtype=np.float32)

        outputs = self.session.run(None, {self._input_name: blob})
        result = np.asarray(outputs[0], dtype=np.float32)
        if result.ndim == 4:
            result = result[0]  # (3, H, W)
        if result.ndim != 3:
            raise ValueError(f"Unexpected restorer output shape: {result.shape}")

        result = np.clip(result, -1.0, 1.0)
        result = (result + 1.0) / 2.0
        result = result.transpose(1, 2, 0)
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")
        if image.dtype == np.uint8:
            return image
        return np.clip(image, 0, 255).astype(np.uint8, copy=False)

    @staticmethod
    def _import_onnxruntime() -> Any:
        return import_onnxruntime()

    @staticmethod
    def _resolve_model_path(model_path: str | Path | None) -> Path:
        path = Path(model_path) if model_path is not None else DEFAULT_RESTORER_ONNX
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        return path

    def _create_session(
        self,
        providers: Sequence[str] | None,
        intra_op_num_threads: int | None,
    ) -> Any:
        options = self._ort.SessionOptions()
        options.graph_optimization_level = (
            self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if isinstance(intra_op_num_threads, int) and intra_op_num_threads > 0:
            options.intra_op_num_threads = int(intra_op_num_threads)

        available = set(self._ort.get_available_providers())
        requested = list(providers) if providers is not None else [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        resolved = [str(name) for name in requested if str(name) in available]
        if not resolved:
            raise RuntimeError(
                "No requested ONNX Runtime provider is available. "
                f"Requested={requested}, Available={sorted(available)}"
            )

        return self._ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=resolved,
        )

    def _resolve_model_size(self) -> tuple[int, int]:
        shape = list(self.session.get_inputs()[0].shape)
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            if shape[2] > 0 and shape[3] > 0:
                return int(shape[3]), int(shape[2])  # (width, height) for cv2.resize
        return (512, 512)
