from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ai_core.onnx_runtime import import_onnxruntime

__all__ = [
    "FaceParser",
    "DEFAULT_PARSER_ONNX",
    "REGION_CLASS_INDEX",
    "DEFAULT_FACE_REGIONS",
]

# bisenet_resnet_34.onnx (FaceFusion) ships in onnx/ next to this module.
DEFAULT_PARSER_ONNX: Path = (
    Path(__file__).resolve().parent / "onnx" / "bisenet_resnet_34.onnx"
)

# CelebAMask-HQ 19-class layout used by the BiSeNet face parser.
REGION_CLASS_INDEX: dict[str, int] = {
    "skin": 1,
    "left-eyebrow": 2,
    "right-eyebrow": 3,
    "left-eye": 4,
    "right-eye": 5,
    "glasses": 6,
    "left-ear": 7,
    "right-ear": 8,
    "earring": 9,
    "nose": 10,
    "mouth": 11,
    "upper-lip": 12,
    "lower-lip": 13,
    "neck": 14,
    "necklace": 15,
    "cloth": 16,
    "hair": 17,
    "hat": 18,
}

# Default swap region: facial skin + features only. Excludes hair, glasses, ears,
# neck and hat so those (often occluders) keep the original pixels.
DEFAULT_FACE_REGIONS: tuple[str, ...] = (
    "skin",
    "left-eyebrow",
    "right-eyebrow",
    "left-eye",
    "right-eye",
    "nose",
    "mouth",
    "upper-lip",
    "lower-lip",
)

# ImageNet normalization (the BiSeNet parser was trained with it).
_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class FaceParser:
    """BiSeNet face parser producing a per-face segmentation mask.

    Runs on an aligned RGB face crop and returns a soft (float32 in [0, 1]) mask of
    the selected facial regions, at the crop's own resolution. Used by
    :class:`~ai_core.face_swapping.face_swapper.FaceSwapper` to replace the fixed
    elliptical blend mask with one that hugs the real face and excludes occluders
    (hair, glasses, hands), removing the swap seam.

    Images are handled in RGB to match the rest of the pipeline.
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = DEFAULT_PARSER_ONNX,
        regions: Sequence[str] = DEFAULT_FACE_REGIONS,
        feather_sigma: float = 5.0,
        providers: Sequence[str] | None = None,
        intra_op_num_threads: int | None = None,
    ) -> None:
        self.region_indices = self._resolve_regions(regions)
        self.feather_sigma = max(float(feather_sigma), 0.0)

        self._ort = self._import_onnxruntime()
        self.model_path = self._resolve_model_path(model_path)
        self.session = self._create_session(providers, intra_op_num_threads)
        self._input_name = str(self.session.get_inputs()[0].name)
        self.model_size = self._resolve_model_size()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compute_mask(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Return a soft region mask (float32 [0, 1]) at ``crop_rgb`` resolution.

        ``crop_rgb`` is an aligned RGB face crop (any H x W; resized internally to the
        model size and the mask resized back).
        """
        crop_rgb = self._ensure_rgb_uint8(crop_rgb)
        h, w = crop_rgb.shape[:2]

        class_map = self._parse(crop_rgb)  # (model_h, model_w) int labels
        mask = np.isin(class_map, self.region_indices).astype(np.float32)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return self._feather(mask)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _parse(self, crop_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop_rgb, self.model_size, interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - _MEAN) / _STD
        blob = blob.transpose(2, 0, 1)[None, ...]
        blob = np.ascontiguousarray(blob, dtype=np.float32)

        outputs = self.session.run(None, {self._input_name: blob})
        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.ndim == 4:
            logits = logits[0]  # (num_classes, H, W)
        if logits.ndim != 3:
            raise ValueError(f"Unexpected parser output shape: {logits.shape}")
        return logits.argmax(axis=0)

    def _feather(self, mask: np.ndarray) -> np.ndarray:
        if self.feather_sigma <= 0.0:
            return np.clip(mask, 0.0, 1.0)
        # Soften the boundary inward while keeping the interior solid.
        blurred = cv2.GaussianBlur(mask.clip(0.0, 1.0), (0, 0), self.feather_sigma)
        return ((blurred.clip(0.5, 1.0) - 0.5) * 2.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_regions(regions: Sequence[str]) -> np.ndarray:
        if not regions:
            raise ValueError("regions must not be empty")
        indices: list[int] = []
        for region in regions:
            key = str(region).strip().lower()
            if key not in REGION_CLASS_INDEX:
                raise ValueError(
                    f"Unknown face region {region!r}. "
                    f"Valid regions: {sorted(REGION_CLASS_INDEX)}"
                )
            indices.append(REGION_CLASS_INDEX[key])
        return np.asarray(sorted(set(indices)), dtype=np.int64)

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
        path = Path(model_path) if model_path is not None else DEFAULT_PARSER_ONNX
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
