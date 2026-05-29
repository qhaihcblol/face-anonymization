from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Sequence, cast

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignedFace, AlignMode, FaceAligner
from ai_core.face_detection.face_detector import (
    FaceDetection,
    FaceDetector,
    FaceLandmarks,
)

__all__ = ["FaceSwapper", "DEFAULT_SOURCE_FACE"]

# source_img.png ships next to this module.
DEFAULT_SOURCE_FACE: Path = Path(__file__).resolve().parent / "source_img.png"

# blendswap_256.onnx lives in the FaceFusion model hub.
_HF_REPO_ID = "facefusion/models-3.0.0"
_HF_FILENAME = "blendswap_256.onnx"


class FaceSwapper:
    """Face swapper backed by the BlendSwap (blendface) ONNX model.

    BlendSwap takes two inputs:
      * ``source`` : the identity face aligned to the ArcFace 112x112 template.
      * ``target`` : the face to be replaced, aligned to the FFHQ 256x256 template.

    The system :class:`FaceAligner` templates are identical to the ones BlendSwap
    was trained on (``insightface`` == ``arcface_112_v2`` and ``ffhq`` == FaceFusion's
    ``ffhq_512``), so the aligner is reused directly for both crops.

    All images are handled in RGB to match the rest of the pipeline
    (``FaceDetector`` / ``FaceAligner`` are fed RGB frames).
    """

    # BlendSwap normalization: RGB, scaled to [0, 1] (mean 0 / std 1).
    _SOURCE_SIZE: tuple[int, int] = (112, 112)
    _TARGET_SIZE: tuple[int, int] = (256, 256)

    def __init__(
        self,
        detector: FaceDetector,
        *,
        model_path: str | Path | None = None,
        source_path: str | Path = DEFAULT_SOURCE_FACE,
        mask_blur: float = 0.1,
        providers: Sequence[str] | None = None,
        intra_op_num_threads: int | None = None,
        hf_repo_id: str = _HF_REPO_ID,
        hf_filename: str = _HF_FILENAME,
    ) -> None:
        if not isinstance(detector, FaceDetector):
            raise TypeError("detector must be a FaceDetector")

        self.detector = detector
        self.source_path = Path(source_path)
        self.mask_blur = float(np.clip(mask_blur, 0.0, 0.49))

        self.source_aligner = FaceAligner(self._SOURCE_SIZE, AlignMode.INSIGHTFACE)
        self.target_aligner = FaceAligner(self._TARGET_SIZE, AlignMode.FFHQ)

        self._ort = self._import_onnxruntime()
        self.model_path = self._resolve_model_path(model_path, hf_repo_id, hf_filename)
        self.session = self._create_session(providers, intra_op_num_threads)
        self._source_input_name, self._target_input_name = self._resolve_input_names()

        # Lazily prepared identity tensor (1, 3, 112, 112).
        self._source_blob: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def prepare_source(self, source_image: np.ndarray | None = None) -> np.ndarray:
        """Detect and align the identity face, returning the cached source blob.

        ``source_image`` is expected in RGB. When ``None`` the image referenced by
        ``self.source_path`` is loaded (and converted BGR->RGB).
        """
        if source_image is None:
            bgr = cv2.imread(str(self.source_path))
            if bgr is None:
                raise FileNotFoundError(f"Cannot read source image: {self.source_path}")
            source_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        source_image = self._ensure_rgb_uint8(source_image)
        detections = self.detector.detect(source_image)
        if not detections:
            raise ValueError(f"No face detected in source image: {self.source_path}")

        best = max(detections, key=lambda det: det.score)
        _, crop = self.source_aligner.align_and_warp(source_image, best)
        self._source_blob = self._to_blob(crop)
        return self._source_blob

    def swap_face(
        self,
        image: np.ndarray,
        aligned_faces: Sequence[AlignedFace],
    ) -> np.ndarray:
        """Swap every aligned face in ``image`` with the source identity.

        Parameters
        ----------
        image:
            Full RGB frame the ``aligned_faces`` were detected in.
        aligned_faces:
            Faces to replace, as produced by :class:`FaceAligner`. Each carries the
            affine ``matrix`` and aligned ``landmarks`` used to recover the original
            5-point landmarks, so any aligner mode/size is accepted.

        Returns
        -------
        np.ndarray
            A copy of ``image`` with all detected faces swapped.
        """
        image = self._ensure_rgb_uint8(image)
        if not aligned_faces:
            return image.copy()

        source_blob = self._source_blob
        if source_blob is None:
            source_blob = self.prepare_source()

        output = image.copy()
        for aligned in aligned_faces:
            detection = self._recover_detection(aligned)
            target_aligned, target_crop = self.target_aligner.align_and_warp(
                image, detection
            )
            swapped_crop = self._run_model(source_blob, target_crop)
            output = self._paste_back(output, swapped_crop, target_aligned.matrix)
        return output

    # ------------------------------------------------------------------ #
    # Model I/O
    # ------------------------------------------------------------------ #
    def _run_model(self, source_blob: np.ndarray, target_crop: np.ndarray) -> np.ndarray:
        target_blob = self._to_blob(target_crop)
        outputs = self.session.run(
            None,
            {
                self._source_input_name: source_blob,
                self._target_input_name: target_blob,
            },
        )
        result = np.asarray(outputs[0], dtype=np.float32)[0]  # (3, H, W)
        result = result.transpose(1, 2, 0)
        result = np.clip(result, 0.0, 1.0) * 255.0
        return result.astype(np.uint8)

    def _to_blob(self, crop: np.ndarray) -> np.ndarray:
        # RGB crop -> (1, 3, H, W) float32 in [0, 1]; mean 0 / std 1.
        blob = crop.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]
        return np.ascontiguousarray(blob, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def _recover_detection(self, aligned: AlignedFace) -> FaceDetection:
        """Recover original-frame landmarks/bbox from an aligned face."""
        aligned_points = aligned.landmarks.as_array()
        original_points = FaceAligner.inverse_transform_points(
            aligned_points, aligned.matrix
        )
        original_bbox = FaceAligner.inverse_transform_bbox(aligned.bbox, aligned.matrix)
        landmarks = FaceLandmarks(
            left_eye=(float(original_points[0, 0]), float(original_points[0, 1])),
            right_eye=(float(original_points[1, 0]), float(original_points[1, 1])),
            nose=(float(original_points[2, 0]), float(original_points[2, 1])),
            left_mouth=(float(original_points[3, 0]), float(original_points[3, 1])),
            right_mouth=(float(original_points[4, 0]), float(original_points[4, 1])),
        )
        return FaceDetection(
            bbox=original_bbox,
            score=float(aligned.score),
            landmarks=landmarks,
        )

    def _paste_back(
        self,
        base: np.ndarray,
        swapped_crop: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """Warp the swapped crop back to frame space and feather-blend it in."""
        inverse = FaceAligner.invert_matrix(matrix)
        h, w = base.shape[:2]

        warped = cv2.warpAffine(
            swapped_crop,
            inverse,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        crop_mask = self._feather_mask(swapped_crop.shape[:2])
        warped_mask = cv2.warpAffine(
            crop_mask,
            inverse,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        alpha = warped_mask[..., None]
        blended = base.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _feather_mask(self, size: tuple[int, int]) -> np.ndarray:
        """Soft-edged box mask (float32 in [0, 1]) for seamless blending."""
        h, w = int(size[0]), int(size[1])
        mask = np.ones((h, w), dtype=np.float32)
        border = max(int(round(min(h, w) * self.mask_blur)), 1)
        mask[:border, :] = 0.0
        mask[-border:, :] = 0.0
        mask[:, :border] = 0.0
        mask[:, -border:] = 0.0
        kernel = border * 2 + 1
        return cv2.GaussianBlur(mask, (kernel, kernel), 0)

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
        try:
            return cast(Any, importlib.import_module("onnxruntime"))
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required. Install `onnxruntime` or `onnxruntime-gpu`."
            ) from exc

    @staticmethod
    def _resolve_model_path(
        model_path: str | Path | None,
        repo_id: str,
        filename: str,
    ) -> Path:
        if model_path is not None:
            path = Path(model_path)
            if not path.is_file():
                raise FileNotFoundError(f"ONNX model not found: {path}")
            return path

        try:
            hub = importlib.import_module("huggingface_hub")
        except ImportError as exc:
            raise ImportError(
                "huggingface-hub is required to auto-download the BlendSwap model. "
                "Install `huggingface-hub` or pass model_path explicitly."
            ) from exc

        return Path(hub.hf_hub_download(repo_id=repo_id, filename=filename))

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

    def _resolve_input_names(self) -> tuple[str, str]:
        """Map the two model inputs to (source, target) by name or spatial size."""
        source_name: str | None = None
        target_name: str | None = None

        for inp in self.session.get_inputs():
            name = str(inp.name)
            lowered = name.lower()
            width = inp.shape[-1] if isinstance(inp.shape[-1], int) else None

            if "source" in lowered or width == self._SOURCE_SIZE[0]:
                source_name = name
            elif "target" in lowered or width == self._TARGET_SIZE[0]:
                target_name = name

        inputs = self.session.get_inputs()
        if len(inputs) != 2:
            raise ValueError(
                f"BlendSwap model must have exactly 2 inputs, got {len(inputs)}"
            )
        # Fall back to declared order: BlendSwap exports [source, target].
        if source_name is None:
            source_name = str(inputs[0].name)
        if target_name is None:
            target_name = str(inputs[1].name)
        if source_name == target_name:
            raise ValueError("Could not distinguish source and target model inputs")
        return source_name, target_name
