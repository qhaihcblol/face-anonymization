from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Sequence

import cv2
import numpy as np

from ai_core.face_swapping.face_swapper import DEFAULT_SOURCE_FACE, FaceSwapper
from ai_core.face_detection.face_detector import FaceDetection, FaceLandmarks

if TYPE_CHECKING:
    from ai_core.face_alignment.face_aligner import AlignedFace, FaceAligner
    from ai_core.face_parsing.face_parser import FaceParser

SOURCE_FACE = DEFAULT_SOURCE_FACE


class AnonymizationMethod(Enum):
    NONE = "none"
    BLUR = "blur"
    PIXELATE = "pixelate"
    MASK = "mask"
    BLACKOUT = "blackout"
    SWAP = "swap"


@dataclass(slots=True)
class ObfuscationParams:
    """Tunable knobs for the no-model obfuscation methods (blur / pixelate / mask).

    Pulled out of :class:`FaceAnonymizer` so a single (heavyweight) anonymizer can be
    reused across many videos while each run picks its own strengths via
    ``FaceAnonymizer.anonymize(..., params=...)`` — no need to rebuild the instance
    (and reload the parser/aligner) per edit. Values are normalized on construction so
    callers can pass raw UI input without pre-validating it.
    """

    blur_strength: int = 31
    pixelation_level: int = 16
    mask_color: tuple[int, int, int] = (160, 160, 160)
    # Hardening so blur/pixelate cannot be recovered (quantize + unstored noise).
    irreversible: bool = True
    noise_strength: float = 12.0
    quantization_levels: int = 8
    # Soft-edge width (px std-dev) for the ellipse fallback mask.
    mask_feather: float = 4.0

    def __post_init__(self) -> None:
        blur = max(int(self.blur_strength), 3)
        if blur % 2 == 0:
            blur += 1  # cv2.GaussianBlur requires an odd kernel size
        self.blur_strength = blur
        self.pixelation_level = max(int(self.pixelation_level), 4)
        self.mask_color = tuple(int(np.clip(c, 0, 255)) for c in self.mask_color)
        self.irreversible = bool(self.irreversible)
        self.noise_strength = max(float(self.noise_strength), 0.0)
        self.quantization_levels = max(int(self.quantization_levels), 0)
        self.mask_feather = max(float(self.mask_feather), 0.0)


class FaceAnonymizer:
    def __init__(
        self,
        blur_strength: int = 31,
        pixelation_level: int = 16,
        mask_color: tuple[int, int, int] = (160, 160, 160),
        face_swapper: FaceSwapper | None = None,
        face_parser: "FaceParser | None" = None,
        face_aligner: "FaceAligner | None" = None,
        irreversible: bool = True,
        noise_strength: float = 12.0,
        quantization_levels: int = 8,
        mask_feather: float = 4.0,
    ) -> None:
        # Default obfuscation knobs for this instance. ``anonymize(..., params=...)``
        # overrides them per call, so one anonymizer can serve many videos/edits.
        self.params = ObfuscationParams(
            blur_strength=blur_strength,
            pixelation_level=pixelation_level,
            mask_color=mask_color,
            irreversible=irreversible,
            noise_strength=noise_strength,
            quantization_levels=quantization_levels,
            mask_feather=mask_feather,
        )
        self.face_swapper = face_swapper

        # Optional precise face-region masking: a BiSeNet parser run on an aligned
        # crop replaces the coarse ellipse with a mask that hugs the real face.
        # Requires an aligner to produce the crop and warp the mask back to frame.
        self.face_parser = face_parser
        self.face_aligner = face_aligner

        # Unseeded RNG on purpose: the noise must not be reproducible, otherwise it
        # could be subtracted back out.
        self._rng = np.random.default_rng()

    @staticmethod
    def _valid_bbox(
        image: np.ndarray,
        det: dict[str, Any],
    ) -> tuple[int, int, int, int] | None:
        """Clip a detection's bbox to the frame, or None if missing/degenerate."""
        h, w = image.shape[:2]
        bbox = np.asarray(det.get("bbox", []), dtype=np.float32)
        if bbox.shape != (4,):
            return None

        x1, y1, x2, y2 = bbox
        x1 = int(np.clip(np.floor(x1), 0, max(w - 1, 0)))
        y1 = int(np.clip(np.floor(y1), 0, max(h - 1, 0)))
        x2 = int(np.clip(np.ceil(x2), 1, w))
        y2 = int(np.clip(np.ceil(y2), 1, h))

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _iter_valid_bboxes(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> list[tuple[int, int, int, int]]:
        boxes: list[tuple[int, int, int, int]] = []
        for det in detections:
            bbox = self._valid_bbox(image, det)
            if bbox is not None:
                boxes.append(bbox)
        return boxes

    @staticmethod
    def _landmarks_from(det: dict[str, Any]) -> np.ndarray | None:
        """Extract a (5, 2) landmark array from a detection/track dict, or None."""
        raw = det.get("landmarks")
        if raw is None:
            return None
        points = np.asarray(raw, dtype=np.float32)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            return None
        return points

    def _ellipse_face_mask(
        self,
        bbox: tuple[int, int, int, int],
        shape: tuple[int, int],
        params: ObfuscationParams,
    ) -> np.ndarray:
        """Soft elliptical mask (float32 [0, 1]) for one bbox — the coarse fallback."""
        x1, y1, x2, y2 = bbox
        mask = np.zeros(shape, dtype=np.float32)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        axis_x = max((x2 - x1) // 2, 1)
        axis_y = max(int((y2 - y1) * 0.58), 1)
        cv2.ellipse(mask, (cx, cy), (axis_x, axis_y), 0, 0, 360, 1.0, -1)
        if params.mask_feather > 0.0:
            mask = cv2.GaussianBlur(mask, (0, 0), params.mask_feather)
        return mask

    def _parser_face_mask(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        bbox: tuple[int, int, int, int],
        shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Precise face-region mask (float32 [0, 1]) via align -> parser -> warp back.

        Returns None on any failure (degenerate landmarks, empty parse) so the caller
        can fall back to the ellipse and never under-cover the face.
        """
        if self.face_parser is None or self.face_aligner is None:
            return None

        h, w = shape
        try:
            detection = FaceDetection(
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                score=1.0,
                landmarks=FaceLandmarks(
                    left_eye=(float(landmarks[0, 0]), float(landmarks[0, 1])),
                    right_eye=(float(landmarks[1, 0]), float(landmarks[1, 1])),
                    nose=(float(landmarks[2, 0]), float(landmarks[2, 1])),
                    left_mouth=(float(landmarks[3, 0]), float(landmarks[3, 1])),
                    right_mouth=(float(landmarks[4, 0]), float(landmarks[4, 1])),
                ),
            )
            aligned = self.face_aligner.align_detection(detection)
            crop_bgr = self.face_aligner.warp_face(image, aligned.matrix)
            # Frames flow through this path in BGR; the parser expects RGB.
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            region = self.face_parser.compute_mask(crop_rgb)
        except (ValueError, cv2.error):
            return None

        # The parser found (almost) no face in the crop -> let the ellipse cover it.
        if float(region.sum()) < 16.0:
            return None

        inverse = self.face_aligner.invert_matrix(aligned.matrix)
        warped = cv2.warpAffine(
            np.asarray(region, dtype=np.float32),
            inverse,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return np.clip(warped, 0.0, 1.0)

    def _region_mask(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
        params: ObfuscationParams,
    ) -> np.ndarray:
        """Combined soft face mask (float32 [0, 1]) over all detections.

        Each face uses the BiSeNet parser when landmarks and a parser are available
        (precise, hugs the real face), otherwise an elliptical bound. The parser mask
        also falls back to the ellipse on failure, so the face is never under-covered.
        Faces are unioned (per-pixel max).
        """
        shape = image.shape[:2]
        mask = np.zeros(shape, dtype=np.float32)
        for det in detections:
            bbox = self._valid_bbox(image, det)
            if bbox is None:
                continue

            face_mask: np.ndarray | None = None
            landmarks = self._landmarks_from(det)
            if landmarks is not None:
                face_mask = self._parser_face_mask(image, landmarks, bbox, shape)
            if face_mask is None:
                face_mask = self._ellipse_face_mask(bbox, shape, params)
            mask = np.maximum(mask, face_mask)
        return mask

    @staticmethod
    def _composite(
        image: np.ndarray,
        content: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Alpha-blend ``content`` over ``image`` using a soft float mask."""
        alpha = mask[..., None]
        blended = image.astype(np.float32) * (1.0 - alpha) + content.astype(
            np.float32
        ) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _destroy(
        self, region: np.ndarray, params: ObfuscationParams
    ) -> np.ndarray:
        """Make an obfuscated region non-recoverable.

        Quantization is a lossy, non-linear step that discards the fine gradient
        information a deconvolution would need (and it survives temporal averaging
        across video frames). Additive, unseeded, unstored noise then makes any
        inverse problem ill-posed. Together they stop blur/pixelation from being
        decoded back to the original face.
        """
        if not params.irreversible:
            return region

        out = region.astype(np.float32)
        if params.quantization_levels > 0:
            step = 256.0 / params.quantization_levels
            out = np.floor(out / step) * step + step / 2.0
        if params.noise_strength > 0.0:
            out = out + self._rng.normal(0.0, params.noise_strength, size=out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _none(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
        params: ObfuscationParams,
    ) -> np.ndarray:
        return image

    def _blur(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
        params: ObfuscationParams,
    ) -> np.ndarray:
        mask = self._region_mask(image, detections, params)
        if not np.any(mask):
            return image

        blurred = cv2.GaussianBlur(image, (params.blur_strength, params.blur_strength), 0)
        blurred = self._destroy(blurred, params)
        return self._composite(image, blurred, mask)

    def _pixelate(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
        params: ObfuscationParams,
    ) -> np.ndarray:
        mask = self._region_mask(image, detections, params)
        if not np.any(mask):
            return image

        # Mosaic the whole bbox region, then composite only the soft face mask back so
        # the pixelation follows the real face boundary instead of a rectangle.
        pixelated = image.copy()
        for x1, y1, x2, y2 in self._iter_valid_bboxes(image, detections):
            face = pixelated[y1:y2, x1:x2]
            h, w = face.shape[:2]
            if h < 2 or w < 2:
                continue

            scale = max(min(h, w) // params.pixelation_level, 1)
            small_w = max(w // scale, 1)
            small_h = max(h // scale, 1)

            small = cv2.resize(
                face, (small_w, small_h), interpolation=cv2.INTER_LINEAR
            )
            # Degrade the low-res block averages (the part that still leaks identity)
            # before upscaling, so the mosaic cannot be reconstructed.
            small = self._destroy(small, params)
            blocks = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            pixelated[y1:y2, x1:x2] = blocks
        return self._composite(image, pixelated, mask)

    def _mask(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
        params: ObfuscationParams,
    ) -> np.ndarray:
        mask = self._region_mask(image, detections, params)
        if not np.any(mask):
            return image

        solid = np.empty_like(image)
        solid[:] = params.mask_color
        return self._composite(image, solid, mask)

    def _blackout(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
        params: ObfuscationParams,
    ) -> np.ndarray:
        mask = self._region_mask(image, detections, params)
        if not np.any(mask):
            return image

        return self._composite(image, np.zeros_like(image), mask)

    def swap_face(
        self,
        image: np.ndarray,
        aligned_faces: Sequence["AlignedFace"],
        source_blob: np.ndarray | None = None,
    ) -> np.ndarray:
        """Replace every aligned face in ``image`` with a source identity.

        Uses the BlendSwap (blendface) model via the configured
        :class:`~ai_core.face_swapping.face_swapper.FaceSwapper`. ``aligned_faces`` are
        the target faces (as produced by ``FaceAligner``); ``source_blob`` selects the
        identity to paste (from ``FaceSwapper.prepare_source``), or ``None`` for the
        swapper's bundled default (``source_img.png``).
        """
        if self.face_swapper is None:
            raise RuntimeError(
                "Face swap requires a FaceSwapper. Construct FaceAnonymizer with "
                "face_swapper=FaceSwapper(detector=...)."
            )
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")
        return self.face_swapper.swap_face(image, aligned_faces, source_blob)

    def anonymize(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
        method: AnonymizationMethod | str,
        params: ObfuscationParams | None = None,
    ) -> np.ndarray:
        """Obfuscate every detected face in ``image`` with ``method``.

        ``params`` overrides this instance's default :class:`ObfuscationParams` for
        this call only (so one anonymizer can serve many videos with different
        strengths); pass ``None`` to use the instance defaults.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")

        method_value = method
        if isinstance(method, str):
            method_value = AnonymizationMethod(method.strip().lower())

        if not isinstance(method_value, AnonymizationMethod):
            raise TypeError("method must be AnonymizationMethod or str")

        if method_value is AnonymizationMethod.SWAP:
            raise ValueError(
                "SWAP operates on aligned faces, not bbox detections. "
                "Call FaceAnonymizer.swap_face(image, aligned_faces) instead."
            )

        resolved = params if params is not None else self.params
        method_map = {
            AnonymizationMethod.NONE: self._none,
            AnonymizationMethod.BLUR: self._blur,
            AnonymizationMethod.PIXELATE: self._pixelate,
            AnonymizationMethod.MASK: self._mask,
            AnonymizationMethod.BLACKOUT: self._blackout,
        }
        return method_map[method_value](image, detections, resolved)