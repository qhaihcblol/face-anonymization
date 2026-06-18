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


class MaskShape(Enum):
    """How the obfuscation region for one face is computed.

    * ``PARSER`` — a BiSeNet semantic-segmentation mask on an aligned crop: it hugs
      the real face (best quality), but runs a model inference *per face, per frame*.
    * ``ELLIPSE`` — a soft elliptical region derived from the bounding box: coarse,
      but model-free and dramatically cheaper. This is the main real-time FPS lever
      on the live path (it skips the per-face parser entirely).
    """

    PARSER = "parser"
    ELLIPSE = "ellipse"


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
    # How the face region is masked: precise BiSeNet parse (PARSER) vs cheap ellipse
    # (ELLIPSE). The parser hugs the real face but runs a model per face per frame;
    # the ellipse is model-free and far faster — the main real-time lever for live.
    mask_shape: MaskShape | str = MaskShape.PARSER

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
        # Accept raw UI input (e.g. "ellipse") and coerce to the enum.
        self.mask_shape = (
            self.mask_shape
            if isinstance(self.mask_shape, MaskShape)
            else MaskShape(str(self.mask_shape).strip().lower())
        )


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

    @staticmethod
    def _bbox_ellipse_geometry(
        bbox: tuple[int, int, int, int],
    ) -> tuple[tuple[int, int], tuple[int, int], float]:
        """Fallback ellipse geometry that stays centered on the current bbox."""
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        center = (int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5)))
        axes = (max(int(round(width * 0.5)), 1), max(int(round(height * 0.5)), 1))
        return center, axes, 0.0

    @staticmethod
    def _landmark_ellipse_geometry(
        bbox: tuple[int, int, int, int],
        landmarks: np.ndarray | None,
    ) -> tuple[tuple[int, int], tuple[int, int], float] | None:
        """Estimate a face ellipse from five RetinaFace landmarks.

        The eye line supplies rotation; eyes, nose, and mouth supply a soft center
        and scale. The current tracker bbox still clamps the result so stale
        landmarks on predict-only live frames cannot drag the mask away from the
        face.
        """
        if landmarks is None:
            return None

        points = np.asarray(landmarks, dtype=np.float32)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            return None

        x1, y1, x2, y2 = bbox
        width = max(float(x2 - x1), 1.0)
        height = max(float(y2 - y1), 1.0)

        pad_x = width * 0.25
        pad_y = height * 0.25
        if (
            np.any(points[:, 0] < float(x1) - pad_x)
            or np.any(points[:, 0] > float(x2) + pad_x)
            or np.any(points[:, 1] < float(y1) - pad_y)
            or np.any(points[:, 1] > float(y2) + pad_y)
        ):
            return None

        left_eye, right_eye, nose, left_mouth, right_mouth = points
        eye_vec = right_eye - left_eye
        eye_dist = float(np.linalg.norm(eye_vec))
        if eye_dist < max(4.0, width * 0.08) or eye_dist > width * 0.95:
            return None

        eye_mid = (left_eye + right_eye) * 0.5
        mouth_mid = (left_mouth + right_mouth) * 0.5
        mouth_vec = right_mouth - left_mouth
        mouth_width = float(np.linalg.norm(mouth_vec))

        eye_unit = eye_vec / eye_dist
        vertical_unit = np.asarray([-eye_unit[1], eye_unit[0]], dtype=np.float32)
        if float(np.dot(mouth_mid - eye_mid, vertical_unit)) < 0.0:
            vertical_unit *= -1.0
        eye_to_mouth = abs(float(np.dot(mouth_mid - eye_mid, vertical_unit)))
        if eye_to_mouth < max(4.0, height * 0.10) or eye_to_mouth > height * 0.90:
            return None

        bbox_center = np.asarray(
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
            dtype=np.float32,
        )
        landmark_center = eye_mid * 0.36 + nose * 0.28 + mouth_mid * 0.36
        center = bbox_center * 0.65 + landmark_center * 0.35
        max_shift = np.asarray([width * 0.10, height * 0.10], dtype=np.float32)
        center = np.clip(center, bbox_center - max_shift, bbox_center + max_shift)

        axis_x_est = max(eye_dist * 1.18, mouth_width * 1.65, width * 0.44)
        axis_y_est = max(eye_to_mouth * 1.55, height * 0.48)
        axis_x = int(round(float(np.clip(axis_x_est, width * 0.40, width * 0.50))))
        axis_y = int(round(float(np.clip(axis_y_est, height * 0.46, height * 0.50))))
        angle = float(np.degrees(np.arctan2(float(eye_vec[1]), float(eye_vec[0]))))

        return (
            (int(round(float(center[0]))), int(round(float(center[1])))),
            (max(axis_x, 1), max(axis_y, 1)),
            angle,
        )

    def _ellipse_face_mask(
        self,
        bbox: tuple[int, int, int, int],
        shape: tuple[int, int],
        params: ObfuscationParams,
        landmarks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Soft elliptical mask (float32 [0, 1]) for one face.

        When five landmarks are available, the ellipse follows the face pose instead
        of being only bbox-centered. If landmarks are absent or implausible, use the
        tighter bbox fallback.
        """
        mask = np.zeros(shape, dtype=np.float32)
        center, axes, angle = self._landmark_ellipse_geometry(
            bbox, landmarks
        ) or self._bbox_ellipse_geometry(bbox)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
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

        ``params.mask_shape`` selects the strategy per face:

        * ``PARSER`` (default): a BiSeNet mask that hugs the real face when landmarks
          and a parser are available, falling back to the ellipse on failure so the
          face is never under-covered. Precise, but a model inference per face.
        * ``ELLIPSE``: the coarse elliptical bound only — no parser, no model call.
          Far cheaper, which is what makes it the live-path FPS lever.

        Faces are unioned (per-pixel max).
        """
        shape = image.shape[:2]
        mask = np.zeros(shape, dtype=np.float32)
        for det in detections:
            bbox = self._valid_bbox(image, det)
            if bbox is None:
                continue

            landmarks = self._landmarks_from(det)
            face_mask: np.ndarray | None = None
            if params.mask_shape is MaskShape.PARSER:
                if landmarks is not None:
                    face_mask = self._parser_face_mask(image, landmarks, bbox, shape)
            if face_mask is None:
                face_mask = self._ellipse_face_mask(
                    bbox,
                    shape,
                    params,
                    landmarks,
                )
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
