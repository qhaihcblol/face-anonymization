from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from ai_core.face_detection.face_detector import FaceDetection, FaceLandmarks

__all__ = ["AlignedFace", "FaceAligner"]


@dataclass(slots=True)
class AlignedFace:
    bbox: tuple[float, float, float, float]
    score: float
    landmarks: FaceLandmarks
    matrix: np.ndarray

    def as_detection(self) -> FaceDetection:
        return FaceDetection(
            bbox=self.bbox,
            score=self.score,
            landmarks=self.landmarks,
        )


class FaceAligner:
    """
    Align face coordinates from detector landmarks to a canonical face space.

    Input is the detector output (`FaceDetection`). Output coordinates are in
    the aligned target space, not in the original image space.
    
    Use `scale_factor` < 1.0 to make the face smaller in the output and include
    more background around the face.
    """

    _BASE_SIZE: tuple[int, int] = (112, 112)
    _BASE_REFERENCE: np.ndarray = np.asarray(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        output_size: tuple[int, int] = (112, 112),
        reference_landmarks: np.ndarray | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        self.output_size = self._validate_output_size(output_size)
        self.scale_factor = self._validate_scale_factor(scale_factor)
        
        if reference_landmarks is None:
            self.reference_landmarks = self._scaled_reference(self.output_size, self.scale_factor)
        else:
            self.reference_landmarks = self._validate_landmarks(reference_landmarks)

    def align(self, detections: Sequence[FaceDetection]) -> list[AlignedFace]:
        return [self.align_detection(det) for det in detections]

    def align_detection(self, detection: FaceDetection) -> AlignedFace:
        src_landmarks = self._validate_landmarks(detection.landmarks.as_array())
        matrix = self._estimate_matrix(src_landmarks)

        aligned_landmarks = self.transform_points(src_landmarks, matrix)
        aligned_bbox = self.transform_bbox(detection.bbox, matrix)

        return AlignedFace(
            bbox=aligned_bbox,
            score=float(detection.score),
            landmarks=self._to_landmarks(aligned_landmarks),
            matrix=matrix,
        )

    def align_and_warp(
        self,
        image: np.ndarray,
        detection: FaceDetection,
    ) -> tuple[AlignedFace, np.ndarray]:
        aligned = self.align_detection(detection)
        aligned_image = self.warp_face(image, aligned.matrix)
        return aligned, aligned_image

    def align_and_warp_batch(
        self,
        image: np.ndarray,
        detections: Sequence[FaceDetection],
    ) -> list[tuple[AlignedFace, np.ndarray]]:
        return [self.align_and_warp(image, det) for det in detections]

    def _estimate_matrix(self, landmarks: np.ndarray) -> np.ndarray:
        matrix, _ = cv2.estimateAffinePartial2D(
            landmarks,
            self.reference_landmarks,
            method=cv2.LMEDS,
        )
        if matrix is None:
            raise ValueError("Cannot estimate face alignment matrix")
        return np.asarray(matrix, dtype=np.float32)

    @staticmethod
    def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        matrix = np.asarray(matrix, dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if matrix.shape != (2, 3):
            raise ValueError("matrix must have shape (2, 3)")

        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.concatenate([points, ones], axis=1)
        return points_h @ matrix.T

    @staticmethod
    def invert_matrix(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (2, 3):
            raise ValueError("matrix must have shape (2, 3)")
        return np.asarray(cv2.invertAffineTransform(matrix), dtype=np.float32)

    @classmethod
    def inverse_transform_points(
        cls,
        points: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:
        inverse_matrix = cls.invert_matrix(matrix)
        return cls.transform_points(points, inverse_matrix)

    @classmethod
    def transform_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        matrix: np.ndarray,
    ) -> tuple[float, float, float, float]:
        box = np.asarray(bbox, dtype=np.float32)
        if box.shape != (4,):
            raise ValueError("bbox must have shape (4,)")

        x1, y1, x2, y2 = box
        corners = np.asarray(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )
        aligned = cls.transform_points(corners, matrix)
        min_xy = aligned.min(axis=0)
        max_xy = aligned.max(axis=0)
        return (
            float(min_xy[0]),
            float(min_xy[1]),
            float(max_xy[0]),
            float(max_xy[1]),
        )

    @classmethod
    def inverse_transform_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        matrix: np.ndarray,
    ) -> tuple[float, float, float, float]:
        inverse_matrix = cls.invert_matrix(matrix)
        return cls.transform_bbox(bbox, inverse_matrix)

    @classmethod
    def _scaled_reference(
        cls,
        output_size: tuple[int, int],
        scale_factor: float = 1.0,
    ) -> np.ndarray:
        out_w, out_h = output_size
        base_w, base_h = cls._BASE_SIZE
        
        # Calculate center of the reference landmarks
        center = cls._BASE_REFERENCE.mean(axis=0)
        
        # Scale landmarks around the center
        scaled = (cls._BASE_REFERENCE - center) * scale_factor + center
        
        # Apply output size scaling
        scale = np.asarray([out_w / base_w, out_h / base_h], dtype=np.float32)
        return scaled * scale

    @staticmethod
    def _validate_scale_factor(scale_factor: float) -> float:
        scale = float(scale_factor)
        if scale <= 0:
            raise ValueError("scale_factor must be > 0")
        return scale

    @staticmethod
    def _validate_output_size(output_size: tuple[int, int]) -> tuple[int, int]:
        if len(output_size) != 2:
            raise ValueError("output_size must be (width, height)")

        width, height = int(output_size[0]), int(output_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("output_size values must be > 0")
        return width, height

    @staticmethod
    def _validate_landmarks(landmarks: np.ndarray) -> np.ndarray:
        points = np.asarray(landmarks, dtype=np.float32)
        if points.shape != (5, 2):
            raise ValueError("landmarks must have shape (5, 2)")
        if not np.isfinite(points).all():
            raise ValueError("landmarks must contain finite values")
        return points

    @staticmethod
    def _to_landmarks(points: np.ndarray) -> FaceLandmarks:
        return FaceLandmarks(
            left_eye=(float(points[0, 0]), float(points[0, 1])),
            right_eye=(float(points[1, 0]), float(points[1, 1])),
            nose=(float(points[2, 0]), float(points[2, 1])),
            left_mouth=(float(points[3, 0]), float(points[3, 1])),
            right_mouth=(float(points[4, 0]), float(points[4, 1])),
        )

    def warp_face(
        self,
        image: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Warp image into canonical aligned face space.

        Args:
            image:
                Source image in HWC format.
            matrix:
                Affine transform matrix with shape (2, 3).

        Returns:
            Aligned face image with shape (H, W, C).
        """
        if image.ndim not in (2, 3):
            raise ValueError("image must have shape (H, W) or (H, W, C)")

        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (2, 3):
            raise ValueError("matrix must have shape (2, 3)")

        width, height = self.output_size

        return cv2.warpAffine(
            image,
            matrix,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def warp_back_to_frame(
        self,
        aligned_image: np.ndarray,
        inverse_matrix: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Warp aligned-face output back to original frame space.

        Args:
            aligned_image:
                Face image in aligned space.
            inverse_matrix:
                Affine transform from aligned space -> original frame space.
            frame_shape:
                Shape of original frame (H, W) or (H, W, C).

        Returns:
            Tuple of:
            - warped image in original frame size
            - uint8 binary mask (255 where aligned image is projected)
        """
        if aligned_image.ndim not in (2, 3):
            raise ValueError("aligned_image must have shape (H, W) or (H, W, C)")

        if len(frame_shape) < 2:
            raise ValueError("frame_shape must contain at least (H, W)")
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        if frame_h <= 0 or frame_w <= 0:
            raise ValueError("frame_shape values must be > 0")

        inverse_matrix = np.asarray(inverse_matrix, dtype=np.float32)
        if inverse_matrix.shape != (2, 3):
            raise ValueError("inverse_matrix must have shape (2, 3)")

        warped = cv2.warpAffine(
            aligned_image,
            inverse_matrix,
            dsize=(frame_w, frame_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        source_mask = np.full(aligned_image.shape[:2], 255, dtype=np.uint8)
        mask = cv2.warpAffine(
            source_mask,
            inverse_matrix,
            dsize=(frame_w, frame_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped, mask
