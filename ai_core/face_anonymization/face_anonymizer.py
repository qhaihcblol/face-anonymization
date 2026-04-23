from enum import Enum
from typing import Any

import cv2
import numpy as np


class AnonymizationMethod(Enum):
    NONE = "none"
    BLUR = "blur"
    PIXELATE = "pixelate"
    MASK = "mask"
    BLACKOUT = "blackout"


class FaceAnonymizer:
    def __init__(
        self,
        blur_strength: int = 31,
        pixelation_level: int = 16,
        mask_color: tuple[int, int, int] = (160, 160, 160),
    ) -> None:
        self.blur_strength = max(int(blur_strength), 3)
        if self.blur_strength % 2 == 0:
            self.blur_strength += 1

        self.pixelation_level = max(int(pixelation_level), 4)
        self.mask_color = tuple(int(np.clip(c, 0, 255)) for c in mask_color)

    def _iter_valid_bboxes(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> list[tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []

        for det in detections:
            bbox = np.asarray(det.get("bbox", []), dtype=np.float32)
            if bbox.shape != (4,):
                continue

            x1, y1, x2, y2 = bbox
            x1 = int(np.clip(np.floor(x1), 0, max(w - 1, 0)))
            y1 = int(np.clip(np.floor(y1), 0, max(h - 1, 0)))
            x2 = int(np.clip(np.ceil(x2), 1, w))
            y2 = int(np.clip(np.ceil(y2), 1, h))

            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))

        return boxes

    def _ellipse(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for x1, y1, x2, y2 in self._iter_valid_bboxes(image, detections):
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            axis_x = max((x2 - x1) // 2, 1)
            axis_y = max(int((y2 - y1) * 0.58), 1)
            cv2.ellipse(mask, (cx, cy), (axis_x, axis_y), 0, 0, 360, 255, -1)
        return mask

    def _none(self, image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        return image

    def _blur(self, image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        mask = self._ellipse(image, detections)
        if not np.any(mask):
            return image

        blurred = cv2.GaussianBlur(image, (self.blur_strength, self.blur_strength), 0)
        output = image.copy()
        output[mask > 0] = blurred[mask > 0]
        return output

    def _pixelate(
        self, image: np.ndarray, detections: list[dict[str, Any]]
    ) -> np.ndarray:
        output = image.copy()
        for x1, y1, x2, y2 in self._iter_valid_bboxes(output, detections):
            face = output[y1:y2, x1:x2]
            h, w = face.shape[:2]
            if h < 2 or w < 2:
                continue

            scale = max(min(h, w) // self.pixelation_level, 1)
            small_w = max(w // scale, 1)
            small_h = max(h // scale, 1)

            pixelated = cv2.resize(
                face, (small_w, small_h), interpolation=cv2.INTER_LINEAR
            )
            pixelated = cv2.resize(pixelated, (w, h), interpolation=cv2.INTER_NEAREST)
            output[y1:y2, x1:x2] = pixelated
        return output

    def _mask(self, image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        mask = self._ellipse(image, detections)
        if not np.any(mask):
            return image

        output = image.copy()
        output[mask > 0] = self.mask_color
        return output

    def _blackout(
        self, image: np.ndarray, detections: list[dict[str, Any]]
    ) -> np.ndarray:
        mask = self._ellipse(image, detections)
        if not np.any(mask):
            return image

        output = image.copy()
        output[mask > 0] = (0, 0, 0)
        return output

    def anonymize_without_model(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
        method: AnonymizationMethod | str,
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")

        method_value = method
        if isinstance(method, str):
            method_value = AnonymizationMethod(method.strip().lower())

        if not isinstance(method_value, AnonymizationMethod):
            raise TypeError("method must be AnonymizationMethod or str")

        method_map = {
            AnonymizationMethod.NONE: self._none,
            AnonymizationMethod.BLUR: self._blur,
            AnonymizationMethod.PIXELATE: self._pixelate,
            AnonymizationMethod.MASK: self._mask,
            AnonymizationMethod.BLACKOUT: self._blackout,
        }
        return method_map[method_value](image, detections)
