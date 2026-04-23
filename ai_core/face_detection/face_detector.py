from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Sequence, cast

import cv2
import numpy as np

__all__ = ["FaceLandmarks", "FaceDetection", "FaceDetector"]


@dataclass(slots=True)
class FaceLandmarks:
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    left_mouth: tuple[float, float]
    right_mouth: tuple[float, float]

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.left_eye,
                self.right_eye,
                self.nose,
                self.left_mouth,
                self.right_mouth,
            ],
            dtype=np.float32,
        )


@dataclass(slots=True)
class FaceDetection:
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: float
    landmarks: FaceLandmarks

    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass(frozen=True)
class _PreprocessMeta:
    inv_scale: float
    original_width: int
    original_height: int


class FaceDetector:
    def __init__(
        self,
        onnx_path: str | Path,
        *,
        image_size: int | None = None,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.4,
        top_k: int = 5000,
        keep_top_k: int = 750,
        pad_value: tuple[int, int, int] = (104, 117, 123),
        providers: Sequence[str] | None = None,
        intra_op_num_threads: int | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if keep_top_k < 0:
            raise ValueError("keep_top_k must be >= 0")

        self._ort = self._import_onnxruntime()
        self.session = self._create_session(
            onnx_path=self.onnx_path,
            providers=providers,
            intra_op_num_threads=intra_op_num_threads,
        )

        self.input_name = str(self.session.get_inputs()[0].name)
        outputs = self.session.get_outputs()
        if len(outputs) != 3:
            raise ValueError("ONNX model must have exactly 3 outputs")
        self.output_names = [str(item.name) for item in outputs]

        self.image_size = self._resolve_image_size(image_size)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.keep_top_k = int(keep_top_k)
        self.pad_value = tuple(int(np.clip(v, 0, 255)) for v in pad_value)

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        input_tensor, meta = self._preprocess(image)
        boxes, scores, landmarks = self._forward(input_tensor)
        return self._postprocess(boxes, scores, landmarks, meta)

    def draw(
        self,
        image: np.ndarray,
        detections: list[FaceDetection],
        *,
        box_color: tuple[int, int, int] = (0, 255, 0),
        landmark_color: tuple[int, int, int] = (255, 0, 0),
        text_color: tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2,
        radius: int = 2,
        draw_score: bool = True,
    ) -> np.ndarray:
        canvas = self._ensure_bgr_uint8(image).copy()
        line_thickness = max(int(thickness), 1)
        point_radius = max(int(radius), 1)

        for det in detections:
            x1, y1, x2, y2 = [int(round(float(v))) for v in det.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, line_thickness)

            points = det.landmarks.as_array()
            for px, py in points:
                cv2.circle(
                    canvas,
                    (int(round(float(px))), int(round(float(py)))),
                    point_radius,
                    landmark_color,
                    thickness=-1,
                )

            if draw_score:
                cv2.putText(
                    canvas,
                    f"{det.score:.3f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    text_color,
                    line_thickness,
                    lineType=cv2.LINE_AA,
                )

        return canvas

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, _PreprocessMeta]:
        image_bgr = self._ensure_bgr_uint8(image)
        h, w = image_bgr.shape[:2]

        side = max(h, w)
        scale = float(self.image_size) / float(side)
        resized_w = min(self.image_size, max(1, int(round(w * scale))))
        resized_h = min(self.image_size, max(1, int(round(h * scale))))

        resized = cv2.resize(
            image_bgr,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.empty((self.image_size, self.image_size, 3), dtype=np.uint8)
        canvas[...] = self.pad_value
        canvas[:resized_h, :resized_w] = resized

        tensor = np.ascontiguousarray(
            canvas.transpose(2, 0, 1)[None, ...],
            dtype=np.float32,
        )

        meta = _PreprocessMeta(
            inv_scale=float(side) / float(self.image_size),
            original_width=int(w),
            original_height=int(h),
        )
        return tensor, meta

    def _forward(self, input_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor},
        )
        if len(outputs) != 3:
            raise RuntimeError("Unexpected ONNX outputs")

        boxes, scores, landmarks = outputs
        if boxes.ndim != 3 or boxes.shape[2] != 4:
            raise ValueError(f"Invalid boxes shape: {boxes.shape}")
        if scores.ndim != 2:
            raise ValueError(f"Invalid scores shape: {scores.shape}")
        if landmarks.ndim not in {3, 4}:
            raise ValueError(f"Invalid landmarks shape: {landmarks.shape}")

        return boxes, scores, landmarks

    def _postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        landmarks: np.ndarray,
        meta: _PreprocessMeta,
    ) -> list[FaceDetection]:
        boxes_1 = np.asarray(boxes[0], dtype=np.float32)
        scores_1 = np.asarray(scores[0], dtype=np.float32)

        if landmarks.ndim == 4:
            landmarks_1 = np.asarray(landmarks[0], dtype=np.float32)
        else:
            if landmarks.shape[2] != 10:
                raise ValueError(f"Invalid landmarks shape: {landmarks.shape}")
            landmarks_1 = np.asarray(landmarks[0], dtype=np.float32).reshape(-1, 5, 2)

        valid = scores_1 > self.conf_threshold
        if not bool(np.any(valid)):
            return []

        boxes_1 = boxes_1[valid]
        scores_1 = scores_1[valid]
        landmarks_1 = landmarks_1[valid]

        if self.top_k > 0 and scores_1.size > self.top_k:
            top = min(self.top_k, scores_1.size)
            idx = np.argpartition(scores_1, -top)[-top:]
            order = idx[np.argsort(scores_1[idx])[::-1]]
        else:
            order = np.argsort(scores_1)[::-1]

        boxes_1 = boxes_1[order]
        scores_1 = scores_1[order]
        landmarks_1 = landmarks_1[order]

        keep = self._nms(boxes_1, scores_1, self.nms_threshold)
        if self.keep_top_k > 0:
            keep = keep[: self.keep_top_k]
        if keep.size == 0:
            return []

        boxes_kept = boxes_1[keep]
        scores_kept = scores_1[keep]
        landmarks_kept = landmarks_1[keep]

        inv_scale = meta.inv_scale
        boxes_kept[:, 0::2] *= inv_scale
        boxes_kept[:, 1::2] *= inv_scale
        landmarks_kept[:, :, 0] *= inv_scale
        landmarks_kept[:, :, 1] *= inv_scale

        max_x = max(float(meta.original_width - 1), 0.0)
        max_y = max(float(meta.original_height - 1), 0.0)
        boxes_kept[:, 0::2] = np.clip(boxes_kept[:, 0::2], 0.0, max_x)
        boxes_kept[:, 1::2] = np.clip(boxes_kept[:, 1::2], 0.0, max_y)
        landmarks_kept[:, :, 0] = np.clip(landmarks_kept[:, :, 0], 0.0, max_x)
        landmarks_kept[:, :, 1] = np.clip(landmarks_kept[:, :, 1], 0.0, max_y)

        detections: list[FaceDetection] = []
        for box, score, lm in zip(boxes_kept, scores_kept, landmarks_kept):
            landmarks_obj = FaceLandmarks(
                left_eye=(float(lm[0, 0]), float(lm[0, 1])),
                right_eye=(float(lm[1, 0]), float(lm[1, 1])),
                nose=(float(lm[2, 0]), float(lm[2, 1])),
                left_mouth=(float(lm[3, 0]), float(lm[3, 1])),
                right_mouth=(float(lm[4, 0]), float(lm[4, 1])),
            )
            detections.append(
                FaceDetection(
                    bbox=(
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ),
                    score=float(score),
                    landmarks=landmarks_obj,
                )
            )
        return detections

    def _ensure_bgr_uint8(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3) in BGR format")
        if image.size == 0:
            raise ValueError("image must not be empty")

        if image.dtype == np.uint8 and image.flags.c_contiguous:
            return image

        out = image
        if out.dtype != np.uint8:
            if np.issubdtype(out.dtype, np.floating):
                max_value = float(np.nanmax(out)) if out.size > 0 else 0.0
                if max_value <= 1.0:
                    out = out * 255.0
            out = np.clip(out, 0, 255).astype(np.uint8, copy=False)

        return np.ascontiguousarray(out)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.int64)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
        order = np.argsort(scores)[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            inter_w = np.maximum(xx2 - xx1, 0.0)
            inter_h = np.maximum(yy2 - yy1, 0.0)
            inter = inter_w * inter_h
            union = areas[i] + areas[rest] - inter
            iou = inter / np.maximum(union, 1e-12)

            order = rest[iou <= threshold]

        return np.asarray(keep, dtype=np.int64)

    @staticmethod
    def _import_onnxruntime() -> Any:
        try:
            return cast(Any, importlib.import_module("onnxruntime"))
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required. Install `onnxruntime` or `onnxruntime-gpu`."
            ) from exc

    def _create_session(
        self,
        *,
        onnx_path: Path,
        providers: Sequence[str] | None,
        intra_op_num_threads: int | None,
    ) -> Any:
        options = self._ort.SessionOptions()
        options.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
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
            str(onnx_path),
            sess_options=options,
            providers=resolved,
        )

    def _resolve_image_size(self, image_size: int | None) -> int:
        if image_size is not None:
            if image_size <= 0:
                raise ValueError("image_size must be > 0")
            return int(image_size)

        shape = list(self.session.get_inputs()[0].shape)
        if len(shape) != 4:
            raise ValueError("Model input must be 4D NCHW")

        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and h == w:
            return int(h)

        raise ValueError(
            "Cannot infer image_size from model input. Pass image_size explicitly."
        )
