from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_detection.face_detector import (
    FaceDetection,
    FaceDetector,
    FaceLandmarks,
)
from ai_core.face_swapping.face_swapper import FaceSwapper

__all__ = ["FaceSwapStabilizer", "OneEuroFilter"]


def _alpha(cutoff: float, freq: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    te = 1.0 / freq
    return 1.0 / (1.0 + tau / te)


class OneEuroFilter:
    """Vectorized 1-Euro filter (Casiez et al.).

    Smooths a fixed-shape signal while staying responsive to fast motion: it filters
    hard when the signal is near-static (killing jitter) and loosens when it moves
    fast (avoiding lag). Used here on the 5 facial landmarks so the per-frame
    alignment stops shimmering.
    """

    def __init__(
        self,
        freq: float,
        min_cutoff: float = 1.0,
        beta: float = 0.3,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = max(float(freq), 1e-3)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def filter(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            return x.copy()

        dx = (x - self._x_prev) * self.freq
        a_d = _alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, self.freq)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


@dataclass
class _FaceTrack:
    track_id: int
    bbox: np.ndarray  # last seen (x1, y1, x2, y2)
    landmark_filter: OneEuroFilter
    missed: int = 0
    prev_crop: np.ndarray | None = field(default=None)
    prev_mask: np.ndarray | None = field(default=None)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    return float(inter / max(area_a + area_b - inter, 1e-9))


class FaceSwapStabilizer:
    """Temporally stable video face swap on top of :class:`FaceSwapper`.

    Removes the flicker/jitter/seam seen when swapping each frame independently:

    * **Jitter / swim** — faces are tracked across frames (greedy IoU) and their 5
      landmarks are smoothed with a per-track :class:`OneEuroFilter` before alignment,
      so the warp no longer shimmers.
    * **Flicker** — optionally EMA-blends each newly swapped *aligned* crop with the
      track's previous crop (crops are registered to the same template, so this is a
      safe temporal average).
    * **Boundary breathing** — optionally EMA-blends the per-frame blend mask with the
      track's previous mask. The parser mask is recomputed each frame and its edge can
      jitter (e.g. around glasses/hairline); smoothing it in the registered aligned
      space steadies the blend boundary.
    * **Seam** — handled by :class:`FaceSwapper`'s elliptical mask + color transfer.

    The object is stateful: call :meth:`reset` between independent videos.
    """

    def __init__(
        self,
        detector: FaceDetector,
        swapper: FaceSwapper,
        *,
        freq: float = 25.0,
        min_cutoff: float = 0.5,
        beta: float = 0.05,
        output_smooth: float = 0.4,
        mask_smooth: float = 0.5,
        iou_threshold: float = 0.3,
        max_missed: int = 8,
        source_blob: np.ndarray | None = None,
    ) -> None:
        self.detector = detector
        self.swapper = swapper
        self.aligner: FaceAligner = swapper.target_aligner
        # Identity pasted onto every face for this run (None = swapper's default).
        # Held for the whole run since the identity is constant across frames.
        self.source_blob = source_blob
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.output_smooth = float(np.clip(output_smooth, 0.0, 1.0))
        self.mask_smooth = float(np.clip(mask_smooth, 0.0, 1.0))
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)

        self._tracks: list[_FaceTrack] = []
        self._next_id = 0
        self.last_face_count = 0

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0
        self.last_face_count = 0

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Swap all faces in a BGR frame with temporal stabilization.

        Returns a BGR frame (matching VideoIO), so it drops into the video pipeline.
        """
        detections = self.detector.detect(frame_bgr)
        self.last_face_count = len(detections)

        assignments = self._associate(detections)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        output = frame_rgb.copy()

        for det, track in assignments:
            smoothed = track.landmark_filter.filter(det.landmarks.as_array().reshape(-1))
            landmarks = smoothed.reshape(5, 2).astype(np.float32)

            detection = FaceDetection(
                bbox=det.bbox,
                score=det.score,
                landmarks=FaceLandmarks(
                    left_eye=(float(landmarks[0, 0]), float(landmarks[0, 1])),
                    right_eye=(float(landmarks[1, 0]), float(landmarks[1, 1])),
                    nose=(float(landmarks[2, 0]), float(landmarks[2, 1])),
                    left_mouth=(float(landmarks[3, 0]), float(landmarks[3, 1])),
                    right_mouth=(float(landmarks[4, 0]), float(landmarks[4, 1])),
                ),
            )
            aligned = self.aligner.align_detection(detection)
            swapped_crop, mask = self.swapper.swap_aligned(
                frame_rgb, aligned, self.source_blob
            )

            # Smoothing is valid because landmarks are filtered, so consecutive crops
            # and masks are registered to the same aligned template.
            if self.output_smooth > 0.0 and track.prev_crop is not None:
                swapped_crop = cv2.addWeighted(
                    swapped_crop,
                    1.0 - self.output_smooth,
                    track.prev_crop,
                    self.output_smooth,
                    0.0,
                )
            track.prev_crop = swapped_crop

            if self.mask_smooth > 0.0 and track.prev_mask is not None:
                mask = cv2.addWeighted(
                    mask,
                    1.0 - self.mask_smooth,
                    track.prev_mask,
                    self.mask_smooth,
                    0.0,
                )
            track.prev_mask = mask

            output = self.swapper.paste_back(output, swapped_crop, aligned.matrix, mask)

        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    def _associate(
        self,
        detections: list[FaceDetection],
    ) -> list[tuple[FaceDetection, _FaceTrack]]:
        """Greedy IoU matching of detections to existing tracks."""
        existing = self._tracks
        det_boxes = [np.asarray(d.bbox, dtype=np.float32) for d in detections]
        unmatched_dets = set(range(len(detections)))
        matched_tracks: set[int] = set()
        assignments: list[tuple[FaceDetection, _FaceTrack]] = []

        # Greedily pick the highest-IoU (track, detection) pairs above threshold.
        pairs: list[tuple[float, int, int]] = []
        for t_idx, track in enumerate(existing):
            for d_idx in range(len(detections)):
                iou = _iou(track.bbox, det_boxes[d_idx])
                if iou >= self.iou_threshold:
                    pairs.append((iou, t_idx, d_idx))
        pairs.sort(key=lambda p: p[0], reverse=True)

        for _, t_idx, d_idx in pairs:
            if t_idx in matched_tracks or d_idx not in unmatched_dets:
                continue
            track = existing[t_idx]
            track.bbox = det_boxes[d_idx]
            track.missed = 0
            matched_tracks.add(t_idx)
            unmatched_dets.discard(d_idx)
            assignments.append((detections[d_idx], track))

        # Surviving existing tracks: matched ones, plus unmatched ones still young.
        survivors: list[_FaceTrack] = []
        for t_idx, track in enumerate(existing):
            if t_idx in matched_tracks:
                survivors.append(track)
            else:
                track.missed += 1
                if track.missed <= self.max_missed:
                    survivors.append(track)

        # New tracks for unmatched detections.
        for d_idx in sorted(unmatched_dets):
            track = _FaceTrack(
                track_id=self._next_id,
                bbox=det_boxes[d_idx],
                landmark_filter=OneEuroFilter(
                    self.freq, min_cutoff=self.min_cutoff, beta=self.beta
                ),
            )
            self._next_id += 1
            survivors.append(track)
            assignments.append((detections[d_idx], track))

        self._tracks = survivors
        return assignments
