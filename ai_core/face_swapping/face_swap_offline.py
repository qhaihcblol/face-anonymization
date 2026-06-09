from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ai_core.face_detection.face_detector import (
    FaceDetection,
    FaceDetector,
    FaceLandmarks,
)
from ai_core.face_swapping.face_swap_stabilizer import _iou
from ai_core.face_swapping.face_swapper import FaceSwapper

__all__ = ["OfflineFaceSwapStabilizer"]


@dataclass
class _Observation:
    frame_idx: int
    landmarks: np.ndarray  # (5, 2) raw detector landmarks
    bbox: tuple[float, float, float, float]
    score: float


@dataclass
class _OfflineTrack:
    track_id: int
    bbox: np.ndarray  # last seen (x1, y1, x2, y2)
    missed: int = 0
    observations: list[_Observation] = field(default_factory=list)


@dataclass
class _PlanEntry:
    track_id: int
    landmarks: np.ndarray  # (5, 2) smoothed landmarks
    bbox: tuple[float, float, float, float]
    score: float


class OfflineFaceSwapStabilizer:
    """Two-pass (offline) temporally stable face swap.

    Unlike :class:`FaceSwapStabilizer`, which smooths landmarks causally with a
    1-Euro filter (and therefore always lags fast motion), this planner sees the
    *whole* clip first and smooths each track's landmark trajectory with a
    zero-phase Savitzky-Golay filter. The result has no temporal lag, so the swap
    can be locked tightly to the face without the ghosting that a causal EMA on the
    swapped crop would introduce.

    Usage is two passes over the same frames:

    * **Pass 1** — call :meth:`observe` on every frame, in order. The detector runs
      once per frame and detections are associated to tracks by greedy IoU.
    * call :meth:`finalize` to smooth every track and build the per-frame plan.
    * **Pass 2** — re-iterate the same frames and call :meth:`render` per frame,
      which swaps using the smoothed landmarks.

    Only landmarks/bboxes are retained between passes (a few floats per face), so
    memory stays flat regardless of clip length; the cost is a second video decode.
    """

    def __init__(
        self,
        detector: FaceDetector,
        swapper: FaceSwapper,
        *,
        output_smooth: float = 0.0,
        mask_smooth: float = 0.0,
        iou_threshold: float = 0.3,
        max_gap: int = 8,
        sg_window: int = 9,
        sg_polyorder: int = 2,
        source_blob: np.ndarray | None = None,
    ) -> None:
        self.detector = detector
        self.swapper = swapper
        self.aligner = swapper.target_aligner
        # Identity pasted onto every face for this run (None = swapper's default).
        self.source_blob = source_blob
        # With zero-phase landmark smoothing the crops are already steady, so the
        # crop/mask EMA defaults to off (it would only reintroduce motion blur).
        self.output_smooth = float(np.clip(output_smooth, 0.0, 1.0))
        self.mask_smooth = float(np.clip(mask_smooth, 0.0, 1.0))
        self.iou_threshold = float(iou_threshold)
        self.max_gap = int(max_gap)
        self.sg_window = int(sg_window)
        self.sg_polyorder = int(sg_polyorder)

        self._tracks: list[_OfflineTrack] = []
        self._next_id = 0
        self._frame_idx = 0
        self._plan: dict[int, list[_PlanEntry]] = {}
        self._render_state: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.last_face_count = 0

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0
        self._frame_idx = 0
        self._plan = {}
        self._render_state = {}
        self.last_face_count = 0

    # ------------------------------------------------------------------ #
    # Pass 1: observe
    # ------------------------------------------------------------------ #
    def observe(self, frame_bgr: np.ndarray) -> int:
        """Record detections for one frame (pass 1). Returns the face count."""
        detections = self.detector.detect(frame_bgr)
        self.last_face_count = len(detections)
        self._associate(detections, self._frame_idx)
        self._frame_idx += 1
        return len(detections)

    def _associate(self, detections: list[FaceDetection], frame_idx: int) -> None:
        det_boxes = [np.asarray(d.bbox, dtype=np.float32) for d in detections]
        unmatched_dets = set(range(len(detections)))
        matched_tracks: set[int] = set()

        pairs: list[tuple[float, int, int]] = []
        for t_idx, track in enumerate(self._tracks):
            for d_idx in range(len(detections)):
                iou = _iou(track.bbox, det_boxes[d_idx])
                if iou >= self.iou_threshold:
                    pairs.append((iou, t_idx, d_idx))
        pairs.sort(key=lambda p: p[0], reverse=True)

        for _, t_idx, d_idx in pairs:
            if t_idx in matched_tracks or d_idx not in unmatched_dets:
                continue
            track = self._tracks[t_idx]
            det = detections[d_idx]
            track.bbox = det_boxes[d_idx]
            track.missed = 0
            track.observations.append(
                _Observation(
                    frame_idx=frame_idx,
                    landmarks=det.landmarks.as_array().astype(np.float32),
                    bbox=det.bbox,
                    score=float(det.score),
                )
            )
            matched_tracks.add(t_idx)
            unmatched_dets.discard(d_idx)

        survivors: list[_OfflineTrack] = []
        for t_idx, track in enumerate(self._tracks):
            if t_idx in matched_tracks:
                survivors.append(track)
            else:
                track.missed += 1
                if track.missed <= self.max_gap:
                    survivors.append(track)

        for d_idx in sorted(unmatched_dets):
            det = detections[d_idx]
            track = _OfflineTrack(track_id=self._next_id, bbox=det_boxes[d_idx])
            track.observations.append(
                _Observation(
                    frame_idx=frame_idx,
                    landmarks=det.landmarks.as_array().astype(np.float32),
                    bbox=det.bbox,
                    score=float(det.score),
                )
            )
            self._next_id += 1
            survivors.append(track)

        self._tracks = survivors

    # ------------------------------------------------------------------ #
    # Build the plan
    # ------------------------------------------------------------------ #
    def finalize(self) -> None:
        """Smooth every track's landmark trajectory and build the per-frame plan."""
        self._plan = {}
        for track in self._tracks:
            obs = sorted(track.observations, key=lambda o: o.frame_idx)
            if not obs:
                continue
            coords = np.stack([o.landmarks.reshape(-1) for o in obs], axis=0)  # (N, 10)
            smoothed = self._smooth_sequence(coords)
            for i, observation in enumerate(obs):
                entry = _PlanEntry(
                    track_id=track.track_id,
                    landmarks=smoothed[i].reshape(5, 2).astype(np.float32),
                    bbox=observation.bbox,
                    score=observation.score,
                )
                self._plan.setdefault(observation.frame_idx, []).append(entry)

        # Pass 1 leaves only the (now-consumed) tracks behind; clear them so a stray
        # second finalize() can't double-count.
        self._tracks = []

    def _smooth_sequence(self, coords: np.ndarray) -> np.ndarray:
        """Zero-phase smooth a (N, 10) landmark trajectory along time (axis 0).

        Note: observations are treated as evenly spaced. Tracks may contain short
        gaps (<= ``max_gap`` frames) where the detector missed the face; those frames
        carry no observation and are simply not smoothed across with special care,
        which is acceptable for the short gaps the tracker bridges.
        """
        n = coords.shape[0]
        window = min(self.sg_window, n)
        if window % 2 == 0:
            window -= 1
        if window < 3 or window <= self.sg_polyorder:
            return coords  # too short to smooth meaningfully

        from scipy.signal import savgol_filter

        return savgol_filter(
            coords, window_length=window, polyorder=self.sg_polyorder, axis=0
        )

    # ------------------------------------------------------------------ #
    # Pass 2: render
    # ------------------------------------------------------------------ #
    def render(self, frame_idx: int, frame_bgr: np.ndarray) -> np.ndarray:
        """Swap all planned faces for ``frame_idx`` using smoothed landmarks.

        Returns a BGR frame (matching VideoIO). Frames with no planned face pass
        through unchanged.
        """
        entries = self._plan.get(frame_idx, [])
        self.last_face_count = len(entries)
        if not entries:
            return frame_bgr

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        output = frame_rgb.copy()

        for entry in entries:
            lmk = entry.landmarks
            detection = FaceDetection(
                bbox=entry.bbox,
                score=entry.score,
                landmarks=FaceLandmarks(
                    left_eye=(float(lmk[0, 0]), float(lmk[0, 1])),
                    right_eye=(float(lmk[1, 0]), float(lmk[1, 1])),
                    nose=(float(lmk[2, 0]), float(lmk[2, 1])),
                    left_mouth=(float(lmk[3, 0]), float(lmk[3, 1])),
                    right_mouth=(float(lmk[4, 0]), float(lmk[4, 1])),
                ),
            )
            aligned = self.aligner.align_detection(detection)
            swapped_crop, mask = self.swapper.swap_aligned(
                frame_rgb, aligned, self.source_blob
            )

            prev = self._render_state.get(entry.track_id)
            if prev is not None:
                prev_crop, prev_mask = prev
                if self.output_smooth > 0.0:
                    swapped_crop = cv2.addWeighted(
                        swapped_crop,
                        1.0 - self.output_smooth,
                        prev_crop,
                        self.output_smooth,
                        0.0,
                    )
                if self.mask_smooth > 0.0:
                    mask = cv2.addWeighted(
                        mask, 1.0 - self.mask_smooth, prev_mask, self.mask_smooth, 0.0
                    )
            self._render_state[entry.track_id] = (swapped_crop, mask)

            output = self.swapper.paste_back(
                output, swapped_crop, aligned.matrix, mask
            )

        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
