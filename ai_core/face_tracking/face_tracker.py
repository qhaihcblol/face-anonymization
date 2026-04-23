from __future__ import annotations

import enum
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["ByteTracker"]
from ai_core.face_detection.face_detector import FaceDetection


def _xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] boxes to [cx, cy, w, h]."""
    boxes = np.asarray(boxes, dtype=np.float64)
    out = np.empty_like(boxes)
    out[..., 0] = (boxes[..., 0] + boxes[..., 2]) * 0.5
    out[..., 1] = (boxes[..., 1] + boxes[..., 3]) * 0.5
    out[..., 2] = boxes[..., 2] - boxes[..., 0]
    out[..., 3] = boxes[..., 3] - boxes[..., 1]
    return out


def _cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert one [cx, cy, w, h] box to [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        dtype=np.float32,
    )


class KalmanBoxFilter:
    """
    Constant-velocity Kalman filter for one bounding box.

    State       : [cx, cy, w, h, vx, vy, vw, vh]
    Measurement : [cx, cy, w, h]

    Process and measurement noise scale with current box height.
    """

    _W_POS: float = 1.0 / 20.0
    _W_VEL: float = 1.0 / 160.0

    F: np.ndarray = np.eye(8, dtype=np.float64)
    F[:4, 4:] = np.eye(4)

    H: np.ndarray = np.zeros((4, 8), dtype=np.float64)
    H[:4, :4] = np.eye(4)

    def __init__(self) -> None:
        self.x: np.ndarray = np.zeros(8, dtype=np.float64)
        self.P: np.ndarray = np.eye(8, dtype=np.float64)

    def _Q(self) -> np.ndarray:
        """Process noise covariance."""
        h = max(abs(float(self.x[3])), 1.0)
        sp = self._W_POS * h
        sv = self._W_VEL * h
        diag = np.square([sp, sp, sp, sp, sv, sv, sv, sv])
        return np.diag(diag)

    def _R(self) -> np.ndarray:
        """Measurement noise covariance."""
        h = max(abs(float(self.x[3])), 1.0)
        sp = self._W_POS * h
        return np.diag(np.square([sp, sp, sp, sp]))

    def initiate(self, bbox_cxcywh: np.ndarray) -> None:
        """Initialize from the first detection."""
        cx, cy, w, h = bbox_cxcywh.astype(np.float64)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        h_val = max(abs(h), 1.0)
        sp = 2.0 * self._W_POS * h_val
        sv = 10.0 * self._W_VEL * h_val
        self.P = np.diag(np.square([sp, sp, sp, sp, sv, sv, sv, sv]))

    def predict(self) -> np.ndarray:
        """Advance the state one frame and return predicted [cx, cy, w, h]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self._Q()
        return self.x[:4].copy()

    def update(self, bbox_cxcywh: np.ndarray) -> None:
        """Correct the state with a matched detection."""
        z = bbox_cxcywh.astype(np.float64)
        S = self.H @ self.P @ self.H.T + self._R()
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def mahalanobis_distance(self, measurements_cxcywh: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance to candidate measurements."""
        S = self.H @ self.P @ self.H.T + self._R()
        S_inv = np.linalg.inv(S)

        diff = measurements_cxcywh.astype(np.float64) - (self.H @ self.x)
        return np.sqrt(np.einsum("ni,ij,nj->n", diff, S_inv, diff))


class TrackState(enum.IntEnum):
    """Lifecycle state of a track."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class Track:
    """Single tracked face."""

    _id_counter: int = 0

    def __init__(self, bbox_xyxy: np.ndarray, score: float) -> None:
        Track._id_counter += 1
        self.track_id: int = Track._id_counter

        self.kalman = KalmanBoxFilter()
        self.kalman.initiate(_xyxy_to_cxcywh(bbox_xyxy))

        self.bbox_xyxy: np.ndarray = bbox_xyxy.astype(np.float32)
        self.score: float = float(score)
        self.state: TrackState = TrackState.New

        self.hits: int = 1
        self.consecutive: int = 1
        self.frames_lost: int = 0

    def predict(self) -> None:
        """Run Kalman prediction and update current bbox."""
        pred_cxcywh = self.kalman.predict()
        self.bbox_xyxy = _cxcywh_to_xyxy(pred_cxcywh)

    def update(self, bbox_xyxy: np.ndarray, score: float) -> None:
        """Run Kalman correction with a matched detection."""
        self.kalman.update(_xyxy_to_cxcywh(bbox_xyxy))
        self.bbox_xyxy = _cxcywh_to_xyxy(self.kalman.x[:4])
        self.score = float(score)
        self.hits += 1
        self.consecutive += 1
        self.frames_lost = 0

    def mark_lost(self) -> None:
        """Mark unmatched in current frame."""
        self.consecutive = 0
        self.frames_lost += 1
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Mark expired and ready to purge."""
        self.state = TrackState.Removed

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.Tracked

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox_xyxy.tolist(),
            "score": self.score,
            "state": self.state.name,
        }


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized IoU matrix between two sets of [x1, y1, x2, y2] boxes."""
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)

    ix1 = np.maximum(a[:, 0:1], b[:, 0])
    iy1 = np.maximum(a[:, 1:2], b[:, 1])
    ix2 = np.minimum(a[:, 2:3], b[:, 2])
    iy2 = np.minimum(a[:, 3:4], b[:, 3])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def _associate(
    tracks: list[Track],
    detections: list[tuple[np.ndarray, float]],
    iou_thresh: float,
    gate_mahal: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Associate tracks and detections with Hungarian matching.

    Base cost is 1 - IoU. If Mahalanobis gating is enabled,
    implausible pairs are assigned a large cost so they are rejected.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    boxes_t = np.stack([t.bbox_xyxy for t in tracks])
    boxes_d = np.stack([d[0] for d in detections])

    cost = 1.0 - _iou_matrix(boxes_t, boxes_d)

    if gate_mahal > 0:
        meas_cxcywh = _xyxy_to_cxcywh(boxes_d)
        for i, t in enumerate(tracks):
            dists = t.kalman.mahalanobis_distance(meas_cxcywh)
            cost[i, dists > gate_mahal] = 2.0

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_t: set[int] = set()
    matched_d: set[int] = set()
    matches: list[tuple[int, int]] = []

    iou_cost_thresh = 1.0 - iou_thresh
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] > iou_cost_thresh:
            continue
        matches.append((int(r), int(c)))
        matched_t.add(int(r))
        matched_d.add(int(c))

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_d]

    return matches, unmatched_tracks, unmatched_dets


class ByteTracker:
    """
    ByteTrack-style multi-face tracker.

    Pipeline per frame:
    1) high-score detections with active tracks,
    2) low-score detections with still-unmatched active tracks,
    3) remaining high-score detections with lost tracks,
    4) initialize tracks from unmatched high-score detections.
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        max_lost: int = 30,
        min_hits: int = 3,
        iou_thresh: float = 0.3,
        iou_thresh_low: float = 0.5,
        gate_mahal: float = 9.49,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.iou_thresh_low = iou_thresh_low
        self.gate_mahal = gate_mahal

        self._tracked: list[Track] = []
        self._lost: list[Track] = []
        self.frame_id: int = 0

        Track._id_counter = 0

    def predict_only(self) -> list[dict[str, Any]]:
        """
        Advance tracks by one frame without detector association.

        Useful when detector inference is skipped for speed.
        """
        self.frame_id += 1

        for t in self._tracked + self._lost:
            t.predict()

        still_alive: list[Track] = []
        for t in self._lost:
            t.frames_lost += 1
            if t.frames_lost > self.max_lost:
                t.mark_removed()
            else:
                still_alive.append(t)
        self._lost = still_alive

        return [t.as_dict() for t in self._tracked]

    def update(self, detections: list["FaceDetection"]) -> list[dict[str, Any]]:
        """Process one detection frame and return active tracks."""
        self.frame_id += 1

        # 1) Split detections by confidence.
        dets_high: list[tuple[np.ndarray, float]] = []
        dets_low: list[tuple[np.ndarray, float]] = []
        for d in detections:
            bbox = np.asarray(d.bbox, dtype=np.float32)
            if bbox.shape != (4,):
                continue

            score = float(d.score)
            if score >= self.high_thresh:
                dets_high.append((bbox, score))
            elif score >= self.low_thresh:
                dets_low.append((bbox, score))

        # 2) Predict all track states before matching.
        for t in self._tracked + self._lost:
            t.predict()

        # 3) Stage 1: high-confidence detections vs active tracks.
        matches_1, unmatched_active_idx, unmatched_high_idx = _associate(
            tracks=self._tracked,
            detections=dets_high,
            iou_thresh=self.iou_thresh,
            gate_mahal=self.gate_mahal,
        )
        for t_idx, d_idx in matches_1:
            self._tracked[t_idx].update(*dets_high[d_idx])

        # 4) Stage 2: low-confidence detections vs unmatched active tracks.
        unmatched_active = [self._tracked[i] for i in unmatched_active_idx]
        matches_2, still_unmatched_active_idx, _ = _associate(
            tracks=unmatched_active,
            detections=dets_low,
            iou_thresh=self.iou_thresh_low,
            gate_mahal=self.gate_mahal,
        )
        for t_idx, d_idx in matches_2:
            unmatched_active[t_idx].update(*dets_low[d_idx])

        still_unmatched_active = [
            unmatched_active[i] for i in still_unmatched_active_idx
        ]
        for t in still_unmatched_active:
            t.mark_lost()

        # 5) Stage 3: remaining high-confidence detections vs lost tracks.
        remaining_high = [dets_high[i] for i in unmatched_high_idx]
        matches_3, _, unmatched_high_idx2 = _associate(
            tracks=self._lost,
            detections=remaining_high,
            iou_thresh=self.iou_thresh,
            gate_mahal=self.gate_mahal,
        )

        re_found_lost_idx: set[int] = set()
        for t_idx, d_idx in matches_3:
            t = self._lost[t_idx]
            t.update(*remaining_high[d_idx])
            t.state = TrackState.Tracked
            re_found_lost_idx.add(t_idx)

        still_lost: list[Track] = []
        for t_idx, t in enumerate(self._lost):
            if t_idx in re_found_lost_idx:
                continue
            t.frames_lost += 1
            if t.frames_lost > self.max_lost:
                t.mark_removed()
            else:
                still_lost.append(t)

        # 6) Create tracks from unmatched high-confidence detections.
        new_tracks: list[Track] = []
        for d_idx in unmatched_high_idx2:
            new_tracks.append(Track(*remaining_high[d_idx]))

        # 7) Rebuild pools and apply New -> Tracked promotion.
        candidate_tracked: list[Track] = (
            [
                t
                for t in self._tracked
                if t.state in (TrackState.New, TrackState.Tracked)
            ]
            + [self._lost[i] for i in range(len(self._lost)) if i in re_found_lost_idx]
            + new_tracks
        )
        for t in candidate_tracked:
            if t.state == TrackState.New and t.consecutive >= self.min_hits:
                t.state = TrackState.Tracked

        self._tracked = candidate_tracked
        self._lost = still_lost + still_unmatched_active

        return [t.as_dict() for t in self._tracked]

    def draw(
        self,
        image: "np.ndarray",
        tracks: list[dict[str, Any]],
        *,
        confirmed_only: bool = True,
    ) -> "np.ndarray":
        """Draw track boxes and IDs on a BGR image."""
        import cv2

        canvas = image.copy()
        for tr in tracks:
            if confirmed_only and tr["state"] != TrackState.Tracked.name:
                continue

            x1, y1, x2, y2 = (int(round(v)) for v in tr["bbox"])
            tid = tr["track_id"]
            color = _track_color(tid)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = f"#{tid}  {tr['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def stats(self) -> dict[str, Any]:
        """Return tracker diagnostics for logging/debugging."""
        return {
            "frame_id": self.frame_id,
            "n_tracked": len(self._tracked),
            "n_lost": len(self._lost),
            "n_confirmed": sum(
                1 for t in self._tracked if t.state == TrackState.Tracked
            ),
            "n_new": sum(1 for t in self._tracked if t.state == TrackState.New),
            "next_id": Track._id_counter + 1,
        }


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a track ID."""
    palette = [
        (0, 215, 255),
        (0, 255, 127),
        (255, 128, 0),
        (255, 0, 127),
        (127, 0, 255),
        (0, 127, 255),
        (255, 215, 0),
        (0, 255, 255),
        (255, 0, 255),
        (127, 255, 0),
        (0, 128, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
    ]
    return palette[track_id % len(palette)]
