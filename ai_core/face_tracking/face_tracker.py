"""
byte_tracker.py — ByteTrack face tracker
=========================================
Paper: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
       Zhang et al., ECCV 2022  https://arxiv.org/abs/2110.06864

Algorithm in one sentence:
  Two-stage Hungarian matching: first associate high-confidence detections
  to active tracks, then recover lost tracks using low-confidence detections.

Dependencies: numpy, scipy
"""

from __future__ import annotations

import enum
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["ByteTracker"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Kalman Filter — constant-velocity model for bounding boxes
# ══════════════════════════════════════════════════════════════════════════════


class KalmanBoxFilter:
    """
    Tracks ONE bounding box with a constant-velocity Kalman filter.

    State    (8-dim): [cx, cy, w, h,  vx, vy, vw, vh]
    Measure  (4-dim): [cx, cy, w, h]

    Noise magnitude scales with box height h so small/large faces are
    treated proportionally (idea from DeepSORT / BoT-SORT).
    """

    _W_POS = 1.0 / 20.0  # position noise weight
    _W_VEL = 1.0 / 160.0  # velocity noise weight

    # ── Matrices (class-level, shared across instances) ───────────────────
    # State transition:  x_{t+1} = F * x_t   (position += velocity)
    F = np.eye(8, dtype=np.float64)
    F[:4, 4:] = np.eye(4)

    # Observation:  z = H * x   (observe position only)
    H = np.zeros((4, 8), dtype=np.float64)
    H[:4, :4] = np.eye(4)

    def __init__(self) -> None:
        self.x = np.zeros(8, dtype=np.float64)  # state mean
        self.P = np.eye(8, dtype=np.float64)  # state covariance

    # ── Noise covariances (adaptive to face size) ─────────────────────────

    def _Q(self) -> np.ndarray:
        """Process noise covariance Q."""
        h = max(abs(float(self.x[3])), 1.0)
        sp = self._W_POS * h
        sv = self._W_VEL * h
        return np.diag(np.square([sp, sp, sp, sp, sv, sv, sv, sv]))

    def _R(self) -> np.ndarray:
        """Measurement noise covariance R."""
        h = max(abs(float(self.x[3])), 1.0)
        sp = self._W_POS * h
        return np.diag(np.square([sp, sp, sp, sp]))

    # ── Public API ────────────────────────────────────────────────────────

    def initiate(self, bbox: np.ndarray) -> None:
        """
        Initialize state from the first detection.
        bbox: [cx, cy, w, h]
        """
        cx, cy, w, h = bbox
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Large initial uncertainty for velocity; moderate for position
        h_val = max(abs(h), 1.0)
        sp = 2 * self._W_POS * h_val
        sv = 10 * self._W_VEL * h_val
        self.P = np.diag(np.square([sp, sp, sp, sp, sv, sv, sv, sv]))

    def predict(self) -> np.ndarray:
        """
        Time-update (predict) step.
        Advances state one frame and returns predicted [cx, cy, w, h].
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self._Q()
        return self.x[:4].copy()

    def update(self, bbox: np.ndarray) -> None:
        """
        Measurement-update (correct) step.
        bbox: [cx, cy, w, h] from a matched detection.
        """
        z = bbox.astype(np.float64)
        S = self.H @ self.P @ self.H.T + self._R()  # innovation cov
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def mahalanobis(self, measurements: np.ndarray) -> np.ndarray:
        """
        Mahalanobis distance from current predicted state to each measurement.

        measurements : (N, 4) — [cx, cy, w, h] per candidate
        Returns      : (N,)   — scalar distance per candidate
        Used for gating: discard assignments with distance > chi2 threshold.
        """
        h_val = max(abs(float(self.x[3])), 1.0)
        S = self.H @ self.P @ self.H.T + np.diag(np.square([self._W_POS * h_val] * 4))
        S_inv = np.linalg.inv(S)
        diff = measurements.astype(np.float64) - (self.H @ self.x)  # (N, 4)
        return np.sqrt(np.einsum("ni,ij,nj->n", diff, S_inv, diff))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Track — one tracked face
# ══════════════════════════════════════════════════════════════════════════════


class TrackState(enum.IntEnum):
    New = 0  # Newly created; awaiting min_hits confirmations
    Tracked = 1  # Actively matched in recent frames
    Lost = 2  # Missed; kept alive for re-identification (up to max_lost frames)
    Removed = 3  # Exceeded max_lost; will be purged this frame


class Track:
    """
    One tracked face.

    Lifecycle:
        New → (matched min_hits times) → Tracked
        Tracked → (missed 1 frame)     → Lost
        Lost    → (matched again)       → Tracked
        Lost    → (missed max_lost)     → Removed
    """

    _id_counter = 0

    def __init__(self, bbox_xyxy: np.ndarray, score: float) -> None:
        Track._id_counter += 1
        self.track_id: int = Track._id_counter

        self.kalman = KalmanBoxFilter()
        self.kalman.initiate(_xyxy_to_cxcywh(bbox_xyxy))

        self.bbox_xyxy: np.ndarray = bbox_xyxy.astype(np.float32)
        self.score: float = score
        self.state: TrackState = TrackState.New

        self.hits: int = 1  # total matched frames
        self.consecutive: int = 1  # consecutive matched frames (for min_hits)
        self.frames_lost: int = 0  # consecutive missed frames

    # ── Kalman wrappers ───────────────────────────────────────────────────

    def predict(self) -> None:
        """Call at the start of every frame before association."""
        pred_cxcywh = self.kalman.predict()
        self.bbox_xyxy = _cxcywh_to_xyxy(pred_cxcywh)

    def update(self, bbox_xyxy: np.ndarray, score: float) -> None:
        """Called when this track is matched to a detection."""
        self.kalman.update(_xyxy_to_cxcywh(bbox_xyxy))
        self.bbox_xyxy = _cxcywh_to_xyxy(self.kalman.x[:4])
        self.score = score
        self.hits += 1
        self.consecutive += 1
        self.frames_lost = 0

    def mark_lost(self) -> None:
        self.consecutive = 0
        self.frames_lost += 1
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
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


# ══════════════════════════════════════════════════════════════════════════════
# 3. ByteTracker — main class
# ══════════════════════════════════════════════════════════════════════════════


class ByteTracker:
    """
    ByteTrack multi-face tracker.

    Usage
    -----
    tracker = ByteTracker()

    for frame in video:
        dets = detector.detect(frame)                 # from FaceDetector
        tracks = tracker.update(dets)
        annotated = tracker.draw(frame, tracks)

    Parameters
    ----------
    high_thresh  : Confidence threshold to be a "high" detection (Stage 1).
    low_thresh   : Minimum confidence; detections below this are ignored.
    max_lost     : Frames a lost track survives before removal.
    min_hits     : Consecutive matches needed to confirm a new track.
    iou_thresh   : IoU threshold for Stage 1 association.
    iou_thresh_low : IoU threshold for Stage 2 (lost tracks, lower = stricter).
    gate_mahal   : If > 0, discard assignments with Mahalanobis distance > this.
                   Chi-square 95% for 4 DOF ≈ 9.49.  Set 0 to disable.
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

        self._tracked: list[Track] = []  # Tracked + New tracks
        self._lost: list[Track] = []  # Lost tracks
        self.frame_id: int = 0

        Track._id_counter = 0  # reset IDs on new tracker instance

    # ── Main entry point ──────────────────────────────────────────────────

    def predict_only(self) -> list[dict[str, Any]]:
        """
        Advance one frame without detection association.

        This is useful when detector is intentionally skipped (e.g. detect every
        N frames). Tracks stay in their current states instead of being marked
        Lost due to missing detections on skipped frames.
        """
        self.frame_id += 1
        for t in self._tracked + self._lost:
            t.predict()
        return [t.as_dict() for t in self._tracked]

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process one frame of detections.

        detections : output of FaceDetector.detect() — list of
                     {"bbox": [x1,y1,x2,y2], "score": float, ...}

        Returns    : list of active track dicts with "track_id" added.
        """
        self.frame_id += 1

        # ── Parse detections ──────────────────────────────────────────────
        dets_high: list[tuple[np.ndarray, float]] = []  # (bbox_xyxy, score)
        dets_low: list[tuple[np.ndarray, float]] = []

        for d in detections:
            bbox = np.asarray(d["bbox"], dtype=np.float32)
            score = float(d["score"])
            if score >= self.high_thresh:
                dets_high.append((bbox, score))
            elif score >= self.low_thresh:
                dets_low.append((bbox, score))

        # ── Step 1: Predict all existing tracks ───────────────────────────
        for t in self._tracked + self._lost:
            t.predict()

        # Active = currently Tracked or New tracks (confident enough to be
        # matched in Stage 1); Lost tracks only participate in Stage 2.
        active_tracks = [t for t in self._tracked]

        # ── Step 2 (Stage 1): high-conf dets ↔ active tracks ─────────────
        matches_1, unmatched_tracks_1, unmatched_dets_high = _associate(
            tracks=active_tracks,
            detections=dets_high,
            iou_thresh=self.iou_thresh,
            gate_mahal=self.gate_mahal,
        )

        for t_idx, d_idx in matches_1:
            active_tracks[t_idx].update(*dets_high[d_idx])

        # ── Step 3 (Stage 2): low-conf dets ↔ unmatched active tracks ────
        #  Goal: recover tracks that were missed due to occlusion / motion blur.
        #  We use ONLY the tracks that were not matched in Stage 1.
        #  Key ByteTrack insight: low-conf dets are more likely to be
        #  partially occluded faces of existing tracks than new faces.
        unmatched_active = [active_tracks[i] for i in unmatched_tracks_1]
        matches_2, unmatched_tracks_2, _ = _associate(
            tracks=unmatched_active,
            detections=dets_low,
            iou_thresh=self.iou_thresh_low,  # stricter threshold
            gate_mahal=self.gate_mahal,
        )

        for t_idx, d_idx in matches_2:
            unmatched_active[t_idx].update(*dets_low[d_idx])

        # ── Step 4 (Stage 3): high-conf dets ↔ lost tracks ───────────────
        #  Re-identify lost faces that re-appear.
        still_unmatched_active = [unmatched_active[i] for i in unmatched_tracks_2]
        remaining_high = [dets_high[i] for i in unmatched_dets_high]
        matches_3, _, unmatched_dets_high2 = _associate(
            tracks=self._lost,
            detections=remaining_high,
            iou_thresh=self.iou_thresh,
            gate_mahal=self.gate_mahal,
        )

        for t_idx, d_idx in matches_3:
            self._lost[t_idx].update(*remaining_high[d_idx])
            self._lost[t_idx].state = TrackState.Tracked

        matched_lost = {t_idx for t_idx, _ in matches_3}
        for t_idx, t in enumerate(self._lost):
            if t_idx in matched_lost:
                continue
            t.frames_lost += 1
            if t.frames_lost > self.max_lost:
                t.mark_removed()

        # ── Step 5: Mark unmatched tracks as lost / removed ───────────────
        for t in still_unmatched_active:
            t.mark_lost()

        # ── Step 6: Initialize new tracks from leftover high-conf dets ────
        new_tracks: list[Track] = []
        for d_idx in unmatched_dets_high2:
            t = Track(*remaining_high[d_idx])
            new_tracks.append(t)

        # ── Step 7: Promote New tracks that have enough hits ──────────────
        for t in self._tracked:
            if t.state == TrackState.New:
                if t.consecutive >= self.min_hits:
                    t.state = TrackState.Tracked

        # ── Update internal lists ─────────────────────────────────────────
        #  Tracked: confirmed + new (in warm-up) tracks that are still matched
        self._tracked = (
            [
                t
                for t in self._tracked
                if t.state in (TrackState.Tracked, TrackState.New)
            ]
            + [t for t in self._lost if t.state == TrackState.Tracked]  # re-found
            + new_tracks
        )
        self._lost = [
            t
            for t in self._lost
            if t.state == TrackState.Lost and t.frames_lost <= self.max_lost
        ] + [t for t in still_unmatched_active if t.state == TrackState.Lost]

        # Return only confirmed + warm-up tracks (not Removed)
        return [t.as_dict() for t in self._tracked]

    # ── Draw ──────────────────────────────────────────────────────────────

    def draw(
        self,
        image: "np.ndarray",
        tracks: list[dict[str, Any]],
        *,
        confirmed_only: bool = True,
    ) -> "np.ndarray":
        """
        Draw track boxes and IDs on a BGR ndarray.
        Each track_id gets a deterministic distinct color.
        """
        import cv2

        canvas = image.copy()
        for tr in tracks:
            if confirmed_only and tr["state"] != "Tracked":
                continue

            x1, y1, x2, y2 = (int(round(v)) for v in tr["bbox"])
            tid = tr["track_id"]
            color = _id_color(tid)

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


# ══════════════════════════════════════════════════════════════════════════════
# 4. Association helpers
# ══════════════════════════════════════════════════════════════════════════════


def _associate(
    tracks: list[Track],
    detections: list[tuple[np.ndarray, float]],
    iou_thresh: float,
    gate_mahal: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Hungarian assignment between tracks and detections.

    Returns
    -------
    matches          : [(track_idx, det_idx), ...]
    unmatched_tracks : [track_idx, ...]
    unmatched_dets   : [det_idx, ...]
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    # ── Build cost matrix (IoU distance = 1 - IoU) ────────────────────────
    boxes_t = np.stack([t.bbox_xyxy for t in tracks])  # (T, 4)
    boxes_d = np.stack([d[0] for d in detections])  # (D, 4)
    cost = 1.0 - _iou_matrix(boxes_t, boxes_d)  # (T, D)

    # ── Optional Mahalanobis gating ───────────────────────────────────────
    #  Suppress assignments that are geometrically implausible in Kalman space.
    if gate_mahal > 0:
        meas = _xyxy_to_cxcywh(boxes_d)  # (D, 4)
        for i, t in enumerate(tracks):
            dists = t.kalman.mahalanobis(meas)  # (D,)
            cost[i, dists > gate_mahal] = 1.0 + 1e-6  # impossible

    # ── Hungarian algorithm ───────────────────────────────────────────────
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_t = set()
    matched_d = set()
    matches: list[tuple[int, int]] = []

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] > 1.0 - iou_thresh:  # insufficient overlap → reject
            continue
        matches.append((int(r), int(c)))
        matched_t.add(int(r))
        matched_d.add(int(c))

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_d]

    return matches, unmatched_tracks, unmatched_dets


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorized IoU between two sets of boxes.

    a : (M, 4) — [x1, y1, x2, y2]
    b : (N, 4) — [x1, y1, x2, y2]
    Returns (M, N)
    """
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)  # (M,)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)  # (N,)

    # Intersection
    ix1 = np.maximum(a[:, 0:1], b[:, 0])  # (M, N)
    iy1 = np.maximum(a[:, 1:2], b[:, 1])
    ix2 = np.minimum(a[:, 2:3], b[:, 2])
    iy2 = np.minimum(a[:, 3:4], b[:, 3])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)  # (M, N)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Coordinate converters
# ══════════════════════════════════════════════════════════════════════════════


def _xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """
    [x1, y1, x2, y2] → [cx, cy, w, h]
    Works for both (4,) and (N, 4).
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    out = np.empty_like(boxes)
    out[..., 0] = (boxes[..., 0] + boxes[..., 2]) * 0.5  # cx
    out[..., 1] = (boxes[..., 1] + boxes[..., 3]) * 0.5  # cy
    out[..., 2] = boxes[..., 2] - boxes[..., 0]  # w
    out[..., 3] = boxes[..., 3] - boxes[..., 1]  # h
    return out


def _cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h] → [x1, y1, x2, y2]  (1-D only)."""
    cx, cy, w, h = box
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Misc helpers
# ══════════════════════════════════════════════════════════════════════════════


def _id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, perceptually-distinct BGR color for a track ID."""
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
