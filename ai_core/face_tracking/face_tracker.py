from __future__ import annotations

import enum
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["ByteTracker"]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Coordinate helpers
# ══════════════════════════════════════════════════════════════════════════════


def _xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from corner format to center format.

    Input  : (..., 4)  [x1, y1, x2, y2]
    Output : (..., 4)  [cx, cy, w,  h ]
    Works for both a single box (4,) and a batch (N, 4).
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    out = np.empty_like(boxes)
    out[..., 0] = (boxes[..., 0] + boxes[..., 2]) * 0.5  # cx
    out[..., 1] = (boxes[..., 1] + boxes[..., 3]) * 0.5  # cy
    out[..., 2] = boxes[..., 2] - boxes[..., 0]  # w
    out[..., 3] = boxes[..., 3] - boxes[..., 1]  # h
    return out


def _cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """
    Convert a single center-format box to corner format.

    Input  : (4,)  [cx, cy, w, h]
    Output : (4,)  [x1, y1, x2, y2]
    """
    cx, cy, w, h = box
    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        dtype=np.float32,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Kalman filter — constant-velocity model for bounding boxes
# ══════════════════════════════════════════════════════════════════════════════


class KalmanBoxFilter:
    """
    Single-target constant-velocity Kalman filter.

    State vector  (8-dim): [cx, cy, w, h, vx, vy, vw, vh]
    Measurement   (4-dim): [cx, cy, w, h]

    Noise scales with box height h so that small and large faces are
    treated proportionally (idea from DeepSORT / BoT-SORT).

    Class-level matrices F and H are shared across all instances to
    avoid re-allocating them on every Track creation.
    """

    # Noise weights (tunable hyper-parameters)
    _W_POS: float = 1.0 / 20.0  # position noise weight
    _W_VEL: float = 1.0 / 160.0  # velocity noise weight

    # ── State-transition matrix: x_{t+1} = F @ x_t ───────────────────────
    # [pos]   [I  I] [pos]
    # [vel] = [0  I] [vel]
    F: np.ndarray = np.eye(8, dtype=np.float64)
    F[:4, 4:] = np.eye(4)

    # ── Observation matrix: z = H @ x  (observe position only) ───────────
    H: np.ndarray = np.zeros((4, 8), dtype=np.float64)
    H[:4, :4] = np.eye(4)

    # ── Instance ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.x: np.ndarray = np.zeros(8, dtype=np.float64)  # state mean
        self.P: np.ndarray = np.eye(8, dtype=np.float64)  # state covariance

    # ── Noise covariances (adaptive to current face height) ───────────────

    def _Q(self) -> np.ndarray:
        """Process noise covariance Q — scales with current box height."""
        h = max(abs(float(self.x[3])), 1.0)
        sp = self._W_POS * h
        sv = self._W_VEL * h
        diag = np.square([sp, sp, sp, sp, sv, sv, sv, sv])
        return np.diag(diag)

    def _R(self) -> np.ndarray:
        """Measurement noise covariance R — scales with current box height."""
        h = max(abs(float(self.x[3])), 1.0)
        sp = self._W_POS * h
        return np.diag(np.square([sp, sp, sp, sp]))

    # ── Public API ────────────────────────────────────────────────────────

    def initiate(self, bbox_cxcywh: np.ndarray) -> None:
        """
        Initialise state from the very first detection.

        Parameters
        ----------
        bbox_cxcywh : (4,)  [cx, cy, w, h]

        Initial velocity is zero; position uncertainty is moderate;
        velocity uncertainty is large (we have no velocity estimate yet).
        """
        cx, cy, w, h = bbox_cxcywh.astype(np.float64)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        h_val = max(abs(h), 1.0)
        sp = 2.0 * self._W_POS * h_val  # moderate position uncertainty
        sv = 10.0 * self._W_VEL * h_val  # large velocity uncertainty
        self.P = np.diag(np.square([sp, sp, sp, sp, sv, sv, sv, sv]))

    def predict(self) -> np.ndarray:
        """
        Time-update (predict) step — advance state by one frame.

        Returns
        -------
        predicted_cxcywh : (4,)  [cx, cy, w, h]
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self._Q()
        return self.x[:4].copy()

    def update(self, bbox_cxcywh: np.ndarray) -> None:
        """
        Measurement-update (correct) step.

        Parameters
        ----------
        bbox_cxcywh : (4,)  [cx, cy, w, h]  from the matched detection.

        Standard Kalman update equations:
            S = H P H' + R          (innovation covariance)
            K = P H' S^{-1}         (Kalman gain)
            x = x + K (z - H x)    (state correction)
            P = (I - K H) P         (covariance correction)
        """
        z = bbox_cxcywh.astype(np.float64)
        S = self.H @ self.P @ self.H.T + self._R()  # (4, 4)
        K = self.P @ self.H.T @ np.linalg.inv(S)  # (8, 4)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def mahalanobis_distance(self, measurements_cxcywh: np.ndarray) -> np.ndarray:
        """
        Mahalanobis distance from the current predicted state to each candidate
        measurement.  Used for gating: implausible assignments are discarded
        before Hungarian matching.

        Parameters
        ----------
        measurements_cxcywh : (N, 4)  — candidate [cx, cy, w, h]

        Returns
        -------
        distances : (N,)   scalar Mahalanobis distance per candidate

        Note: We use the SAME _R() as update() so gating is consistent
        with the filter's own noise model.
        """
        # Innovation covariance in measurement space
        S = self.H @ self.P @ self.H.T + self._R()  # (4, 4)
        S_inv = np.linalg.inv(S)  # (4, 4)

        diff = measurements_cxcywh.astype(np.float64) - (self.H @ self.x)  # (N, 4)
        # Vectorised: d_i = sqrt(diff_i' S^{-1} diff_i)
        return np.sqrt(np.einsum("ni,ij,nj->n", diff, S_inv, diff))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Track state machine
# ══════════════════════════════════════════════════════════════════════════════


class TrackState(enum.IntEnum):
    """
    Lifecycle of a single track:

        New  ──(consecutive ≥ min_hits)──▶  Tracked
        Tracked  ──(missed 1 frame)──────▶  Lost
        Lost     ──(matched again)────────▶  Tracked   (frames_lost reset)
        Lost     ──(missed > max_lost)───▶  Removed
    """

    New = 0  # just created; awaiting enough consecutive matches
    Tracked = 1  # confirmed; actively matched in recent frames
    Lost = 2  # missed; kept alive for re-identification
    Removed = 3  # exceeded max_lost; will be purged this frame


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Track — one tracked face
# ══════════════════════════════════════════════════════════════════════════════


class Track:
    """
    Represents one tracked face across frames.

    Attributes
    ----------
    track_id    : Unique, monotonically increasing integer ID.
    bbox_xyxy   : Most recent box in [x1, y1, x2, y2] format.
    score       : Detection confidence of the last matched detection.
    state       : Current lifecycle state (TrackState).
    hits        : Total number of frames this track was matched.
    consecutive : Consecutive matched frames since last miss (used for
                  promotion from New → Tracked).
    frames_lost : Consecutive missed frames (used for expiry).
    """

    _id_counter: int = 0  # class-level monotonic counter; reset by ByteTracker

    def __init__(self, bbox_xyxy: np.ndarray, score: float) -> None:
        Track._id_counter += 1
        self.track_id: int = Track._id_counter

        self.kalman = KalmanBoxFilter()
        self.kalman.initiate(_xyxy_to_cxcywh(bbox_xyxy))

        self.bbox_xyxy: np.ndarray = bbox_xyxy.astype(np.float32)
        self.score: float = float(score)
        self.state: TrackState = TrackState.New

        self.hits: int = 1  # total matched frames
        self.consecutive: int = 1  # consecutive matched frames
        self.frames_lost: int = 0  # consecutive missed frames

    # ── Kalman wrappers ───────────────────────────────────────────────────

    def predict(self) -> None:
        """
        Advance the Kalman filter by one frame.
        Must be called ONCE at the start of each frame, before association.
        Updates self.bbox_xyxy with the predicted position.
        """
        pred_cxcywh = self.kalman.predict()
        self.bbox_xyxy = _cxcywh_to_xyxy(pred_cxcywh)

    def update(self, bbox_xyxy: np.ndarray, score: float) -> None:
        """
        Correct the Kalman filter with a matched detection.
        Resets frames_lost and increments hit counters.
        """
        self.kalman.update(_xyxy_to_cxcywh(bbox_xyxy))
        self.bbox_xyxy = _cxcywh_to_xyxy(self.kalman.x[:4])
        self.score = float(score)
        self.hits += 1
        self.consecutive += 1
        self.frames_lost = 0  # track is active again

    # ── State transitions ─────────────────────────────────────────────────

    def mark_lost(self) -> None:
        """Called when no detection was matched this frame."""
        self.consecutive = 0
        self.frames_lost += 1
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Called when frames_lost exceeds max_lost."""
        self.state = TrackState.Removed

    # ── Serialisation ─────────────────────────────────────────────────────

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
# 5.  Association helpers
# ══════════════════════════════════════════════════════════════════════════════


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU between two sets of axis-aligned boxes.

    Parameters
    ----------
    a : (M, 4)  [x1, y1, x2, y2]
    b : (N, 4)  [x1, y1, x2, y2]

    Returns
    -------
    iou : (M, N)  element [i, j] = IoU(a[i], b[j])
    """
    # Areas
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)  # (M,)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)  # (N,)

    # Intersection corners via broadcasting
    ix1 = np.maximum(a[:, 0:1], b[:, 0])  # (M, N)
    iy1 = np.maximum(a[:, 1:2], b[:, 1])
    ix2 = np.minimum(a[:, 2:3], b[:, 2])
    iy2 = np.minimum(a[:, 3:4], b[:, 3])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)  # (M, N)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def _associate(
    tracks: list[Track],
    detections: list[tuple[np.ndarray, float]],
    iou_thresh: float,
    gate_mahal: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Hungarian (linear sum assignment) matching between tracks and detections.

    Cost matrix  : 1 − IoU   (lower = better match)
    Gating       : entries whose Mahalanobis distance > gate_mahal are set
                   to an impossible cost (> 1.0) so they cannot be assigned.
    Accept rule  : assignment (t, d) is accepted only if IoU ≥ iou_thresh
                   (i.e. cost ≤ 1 − iou_thresh).

    Parameters
    ----------
    tracks      : list of Track objects
    detections  : list of (bbox_xyxy, score) tuples
    iou_thresh  : minimum IoU to accept a match
    gate_mahal  : chi-square threshold for Mahalanobis gating
                  (chi-sq 95% for 4 DOF ≈ 9.49; set ≤ 0 to disable)

    Returns
    -------
    matches          : [(track_idx, det_idx), ...]
    unmatched_tracks : [track_idx, ...]
    unmatched_dets   : [det_idx, ...]
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    boxes_t = np.stack([t.bbox_xyxy for t in tracks])  # (T, 4)
    boxes_d = np.stack([d[0] for d in detections])  # (D, 4)

    # ── Cost matrix: IoU distance ─────────────────────────────────────────
    cost = 1.0 - _iou_matrix(boxes_t, boxes_d)  # (T, D)

    # ── Mahalanobis gating ────────────────────────────────────────────────
    # Convert detection corners to centre format once for all tracks.
    if gate_mahal > 0:
        meas_cxcywh = _xyxy_to_cxcywh(boxes_d)  # (D, 4)
        for i, t in enumerate(tracks):
            dists = t.kalman.mahalanobis_distance(meas_cxcywh)  # (D,)
            # Mark implausible cells with cost > 1 so Hungarian ignores them
            cost[i, dists > gate_mahal] = 2.0

    # ── Hungarian assignment ──────────────────────────────────────────────
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_t: set[int] = set()
    matched_d: set[int] = set()
    matches: list[tuple[int, int]] = []

    iou_cost_thresh = 1.0 - iou_thresh  # cost threshold (lower is better)

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] > iou_cost_thresh:
            # IoU too low (or gated) — reject this pair
            continue
        matches.append((int(r), int(c)))
        matched_t.add(int(r))
        matched_d.add(int(c))

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_d]

    return matches, unmatched_tracks, unmatched_dets


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ByteTracker — main class
# ══════════════════════════════════════════════════════════════════════════════


class ByteTracker:
    """
    ByteTrack multi-face tracker.

    Typical usage
    -------------
    tracker = ByteTracker()

    for frame in video:
        dets   = face_detector.detect(frame)     # list of {"bbox", "score", ...}
        tracks = tracker.update(dets)
        canvas = tracker.draw(frame, tracks)

    Parameters
    ----------
    high_thresh    : Confidence threshold for "high" detections (Stage 1).
    low_thresh     : Minimum confidence; detections below this are ignored.
    max_lost       : Frames a Lost track survives before Removal.
    min_hits       : Consecutive matches required to confirm a New track.
    iou_thresh     : IoU threshold for Stage 1 & 3 acceptance.
    iou_thresh_low : IoU threshold for Stage 2 (lost tracks → stricter).
    gate_mahal     : Mahalanobis distance gate. Chi-sq 95% for 4 DOF ≈ 9.49.
                     Set ≤ 0 to disable gating entirely.
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

        # Internal track pools
        self._tracked: list[Track] = []  # New + Tracked (actively matched)
        self._lost: list[Track] = []  # Lost (waiting for re-ID)
        self.frame_id: int = 0

        Track._id_counter = 0  # reset monotonic ID counter for this tracker

    # ── Predict-only (no detections this frame) ───────────────────────────

    def predict_only(self) -> list[dict[str, Any]]:
        """
        Advance all tracks by one frame without running association.

        Use this when the detector is intentionally skipped (e.g. detect every
        N frames for speed).  frames_lost IS incremented so that Lost tracks
        still expire correctly even during skipped frames.

        Returns the current active (Tracked / New) track dicts.
        """
        self.frame_id += 1

        # Predict positions
        for t in self._tracked + self._lost:
            t.predict()

        # Age lost tracks so they still expire on schedule
        still_alive: list[Track] = []
        for t in self._lost:
            t.frames_lost += 1
            if t.frames_lost > self.max_lost:
                t.mark_removed()
            else:
                still_alive.append(t)
        self._lost = still_alive

        return [t.as_dict() for t in self._tracked]

    # ── Main update (one frame with detections) ───────────────────────────

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process one frame of detections and return active tracks.

        Parameters
        ----------
        detections : output of a face detector — list of dicts, each with
                     at least {"bbox": [x1, y1, x2, y2], "score": float}.

        Returns
        -------
        List of track dicts for all New and Tracked tracks this frame,
        each containing {"track_id", "bbox", "score", "state"}.
        """
        self.frame_id += 1

        # ── Parse and split detections by confidence ──────────────────────
        dets_high: list[tuple[np.ndarray, float]] = []
        dets_low: list[tuple[np.ndarray, float]] = []

        for d in detections:
            bbox = np.asarray(d["bbox"], dtype=np.float32)
            score = float(d["score"])
            if score >= self.high_thresh:
                dets_high.append((bbox, score))
            elif score >= self.low_thresh:
                dets_low.append((bbox, score))
            # detections below low_thresh are discarded entirely

        # ── Step 1: Predict all existing tracks ───────────────────────────
        # Must be done before any association so bbox_xyxy reflects the
        # Kalman-predicted position, not the stale previous-frame position.
        for t in self._tracked + self._lost:
            t.predict()

        # ── Step 2 (Stage 1): high-conf dets ↔ active tracks ─────────────
        # "Active" means the tracks currently in self._tracked (New + Tracked).
        # Lost tracks do NOT participate in Stage 1.
        matches_1, unmatched_active_idx, unmatched_high_idx = _associate(
            tracks=self._tracked,
            detections=dets_high,
            iou_thresh=self.iou_thresh,
            gate_mahal=self.gate_mahal,
        )

        for t_idx, d_idx in matches_1:
            self._tracked[t_idx].update(*dets_high[d_idx])

        # ── Step 3 (Stage 2): low-conf dets ↔ unmatched active tracks ─────
        # ByteTrack insight: a low-conf detection for an unmatched active
        # track is more likely a partially-occluded/blurry face of an existing
        # person than a spurious background response.
        unmatched_active = [self._tracked[i] for i in unmatched_active_idx]

        matches_2, still_unmatched_active_idx, _ = _associate(
            tracks=unmatched_active,
            detections=dets_low,
            iou_thresh=self.iou_thresh_low,  # stricter threshold for low-conf
            gate_mahal=self.gate_mahal,
        )

        for t_idx, d_idx in matches_2:
            unmatched_active[t_idx].update(*dets_low[d_idx])

        # Tracks still unmatched after both Stage 1 and Stage 2 are now Lost
        still_unmatched_active = [
            unmatched_active[i] for i in still_unmatched_active_idx
        ]
        for t in still_unmatched_active:
            t.mark_lost()  # sets state = Lost, increments frames_lost, resets consecutive

        # ── Step 4 (Stage 3): remaining high-conf dets ↔ lost tracks ──────
        # Try to re-identify faces that had disappeared (occlusion, going
        # off-screen briefly, etc.).
        remaining_high = [dets_high[i] for i in unmatched_high_idx]

        matches_3, unmatched_lost_idx, unmatched_high_idx2 = _associate(
            tracks=self._lost,
            detections=remaining_high,
            iou_thresh=self.iou_thresh,
            gate_mahal=self.gate_mahal,
        )

        re_found_lost_idx: set[int] = set()
        for t_idx, d_idx in matches_3:
            t = self._lost[t_idx]
            t.update(*remaining_high[d_idx])  # update() resets frames_lost to 0
            t.state = TrackState.Tracked  # re-confirm immediately
            re_found_lost_idx.add(t_idx)

        # Age lost tracks that were NOT re-identified; remove expired ones
        still_lost: list[Track] = []
        for t_idx, t in enumerate(self._lost):
            if t_idx in re_found_lost_idx:
                continue  # will be moved to _tracked below
            t.frames_lost += 1
            if t.frames_lost > self.max_lost:
                t.mark_removed()  # will be dropped from the pool
            else:
                still_lost.append(t)

        # ── Step 5: Create new tracks for leftover high-conf detections ───
        # These are faces the tracker has never seen before.
        new_tracks: list[Track] = []
        for d_idx in unmatched_high_idx2:
            new_tracks.append(Track(*remaining_high[d_idx]))
            # State is New; they need min_hits consecutive matches before
            # they are promoted to Tracked and returned as confirmed.

        # ── Step 6: Rebuild internal pools ────────────────────────────────
        # Order matters: promotion (New → Tracked) happens INSIDE this rebuild
        # so that new_tracks created this very frame are also checked.

        # Collect all tracks that should stay in the "active" pool
        candidate_tracked: list[Track] = (
            [
                t
                for t in self._tracked
                if t.state in (TrackState.New, TrackState.Tracked)
            ]
            + [self._lost[i] for i in range(len(self._lost)) if i in re_found_lost_idx]
            + new_tracks
        )

        # Promote New → Tracked if consecutive matches ≥ min_hits
        for t in candidate_tracked:
            if t.state == TrackState.New and t.consecutive >= self.min_hits:
                t.state = TrackState.Tracked

        self._tracked = candidate_tracked

        # Lost pool: tracks that became Lost this frame + survivors from before
        self._lost = (
            still_lost
            + still_unmatched_active  # tracks that just became Lost this frame
        )

        # ── Return active tracks (New + Tracked; not Removed) ─────────────
        return [t.as_dict() for t in self._tracked]

    # ── Drawing ───────────────────────────────────────────────────────────

    def draw(
        self,
        image: "np.ndarray",
        tracks: list[dict[str, Any]],
        *,
        confirmed_only: bool = True,
    ) -> "np.ndarray":
        """
        Annotate a BGR image with track bounding boxes and IDs.

        Parameters
        ----------
        image          : OpenCV BGR ndarray (not modified in-place).
        tracks         : Output of update() or predict_only().
        confirmed_only : If True, draw only Tracked (confirmed) tracks;
                         skip New tracks still in warm-up.

        Returns
        -------
        Annotated copy of the image.
        """
        import cv2

        canvas = image.copy()
        for tr in tracks:
            if confirmed_only and tr["state"] != TrackState.Tracked.name:
                continue

            x1, y1, x2, y2 = (int(round(v)) for v in tr["bbox"])
            tid = tr["track_id"]
            color = _track_color(tid)

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Label background + text
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

    # ── Diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of tracker internals for debugging / logging."""
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


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Misc helpers
# ══════════════════════════════════════════════════════════════════════════════


def _track_color(track_id: int) -> tuple[int, int, int]:
    """
    Deterministic, perceptually-distinct BGR color for a given track ID.
    Cycles through a hand-picked palette so IDs are visually distinguishable.
    """
    palette = [
        (0, 215, 255),  # gold
        (0, 255, 127),  # spring green
        (255, 128, 0),  # dodger blue
        (255, 0, 127),  # deep pink
        (127, 0, 255),  # violet
        (0, 127, 255),  # orange
        (255, 215, 0),  # cyan-gold
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (127, 255, 0),  # chartreuse
        (0, 128, 255),  # orange-amber
        (255, 0, 0),  # blue
        (0, 255, 0),  # lime
        (0, 0, 255),  # red
        (255, 255, 0),  # aqua
    ]
    return palette[track_id % len(palette)]
