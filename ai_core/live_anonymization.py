"""Real-time, frame-by-frame face anonymization for live camera streams.

This is the *streaming* counterpart to :class:`~ai_core.video_anonymization.VideoAnonymization`.
Where the offline class is **file-in → file-out** (and muxes audio, supports face
swap, builds a fresh tracker per run), the live class is **frame-in → frame-out**:

* **Stateful.** The tracker persists across calls, so identities and Kalman motion
  carry over between frames of the same session. Call :meth:`reset` to start a new
  session (e.g. a new camera) with clean state.
* **Latency-oriented.** The detector runs only every Nth frame
  (:attr:`LiveVisualConfig.detect_interval`); in-between frames reuse the tracker's
  motion predictions. This is the single biggest real-time lever.
* **Obfuscation only.** Blur / pixelate / mask / blackout / none — no face swap
  (too heavy for real-time CPU) and no audio (a separate real-time DSP problem).

The heavy models (detector, parser, anonymizer) are injected already-loaded so a
caller can share one set of ONNX sessions across every live session — only the
lightweight per-session tracker is owned here.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ai_core.face_anonymization.face_anonymizer import (
    AnonymizationMethod,
    ObfuscationParams,
)
from ai_core.face_tracking.face_tracker import ByteTracker

if TYPE_CHECKING:
    from ai_core.face_anonymization.face_anonymizer import FaceAnonymizer
    from ai_core.face_detection.face_detector import FaceDetector

__all__ = [
    "AnonymizationMethod",
    "LiveFaceAnonymizer",
    "LiveFrameResult",
    "LiveVisualConfig",
    "ObfuscationParams",
]


@dataclass(slots=True)
class LiveVisualConfig:
    """Per-session look of the live filter — the streaming sibling of ``VisualOptions``.

    Mutable: the front-end can change the filter mid-stream (e.g. blur → pixelate)
    and the session keeps its tracker, so identities stay stable across the switch.
    """

    method: AnonymizationMethod | str = AnonymizationMethod.BLUR
    # Run the detector every N frames (>= 1); in-between frames reuse tracker
    # predictions. Higher = lower latency, looser face following. 1 = every frame.
    detect_interval: int = 2
    # Also obfuscate "New" (unconfirmed) tracks, not just confirmed "Tracked" ones.
    # Hides faces a frame or two sooner at the cost of the odd false positive.
    blur_new: bool = False
    # Overlay tracker boxes/ids on the output (server-side debug visualization).
    draw_tracks: bool = False
    # Per-session obfuscation strengths (blur kernel / pixelation / mask colour).
    # None -> use the FaceAnonymizer instance's own configured defaults.
    obfuscation: ObfuscationParams | None = None

    @property
    def resolved_method(self) -> AnonymizationMethod:
        """Coerce ``method`` to an :class:`AnonymizationMethod`, rejecting SWAP.

        Face swap is offline-only (it needs a source identity and per-frame
        landmarks), so routing it here would be a silent no-op; it is rejected early.
        """
        method = self.method
        if isinstance(method, str):
            method = AnonymizationMethod(method.strip().lower())
        if not isinstance(method, AnonymizationMethod):
            raise TypeError(
                "LiveVisualConfig.method must be AnonymizationMethod or str, "
                f"got {type(method).__name__}"
            )
        if method is AnonymizationMethod.SWAP:
            raise ValueError("Face swap is not supported on the live path.")
        return method

    @property
    def resolved_detect_interval(self) -> int:
        return max(int(self.detect_interval), 1)


@dataclass(slots=True)
class LiveFrameResult:
    """One processed frame plus the diagnostics a UI overlay / HUD needs."""

    frame: np.ndarray
    # Tracks actually obfuscated this frame (each a dict: track_id/bbox/score/state).
    tracks: list[dict[str, Any]]
    # Whether the detector ran this frame (vs. a tracker predict-only frame).
    detected: bool
    # Detector latency of the most recent detect frame, in milliseconds.
    detect_ms: float
    # Total wall time spent in :meth:`LiveFaceAnonymizer.process_frame`, in ms.
    process_ms: float


def _fresh_tracker(template: ByteTracker) -> ByteTracker:
    """A new tracker with the same thresholds — clean per-session state."""
    return ByteTracker(
        high_thresh=template.high_thresh,
        low_thresh=template.low_thresh,
        max_lost=template.max_lost,
        min_hits=template.min_hits,
        iou_thresh=template.iou_thresh,
        iou_thresh_low=template.iou_thresh_low,
        gate_mahal=template.gate_mahal,
    )


class LiveFaceAnonymizer:
    """Detect → track → obfuscate one live frame at a time.

    The detector and anonymizer are shared (read-only inference), but the tracker is
    cloned per instance so concurrent sessions never share motion state.
    """

    def __init__(
        self,
        *,
        face_detector: "FaceDetector",
        face_tracker: ByteTracker,
        face_anonymizer: "FaceAnonymizer",
        config: LiveVisualConfig | None = None,
    ) -> None:
        self.face_detector = face_detector
        self.face_anonymizer = face_anonymizer
        # Own a clean tracker; keep the passed-in one as the template for reset().
        self._tracker_template = face_tracker
        self.face_tracker = _fresh_tracker(face_tracker)

        self._config = config if config is not None else LiveVisualConfig()
        self._config.resolved_method  # validate up-front (rejects SWAP)

        self._frame_idx = 0
        self._last_detect_ms = 0.0
        self._tracks: list[dict[str, Any]] = []

    @property
    def config(self) -> LiveVisualConfig:
        return self._config

    def configure(self, config: LiveVisualConfig) -> None:
        """Swap the active filter without dropping tracker state.

        Changing the look mid-stream (blur → pixelate, new strengths, …) keeps face
        identities stable; only :meth:`reset` clears the tracker.
        """
        config.resolved_method  # validate before adopting
        self._config = config

    def reset(self) -> None:
        """Start a fresh session: clean tracker + counters (e.g. a new camera)."""
        self.face_tracker = _fresh_tracker(self._tracker_template)
        self._frame_idx = 0
        self._last_detect_ms = 0.0
        self._tracks = []

    def process_frame(self, frame_bgr: np.ndarray) -> LiveFrameResult:
        """Obfuscate every face in one BGR frame using the current config."""
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            raise ValueError("frame_bgr must be an (H, W, 3) BGR image")

        t0 = time.perf_counter()
        config = self._config
        method = config.resolved_method

        # Pure passthrough: skip detection entirely when there is nothing to draw or
        # obfuscate, so an idle "None" filter costs almost nothing.
        if method is AnonymizationMethod.NONE and not config.draw_tracks:
            self._tracks = []
            self._frame_idx += 1
            return LiveFrameResult(
                frame=frame_bgr,
                tracks=[],
                detected=False,
                detect_ms=0.0,
                process_ms=(time.perf_counter() - t0) * 1000.0,
            )

        run_detect = (self._frame_idx % config.resolved_detect_interval) == 0
        if run_detect:
            t_detect = time.perf_counter()
            detections = self.face_detector.detect(frame_bgr)
            self._last_detect_ms = (time.perf_counter() - t_detect) * 1000.0
            self._tracks = self.face_tracker.update(detections)
        else:
            self._tracks = self.face_tracker.predict_only()

        if config.blur_new:
            tracks_to_anonymize = self._tracks
        else:
            tracks_to_anonymize = [
                track for track in self._tracks if track.get("state") == "Tracked"
            ]

        output = self.face_anonymizer.anonymize(
            frame_bgr,
            tracks_to_anonymize,
            method=method,
            params=config.obfuscation,
        )

        if config.draw_tracks:
            output = self.face_tracker.draw(
                output, self._tracks, confirmed_only=not config.blur_new
            )

        self._frame_idx += 1
        return LiveFrameResult(
            frame=output,
            tracks=tracks_to_anonymize,
            detected=run_detect,
            detect_ms=self._last_detect_ms,
            process_ms=(time.perf_counter() - t0) * 1000.0,
        )
