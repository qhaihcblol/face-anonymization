"""Visualize face tracking over a video (stable IDs + motion trails).

Unlike ``test_face_tracker_compare.py`` (which contrasts detection-only vs.
tracking to *justify* the tracker), this is a plain demonstration of what the
tracker produces: the detector feeds every frame into ByteTrack, and each
confirmed track is drawn with

  * an ID-coloured box (the colour is keyed to ``track_id``, so the same physical
    face keeps its colour for its whole lifetime),
  * a ``#id  score`` label,
  * the five facial landmarks, and
  * a fading motion trail of the box centre over the last ``--trail`` frames,

so a reader can see identities persist and move smoothly across frames. A live
counter shows how many tracks are currently active.

Outputs (under ``outputs/``):
  * ``face_tracking_demo.mp4``  — the full annotated clip.
  * ``face_tracking_demo.png``  — a strip of consecutive frames (the still figure).

Usage:
    python -m test_scripts.test_face_tracking_demo
    python -m test_scripts.test_face_tracking_demo --input test_videos/video_track.mp4
    python -m test_scripts.test_face_tracking_demo --start 80 --count 4 --no-video
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_tracking.face_tracker import ByteTracker, _track_color
from ai_core.video_io.video_io import VideoIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "test_videos" / "video_track.mp4"
DEFAULT_ONNX = PROJECT_ROOT / "ai_core" / "face_detection" / "onnx" / "retinaface_best.onnx"


def _center(bbox) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))


def _draw_track(canvas: np.ndarray, track: dict, trail: deque, thickness: int = 2) -> None:
    """Draw one confirmed track: box + label + landmarks + fading trail."""
    tid = int(track["track_id"])
    color = _track_color(tid)
    x1, y1, x2, y2 = (int(round(v)) for v in track["bbox"])
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    label = f"#{tid}  {float(track.get('score', 0.0)):.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(canvas, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    landmarks = track.get("landmarks")
    if landmarks:
        for px, py in landmarks:
            cv2.circle(canvas, (int(px), int(py)), 2, color, -1, cv2.LINE_AA)

    # Fading motion trail: older points are thinner/dimmer.
    pts = list(trail)
    n = len(pts)
    for i in range(1, n):
        alpha = i / n
        cv2.line(canvas, pts[i - 1], pts[i],
                 tuple(int(c * alpha) for c in color),
                 max(1, int(round(thickness * alpha))), cv2.LINE_AA)


def _hud(canvas: np.ndarray, idx: int, n_active: int) -> None:
    """Top-left heads-up text: frame index + active track count."""
    text = f"frame {idx}   active tracks: {n_active}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(canvas, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


def _annotate(
    video_path: Path,
    detector: FaceDetector,
    tracker: ByteTracker,
    trail_len: int,
):
    """Single pass: yield (frame_index, annotated BGR frame) for the whole clip."""
    vio = VideoIO()
    fps = vio.get_video_metadata(str(video_path)).fps
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))
    frames = []

    for idx, frame in enumerate(vio.iter_frames(str(video_path))):
        tracks = [t for t in tracker.update(detector.detect(frame)) if t.get("state") == "Tracked"]
        canvas = frame.copy()
        for t in tracks:
            tid = int(t["track_id"])
            trails[tid].append(_center(t["bbox"]))
            _draw_track(canvas, t, trails[tid])
        _hud(canvas, idx, len(tracks))
        frames.append((idx, canvas))

    return frames, fps


def _build_strip(frames, start: int, count: int, cell_w: int, gap: int = 6) -> np.ndarray:
    """Horizontal strip of ``count`` consecutive annotated frames from ``start``."""
    chosen = [f for f in frames if start <= f[0] < start + count]
    if not chosen:
        chosen = frames[:count]
    h, w = chosen[0][1].shape[:2]
    cell_h = int(round(cell_w * h / w))
    sep = np.full((cell_h, gap, 3), 255, dtype=np.uint8)
    cells = []
    for i, (_idx, img) in enumerate(chosen):
        if i:
            cells.append(sep)
        cells.append(cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA))
    return np.hstack(cells)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize face tracking over a video.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--conf", type=float, default=0.4, help="Detector confidence threshold.")
    p.add_argument("--trail", type=int, default=24, help="Motion-trail length (frames).")
    p.add_argument("--start", type=int, default=0, help="First frame of the still strip.")
    p.add_argument("--count", type=int, default=4, help="Frames in the still strip (3-4 typical).")
    p.add_argument("--cell-width", type=int, default=420, help="Per-cell width (px) in the strip.")
    p.add_argument("--no-video", action="store_true", help="Skip the annotated video.")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if not args.input.is_file():
        print(f"Input video not found: {args.input}", file=sys.stderr)
        return 1
    args.outdir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(onnx_path=args.onnx, conf_threshold=args.conf)
    tracker = ByteTracker()

    print(f"Tracking faces in {args.input.name}...")
    frames, fps = _annotate(args.input, detector, tracker, max(args.trail, 1))
    print(f"Processed {len(frames)} frames at {fps:.1f} fps.")

    strip = _build_strip(frames, args.start, args.count, args.cell_width)
    strip_path = args.outdir / "face_tracking_demo.png"
    cv2.imwrite(str(strip_path), strip)
    print(f"Saved frame strip -> {strip_path}  ({strip.shape[1]}x{strip.shape[0]})")

    if not args.no_video:
        video_path = args.outdir / "face_tracking_demo.mp4"
        VideoIO().write_frames((img for _i, img in frames), str(video_path), fps=fps)
        print(f"Saved annotated video -> {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
