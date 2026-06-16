"""Export a strip image of a few consecutive frames from a video.

A small visual helper for the thesis: it grabs N (default 3) consecutive frames
starting at a chosen frame index and lays them out side-by-side into a single
PNG, with a small caption (frame index + timestamp) under each tile. Handy for
illustrating temporal behaviour (tracking, flicker, stabilization) in the text.

Usage:
    python -m test_scripts.test_consecutive_frames
    python -m test_scripts.test_consecutive_frames --input test_videos/hai1.mp4 --start 80 --count 4
    python -m test_scripts.test_consecutive_frames --input test_videos/video_track.mp4 --start 89 --count 3 --step 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "test_videos" / "hai1.mp4"


def _grab_frames(cap: cv2.VideoCapture, start: int, count: int, step: int):
    """Return [(frame_index, BGR image), ...] of consecutive frames."""
    frames = []
    for i in range(count):
        idx = start + i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append((idx, frame))
    return frames


def _caption(frame, text: str, height: int = 34):
    """Add a black caption bar with white text under a frame."""
    h, w = frame.shape[:2]
    bar = np.zeros((height, w, 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, height - 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([frame, bar])


def main() -> int:
    p = argparse.ArgumentParser(description="Save a strip of consecutive video frames.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input video.")
    p.add_argument("--start", type=int, default=0, help="First frame index.")
    p.add_argument("--count", type=int, default=3, help="How many frames (3-4 is typical).")
    p.add_argument("--step", type=int, default=1, help="Stride between frames (1 = strictly consecutive).")
    p.add_argument("--gap", type=int, default=8, help="White gap (px) between tiles.")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--out", type=Path, default=None, help="Explicit output PNG path.")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        return 1

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Could not open video: {args.input}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = _grab_frames(cap, args.start, args.count, args.step)
    cap.release()
    if not frames:
        print(f"No frames read (start={args.start}, total={total}).")
        return 1

    tiles = []
    for idx, frame in frames:
        ts = f"  t={idx / fps:.2f}s" if fps > 0 else ""
        tiles.append(_caption(frame, f"frame {idx}{ts}"))

    h = max(t.shape[0] for t in tiles)
    sep = np.full((h, args.gap, 3), 255, dtype=np.uint8)
    row = []
    for i, t in enumerate(tiles):
        if t.shape[0] != h:  # pad shorter tiles to common height
            t = np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 255, dtype=np.uint8)])
        if i:
            row.append(sep)
        row.append(t)
    strip = np.hstack(row)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.out or args.outdir / f"{args.input.stem}_frames_{args.start}-{args.start + (len(frames) - 1) * args.step}.png"
    cv2.imwrite(str(out), strip)
    print(f"Saved {len(frames)} consecutive frames "
          f"({[idx for idx, _ in frames]}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
