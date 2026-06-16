"""Visualize face alignment: original face -> InsightFace / FFHQ canonical crops.

The alignment step warps each detected face onto a *canonical* template so the
five facial landmarks land on fixed reference positions, regardless of the head's
in-plane rotation/scale/position. This figure makes that concrete: one row per
detected face, columns

  * **Original (detected)** — square crop around the face with the 5 detected
    landmarks (the input to the affine fit) and the bounding box.
  * **InsightFace 112x112** — ArcFace/recognition template (eyes ~31% apart),
    the crop fed to the recognizer / swapper.
  * **FFHQ 512x512** — the StyleGAN/restoration template (looser, more forehead +
    chin), the crop fed to GFPGAN-style restoration.

The five landmarks keep a fixed colour across every panel, so a reader can see
the *same* points move onto the *same* canonical locations. On the two aligned
panels the canonical reference target is overlaid as hollow rings — the warped
landmarks (filled dots) snapping onto those rings is the whole point of alignment.

Output (under ``outputs/``):
  * ``face_alignment_demo.png`` — the grid (faces x {original, insightface, ffhq}).

Usage:
    python -m test_scripts.test_face_alignment_demo
    python -m test_scripts.test_face_alignment_demo --input test_images/female_1.jpeg
    python -m test_scripts.test_face_alignment_demo --input test_videos/hai1.mp4 --frame 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.video_io.video_io import VideoIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "test_images" / "female_1.jpeg"
DEFAULT_ONNX = PROJECT_ROOT / "ai_core" / "face_detection" / "onnx" / "retinaface_best.onnx"

# Fixed BGR colour per landmark (left eye, right eye, nose, left mouth, right mouth).
LANDMARK_COLORS = [
    (0, 0, 255),     # left eye  - red
    (0, 165, 255),   # right eye - orange
    (0, 255, 255),   # nose      - yellow
    (0, 255, 0),     # left mouth- green
    (255, 128, 0),   # right mouth- blue
]
PANEL = 360          # display size (px) of every panel
COLUMNS = ["Original (detected)", "InsightFace 112x112", "FFHQ 512x512"]


def _load_image(path: Path, frame: int) -> np.ndarray | None:
    """Load an image, or grab frame ``frame`` if ``path`` is a video."""
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        vio = VideoIO()
        for idx, f in enumerate(vio.iter_frames(str(path))):
            if idx == frame:
                return f
        return None
    return cv2.imread(str(path))


def _draw_landmarks(canvas, pts, *, radius=4, refs=None):
    """Filled dots for ``pts``; if ``refs`` given, hollow rings for the targets."""
    if refs is not None:
        for (rx, ry), color in zip(refs, LANDMARK_COLORS):
            cv2.circle(canvas, (int(round(rx)), int(round(ry))), radius + 4, color, 2, cv2.LINE_AA)
    for (px, py), color in zip(pts, LANDMARK_COLORS):
        cv2.circle(canvas, (int(round(px)), int(round(py))), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(round(px)), int(round(py))), radius, (255, 255, 255), 1, cv2.LINE_AA)


def _square_crop(image, bbox, lmk, margin=0.6):
    """Square crop around ``bbox`` (with margin); return (crop, landmarks-in-crop)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (0.5 + margin)
    sx1, sy1 = int(max(cx - half, 0)), int(max(cy - half, 0))
    sx2, sy2 = int(min(cx + half, w)), int(min(cy + half, h))
    crop = image[sy1:sy2, sx1:sx2].copy()
    shifted = lmk - np.asarray([sx1, sy1], dtype=np.float32)
    return crop, shifted


def _fit_panel(img, lmk, refs=None):
    """Resize ``img`` to PANEL x PANEL and scale the landmarks/refs to match."""
    h, w = img.shape[:2]
    sx, sy = PANEL / w, PANEL / h
    panel = cv2.resize(img, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    scale = np.asarray([sx, sy], dtype=np.float32)
    _draw_landmarks(panel, lmk * scale, refs=(refs * scale) if refs is not None else None)
    return panel


def _header(text, width, height=40, bg=(60, 60, 60)):
    bar = np.full((height, width, 3), bg, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(bar, text, ((width - tw) // 2, height - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return bar


def main() -> int:
    p = argparse.ArgumentParser(description="Visualize FFHQ vs InsightFace alignment.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--conf", type=float, default=0.4, help="Detector confidence threshold.")
    p.add_argument("--frame", type=int, default=0, help="Frame index if input is a video.")
    p.add_argument("--max-faces", type=int, default=4, help="Cap rows so the figure stays readable.")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    args = p.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1
    image = _load_image(args.input, args.frame)
    if image is None:
        print("Could not read image / frame.", file=sys.stderr)
        return 1

    detector = FaceDetector(onnx_path=args.onnx, conf_threshold=args.conf)
    insight = FaceAligner(output_size=(112, 112), mode=AlignMode.INSIGHTFACE)
    ffhq = FaceAligner(output_size=(512, 512), mode=AlignMode.FFHQ)

    detections = detector.detect(image)
    if not detections:
        print("No faces detected.", file=sys.stderr)
        return 1
    detections = sorted(detections, key=lambda d: -d.score)[: args.max_faces]
    print(f"Detected {len(detections)} face(s); rendering alignment grid...")

    sep = np.full((PANEL, 4, 3), 255, dtype=np.uint8)
    rows = []
    for i, det in enumerate(detections):
        src = det.landmarks.as_array()

        crop, crop_lmk = _square_crop(image, det.bbox, src)
        orig_panel = _fit_panel(crop, crop_lmk)

        ins_aligned, ins_img = insight.align_and_warp(image, det)
        ins_panel = _fit_panel(ins_img, ins_aligned.landmarks.as_array(),
                               refs=insight.reference_landmarks)

        ffhq_aligned, ffhq_img = ffhq.align_and_warp(image, det)
        ffhq_panel = _fit_panel(ffhq_img, ffhq_aligned.landmarks.as_array(),
                                refs=ffhq.reference_landmarks)

        row = np.hstack([orig_panel, sep, ins_panel, sep, ffhq_panel])
        # Left margin with the face index.
        label = np.full((row.shape[0], 44, 3), 255, dtype=np.uint8)
        cv2.putText(label, f"#{i}", (6, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(np.hstack([label, row]))

    body = np.vstack([np.vstack([r, np.full((4, r.shape[1], 3), 255, dtype=np.uint8)]) for r in rows])

    # Column headers aligned over the three panels (after the 44px index margin).
    head_cells = [_header("", 44)]
    for c, name in enumerate(COLUMNS):
        head_cells.append(_header(name, PANEL))
        if c < len(COLUMNS) - 1:
            head_cells.append(np.full((40, 4, 3), 255, dtype=np.uint8))
    header = np.hstack(head_cells)
    grid = np.vstack([header, body])

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "face_alignment_demo.png"
    cv2.imwrite(str(out), grid)
    print(f"Saved alignment grid -> {out}  ({grid.shape[1]}x{grid.shape[0]})")
    print("Rings = canonical reference targets; dots = warped landmarks "
          "(dots on rings = correct alignment).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
