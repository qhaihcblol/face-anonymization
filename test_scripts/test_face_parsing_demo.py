"""Visualize face parsing: the BiSeNet semantic segmentation of a face.

Where ``test_face_parsing_compare.py`` argues *why* the parser mask beats a coarse
ellipse, this script simply shows *what the parser sees*: the full CelebAMask-HQ
19-class segmentation. Each detected face is FFHQ-aligned to a 512x512 crop (the
resolution the parser runs at in the pipeline), parsed into per-pixel class
labels, and rendered as

  * **Aligned face** — the RGB crop fed to BiSeNet.
  * **Segmentation** — every class painted a fixed colour (skin, eyes, brows,
    nose, lips, hair, ears, neck, glasses, ...).
  * **Overlay** — the segmentation alpha-blended over the face so the boundaries
    are visible on the real pixels.

A colour legend lists the classes actually present in the image. This is the raw
material the swap/blur mask is built from: ``DEFAULT_FACE_REGIONS`` (skin + facial
features, no hair/ears/neck) is just a subset of these classes unioned together.

Output (under ``outputs/``):
  * ``face_parsing_demo.png`` — grid (faces x {aligned, segmentation, overlay}) + legend.

Usage:
    python -m test_scripts.test_face_parsing_demo
    python -m test_scripts.test_face_parsing_demo --input test_images/male_1.jpeg
    python -m test_scripts.test_face_parsing_demo --input test_videos/video_track.mp4 --frame 100 --max-faces 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_parsing.face_parser import REGION_CLASS_INDEX, FaceParser
from ai_core.video_io.video_io import VideoIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "test_images" / "female_1.jpeg"
DEFAULT_ONNX = PROJECT_ROOT / "ai_core" / "face_detection" / "onnx" / "retinaface_best.onnx"

# Fixed BGR colour per CelebAMask-HQ class (index 0 = background -> black).
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 153, 255),      # skin       - amber
    2: (0, 255, 128),      # l-eyebrow  - green
    3: (0, 255, 0),        # r-eyebrow  - green
    4: (255, 255, 0),      # l-eye      - cyan
    5: (255, 200, 0),      # r-eye      - cyan
    6: (255, 0, 255),      # glasses    - magenta
    7: (180, 120, 0),      # l-ear      - teal
    8: (180, 80, 0),       # r-ear      - teal
    9: (200, 0, 200),      # earring
    10: (0, 0, 255),       # nose       - red
    11: (0, 80, 255),      # mouth      - orange
    12: (80, 0, 200),      # upper-lip  - dark red
    13: (160, 0, 160),     # lower-lip  - purple
    14: (255, 128, 0),     # neck       - blue
    15: (255, 0, 128),     # necklace
    16: (128, 128, 128),   # cloth      - gray
    17: (60, 60, 200),     # hair       - brick
    18: (0, 200, 200),     # hat        - olive
}
INDEX_TO_NAME = {v: k for k, v in REGION_CLASS_INDEX.items()}
PANEL = 360


def _load_image(path: Path, frame: int) -> np.ndarray | None:
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        for idx, f in enumerate(VideoIO().iter_frames(str(path))):
            if idx == frame:
                return f
        return None
    return cv2.imread(str(path))


def _class_map(parser: FaceParser, crop_bgr: np.ndarray) -> np.ndarray:
    """Full per-pixel class labels at the crop resolution (uses BiSeNet directly)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    labels = parser._parse(crop_rgb)  # (model_h, model_w) int labels
    h, w = crop_bgr.shape[:2]
    return cv2.resize(labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)


def _colorize(class_map: np.ndarray) -> np.ndarray:
    seg = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for idx, color in CLASS_COLORS.items():
        seg[class_map == idx] = color
    return seg


def _header(text: str, width: int, height: int = 38, bg=(60, 60, 60)) -> np.ndarray:
    bar = np.full((height, width, 3), bg, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(bar, text, ((width - tw) // 2, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return bar


def _legend(present: set[int], width: int) -> np.ndarray:
    """Swatch + name for every class present, wrapped to ``width``."""
    items = [(i, INDEX_TO_NAME.get(i, f"class {i}")) for i in sorted(present) if i != 0]
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    rows, cur, x = [], [], 12
    for i, name in items:
        (tw, _), _ = cv2.getTextSize(name, font, fs, th)
        cell_w = 24 + tw + 24
        if x + cell_w > width and cur:
            rows.append(cur); cur, x = [], 12
        cur.append((i, name, x, cell_w)); x += cell_w
    if cur:
        rows.append(cur)
    bar = np.full((max(len(rows), 1) * 30 + 8, width, 3), 245, dtype=np.uint8)
    for r, row in enumerate(rows):
        y = 24 + r * 30
        for i, name, x, _ in row:
            cv2.rectangle(bar, (x, y - 14, 18, 16), CLASS_COLORS.get(i, (0, 0, 0)), -1)
            cv2.rectangle(bar, (x, y - 14, 18, 16), (80, 80, 80), 1)
            cv2.putText(bar, name, (x + 24, y), font, fs, (30, 30, 30), th, cv2.LINE_AA)
    return bar


def main() -> int:
    p = argparse.ArgumentParser(description="Visualize BiSeNet face parsing (19-class segmentation).")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--conf", type=float, default=0.4, help="Detector confidence threshold.")
    p.add_argument("--frame", type=int, default=0, help="Frame index if input is a video.")
    p.add_argument("--max-faces", type=int, default=4, help="Cap rows so the figure stays readable.")
    p.add_argument("--alpha", type=float, default=0.5, help="Overlay blend strength.")
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
    parser = FaceParser()
    aligner = FaceAligner(output_size=(512, 512), mode=AlignMode.FFHQ)

    detections = detector.detect(image)
    if not detections:
        print("No faces detected.", file=sys.stderr)
        return 1
    detections = sorted(detections, key=lambda d: -d.score)[: args.max_faces]
    print(f"Detected {len(detections)} face(s); parsing...")

    sep = np.full((PANEL, 4, 3), 255, dtype=np.uint8)
    rows, present = [], set()
    for i, det in enumerate(detections):
        _, crop = aligner.align_and_warp(image, det)
        class_map = _class_map(parser, crop)
        present.update(np.unique(class_map).tolist())
        seg = _colorize(class_map)
        overlay = cv2.addWeighted(crop, 1.0 - args.alpha, seg, args.alpha, 0.0)

        panels = [cv2.resize(im, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
                  for im in (crop, seg, overlay)]
        row = np.hstack([panels[0], sep, panels[1], sep, panels[2]])
        label = np.full((row.shape[0], 44, 3), 255, dtype=np.uint8)
        cv2.putText(label, f"#{i}", (6, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(np.hstack([label, row]))

    body = np.vstack([np.vstack([r, np.full((4, r.shape[1], 3), 255, dtype=np.uint8)]) for r in rows])
    total_w = body.shape[1]

    head_cells = [_header("", 44)]
    for c, name in enumerate(["Aligned face", "Segmentation", "Overlay"]):
        head_cells.append(_header(name, PANEL))
        if c < 2:
            head_cells.append(np.full((38, 4, 3), 255, dtype=np.uint8))
    header = np.hstack(head_cells)

    grid = np.vstack([header, body, _legend(present, total_w)])

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "face_parsing_demo.png"
    cv2.imwrite(str(out), grid)
    names = [INDEX_TO_NAME.get(i, str(i)) for i in sorted(present) if i != 0]
    print(f"Classes found: {', '.join(names)}")
    print(f"Saved parsing grid -> {out}  ({grid.shape[1]}x{grid.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
