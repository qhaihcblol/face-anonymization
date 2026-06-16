"""Face-swap anonymization demo: source identity | original | swapped.

The swap path replaces every detected face in the frame with a *source identity*
(BlendSwap), unlike blur/pixelate/blackout which only obscure. Because the result
depends on which identity is injected, the output image is a triptych so the swap
is self-explanatory:

  * **Source**   — the identity face being injected (the swapper's reference).
  * **Original** — the input frame.
  * **Swapped**  — every face replaced by the source identity.

Pass several ``--source`` images to get one triptych per source identity.

Outputs (under ``outputs/``, prefix configurable with --prefix):
  * ``swap_<source-stem>.png`` — source | original | swapped, one per source.

Usage:
    python -m test_scripts.test_face_swap_demo
    python -m test_scripts.test_face_swap_demo --input test_images/female_1.jpeg --source test_images/source.png
    python -m test_scripts.test_face_swap_demo --input test_videos/video_track.mp4 --frame 100 \
        --source test_images/source.png test_images/male_1.jpeg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_parsing.face_parser import FaceParser
from ai_core.face_swapping.face_swapper import FaceSwapper
from ai_core.video_io.video_io import VideoIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "test_images" / "female_1.jpeg"
DEFAULT_SOURCE = PROJECT_ROOT / "test_images" / "source.png"
DEFAULT_ONNX = PROJECT_ROOT / "ai_core" / "face_detection" / "onnx" / "retinaface_best.onnx"


def _load_image(path: Path, frame: int):
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        for idx, f in enumerate(VideoIO().iter_frames(str(path))):
            if idx == frame:
                return f
        return None
    return cv2.imread(str(path))


def _fit_h(img, height: int):
    """Resize ``img`` to a fixed height (keeps aspect)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (int(round(w * height / h)), height), interpolation=cv2.INTER_AREA)


def _label(img, text: str, height: int = 38, bg=(60, 60, 60)):
    bar = np.full((height, img.shape[1], 3), bg, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(bar, text, ((img.shape[1] - tw) // 2, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def _triptych(source, original, swapped):
    """Source | Original | Swapped, normalised to a common height with titles."""
    h = max(original.shape[0], 360)
    cells = [_label(_fit_h(source, h), "Source"),
             _label(_fit_h(original, h), "Original"),
             _label(_fit_h(swapped, h), "Swapped")]
    sep = np.full((cells[0].shape[0], 6, 3), 255, dtype=np.uint8)
    out = []
    for i, c in enumerate(cells):
        if i:
            out.append(sep)
        out.append(c)
    return np.hstack(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Face-swap demo: source | original | swapped.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--source", type=Path, nargs="+", default=[DEFAULT_SOURCE],
                   help="One or more source identity images (one triptych each).")
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--conf", type=float, default=0.4, help="Detector confidence threshold.")
    p.add_argument("--frame", type=int, default=0, help="Frame index if input is a video.")
    p.add_argument("--prefix", type=str, default="swap", help="Output file-name prefix.")
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
    swapper = FaceSwapper(detector=detector, face_parser=parser)
    aligner = FaceAligner(output_size=(256, 256), mode=AlignMode.FFHQ)

    detections = detector.detect(image)
    if not detections:
        print("No faces detected in input.", file=sys.stderr)
        return 1
    print(f"Detected {len(detections)} face(s) in {args.input.name}.")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aligned = aligner.align(detections)

    args.outdir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for src_path in args.source:
        src_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if src_bgr is None:
            print(f"Warning: cannot read source {src_path}", file=sys.stderr)
            continue
        try:
            src_blob = swapper.prepare_source(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB))
        except ValueError as exc:
            print(f"Warning: {exc}", file=sys.stderr)
            continue

        swapped = cv2.cvtColor(swapper.swap_face(rgb, aligned, src_blob), cv2.COLOR_RGB2BGR)
        triptych = _triptych(src_bgr, image, swapped)
        out_path = args.outdir / f"{args.prefix}_{src_path.stem}.png"
        cv2.imwrite(str(out_path), triptych)
        saved.append(out_path)

    if not saved:
        print("No swaps produced (no usable source).", file=sys.stderr)
        return 1
    print("Saved (source | original | swapped):")
    for path in saved:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
