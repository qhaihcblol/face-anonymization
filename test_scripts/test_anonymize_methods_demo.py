"""Apply the no-model anonymization methods and save original/method pairs.

Runs the audio-free, swap-free anonymization path on every detected face and
writes one paired image per method (original on the left, anonymized on the
right, with labels):

  * **Blur**      — Gaussian blur over the face region.
  * **Pixelate**  — mosaic / down-then-up sampling of the face region.
  * **Blackout**  — solid block covering the face region ("che khoi").

All methods share the same region mask (parser mask when landmarks + parser are
available, else the soft ellipse), so the only difference between outputs is the
fill, not the coverage.

Outputs (under ``outputs/``, prefix configurable with --prefix):
  * ``anonymize_blur.png``      — original | blur
  * ``anonymize_pixelate.png``  — original | pixelate
  * ``anonymize_blackout.png``  — original | blackout

Usage:
    python -m test_scripts.test_anonymize_methods_demo
    python -m test_scripts.test_anonymize_methods_demo --input test_images/image.png
    python -m test_scripts.test_anonymize_methods_demo --input test_videos/video_track.mp4 --frame 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_anonymization.face_anonymizer import AnonymizationMethod, FaceAnonymizer
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_parsing.face_parser import FaceParser
from ai_core.video_io.video_io import VideoIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "test_images" / "image.png"
DEFAULT_ONNX = PROJECT_ROOT / "ai_core" / "face_detection" / "onnx" / "retinaface_best.onnx"

# (file-name key, label, method).
METHODS = [
    ("blur", "Blur", AnonymizationMethod.BLUR),
    ("pixelate", "Pixelate", AnonymizationMethod.PIXELATE),
    ("blackout", "Blackout (che khoi)", AnonymizationMethod.BLACKOUT),
]


def _load_image(path: Path, frame: int):
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        for idx, f in enumerate(VideoIO().iter_frames(str(path))):
            if idx == frame:
                return f
        return None
    return cv2.imread(str(path))


def _label(img, text: str, height: int = 38, bg=(60, 60, 60)):
    """Stack a titled bar on top of ``img`` (bar matches the image width)."""
    bar = np.full((height, img.shape[1], 3), bg, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(bar, text, ((img.shape[1] - tw) // 2, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def _pair(original, processed, method_label: str):
    """Side-by-side original | processed, each with a title and a white gutter."""
    left = _label(original, "Original")
    right = _label(processed, method_label)
    sep = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
    return np.hstack([left, sep, right])


def main() -> int:
    p = argparse.ArgumentParser(description="Save original/blur, original/pixelate, original/blackout pairs.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--conf", type=float, default=0.4, help="Detector confidence threshold.")
    p.add_argument("--frame", type=int, default=0, help="Frame index if input is a video.")
    p.add_argument("--prefix", type=str, default="anonymize", help="Output file-name prefix.")
    p.add_argument("--no-parser", action="store_true",
                   help="Use the ellipse mask instead of the BiSeNet parser mask.")
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
    if args.no_parser:
        anonymizer = FaceAnonymizer()
    else:
        anonymizer = FaceAnonymizer(
            face_parser=FaceParser(),
            face_aligner=FaceAligner(output_size=(512, 512), mode=AlignMode.FFHQ),
        )

    detections = detector.detect(image)
    if not detections:
        print("No faces detected.", file=sys.stderr)
        return 1
    print(f"Detected {len(detections)} face(s); applying {[m[0] for m in METHODS]}...")
    dets = [{"bbox": list(d.bbox), "landmarks": d.landmarks.as_array().tolist()} for d in detections]

    args.outdir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for name, label, method in METHODS:
        out = anonymizer.anonymize(image, dets, method=method)
        pair = _pair(image, out, label)
        out_path = args.outdir / f"{args.prefix}_{name}.png"
        cv2.imwrite(str(out_path), pair)
        saved.append(out_path)

    print("Saved pairs (original | method):")
    for path in saved:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
