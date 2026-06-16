"""Simple face-detection demo.

Runs ``FaceDetector`` (RetinaFace) on a single image, then draws the bounding box,
5-point landmarks, and confidence score for every detected face and saves the
annotated result.

Usage:
    # Defaults: test_images/image.png -> outputs/detected_image.png
    python -m test_scripts.test_face_detection

    # Pick a different image / output / threshold.
    python -m test_scripts.test_face_detection \
        --input test_images/male_1.jpeg --output outputs/out.png --conf 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from ai_core.face_detection.face_detector import FaceDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect faces in an image and draw bbox / landmarks / score."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "test_images" / "image.png",
        help="Input image path (default: test_images/image.png)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: outputs/detected_<input_stem>.png)",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=PROJECT_ROOT
        / "ai_core"
        / "face_detection"
        / "onnx"
        / "retinaface_best.onnx",
        help="Path to the RetinaFace ONNX model (default: bundled onnx/ model)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold (default: 0.4)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if not args.input.is_file():
        print(f"Input image not found: {args.input}", file=sys.stderr)
        return 1

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to read image: {args.input}", file=sys.stderr)
        return 1

    output = args.output or (
        PROJECT_ROOT / "outputs" / f"detected_{args.input.stem}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(onnx_path=args.onnx, conf_threshold=args.conf)
    detections = detector.detect(image)

    print(f"Detected {len(detections)} face(s) in {args.input.name}")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        print(
            f"  [{i}] score={det.score:.3f} "
            f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )

    annotated = detector.draw(image, detections,thickness=1, radius=2,text_color=(0, 255, 0))
    if not cv2.imwrite(str(output), annotated):
        print(f"Failed to write output: {output}", file=sys.stderr)
        return 1

    print(f"Saved annotated image -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
