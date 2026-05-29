"""Manual example: swap every face in an image with source_img.png.

Usage:
    python tests/test_face_swap.py \
        --image test_images/test5.jpeg \
        [--source ai_core/face_anonymization/source_img.png] \
        [--model /path/to/blendswap_256.onnx]

If --model is omitted the BlendSwap model is downloaded from Hugging Face
(facefusion/models-3.0.0, ~1.66 GB) and cached locally.
"""

import argparse
from pathlib import Path

import cv2

from ai_core.face_alignment.face_aligner import FaceAligner
from ai_core.face_anonymization.face_anonymizer import FaceAnonymizer
from ai_core.face_anonymization.face_swapper import DEFAULT_SOURCE_FACE, FaceSwapper
from ai_core.face_detection.face_detector import FaceDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swap all faces in an image.")
    parser.add_argument("--image", default="test_images/test5.jpeg")
    parser.add_argument(
        "--onnx", default="ai_core/face_detection/onnx/retinaface_best.onnx"
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_FACE))
    parser.add_argument(
        "--model",
        default=None,
        help="Path to blendswap_256.onnx (downloaded from HF if omitted).",
    )
    parser.add_argument("--output", default="outputs/swapped.jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detector = FaceDetector(onnx_path=args.onnx)
    aligner = FaceAligner(output_size=(256, 256), mode="ffhq")
    swapper = FaceSwapper(
        detector=detector,
        model_path=args.model,
        source_path=args.source,
    )
    anonymizer = FaceAnonymizer(face_swapper=swapper)

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"Failed to read image: {args.image}")
        return
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    detections = detector.detect(image)
    print(f"Detected faces: {len(detections)}")
    if not detections:
        return

    aligned_faces = aligner.align(detections)
    swapped = anonymizer.swap_face(image, aligned_faces)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
