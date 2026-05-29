from __future__ import annotations

import argparse
from pathlib import Path

from ai_core.face_alignment.face_aligner import FaceAligner
from ai_core.face_anonymization.face_anonymizer import FaceAnonymizer
from ai_core.face_anonymization.face_swapper import DEFAULT_SOURCE_FACE, FaceSwapper
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_tracking.face_tracker import ByteTracker
from ai_core.video_anonymization import VideoAnonymization
from ai_core.video_io.video_io import VideoIO


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline face-swap pipeline using "
            "VideoAnonymization.anonymize_video_with_model"
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: outputs/swapped_<input_stem>.mp4)",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "ai_core"
        / "face_detection"
        / "onnx"
        / "retinaface_best.onnx",
        help="Path to RetinaFace ONNX model",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(DEFAULT_SOURCE_FACE),
        help="Source identity image",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to blendswap_256.onnx (downloaded from HF if omitted)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help="Optional sampling FPS for processing/output",
    )
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument(
        "--codec", type=str, default="H264", help="Output video FourCC codec (4 chars)"
    )
    parser.add_argument(
        "--progress-every", type=int, default=60, help="Print progress every N frames"
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    detector = FaceDetector(onnx_path=args.onnx)
    swapper = FaceSwapper(
        detector=detector,
        model_path=args.model,
        source_path=args.source,
    )
    anonymizer = FaceAnonymizer(face_swapper=swapper)
    video_anonymization = VideoAnonymization(
        video_io=VideoIO(),
        face_detector=detector,
        face_tracker=ByteTracker(),
        face_anonymizer=anonymizer,
        face_aligner=FaceAligner(output_size=(256, 256), mode="ffhq"),
    )

    result = video_anonymization.anonymize_video_with_model(
        input_path=args.input,
        output_path=args.output,
        target_fps=args.target_fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        codec=args.codec,
        progress_every=args.progress_every,
    )
    print(f"Saved output to: {result.output_path}")


if __name__ == "__main__":
    main()
