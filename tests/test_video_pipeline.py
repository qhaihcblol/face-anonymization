from __future__ import annotations

import argparse
import sys
from pathlib import Path


from ai_core.face_anonymization.face_anonymizer import (
    AnonymizationMethod,
    FaceAnonymizer,
)
from ai_core.video_anonymization import VideoAnonymization
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_tracking.face_tracker import ByteTracker
from ai_core.video_io.video_io import VideoIO


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline pipeline using VideoAnonymization.anonymize_video_without_model"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: outputs/anonymized_<input_stem>.mp4)",
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
        "--method",
        type=str,
        choices=[method.value for method in AnonymizationMethod],
        default=AnonymizationMethod.BLUR.value,
        help="Anonymization method",
    )
    parser.add_argument("--blur-strength", type=int, default=31)
    parser.add_argument("--pixelation-level", type=int, default=16)
    parser.add_argument(
        "--mask-color",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=(160, 160, 160),
        help="Mask color in BGR",
    )
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=1,
        help="Run detector every N frames (1 = detect every frame)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help="Optional sampling FPS for processing/output",
    )
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--high-thresh", type=float, default=0.6)
    parser.add_argument("--low-thresh", type=float, default=0.1)
    parser.add_argument("--max-lost", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument(
        "--blur-new",
        action="store_true",
        help="Also anonymize New tracks (default: only confirmed Tracked)",
    )
    parser.add_argument(
        "--draw-tracks",
        action="store_true",
        help="Overlay tracker boxes/ids on output video",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Output video FourCC codec (4 chars)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=60,
        help="Print progress every N frames",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    input_path = args.input

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    method = AnonymizationMethod(args.method)
    detector = FaceDetector(onnx_path=args.onnx)
    tracker = ByteTracker(
        high_thresh=args.high_thresh,
        low_thresh=args.low_thresh,
        max_lost=args.max_lost,
        min_hits=args.min_hits,
    )
    anonymizer = FaceAnonymizer(
        blur_strength=args.blur_strength,
        pixelation_level=args.pixelation_level,
        mask_color=tuple(args.mask_color),
    )
    video_anonymization = VideoAnonymization(
        video_io=VideoIO(),
        face_detector=detector,
        face_tracker=tracker,
        face_anonymizer=anonymizer,
    )

    result = video_anonymization.anonymize_video_without_model(
        input_path=input_path,
        output_path=args.output,
        method=method,
        detect_interval=args.detect_interval,
        target_fps=args.target_fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        blur_new=args.blur_new,
        draw_tracks=args.draw_tracks,
        codec=args.codec,
        progress_every=args.progress_every,
    )
    print(f"Saved output to: {result.output_path}")


if __name__ == "__main__":
    main()
