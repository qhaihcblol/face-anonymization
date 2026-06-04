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
        "--reversible",
        action="store_true",
        help="Disable irreversible hardening of blur/pixelate (NOT privacy-safe)",
    )
    parser.add_argument(
        "--precise-mask",
        action="store_true",
        help=(
            "Use the BiSeNet face parser for a precise, face-hugging mask "
            "(blur/pixelate/mask/blackout) instead of a coarse ellipse"
        ),
    )
    parser.add_argument(
        "--parser-model",
        type=Path,
        default=None,
        help="Path to bisenet_resnet_34.onnx (default: bundled onnx/ model)",
    )
    parser.add_argument(
        "--mask-feather",
        type=float,
        default=4.0,
        help="Soft-edge width (px) for the ellipse fallback mask",
    )
    parser.add_argument(
        "--noise-strength",
        type=float,
        default=12.0,
        help="Std-dev of noise injected into blur/pixelate regions (0 = off)",
    )
    parser.add_argument(
        "--quantization-levels",
        type=int,
        default=8,
        help="Tonal levels for blur/pixelate quantization (0 = off, lower = stronger)",
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
        default="H264",
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

    face_parser = None
    face_aligner = None
    if args.precise_mask:
        # Lazy imports: only needed (and only download the parser model) when the
        # precise-mask path is requested.
        from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
        from ai_core.face_parsing.face_parser import FaceParser

        face_parser = FaceParser(model_path=args.parser_model)
        # FFHQ-512 gives the parser a large, well-centred crop to segment.
        face_aligner = FaceAligner(output_size=(512, 512), mode=AlignMode.FFHQ)
        print("Precise mask: BiSeNet face parser enabled (FFHQ-512 alignment)")

    anonymizer = FaceAnonymizer(
        blur_strength=args.blur_strength,
        pixelation_level=args.pixelation_level,
        mask_color=tuple(args.mask_color),
        face_parser=face_parser,
        face_aligner=face_aligner,
        irreversible=not args.reversible,
        noise_strength=args.noise_strength,
        quantization_levels=args.quantization_levels,
        mask_feather=args.mask_feather,
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
