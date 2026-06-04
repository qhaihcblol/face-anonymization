from __future__ import annotations

import argparse
from pathlib import Path

from ai_core.face_alignment.face_aligner import FaceAligner
from ai_core.face_anonymization.face_anonymizer import FaceAnonymizer
from ai_core.face_parsing.face_parser import FaceParser
from ai_core.face_restoration.face_restorer import FaceRestorer
from ai_core.face_swapping.face_swapper import DEFAULT_SOURCE_FACE, FaceSwapper
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
        help="Path to blendswap_256.onnx (default: bundled onnx/ model)",
    )
    parser.add_argument(
        "--no-region-mask",
        action="store_true",
        help="Disable BiSeNet face-parsing mask (use the elliptical mask only)",
    )
    parser.add_argument(
        "--parser-model",
        type=Path,
        default=None,
        help="Path to bisenet_resnet_34.onnx (default: bundled onnx/ model)",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Disable GFPGAN face restoration (swapped face stays soft)",
    )
    parser.add_argument(
        "--restore-model",
        type=Path,
        default=None,
        help="Path to gfpgan_1.4.onnx (default: bundled onnx/ model)",
    )
    parser.add_argument(
        "--restore-blend",
        type=float,
        default=0.8,
        help="How much restored detail to mix back (0..1)",
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
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Disable temporal stabilization (per-frame independent swap)",
    )
    parser.add_argument(
        "--smooth-min-cutoff",
        type=float,
        default=0.5,
        help="One-Euro min cutoff (lower = smoother landmarks)",
    )
    parser.add_argument(
        "--smooth-beta",
        type=float,
        default=0.05,
        help="One-Euro beta (higher = more responsive to fast motion)",
    )
    parser.add_argument(
        "--output-smooth",
        type=float,
        default=None,
        help=(
            "EMA weight on the swapped crop (0 = off, higher = less flicker). "
            "Unset = per-mode default: 0.4 for online, 0.25 for offline."
        ),
    )
    parser.add_argument(
        "--mask-smooth",
        type=float,
        default=0.5,
        help="EMA weight on the blend mask (0 = off, higher = steadier edges)",
    )
    parser.add_argument(
        "--smoothing",
        choices=("online", "offline"),
        default="online",
        help=(
            "Landmark smoothing mode. 'online' = causal 1-Euro (default); "
            "'offline' = 2-pass zero-phase (no lag, recommended for uploads)"
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Do not mux the source audio into the output (silent video)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    detector = FaceDetector(onnx_path=args.onnx)
    face_parser = (
        None
        if args.no_region_mask
        else FaceParser(model_path=args.parser_model)
    )
    face_restorer = (
        None
        if args.no_restore
        else FaceRestorer(model_path=args.restore_model, blend=args.restore_blend)
    )
    swapper = FaceSwapper(
        detector=detector,
        model_path=args.model,
        source_path=args.source,
        face_parser=face_parser,
        face_restorer=face_restorer,
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
        stabilize=not args.no_stabilize,
        smoothing=args.smoothing,
        smooth_min_cutoff=args.smooth_min_cutoff,
        smooth_beta=args.smooth_beta,
        output_smooth=args.output_smooth,
        mask_smooth=args.mask_smooth,
        keep_audio=not args.no_audio,
    )
    print(f"Saved output to: {result.output_path}")


if __name__ == "__main__":
    main()
