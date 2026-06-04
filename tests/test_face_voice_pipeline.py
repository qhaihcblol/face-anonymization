"""Combined demo: face swap + voice conversion in a single pass.

Wires the model-based face swap (BlendSwap) together with kNN-VC voice conversion
and runs both through ``VideoAnonymization.anonymize_video_with_model`` — the output
video has a swapped identity AND an anonymized voice.

This is the union of two single-concern scripts:
  * tests/test_face_swap_video_pipeline.py  (face swap only)
  * tests/test_voice_anonymize.py           (voice only)

Prerequisites:
  * Face swap models bundled in each module's onnx/: blendswap_256, bisenet, gfpgan.
  * Voice models exported once: ``python tools/export_knnvc_onnx.py`` ->
    ai_core/voice_anonymization/onnx/{wavlm_encoder,hifigan_vocoder}.onnx
  * A target voice at ai_core/voice_anonymization/reference_voice.wav (or pass
    --reference-voice). The audio analog of --source for the face.

Usage:
    # Swap face to source_img + convert voice to the bundled reference, in one pass.
    python -m tests.test_face_voice_pipeline --input clip.mp4

    # Pick the source identity and a different reference voice; trim to 5 s.
    python -m tests.test_face_voice_pipeline --input clip.mp4 \
        --source my_face.png --reference-voice target.wav --start-sec 0 --end-sec 5

    # Use a DSP voice method instead of the ONNX model (no voice weights needed).
    python -m tests.test_face_voice_pipeline --input clip.mp4 --voice-method mcadams
"""
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
from ai_core.voice_anonymization.voice_anonymizer import (
    VoiceAnonymizationMethod,
    VoiceAnonymizer,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Face swap + voice conversion in one pass (anonymize_video_with_model)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: outputs/swapped_<input_stem>.mp4)",
    )

    # --- Face swap ---
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
        help="Source identity image (the face to swap in)",
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
        "--no-stabilize",
        action="store_true",
        help="Disable temporal stabilization (per-frame independent swap)",
    )
    parser.add_argument(
        "--smoothing",
        choices=("online", "offline"),
        default="online",
        help="Landmark smoothing: 'online' causal 1-Euro (default) or 'offline' 2-pass",
    )
    parser.add_argument("--smooth-min-cutoff", type=float, default=0.5)
    parser.add_argument("--smooth-beta", type=float, default=0.05)
    parser.add_argument("--output-smooth", type=float, default=None)
    parser.add_argument("--mask-smooth", type=float, default=0.5)

    # --- Voice ---
    parser.add_argument(
        "--voice-method",
        type=str,
        choices=[m.value for m in VoiceAnonymizationMethod],
        default=VoiceAnonymizationMethod.CONVERT.value,
        help="Voice method (default: convert = kNN-VC ONNX model)",
    )
    parser.add_argument(
        "--reference-voice",
        type=Path,
        default=None,
        help="Target voice wav (kNN-VC); defaults to the bundled reference_voice.wav",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=None,
        help="WavLM encoder ONNX (default: exported onnx/wavlm_encoder.onnx)",
    )
    parser.add_argument(
        "--vocoder",
        type=Path,
        default=None,
        help="HiFi-GAN vocoder ONNX (default: exported onnx/hifigan_vocoder.onnx)",
    )
    parser.add_argument(
        "--topk", type=int, default=4, help="kNN neighbours to average (kNN-VC)"
    )
    parser.add_argument(
        "--mcadams-alpha", type=float, default=0.8, help="McAdams coefficient (DSP)"
    )
    parser.add_argument(
        "--pitch-steps", type=float, default=-4.0, help="Pitch shift semitones (DSP)"
    )
    parser.add_argument(
        "--formant-shift", type=float, default=1.2, help="Formant scale (DSP)"
    )

    # --- Common ---
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument(
        "--codec", type=str, default="H264", help="Output video FourCC (4 chars)"
    )
    parser.add_argument("--progress-every", type=int, default=60)
    return parser


def _build_voice_anonymizer(args: argparse.Namespace) -> VoiceAnonymizer:
    voice_converter = None
    if args.voice_method == VoiceAnonymizationMethod.CONVERT.value:
        # Lazy import: only the model path needs onnxruntime / the exported ONNX.
        from ai_core.voice_anonymization.voice_converter import (
            DEFAULT_REFERENCE_VOICE,
            VoiceConverter,
        )

        kwargs: dict = {
            "topk": args.topk,
            "reference_voice_path": args.reference_voice or DEFAULT_REFERENCE_VOICE,
        }
        if args.encoder is not None:
            kwargs["encoder_onnx_path"] = args.encoder
        if args.vocoder is not None:
            kwargs["vocoder_onnx_path"] = args.vocoder
        voice_converter = VoiceConverter(**kwargs)
    return VoiceAnonymizer(
        mcadams_alpha=args.mcadams_alpha,
        pitch_steps=args.pitch_steps,
        formant_shift=args.formant_shift,
        voice_converter=voice_converter,
    )


def main() -> None:
    args = _build_parser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    detector = FaceDetector(onnx_path=args.onnx)
    face_parser = None if args.no_region_mask else FaceParser(model_path=args.parser_model)
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

    video_anonymization = VideoAnonymization(
        video_io=VideoIO(),
        face_detector=detector,
        face_tracker=ByteTracker(),
        face_anonymizer=FaceAnonymizer(face_swapper=swapper),
        face_aligner=FaceAligner(output_size=(256, 256), mode="ffhq"),
        voice_anonymizer=_build_voice_anonymizer(args),
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
        keep_audio=True,
        anonymize_voice=True,
        voice_method=args.voice_method,
    )
    print(f"Saved output to: {result.output_path}")


if __name__ == "__main__":
    main()
