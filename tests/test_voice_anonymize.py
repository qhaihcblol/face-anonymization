"""Manual demo: anonymize a speaker's voice (no-model DSP or ONNX model path).

Two directions, mirroring the face pipeline:

  * No model (librosa DSP) — pitch / formant / pitch_formant. Runs anywhere, no
    weights to download.
  * Model (ONNX voice conversion) — ``--method convert --model path/to/vc.onnx``.

Usage:
    # Quick audition: just process the audio track to a WAV (no video re-encode).
    python tests/test_voice_anonymize.py --input clip.mp4 --audio-only \
        --method pitch_formant --pitch-steps -4 --formant-shift 1.2

    # Full video: faces left untouched (method=none), only the voice is anonymized.
    python tests/test_voice_anonymize.py --input clip.mp4 --method pitch

    # Model path (kNN-VC): you supply the encoder + vocoder ONNX and a reference voice.
    python tests/test_voice_anonymize.py --input clip.mp4 --audio-only --method convert \
        --encoder wavlm.onnx --vocoder hifigan.onnx --reference-voice target.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_core.face_anonymization.face_anonymizer import FaceAnonymizer
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
        description="Anonymize the voice in a video/audio file (DSP or ONNX model)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input media path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: outputs/voice_<stem>.wav or the pipeline default)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=[m.value for m in VoiceAnonymizationMethod],
        default=VoiceAnonymizationMethod.MCADAMS.value,
        help="Voice anonymization method (default: mcadams, non-invertible)",
    )
    parser.add_argument(
        "--mcadams-alpha",
        type=float,
        default=0.8,
        help="McAdams coefficient (<1); how hard to warp formants",
    )
    parser.add_argument(
        "--pitch-steps",
        type=float,
        default=-4.0,
        help="Pitch shift in semitones (DSP methods); negative = lower",
    )
    parser.add_argument(
        "--formant-shift",
        type=float,
        default=1.2,
        help="Formant scale (DSP methods); >1 raises formants, <1 lowers",
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
        "--reference-voice",
        type=Path,
        default=None,
        help="Path to reference_voice.wav (target identity); defaults to the bundled one",
    )
    parser.add_argument(
        "--topk", type=int, default=4, help="kNN neighbours to average (kNN-VC)"
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Process just the audio track to a WAV (skip video re-encode)",
    )
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "ai_core"
        / "face_detection"
        / "onnx"
        / "retinaface_best.onnx",
        help="Path to RetinaFace ONNX (full-video mode only)",
    )
    parser.add_argument(
        "--codec", type=str, default="H264", help="Output video FourCC (4 chars)"
    )
    parser.add_argument("--progress-every", type=int, default=60)
    return parser


def _build_voice_anonymizer(args: argparse.Namespace) -> VoiceAnonymizer:
    voice_converter = None
    if args.method == VoiceAnonymizationMethod.CONVERT.value:
        # Lazy import: only the model path needs onnxruntime wired up here.
        from ai_core.voice_anonymization.voice_converter import (
            DEFAULT_REFERENCE_VOICE,
            VoiceConverter,
        )

        # Encoder/vocoder default to the exported onnx/ files when not overridden.
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
        raise FileNotFoundError(f"Input not found: {args.input}")

    voice_anonymizer = _build_voice_anonymizer(args)

    if args.audio_only:
        video_io = VideoIO()
        if not video_io.has_audio(str(args.input)):
            raise SystemExit(f"No audio stream in: {args.input}")

        waveform, sample_rate = video_io.extract_audio(
            str(args.input), start_sec=args.start_sec, end_sec=args.end_sec
        )
        print(
            f"Extracted audio: {waveform.shape[0]} samples, "
            f"{waveform.shape[1]} ch, {sample_rate} Hz"
        )
        processed = voice_anonymizer.process(
            waveform, sample_rate, method=args.method
        )
        output = args.output or Path("outputs") / f"voice_{args.input.stem}.wav"
        path = video_io.write_audio(processed, sample_rate, str(output))
        print(f"Saved anonymized audio to: {path}")
        return

    # Full video: leave faces untouched (method=none) and anonymize only the voice.
    detector = FaceDetector(onnx_path=args.onnx)
    pipeline = VideoAnonymization(
        video_io=VideoIO(),
        face_detector=detector,
        face_tracker=ByteTracker(),
        face_anonymizer=FaceAnonymizer(),
        voice_anonymizer=voice_anonymizer,
    )
    result = pipeline.anonymize_video_without_model(
        input_path=args.input,
        output_path=args.output,
        method="none",
        detect_interval=30,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        codec=args.codec,
        progress_every=args.progress_every,
        keep_audio=True,
        anonymize_voice=True,
        voice_method=args.method,
    )
    print(f"Saved output to: {result.output_path}")


if __name__ == "__main__":
    main()
