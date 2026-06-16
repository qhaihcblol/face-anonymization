"""Evaluation of the voice-anonymization methods.

The audio counterpart to ``test_anonymize_nonswap_eval.py`` / ``..._swap_eval.py``:
the same two-axis story (privacy vs. utility), measured for speech instead of faces.

Methods compared (see :mod:`ai_core.voice_anonymization.voice_anonymizer`):

* ``mcadams``        — VoicePrivacy DSP baseline, formant warp (keeps pitch + length).
* ``pitch``          — semitone pitch shift.
* ``formant``        — spectral-envelope (formant) warp.
* ``pitch_formant``  — formant then pitch.
* ``convert``        — kNN-VC neural conversion to the bundled reference identity.

Two axes:

* **Speaker de-identification (privacy).** The headline metric, the audio analog of
  the ArcFace-cosine figure in the face chapters. We embed the *original* and the
  *anonymized* clip with Resemblyzer (GE2E speaker encoder) and report cosine
  similarity. Because we only have two speakers, we don't draw an EER curve; instead
  we anchor the bars with two data-driven reference lines computed from the same
  clips: a **same-speaker** band (cosine between two halves of one original clip) and
  a **different-speaker** band (female-original vs. male-original). A method has
  de-identified the voice when its cosine drops from the same-speaker band down toward
  the different-speaker band. For ``convert`` we additionally report cosine to the
  *target* reference identity (it should resemble the target, not the source).
* **Intelligibility (utility).** Word Error Rate from a faster-whisper transcript:
  we transcribe the *original* clip as the pseudo-reference text, transcribe each
  anonymized clip, and report WER. Higher WER = more linguistic content destroyed.
  A good method de-identifies (low speaker cosine) without wrecking intelligibility
  (low WER).

Figures (under ``outputs/``):
  * ``voice_spectrograms.png``    — A. voices x {original, methods}: mel-spectrograms.
  * ``voice_identity_leakage.png``— B. speaker cosine vs source per method (+ refs).
  * ``voice_intelligibility.png`` — C. WER per method (lower = better utility).
  * ``voice_privacy_utility.png`` — D. scatter: WER (x) vs speaker cosine (y).
  * ``voice_f0_contour.png``      — E. F0 (pitch) tracks: proves mcadams/formant are
                                       pitch-preserving (lip-safe) vs. pitch shifting.

Anonymized clips are also written to ``outputs/voice/`` so they can be listened to.

Usage:
    python -m test_scripts.test_voice_anonymize_eval
    python -m test_scripts.test_voice_anonymize_eval --max-seconds 30 --whisper base
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ai_core.voice_anonymization.voice_anonymizer import (
    VoiceAnonymizationMethod,
    VoiceAnonymizer,
    VoiceParams,
)
from ai_core.voice_anonymization.voice_converter import (
    DEFAULT_REFERENCE_VOICE,
    VoiceConverter,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOICES = [
    PROJECT_ROOT / "test_voices" / "voices_female_1.wav",
    PROJECT_ROOT / "test_voices" / "voices_male_2.wav",
]

# (column label, method) for the figures + metrics.
METHODS = [
    ("mcadams", VoiceAnonymizationMethod.MCADAMS),
    ("pitch", VoiceAnonymizationMethod.PITCH),
    ("formant", VoiceAnonymizationMethod.FORMANT),
    ("pitch_formant", VoiceAnonymizationMethod.PITCH_FORMANT),
    ("convert", VoiceAnonymizationMethod.CONVERT),
]
COLORS = {
    "mcadams": "#264653",
    "pitch": "#2a9d8f",
    "formant": "#e9c46a",
    "pitch_formant": "#f4a261",
    "convert": "#e76f51",
}


@dataclass
class Voice:
    tag: str
    wav: np.ndarray            # mono float32, native sr
    sr: int
    emb: np.ndarray            # speaker embedding of the original
    ref_text: str             # whisper transcript of the original (pseudo-reference)
    outputs: dict = field(default_factory=dict)   # method -> anonymized waveform


# --------------------------------------------------------------------------- #
# IO / model wrappers.                                                         #
# --------------------------------------------------------------------------- #
def _load_wav(path: Path, max_seconds: float) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if max_seconds > 0:
        wav = wav[: int(max_seconds * sr)]
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class _Speaker:
    """Resemblyzer GE2E speaker encoder -> 256-d embedding + cosine."""

    def __init__(self) -> None:
        from resemblyzer import VoiceEncoder
        self._enc = VoiceEncoder("cpu")

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        from resemblyzer import preprocess_wav
        proc = preprocess_wav(wav.astype(np.float32), source_sr=int(sr))
        if proc.size == 0:
            return np.zeros(256, dtype=np.float32)
        return self._enc.embed_utterance(proc).astype(np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


class _Asr:
    """faster-whisper transcriber -> WER against a reference transcript."""

    def __init__(self, model_name: str) -> None:
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self, wav: np.ndarray, sr: int) -> str:
        if sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, 16000)
            wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
        segments, _ = self._model.transcribe(wav.astype(np.float32), language="en", beam_size=1)
        return " ".join(s.text for s in segments)

    @staticmethod
    def wer(reference: str, hypothesis: str) -> float:
        import jiwer
        ref, hyp = _normalize_text(reference), _normalize_text(hypothesis)
        if not ref:
            return float("nan")
        if not hyp:
            return 1.0
        return float(jiwer.wer(ref, hyp))


# --------------------------------------------------------------------------- #
# Figures.                                                                     #
# --------------------------------------------------------------------------- #
def _mel_db(wav: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    return librosa.power_to_db(mel, ref=np.max)


def _fig_spectrograms(voices, anon, converter, params, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["original"] + [m[0] for m in METHODS]
    nrows, ncols = len(voices), len(cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows), squeeze=False)
    for r, v in enumerate(voices):
        clips = [v.wav] + [v.outputs[m[0]] for m in METHODS]
        for c, (name, clip) in enumerate(zip(cols, clips)):
            ax = axes[r][c]
            ax.imshow(_mel_db(clip, v.sr), origin="lower", aspect="auto", cmap="magma",
                      vmin=-80, vmax=0)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.set_ylabel(v.tag, fontsize=9)
    fig.suptitle("Mel-spectrograms — time axis (width) unchanged = lip-sync safe", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_identity_leakage(cos_by_method, same_ref, diff_ref, convert_target, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [m[0] for m in METHODS]
    cos = [float(np.mean(cos_by_method[n])) for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(names, cos, color=[COLORS[n] for n in names])
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.axhline(same_ref, color="#444", ls="--", lw=1.2, label=f"same speaker ~{same_ref:.2f}")
    ax.axhline(diff_ref, color="red", ls="--", lw=1.2, label=f"different speaker ~{diff_ref:.2f}")
    if convert_target is not None:
        ax.scatter(["convert"], [convert_target], marker="*", s=180, color="black", zorder=5,
                   label=f"convert vs TARGET {convert_target:.2f}")
    ax.set_ylabel("Resemblyzer cosine vs original speaker")
    ax.set_title("Speaker de-identification (lower toward 'different speaker' = better)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_intelligibility(wer_by_method, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [m[0] for m in METHODS]
    wer = [float(np.nanmean(wer_by_method[n])) for n in names]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(names, wer, color=[COLORS[n] for n in names])
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel("Word Error Rate vs original transcript")
    ax.set_title("Intelligibility loss (lower = better utility)")
    ax.set_ylim(0, max(1.0, max(wer) * 1.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_privacy_utility(cos_by_method, wer_by_method, diff_ref, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for name, _ in METHODS:
        x = float(np.nanmean(wer_by_method[name]))
        y = float(np.mean(cos_by_method[name]))
        ax.scatter(x, y, s=130, color=COLORS[name], zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.axhline(diff_ref, color="red", ls="--", lw=1.0, label=f"different speaker ~{diff_ref:.2f}")
    ax.set_xlabel("WER (utility loss)  ->")
    ax.set_ylabel("speaker cosine (privacy leakage)  ->")
    ax.set_title("Privacy-utility trade-off (ideal = bottom-left)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _f0_track(wav: np.ndarray, sr: int):
    import librosa
    f0, _, _ = librosa.pyin(wav, sr=sr, fmin=65.0, fmax=1000.0,
                            frame_length=2048, hop_length=256)
    times = librosa.times_like(f0, sr=sr, hop_length=256)
    return times, f0


def _fig_f0_contour(voices, out_path, window=(2.0, 7.0)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = len(voices)
    fig, axes = plt.subplots(nrows, 1, figsize=(9.0, 3.0 * nrows), squeeze=False)
    for r, v in enumerate(voices):
        ax = axes[r][0]
        t, f0 = _f0_track(v.wav, v.sr)
        ax.plot(t, f0, color="black", lw=2.0, label="original")
        for name, _ in METHODS:
            t2, f02 = _f0_track(v.outputs[name], v.sr)
            ax.plot(t2, f02, color=COLORS[name], lw=1.0, alpha=0.8, label=name)
        ax.set_xlim(*window)
        ax.set_ylabel("F0 (Hz)")
        ax.set_title(f"{v.tag}: pitch track (mcadams/formant overlap original = pitch-preserving)",
                     fontsize=10)
        if r == 0:
            ax.legend(fontsize=8, ncol=6, loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1][0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate voice-anonymization methods.")
    p.add_argument("--inputs", type=Path, nargs="*", default=None)
    p.add_argument("--max-seconds", type=float, default=20.0,
                   help="Trim each clip (keeps the CONVERT/whisper passes fast). 0 = full.")
    p.add_argument("--whisper", type=str, default="base", help="faster-whisper model size.")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_VOICE,
                   help="kNN-VC target identity wav (for the CONVERT method).")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    return p


def main() -> int:
    warnings.filterwarnings("ignore")
    args = _build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    clip_dir = args.outdir / "voice"
    clip_dir.mkdir(parents=True, exist_ok=True)

    paths = args.inputs if args.inputs else DEFAULT_VOICES
    print("Loading models (resemblyzer + faster-whisper + kNN-VC ONNX)...")
    speaker = _Speaker()
    asr = _Asr(args.whisper)
    converter = VoiceConverter(reference_voice_path=args.reference)
    anon = VoiceAnonymizer(voice_converter=converter)
    params = VoiceParams()

    # Build the voices: load, embed original, transcribe original as pseudo-reference.
    voices: list[Voice] = []
    for p in paths:
        wav, sr = _load_wav(p, args.max_seconds)
        if wav.size == 0:
            print(f"Warning: empty/missing {p}", file=sys.stderr)
            continue
        v = Voice(tag=Path(p).stem.replace("voices_", ""), wav=wav, sr=sr,
                  emb=speaker.embed(wav, sr), ref_text=asr.transcribe(wav, sr))
        voices.append(v)
    if not voices:
        print("No voices loaded.", file=sys.stderr)
        return 1
    print(f"Voices: {[v.tag for v in voices]}")

    # Anonymize every voice with every method (and persist the clip for listening).
    for v in voices:
        for name, method in METHODS:
            out = anon.process(v.wav, v.sr, method=method, params=params)
            v.outputs[name] = np.asarray(out, dtype=np.float32)
            sf.write(str(clip_dir / f"{v.tag}_{name}.wav"), v.outputs[name], v.sr)
        sf.write(str(clip_dir / f"{v.tag}_original.wav"), v.wav, v.sr)

    # Metrics.
    cos_method = {m[0]: [] for m in METHODS}
    wer_method = {m[0]: [] for m in METHODS}
    convert_target_cos: list[float] = []
    target_emb = speaker.embed(*_load_wav(args.reference, args.max_seconds))

    for v in voices:
        for name, _ in METHODS:
            clip = v.outputs[name]
            cos_method[name].append(speaker.cosine(v.emb, speaker.embed(clip, v.sr)))
            wer_method[name].append(asr.wer(v.ref_text, asr.transcribe(clip, v.sr)))
        convert_target_cos.append(speaker.cosine(target_emb, speaker.embed(v.outputs["convert"], v.sr)))

    # Data-driven anchors: same-speaker (two halves of one original) and
    # different-speaker (voice 0 vs voice 1 originals).
    same_vals = []
    for v in voices:
        half = v.wav.shape[0] // 2
        if half > v.sr:  # need a usable chunk
            same_vals.append(speaker.cosine(speaker.embed(v.wav[:half], v.sr),
                                             speaker.embed(v.wav[half:], v.sr)))
    same_ref = float(np.mean(same_vals)) if same_vals else 0.85
    diff_ref = (float(speaker.cosine(voices[0].emb, voices[1].emb))
                if len(voices) >= 2 else 0.0)
    convert_target = float(np.mean(convert_target_cos)) if convert_target_cos else None

    # Figures.
    _fig_spectrograms(voices, anon, converter, params, args.outdir / "voice_spectrograms.png")
    _fig_identity_leakage(cos_method, same_ref, diff_ref, convert_target,
                          args.outdir / "voice_identity_leakage.png")
    _fig_intelligibility(wer_method, args.outdir / "voice_intelligibility.png")
    _fig_privacy_utility(cos_method, wer_method, diff_ref,
                         args.outdir / "voice_privacy_utility.png")
    _fig_f0_contour(voices, args.outdir / "voice_f0_contour.png")

    # Console report.
    print(f"\nSpeaker anchors: same-speaker ~{same_ref:.3f}, different-speaker ~{diff_ref:.3f}")
    print("\nSpeaker cosine vs original (lower = better de-identification):")
    print(f"{'voice':10s} " + " ".join(f"{m[0]:>13s}" for m in METHODS))
    for i, v in enumerate(voices):
        print(f"{v.tag:10s} " + " ".join(f"{cos_method[m[0]][i]:13.3f}" for m in METHODS))
    print(f"{'MEAN':10s} " + " ".join(f"{np.mean(cos_method[m[0]]):13.3f}" for m in METHODS))
    if convert_target is not None:
        print(f"convert vs TARGET identity (higher = took on target): {convert_target:.3f}")

    print("\nWER vs original transcript (lower = better intelligibility):")
    print(f"{'voice':10s} " + " ".join(f"{m[0]:>13s}" for m in METHODS))
    for i, v in enumerate(voices):
        print(f"{v.tag:10s} " + " ".join(f"{wer_method[m[0]][i]:13.3f}" for m in METHODS))
    print(f"{'MEAN':10s} " + " ".join(f"{np.nanmean(wer_method[m[0]]):13.3f}" for m in METHODS))

    print(f"\nSaved -> voice_spectrograms.png, voice_identity_leakage.png, "
          f"voice_intelligibility.png, voice_privacy_utility.png, voice_f0_contour.png "
          f"(in {args.outdir}); clips in {clip_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
