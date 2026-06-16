"""Spectrogram comparison figures for the voice-anonymization chapter.

Setup (as fixed by the thesis): ``source_voices_female_1.wav`` is the SOURCE speaker
and ``source_voices_male_2.wav`` is the TARGET identity for kNN-VC.

Two figures (axes: x = time, y = frequency, brightness = intensity in dB):

  * ``voice_spec_dsp.png``    — source spectrograms side by side:
        original | mcadams | pitch | formant | pitch_formant
  * ``voice_spec_knnvc.png``  — kNN-VC: source (original) vs. converted (+ the target
        for reference), to judge how *natural* the neural conversion is next to DSP.

Usage:
    python -m test_scripts.test_voice_spectrogram_compare
    python -m test_scripts.test_voice_spectrogram_compare --start 2 --duration 6
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from ai_core.voice_anonymization.voice_anonymizer import (
    VoiceAnonymizationMethod,
    VoiceAnonymizer,
    VoiceParams,
)
from ai_core.voice_anonymization.voice_converter import VoiceConverter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_WAV = PROJECT_ROOT / "test_voices" / "voices_female_1.wav"
TARGET_WAV = PROJECT_ROOT / "test_voices" / "voices_male_2.wav"

DSP_METHODS = [
    ("mcadams", VoiceAnonymizationMethod.MCADAMS),
    ("pitch", VoiceAnonymizationMethod.PITCH),
    ("formant", VoiceAnonymizationMethod.FORMANT),
    ("pitch_formant", VoiceAnonymizationMethod.PITCH_FORMANT),
]

HOP = 256
N_FFT = 1024


def _load(path: Path, start: float, duration: float) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    a = int(start * sr)
    b = a + int(duration * sr) if duration > 0 else wav.shape[0]
    return np.ascontiguousarray(wav[a:b], dtype=np.float32), int(sr)


def _draw(ax, wav, sr, label, show_x, fmax):
    """One full-width mel-spectrogram strip (perceptual y, capped at ``fmax``)."""
    import librosa
    import librosa.display
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                          n_mels=96, fmax=fmax)
    db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(db, sr=sr, hop_length=HOP, x_axis="time",
                                   y_axis="mel", fmax=fmax, ax=ax, cmap="magma",
                                   vmin=-80, vmax=0)
    ax.set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center", labelpad=42)
    ax.set_xlabel("time (s)" if show_x else "")
    if not show_x:
        ax.set_xticklabels([])
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    return img


def _strips(panels, sr, fmax, suptitle, out_path):
    """Stack panels vertically as full-width strips (time gets the full width)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11.0, 1.5 * n + 0.6), squeeze=False)
    last = None
    for r, (label, wav) in enumerate(panels):
        last = _draw(axes[r][0], wav, sr, label, show_x=(r == n - 1), fmax=fmax)
    fig.colorbar(last, ax=axes[:, 0], format="%+2.0f dB", fraction=0.015, pad=0.01)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_ltas(panels, sr, fmax, out_path):
    """Long-term average spectrum: one curve per method (intuitive formant view)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    keep = freqs <= fmax
    palette = ["#222", "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    for (label, wav), color in zip(panels, palette):
        mag = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP)).mean(axis=1)
        db = librosa.amplitude_to_db(mag, ref=np.max)
        lw = 2.4 if label.startswith("original") else 1.4
        ax.plot(freqs[keep], db[keep], color=color, lw=lw, label=label)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("average magnitude (dB)")
    ax.set_title("Long-term average spectrum — where each method moves the energy/formants")
    ax.set_xlim(0, fmax)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description="Voice spectrogram comparison figures.")
    p.add_argument("--source", type=Path, default=SOURCE_WAV)
    p.add_argument("--target", type=Path, default=TARGET_WAV)
    p.add_argument("--start", type=float, default=2.0, help="Clip start (s) for readability.")
    p.add_argument("--duration", type=float, default=3.0, help="Clip length (s). 0 = full.")
    p.add_argument("--fmax", type=float, default=8000.0, help="Top of the mel/LTAS axis (Hz).")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    src, sr = _load(args.source, args.start, args.duration)
    tgt, _ = _load(args.target, args.start, args.duration)

    params = VoiceParams()
    converter = VoiceConverter(reference_voice_path=args.target)  # male_2 is the target
    anon = VoiceAnonymizer(voice_converter=converter)

    # ----- Figure 1: DSP methods as stacked full-width strips ------------------
    dsp = [("original (female_1)", src)]
    for name, method in DSP_METHODS:
        dsp.append((name, anon.process(src, sr, method=method, params=params)))
    _strips(dsp, sr, args.fmax,
            "DSP voice anonymization — mel-spectrograms (x = time, y = freq, brightness = intensity)",
            args.outdir / "voice_spec_dsp.png")

    # ----- Figure 2: kNN-VC source vs converted (+ target reference) -----------
    converted = anon.process(src, sr, method=VoiceAnonymizationMethod.CONVERT, params=params)
    _strips([("source (female_1)", src),
             ("kNN-VC converted", converted),
             ("target (male_2)", tgt)], sr, args.fmax,
            "kNN-VC conversion vs DSP — naturalness of the resynthesized speech",
            args.outdir / "voice_spec_knnvc.png")

    # ----- Figure 3: long-term average spectrum (intuitive formant view) -------
    _fig_ltas(dsp, sr, args.fmax, args.outdir / "voice_ltas.png")

    print(f"Saved -> voice_spec_dsp.png, voice_spec_knnvc.png, voice_ltas.png (in {args.outdir})")
    print(f"Source: {args.source.name} | Target (kNN-VC): {args.target.name} | "
          f"window: {args.start:.1f}-{args.start + args.duration:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
