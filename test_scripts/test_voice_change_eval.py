"""How much / how does each method change the voice *relative to the original*.

Unlike the side-by-side spectrograms (which show each result on its own), these two
figures make the **change from the original** the subject:

  * ``voice_change_spectrogram.png`` — top strip = the original mel-spectrogram; each
    strip below it is the **difference** (method - original) in dB on a diverging
    colormap: RED = energy this method *added*, BLUE = energy it *removed*. You read
    off exactly which frequencies/times were altered. (DSP methods and kNN-VC all keep
    the duration, so frames line up and the subtraction is meaningful.)
  * ``voice_change_metrics.png`` — the change decomposed into two intuitive numbers:
    LEFT = **timbre change** (mel-cepstral distortion, MCD, from the original; higher =
    more changed), RIGHT = **pitch change** (median F0 shift in semitones; ~0 = pitch
    preserved). Together: mcadams/formant move timbre but not pitch; pitch moves pitch.

Source = ``voices_female_1.wav``; kNN-VC target = ``voices_male_2.wav``.

Usage:
    python -m test_scripts.test_voice_change_eval
    python -m test_scripts.test_voice_change_eval --start 2 --duration 3 --fmax 8000
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

METHODS = [
    ("mcadams", VoiceAnonymizationMethod.MCADAMS),
    ("pitch", VoiceAnonymizationMethod.PITCH),
    ("formant", VoiceAnonymizationMethod.FORMANT),
    ("pitch_formant", VoiceAnonymizationMethod.PITCH_FORMANT),
    ("convert", VoiceAnonymizationMethod.CONVERT),
]
COLORS = {
    "mcadams": "#264653", "pitch": "#2a9d8f", "formant": "#e9c46a",
    "pitch_formant": "#f4a261", "convert": "#e76f51",
}
N_FFT = 1024
HOP = 256


def _load(path: Path, start: float, duration: float) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    a = int(start * sr)
    b = a + int(duration * sr) if duration > 0 else wav.shape[0]
    return np.ascontiguousarray(wav[a:b], dtype=np.float32), int(sr)


def _mel_db(wav, sr, fmax):
    import librosa
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=96, fmax=fmax)
    return librosa.power_to_db(mel, ref=np.max)


def _lsd(orig, other, sr):
    """Log-spectral distance (dB) between two waveforms — overall spectral change.

    Per frame: RMS over frequency of the log-magnitude (dB) difference; averaged over
    the frames that are energetic in *both* clips so silent gaps don't skew it. A
    standard, directly-interpretable measure of how far the spectrum moved.
    """
    import librosa
    A = np.abs(librosa.stft(orig, n_fft=N_FFT, hop_length=HOP))
    B = np.abs(librosa.stft(other, n_fft=N_FFT, hop_length=HOP))
    t = min(A.shape[1], B.shape[1])
    A, B = A[:, :t], B[:, :t]
    ad = 20.0 * np.log10(np.maximum(A, A.max() * 1e-4))
    bd = 20.0 * np.log10(np.maximum(B, B.max() * 1e-4))
    per_frame = np.sqrt(np.mean((ad - bd) ** 2, axis=0))
    ea = librosa.feature.rms(y=orig, frame_length=N_FFT, hop_length=HOP)[0][:t]
    eb = librosa.feature.rms(y=other, frame_length=N_FFT, hop_length=HOP)[0][:t]
    active = (ea > ea.max() * 0.05) & (eb > eb.max() * 0.05)
    if active.sum() < 5:
        active = np.ones(t, dtype=bool)
    return float(per_frame[active].mean())


def _median_f0(wav, sr):
    import librosa
    f0, _, _ = librosa.pyin(wav, sr=sr, fmin=65.0, fmax=1000.0,
                            frame_length=2048, hop_length=HOP)
    voiced = f0[np.isfinite(f0)]
    return float(np.median(voiced)) if voiced.size else float("nan")


def _semitone_shift(orig, other, sr):
    f0a, f0b = _median_f0(orig, sr), _median_f0(other, sr)
    if not (np.isfinite(f0a) and np.isfinite(f0b)) or f0a <= 0 or f0b <= 0:
        return float("nan")
    return float(12.0 * np.log2(f0b / f0a))


def _fig_diff(original, outs, sr, fmax, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig_db = _mel_db(original, sr, fmax)
    n = 1 + len(METHODS)
    fig, axes = plt.subplots(n, 1, figsize=(11.0, 1.5 * n + 0.6), squeeze=False)

    import librosa.display
    img0 = librosa.display.specshow(orig_db, sr=sr, hop_length=HOP, x_axis="time",
                                    y_axis="mel", fmax=fmax, ax=axes[0][0],
                                    cmap="magma", vmin=-80, vmax=0)
    axes[0][0].set_ylabel("original", fontsize=10, rotation=0, ha="right", va="center", labelpad=46)
    fig.colorbar(img0, ax=axes[0][0], format="%+2.0f dB", fraction=0.015, pad=0.01)

    last = None
    for r, (name, _) in enumerate(METHODS, start=1):
        diff = _mel_db(outs[name], sr, fmax) - orig_db
        t = min(diff.shape[1], orig_db.shape[1])
        last = librosa.display.specshow(diff[:, :t], sr=sr, hop_length=HOP, x_axis="time",
                                        y_axis="mel", fmax=fmax, ax=axes[r][0],
                                        cmap="RdBu_r", vmin=-30, vmax=30)
        axes[r][0].set_ylabel(f"{name}\n- original", fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=46)
        axes[r][0].set_yticks([])
        if r != n - 1:
            axes[r][0].set_xticklabels([])
            axes[r][0].set_xlabel("")
    fig.colorbar(last, ax=axes[1:, 0], format="%+2.0f dB", fraction=0.015, pad=0.01,
                 label="energy vs original (red +, blue -)")
    axes[0][0].set_xticklabels([]); axes[0][0].set_xlabel("")
    fig.suptitle("How each method changes the voice vs the original (difference spectrogram)",
                 fontsize=12)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_metrics(lsd, semis, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [m[0] for m in METHODS]
    cols = [COLORS[n] for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    b1 = ax1.bar(names, [lsd[n] for n in names], color=cols)
    ax1.bar_label(b1, fmt="%.1f", padding=3)
    ax1.set_ylabel("log-spectral distance (dB) vs original")
    ax1.set_title("Spectral change (higher = more changed)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis="x", labelrotation=15)

    sv = [semis[n] for n in names]
    b2 = ax2.bar(names, sv, color=cols)
    ax2.bar_label(b2, fmt="%+.1f", padding=3)
    ax2.axhline(0, color="#444", lw=1.0)
    lo, hi = min(sv + [0]), max(sv + [0])
    ax2.set_ylim(lo - 1.2, hi + 1.2)  # headroom so the value labels aren't clipped
    ax2.set_ylabel("median F0 shift (semitones)")
    ax2.set_title("Pitch change (0 = pitch preserved)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis="x", labelrotation=15)

    fig.suptitle("Amount of voice change relative to the original", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description="Visualize voice change vs the original.")
    p.add_argument("--source", type=Path, default=SOURCE_WAV)
    p.add_argument("--target", type=Path, default=TARGET_WAV)
    p.add_argument("--start", type=float, default=2.0)
    p.add_argument("--duration", type=float, default=3.0, help="0 = full clip.")
    p.add_argument("--fmax", type=float, default=8000.0)
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    src, sr = _load(args.source, args.start, args.duration)
    params = VoiceParams()
    converter = VoiceConverter(reference_voice_path=args.target)
    anon = VoiceAnonymizer(voice_converter=converter)

    outs = {name: np.asarray(anon.process(src, sr, method=method, params=params),
                             dtype=np.float32)
            for name, method in METHODS}

    _fig_diff(src, outs, sr, args.fmax, args.outdir / "voice_change_spectrogram.png")

    lsd = {n: _lsd(src, outs[n], sr) for n, _ in METHODS}
    semis = {n: _semitone_shift(src, outs[n], sr) for n, _ in METHODS}
    _fig_metrics(lsd, semis, args.outdir / "voice_change_metrics.png")

    print("Voice change vs original (source: %s):" % args.source.name)
    print(f"{'method':14s} {'LSD(dB)':>9s} {'F0 shift(st)':>13s}")
    for n, _ in METHODS:
        print(f"{n:14s} {lsd[n]:9.2f} {semis[n]:13.2f}")
    print(f"\nSaved -> voice_change_spectrogram.png, voice_change_metrics.png (in {args.outdir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
