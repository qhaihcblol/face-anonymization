from __future__ import annotations

import importlib
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ai_core.voice_anonymization.voice_converter import VoiceConverter

__all__ = ["VoiceAnonymizationMethod", "VoiceAnonymizer"]


class VoiceAnonymizationMethod(Enum):
    NONE = "none"
    MCADAMS = "mcadams"  # default: non-invertible formant warp (preserves pitch)
    PITCH = "pitch"
    FORMANT = "formant"
    PITCH_FORMANT = "pitch_formant"
    CONVERT = "convert"  # model-based (ONNX voice conversion)


class VoiceAnonymizer:

    def __init__(
        self,
        *,
        mcadams_alpha: float = 0.8,
        lpc_order: int = 0,
        pitch_steps: float = -4.0,
        formant_shift: float = 1.2,
        n_fft: int = 1024,
        hop_length: int = 256,
        lifter: int = 30,
        voice_converter: "VoiceConverter | None" = None,
    ) -> None:
        self.mcadams_alpha = float(mcadams_alpha)
        if self.mcadams_alpha <= 0.0:
            raise ValueError(f"mcadams_alpha must be > 0, got {mcadams_alpha}")
        # 0 -> auto (sr // 1000 + 2), resolved per call since it depends on the rate.
        self.lpc_order = int(lpc_order)

        self.pitch_steps = float(pitch_steps)

        self.formant_shift = float(formant_shift)
        if self.formant_shift <= 0.0:
            raise ValueError(f"formant_shift must be > 0, got {formant_shift}")

        self.n_fft = max(int(n_fft), 32)
        self.hop_length = max(int(hop_length), 1)
        self.lifter = max(int(lifter), 1)

        # The model backend (optional); only required for the CONVERT method.
        self.voice_converter = voice_converter

        # librosa is heavy to import (numba), so defer it until a DSP method runs —
        # the NONE / CONVERT paths never pay for it.
        self._librosa_module: Any = None

    @property
    def _librosa(self) -> Any:
        if self._librosa_module is None:
            self._librosa_module = importlib.import_module("librosa")
        return self._librosa_module

    @staticmethod
    def _coerce_method(
        method: VoiceAnonymizationMethod | str,
    ) -> VoiceAnonymizationMethod:
        if isinstance(method, VoiceAnonymizationMethod):
            return method
        if isinstance(method, str):
            return VoiceAnonymizationMethod(method.strip().lower())
        raise TypeError(
            "method must be VoiceAnonymizationMethod or str, "
            f"got {type(method).__name__}"
        )

    def process(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        method: VoiceAnonymizationMethod | str = VoiceAnonymizationMethod.MCADAMS,
    ) -> np.ndarray:
        """Anonymize ``waveform`` and return it with the same shape, rate and length.

        Args:
            waveform: float32 array of shape ``(n_samples,)`` or
                ``(n_samples, channels)`` in roughly [-1, 1].
            sample_rate: Sample rate of ``waveform`` in Hz.
            method: Which transform to apply. DSP methods run per channel; ``CONVERT``
                delegates to the configured :class:`VoiceConverter`.

        Returns:
            The processed waveform, same dtype/shape/length as the input.

        Raises:
            ValueError: If the waveform shape or sample rate is invalid.
            RuntimeError: If ``CONVERT`` is requested without a ``VoiceConverter``.
        """
        method_value = self._coerce_method(method)

        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        sample_rate = int(sample_rate)

        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim == 1:
            was_1d = True
            audio_2d = audio[:, None]
        elif audio.ndim == 2 and audio.shape[1] >= 1:
            was_1d = False
            audio_2d = audio
        else:
            raise ValueError(
                "waveform must be 1-D or 2-D (n_samples, channels >= 1), "
                f"got shape {audio.shape}"
            )

        n_samples = audio_2d.shape[0]
        if method_value is VoiceAnonymizationMethod.NONE or n_samples == 0:
            return audio.copy()

        if method_value is VoiceAnonymizationMethod.CONVERT:
            out_2d = self._convert(audio_2d, sample_rate)
        else:
            channels = [
                self._anonymize_channel(
                    np.ascontiguousarray(audio_2d[:, c]), method_value, sample_rate
                )
                for c in range(audio_2d.shape[1])
            ]
            out_2d = np.stack(channels, axis=1)

        # Guarantee the length contract even if a backend drifts by a few samples,
        # otherwise the muxed audio would slowly desync from the video.
        out_2d = self._match_length(out_2d, n_samples)
        return out_2d[:, 0] if was_1d else out_2d

    def _anonymize_channel(
        self,
        y: np.ndarray,
        method: VoiceAnonymizationMethod,
        sample_rate: int,
    ) -> np.ndarray:
        if method is VoiceAnonymizationMethod.MCADAMS:
            return self._mcadams_shift(y, sample_rate)
        if method is VoiceAnonymizationMethod.PITCH:
            return self._pitch_shift(y, sample_rate)
        if method is VoiceAnonymizationMethod.FORMANT:
            return self._formant_shift(y, sample_rate)
        if method is VoiceAnonymizationMethod.PITCH_FORMANT:
            # Formant first (on the original pitch), then pitch — pitch_shift resamples
            # internally, so doing it last keeps the warped envelope intact.
            return self._pitch_shift(self._formant_shift(y, sample_rate), sample_rate)
        raise ValueError(f"Unsupported DSP method: {method}")

    def _pitch_shift(self, y: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.pitch_steps == 0.0:
            return y
        shifted = self._librosa.effects.pitch_shift(
            y, sr=sample_rate, n_steps=self.pitch_steps
        )
        return np.asarray(shifted, dtype=np.float32)

    def _formant_shift(self, y: np.ndarray, sample_rate: int) -> np.ndarray:
        """Shift the spectral envelope (formants) without moving the pitch.

        Separates each STFT frame into a smooth spectral envelope (cepstral
        low-quefrency liftering) and the fine harmonic structure (excitation). Only
        the envelope is warped along the frequency axis by ``formant_shift``; the
        excitation — which carries pitch — is left where it is, then the two are
        recombined. ``formant_shift`` > 1 raises formants (smaller-sounding vocal
        tract), < 1 lowers them.
        """
        if self.formant_shift == 1.0 or y.size == 0:
            return y

        librosa = self._librosa
        spec = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(spec)
        phase = np.angle(spec)
        eps = 1e-8

        log_mag = np.log(mag + eps)
        num_bins = log_mag.shape[0]  # n_fft // 2 + 1

        # Real cepstrum along the frequency axis, then keep only low quefrency to
        # recover the smooth envelope (drops the harmonic ripple = excitation).
        cepstrum = np.fft.irfft(log_mag, n=2 * (num_bins - 1), axis=0)
        lifter_win = np.zeros(cepstrum.shape[0], dtype=np.float32)
        keep = min(self.lifter, cepstrum.shape[0] // 2)
        lifter_win[:keep] = 1.0
        if keep > 1:
            lifter_win[-(keep - 1):] = 1.0  # symmetric low-quefrency window
        log_env = np.fft.rfft(cepstrum * lifter_win[:, None], axis=0).real
        envelope = np.exp(log_env)
        excitation = mag / (envelope + eps)

        # Warp the envelope: new_env[f] = env[f / formant_shift].
        bins = np.arange(num_bins, dtype=np.float32)
        source_bins = bins / self.formant_shift
        warped = np.empty_like(envelope)
        for frame in range(envelope.shape[1]):
            warped[:, frame] = np.interp(source_bins, bins, envelope[:, frame])

        new_spec = (warped * excitation) * np.exp(1j * phase)
        out = librosa.istft(
            new_spec, hop_length=self.hop_length, length=int(y.shape[0])
        )
        return np.asarray(out, dtype=np.float32)

    def _mcadams_shift(self, y: np.ndarray, sample_rate: int) -> np.ndarray:
        """McAdams-coefficient transform: warp formants, keep pitch and duration.

        The VoicePrivacy DSP baseline. Per windowed frame:

        1. Fit an LPC all-pole model and find its poles.
        2. Raise each complex pole's *angle* to the power ``mcadams_alpha`` (keeping the
           radius, so the filter stays stable), which moves the formant frequencies.
        3. Inverse-filter the frame with the *original* LPC to get the excitation
           (which carries pitch), then re-synthesize with the *modified* LPC.
        4. Windowed overlap-add (WOLA), normalized by the squared-window envelope.

        Pitch and length are untouched (lip-safe), and the pole-angle warp is lossy /
        non-linear, so the result is not cheaply invertible.
        """
        alpha = self.mcadams_alpha
        if alpha == 1.0 or y.size == 0:
            return y

        from scipy.signal import lfilter

        librosa = self._librosa
        order = self.lpc_order if self.lpc_order > 0 else int(sample_rate / 1000) + 2
        win_len = int(0.020 * sample_rate)  # 20 ms analysis frame
        if win_len < order + 1 or y.shape[0] <= win_len:
            return y  # too short to model — leave it untouched
        shift = max(win_len // 2, 1)  # 50% overlap
        window = np.hanning(win_len).astype(np.float64)

        signal = y.astype(np.float64)
        padded = y.shape[0] + win_len
        out = np.zeros(padded, dtype=np.float64)
        norm = np.zeros(padded, dtype=np.float64)
        eps = 1e-9

        starts = range(0, y.shape[0] - win_len + 1, shift)
        for idx in starts:
            frame = signal[idx : idx + win_len] * window
            try:
                lpc = librosa.lpc(frame + eps, order=order)
            except FloatingPointError:
                lpc = None

            frame_rec = frame
            if lpc is not None and np.all(np.isfinite(lpc)):
                poles = np.roots(lpc)
                new_poles = poles.copy()
                upper = (np.abs(poles.imag) > 1e-9) & (poles.imag > 0)
                lower = (np.abs(poles.imag) > 1e-9) & (poles.imag < 0)
                # Conjugate pairs stay conjugate (real LPC out), filter stays stable.
                up_ang = np.clip(np.angle(poles[upper]) ** alpha, 0.0, np.pi)
                new_poles[upper] = np.abs(poles[upper]) * np.exp(1j * up_ang)
                lo_ang = np.clip((-np.angle(poles[lower])) ** alpha, 0.0, np.pi)
                new_poles[lower] = np.abs(poles[lower]) * np.exp(-1j * lo_ang)
                lpc_new = np.real(np.poly(new_poles))
                residual = lfilter(lpc, [1.0], frame)
                candidate = lfilter([1.0], lpc_new, residual)
                if np.all(np.isfinite(candidate)):
                    frame_rec = candidate

            out[idx : idx + win_len] += frame_rec * window
            norm[idx : idx + win_len] += window * window

        norm[norm < 1e-8] = 1.0
        return (out[: y.shape[0]] / norm[: y.shape[0]]).astype(np.float32)

    def _convert(self, audio_2d: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.voice_converter is None:
            raise RuntimeError(
                "CONVERT requires a VoiceConverter. Construct VoiceAnonymizer with "
                "voice_converter=VoiceConverter(encoder_onnx_path=..., "
                "vocoder_onnx_path=...)."
            )
        converted = self.voice_converter.convert(audio_2d, sample_rate)
        converted = np.asarray(converted, dtype=np.float32)
        if converted.ndim == 1:
            converted = converted[:, None]
        return converted

    @staticmethod
    def _match_length(audio_2d: np.ndarray, n_samples: int) -> np.ndarray:
        current = audio_2d.shape[0]
        if current == n_samples:
            return audio_2d
        if current > n_samples:
            return audio_2d[:n_samples]
        pad = np.zeros((n_samples - current, audio_2d.shape[1]), dtype=audio_2d.dtype)
        return np.concatenate([audio_2d, pad], axis=0)
