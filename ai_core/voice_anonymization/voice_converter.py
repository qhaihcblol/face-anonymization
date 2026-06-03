from __future__ import annotations

import importlib
from math import gcd
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "DEFAULT_ENCODER_ONNX",
    "DEFAULT_VOCODER_ONNX",
    "DEFAULT_REFERENCE_VOICE",
    "VoiceConverter",
]

_MODULE_DIR = Path(__file__).resolve().parent

# kNN-VC ONNX models live in ai_core/voice_anonymization/onnx/ (mirrors the
# face_detection/onnx/ convention). Produced once by tools/export_knnvc_onnx.py.
DEFAULT_ENCODER_ONNX: Path = _MODULE_DIR / "onnx" / "wavlm_encoder.onnx"
DEFAULT_VOCODER_ONNX: Path = _MODULE_DIR / "onnx" / "hifigan_vocoder.onnx"

# reference_voice.wav (the target pseudo-identity) ships next to this module — the
# audio analog of face_anonymization/source_img.png. Supplied by the user.
DEFAULT_REFERENCE_VOICE: Path = _MODULE_DIR / "reference_voice.wav"

# WavLM (and the matched HiFi-GAN) operate at 16 kHz.
TARGET_SAMPLE_RATE = 16000


class VoiceConverter:
    """kNN-VC voice conversion — convert any speaker to the reference identity.

    Implements kNN-VC (Baas et al., 2023): two ONNX models plus a non-parametric
    matching step.

    1. **Encoder** (WavLM) maps a 16 kHz waveform to a sequence of frame features.
    2. The same encoder run on the **reference voice** builds a "matching set",
       prepared once — the audio analog of ``FaceSwapper.prepare_source()``.
    3. Each source frame feature is replaced by the mean of its ``topk`` nearest
       neighbours (cosine) in the matching set. This is frame-synchronous, so the
       **pitch contour and duration are preserved** (lip-safe).
    4. **Vocoder** (prematched HiFi-GAN) resynthesizes the waveform.

    The identity lives entirely in ``reference_voice.wav`` — no per-speaker training
    is needed; swap the reference to change the pseudo-voice.

    Assumed ONNX I/O (confirm against your export, like the face models were matched
    to FaceFusion): encoder ``(1, n_samples) -> (1, T, D)`` (WavLM features, e.g.
    layer 6); vocoder ``(1, T, D) -> (1, n_out)``. If an export is channels-first
    (``(1, D, T)``), adjust :meth:`_encode` / :meth:`_vocode` — they are the only two
    places that touch tensor layout. Verify the logic with a mocked onnxruntime
    (fake encoder/vocoder sessions) — no weights required.
    """

    def __init__(
        self,
        encoder_onnx_path: str | Path = DEFAULT_ENCODER_ONNX,
        vocoder_onnx_path: str | Path = DEFAULT_VOCODER_ONNX,
        *,
        reference_voice_path: str | Path = DEFAULT_REFERENCE_VOICE,
        topk: int = 4,
        providers: Sequence[str] | None = None,
        intra_op_num_threads: int | None = None,
    ) -> None:
        self.encoder_onnx_path = Path(encoder_onnx_path)
        self.vocoder_onnx_path = Path(vocoder_onnx_path)
        for path in (self.encoder_onnx_path, self.vocoder_onnx_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"kNN-VC ONNX model not found: {path}. Run "
                    "`python tools/export_knnvc_onnx.py` once to export the WavLM "
                    "encoder + HiFi-GAN vocoder, or pass encoder_onnx_path= / "
                    "vocoder_onnx_path= explicitly."
                )

        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        self.topk = int(topk)
        self.reference_voice_path = Path(reference_voice_path)

        self._ort = importlib.import_module("onnxruntime")
        self.encoder = self._create_session(
            self.encoder_onnx_path, providers, intra_op_num_threads
        )
        self.vocoder = self._create_session(
            self.vocoder_onnx_path, providers, intra_op_num_threads
        )
        self.encoder_input = str(self.encoder.get_inputs()[0].name)
        self.encoder_output = str(self.encoder.get_outputs()[0].name)
        self.vocoder_input = str(self.vocoder.get_inputs()[0].name)
        self.vocoder_output = str(self.vocoder.get_outputs()[0].name)

        # Raw reference features (N_ref, D); built lazily by prepare_reference().
        self._matching_set: np.ndarray | None = None
        # librosa is only needed to decode the reference wav from disk.
        self._librosa_module: Any = None

    @property
    def _librosa(self) -> Any:
        if self._librosa_module is None:
            self._librosa_module = importlib.import_module("librosa")
        return self._librosa_module

    def _create_session(
        self,
        path: Path,
        providers: Sequence[str] | None,
        intra_op_num_threads: int | None,
    ) -> Any:
        options = self._ort.SessionOptions()
        if intra_op_num_threads is not None:
            options.intra_op_num_threads = int(intra_op_num_threads)
        resolved = (
            list(providers)
            if providers is not None
            else self._ort.get_available_providers()
        )
        return self._ort.InferenceSession(
            str(path), sess_options=options, providers=resolved
        )

    def prepare_reference(
        self,
        reference: np.ndarray | None = None,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        """Encode the reference voice into the kNN matching set (cached).

        Mirrors ``FaceSwapper.prepare_source``: when ``reference`` is omitted the wav
        at ``reference_voice_path`` is loaded and resampled to 16 kHz. Pass an
        in-memory ``reference`` (with its ``sample_rate``) to override the file.

        Returns the ``(N_ref, D)`` matching set.
        """
        if reference is None:
            mono = self._load_reference(self.reference_voice_path)
        else:
            if sample_rate is None:
                raise ValueError(
                    "sample_rate is required when passing a reference array"
                )
            mono = self._to_mono_16k(reference, sample_rate)

        features = self._encode(mono)
        if features.shape[0] == 0:
            raise ValueError(
                f"Reference voice produced no features: {self.reference_voice_path}"
            )
        self._matching_set = features
        return features

    def convert(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Convert ``waveform`` to the reference identity, preserving length.

        Args:
            waveform: float32 array ``(n_samples,)`` or ``(n_samples, channels)``.
            sample_rate: Sample rate of ``waveform`` in Hz.

        Returns:
            The converted waveform, same shape and length as the input. Multi-channel
            input is downmixed to mono for the model and the converted mono is
            broadcast back to the original channel count.
        """
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        sample_rate = int(sample_rate)

        audio = np.asarray(waveform, dtype=np.float32)
        was_1d = audio.ndim == 1
        if was_1d:
            audio = audio[:, None]
        if audio.ndim != 2 or audio.shape[1] < 1:
            raise ValueError(
                "waveform must be 1-D or 2-D (n_samples, channels >= 1), "
                f"got shape {np.asarray(waveform).shape}"
            )

        n_samples, channels = audio.shape
        if n_samples == 0:
            return waveform if was_1d else audio

        if self._matching_set is None:
            self.prepare_reference()

        mono_16k = self._to_mono_16k(audio, sample_rate)
        features = self._encode(mono_16k)                  # (T, D)
        converted = self._knn(features, self._matching_set)  # (T, D)
        wav_16k = self._vocode(converted)                  # (n_out,)

        out_mono = self._resample(wav_16k, TARGET_SAMPLE_RATE, sample_rate)
        out_mono = self._fit_length(out_mono, n_samples)
        if was_1d:
            return out_mono
        return np.repeat(out_mono[:, None], channels, axis=1)

    def _encode(self, mono_16k: np.ndarray) -> np.ndarray:
        """Run the WavLM encoder: ``(n,) -> (T, D)`` frame features."""
        feeds = {self.encoder_input: mono_16k.astype(np.float32)[None, :]}
        out = self.encoder.run([self.encoder_output], feeds)[0]
        out = np.asarray(out, dtype=np.float32)
        return out.reshape(-1, out.shape[-1])

    def _vocode(self, features: np.ndarray) -> np.ndarray:
        """Run the HiFi-GAN vocoder: ``(T, D) -> (n_out,)`` waveform."""
        feeds = {self.vocoder_input: features.astype(np.float32)[None, ...]}
        out = self.vocoder.run([self.vocoder_output], feeds)[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def _knn(self, query: np.ndarray, matching_set: np.ndarray) -> np.ndarray:
        """Replace each query frame with the mean of its ``topk`` cosine neighbours.

        ``query`` is ``(T, D)``, ``matching_set`` is ``(N, D)`` (raw reference
        features). Neighbours are selected by cosine similarity but averaged over the
        *raw* features, so the output stays in the vocoder's expected feature space.
        """
        k = min(self.topk, matching_set.shape[0])
        query_unit = self._l2_normalize(query)
        set_unit = self._l2_normalize(matching_set)
        similarity = query_unit @ set_unit.T                 # (T, N) cosine
        top_idx = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        neighbours = matching_set[top_idx]                   # (T, k, D)
        return neighbours.mean(axis=1).astype(np.float32)

    def _load_reference(self, path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(
                f"Reference voice not found: {path}. Add a reference_voice.wav (the "
                "target pseudo-identity) or pass reference_voice_path=."
            )
        wave, _ = self._librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
        return np.asarray(wave, dtype=np.float32)

    def _to_mono_16k(self, wave: np.ndarray, sample_rate: int) -> np.ndarray:
        mono = np.asarray(wave, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        return self._resample(mono, sample_rate, TARGET_SAMPLE_RATE)

    @staticmethod
    def _l2_normalize(features: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        return features / (norm + 1e-8)

    @staticmethod
    def _resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        if sr_from == sr_to or y.size == 0:
            return np.asarray(y, dtype=np.float32)
        # Polyphase resampling keeps this dependency-light (scipy only).
        from scipy.signal import resample_poly

        divisor = gcd(int(sr_from), int(sr_to))
        return np.asarray(
            resample_poly(y, int(sr_to) // divisor, int(sr_from) // divisor),
            dtype=np.float32,
        )

    @staticmethod
    def _fit_length(y: np.ndarray, n_samples: int) -> np.ndarray:
        if y.shape[0] == n_samples:
            return y
        if y.shape[0] > n_samples:
            return y[:n_samples]
        return np.pad(y, (0, n_samples - y.shape[0]))
