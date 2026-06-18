from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)

# webapp/backend/app/processing/pipeline.py -> parents[4] is the repo root that holds
# the ``ai_core`` package. It is not pip-installed, so make it importable on demand
# regardless of how/where uvicorn is launched.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _ensure_ai_core_importable() -> None:
    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _hex_to_bgr(
    value: Any, default: tuple[int, int, int] = (160, 160, 160)
) -> tuple[int, int, int]:
    """Convert a ``#RRGGBB`` hex colour to an OpenCV ``(B, G, R)`` tuple.

    ai_core fills BGR video frames with this tuple, so the RGB the user picked is
    reordered here. Falls back to neutral grey if the value is missing or malformed.
    """
    text = str(value or "").strip().lstrip("#")
    if len(text) != 6:
        return default
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
    except ValueError:
        return default
    return (b, g, r)


class AnonymizationPipeline:
    """Thin wrapper around ai_core's ``VideoAnonymization``.

    The heavy engine (ONNX models) is built lazily on first use and cached, so the
    web app boots instantly and only pays the model-load cost when the first edit
    runs. ``process_file`` is synchronous (CPU-bound) and is meant to be called from
    a worker thread.
    """

    def __init__(
        self,
        *,
        retinaface_onnx_path: str | None = None,
        restore_blend: float = 0.8,
        knnvc_reference_voice_path: str | None = None,
        knnvc_encoder_onnx_path: str | None = None,
        knnvc_vocoder_onnx_path: str | None = None,
        knnvc_topk: int = 4,
        live_detect_interval: int = 2,
    ) -> None:
        self._retinaface_onnx_path = retinaface_onnx_path
        self._restore_blend = restore_blend
        self._knnvc_reference_voice_path = knnvc_reference_voice_path
        self._knnvc_encoder_onnx_path = knnvc_encoder_onnx_path
        self._knnvc_vocoder_onnx_path = knnvc_vocoder_onnx_path
        self._knnvc_topk = knnvc_topk
        self._live_detect_interval = live_detect_interval
        self._engine: Any | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: "Settings") -> "AnonymizationPipeline":
        return cls(
            retinaface_onnx_path=settings.retinaface_onnx_path,
            knnvc_reference_voice_path=settings.knnvc_reference_voice_path,
            knnvc_encoder_onnx_path=settings.knnvc_encoder_onnx_path,
            knnvc_vocoder_onnx_path=settings.knnvc_vocoder_onnx_path,
            knnvc_topk=settings.knnvc_topk,
            live_detect_interval=settings.live_detect_interval,
        )

    def process_file(
        self,
        source_path: str,
        output_path: str,
        params: dict[str, Any],
        *,
        swap_source_path: str | None = None,
        voice_reference_path: str | None = None,
    ) -> None:
        """Render an anonymized copy of ``source_path`` to ``output_path`` (blocking).

        ``swap_source_path`` / ``voice_reference_path`` are local files the caller has
        already fetched for the selected face / voice. When ``None`` the engine uses
        its bundled defaults. They are passed as values into ai_core's option objects,
        so the shared engine can serve concurrent edits with different identities.
        """
        engine = self._engine_or_build()
        visual, audio = self._build_options(
            params,
            swap_source_path=swap_source_path,
            voice_reference_path=voice_reference_path,
        )
        engine.anonymize_video(
            input_path=source_path,
            output_path=output_path,
            visual=visual,
            audio=audio,
            target_fps=params.get("target_fps"),
            start_sec=params.get("start_sec"),
            end_sec=params.get("end_sec"),
        )

    # ------------------------------------------------------------------ #
    # Live camera (real-time, frame-by-frame)                             #
    # ------------------------------------------------------------------ #
    def create_live_session(self) -> Any:
        """Build one real-time anonymizer for a single live connection.

        Reuses the cached engine's already-loaded detector + anonymizer (so live
        never loads a second copy of the ONNX models); only the lightweight tracker
        is per-session. Blocking on first call — it triggers the lazy model load —
        so call it from a worker thread.
        """
        engine = self._engine_or_build()
        _ensure_ai_core_importable()
        from ai_core.live_anonymization import LiveFaceAnonymizer, LiveVisualConfig

        return LiveFaceAnonymizer(
            face_detector=engine.face_detector,
            face_tracker=engine.face_tracker,
            face_anonymizer=engine.face_anonymizer,
            config=LiveVisualConfig(detect_interval=self._live_detect_interval),
        )

    def build_live_config(self, payload: dict[str, Any]) -> Any:
        """Translate a validated ``LiveConfigMessage`` dict into a ``LiveVisualConfig``.

        ``payload`` comes from :class:`~app.schemas.live.LiveConfigMessage`, so every
        key is present and range-checked; face swap is already excluded there.
        """
        _ensure_ai_core_importable()
        from ai_core.face_anonymization.face_anonymizer import ObfuscationParams
        from ai_core.live_anonymization import LiveVisualConfig

        return LiveVisualConfig(
            method=str(payload.get("visual_method", "none")).strip().lower(),
            detect_interval=int(payload.get("detect_interval", self._live_detect_interval)),
            draw_tracks=bool(payload.get("draw_boxes", False)),
            obfuscation=ObfuscationParams(
                blur_strength=int(payload.get("blur_strength", 31)),
                pixelation_level=int(payload.get("pixelation_level", 16)),
                mask_color=_hex_to_bgr(payload.get("mask_color", "#A0A0A0")),
                # ObfuscationParams coerces the raw string ("parser"/"ellipse").
                mask_shape=str(payload.get("mask_shape", "parser")).strip().lower(),
            ),
        )

    # ------------------------------------------------------------------ #
    # Engine (lazy, cached, thread-safe)                                  #
    # ------------------------------------------------------------------ #
    def _engine_or_build(self) -> Any:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = self._build_engine()
        return self._engine

    def _build_engine(self) -> Any:
        _ensure_ai_core_importable()
        from ai_core.face_alignment.face_aligner import FaceAligner
        from ai_core.face_anonymization.face_anonymizer import FaceAnonymizer
        from ai_core.face_detection.face_detector import FaceDetector
        from ai_core.face_parsing.face_parser import FaceParser
        from ai_core.face_restoration.face_restorer import FaceRestorer
        from ai_core.face_swapping.face_swapper import FaceSwapper
        from ai_core.face_tracking.face_tracker import ByteTracker
        from ai_core.video_anonymization import VideoAnonymization
        from ai_core.video_io.video_io import VideoIO
        from ai_core.voice_anonymization.voice_anonymizer import VoiceAnonymizer

        detector = FaceDetector(onnx_path=self._resolve_retinaface_path())
        parser = FaceParser()  # bundled BiSeNet
        aligner = FaceAligner(output_size=(256, 256), mode="ffhq")
        restorer = FaceRestorer(blend=self._restore_blend)  # bundled GFPGAN
        swapper = FaceSwapper(  # bundled BlendSwap + default source identity
            detector=detector,
            face_parser=parser,
            face_restorer=restorer,
        )
        # One anonymizer handles both paths: obfuscation (parser+aligner -> precise
        # masks) and swap (face_swapper). Voice uses DSP methods by default; the
        # model-based CONVERT method needs a VoiceConverter wired in here.
        anonymizer = FaceAnonymizer(
            face_swapper=swapper,
            face_parser=parser,
            face_aligner=aligner,
        )
        return VideoAnonymization(
            video_io=VideoIO(),
            face_detector=detector,
            face_tracker=ByteTracker(),
            face_anonymizer=anonymizer,
            face_aligner=aligner,
            voice_anonymizer=VoiceAnonymizer(voice_converter=self._build_voice_converter()),
        )

    def _build_voice_converter(self) -> Any | None:
        """Build the kNN-VC converter for the voice ``convert`` method.

        Returns ``None`` (and logs a warning) if the converter cannot be built — e.g.
        the exported ONNX models are missing — so the pipeline still serves every DSP
        voice method; only ``convert`` becomes unavailable.
        """
        _ensure_ai_core_importable()
        try:
            from ai_core.voice_anonymization.voice_converter import VoiceConverter
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "kNN-VC import failed (%s); voice 'convert' disabled, DSP methods OK.",
                exc,
            )
            return None

        kwargs: dict[str, Any] = {"topk": self._knnvc_topk}
        if self._knnvc_reference_voice_path:
            kwargs["reference_voice_path"] = self._knnvc_reference_voice_path
        if self._knnvc_encoder_onnx_path:
            kwargs["encoder_onnx_path"] = self._knnvc_encoder_onnx_path
        if self._knnvc_vocoder_onnx_path:
            kwargs["vocoder_onnx_path"] = self._knnvc_vocoder_onnx_path

        try:
            converter = VoiceConverter(**kwargs)
            logger.info("kNN-VC voice converter loaded.")
            return converter
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "kNN-VC converter unavailable (%s); voice 'convert' disabled, "
                "DSP methods OK.",
                exc,
            )
            return None

    def _resolve_retinaface_path(self) -> str:
        if self._retinaface_onnx_path:
            return self._retinaface_onnx_path
        _ensure_ai_core_importable()
        import ai_core.face_detection.face_detector as detector_module

        bundled = (
            Path(detector_module.__file__).resolve().parent
            / "onnx"
            / "retinaface_best.onnx"
        )
        return str(bundled)

    # ------------------------------------------------------------------ #
    # Params -> ai_core options                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_options(
        params: dict[str, Any],
        *,
        swap_source_path: str | None = None,
        voice_reference_path: str | None = None,
    ) -> tuple[Any, Any]:
        """Translate the persisted edit ``params`` into ai_core option objects.

        ``params`` is the JSON-serialized :class:`VideoEditCreate`, so every key is
        present with a validated value; ``.get(..., default)`` is belt-and-braces for
        rows created before a field existed. The selected face/voice arrive as local
        file paths (already fetched by the caller), not as keys.
        """
        _ensure_ai_core_importable()
        from ai_core.video_anonymization import (
            AudioOptions,
            ObfuscationParams,
            SwapOptions,
            VisualOptions,
            VoiceParams,
        )

        method = str(params.get("visual_method", "blur")).strip().lower()
        if method == "swap":
            # Face swap has its own (model) option object; the obfuscation knobs and
            # the box overlay do not apply to it. ``source_face_path=None`` keeps the
            # engine's bundled default identity.
            visual: Any = SwapOptions(source_face_path=swap_source_path)
        else:
            visual = VisualOptions(
                method=method,
                draw_tracks=bool(params.get("draw_boxes", False)),
                obfuscation=ObfuscationParams(
                    blur_strength=int(params.get("blur_strength", 31)),
                    pixelation_level=int(params.get("pixelation_level", 16)),
                    mask_color=_hex_to_bgr(params.get("mask_color", "#A0A0A0")),
                ),
            )

        audio = AudioOptions(
            keep_audio=bool(params.get("keep_audio", True)),
            anonymize_voice=bool(params.get("anonymize_voice", False)),
            voice_method=str(params.get("voice_method", "mcadams")),
            voice=VoiceParams(
                mcadams_alpha=float(params.get("mcadams_alpha", 0.8)),
                pitch_steps=float(params.get("pitch_steps", -4.0)),
                formant_shift=float(params.get("formant_shift", 1.2)),
            ),
            # Only the CONVERT method consumes this; None keeps the bundled default.
            voice_reference_path=voice_reference_path,
        )
        return visual, audio
