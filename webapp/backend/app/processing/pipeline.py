from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.config import Settings

# webapp/backend/app/processing/pipeline.py -> parents[4] is the repo root that holds
# the ``ai_core`` package. It is not pip-installed, so make it importable on demand
# regardless of how/where uvicorn is launched.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _ensure_ai_core_importable() -> None:
    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


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
    ) -> None:
        self._retinaface_onnx_path = retinaface_onnx_path
        self._restore_blend = restore_blend
        self._engine: Any | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: "Settings") -> "AnonymizationPipeline":
        return cls(retinaface_onnx_path=settings.retinaface_onnx_path)

    def process_file(
        self,
        source_path: str,
        output_path: str,
        params: dict[str, Any],
    ) -> None:
        """Render an anonymized copy of ``source_path`` to ``output_path`` (blocking)."""
        engine = self._engine_or_build()
        visual, audio = self._build_options(params)
        engine.anonymize_video(
            input_path=source_path,
            output_path=output_path,
            visual=visual,
            audio=audio,
            start_sec=params.get("start_sec"),
            end_sec=params.get("end_sec"),
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
            voice_anonymizer=VoiceAnonymizer(),
        )

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
    def _build_options(params: dict[str, Any]) -> tuple[Any, Any]:
        _ensure_ai_core_importable()
        from ai_core.video_anonymization import (
            AudioOptions,
            SwapOptions,
            VisualOptions,
        )

        method = str(params.get("visual_method", "blur")).strip().lower()
        visual = SwapOptions() if method == "swap" else VisualOptions(method=method)
        audio = AudioOptions(
            keep_audio=bool(params.get("keep_audio", True)),
            anonymize_voice=bool(params.get("anonymize_voice", False)),
            voice_method=str(params.get("voice_method", "mcadams")),
        )
        return visual, audio
