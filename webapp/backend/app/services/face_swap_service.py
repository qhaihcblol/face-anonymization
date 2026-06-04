from __future__ import annotations

import asyncio
from typing import Any

from fastapi import status
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.services.video_service import VideoPipelineService
from app.utils.exceptions import AppException


class FaceSwapService:
    """Run the BlendSwap (model-based) face-swap pipeline on uploaded videos.

    Storage, video I/O and the processing lock are shared with
    :class:`VideoPipelineService` so a swap writes to the same output path the
    ``/anonymized`` endpoint already serves. The swap pipeline itself (ONNX model
    download + inference session) is heavy, so it is built lazily on first use to
    keep application startup fast and avoid loading the model when no one swaps.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        video_service: VideoPipelineService,
    ) -> None:
        self.settings = settings
        self.video_service = video_service
        # Holds the heavy FaceAnonymizer (BlendSwap ONNX session). The thin pipeline
        # orchestrator is rebuilt per request so each run gets its own voice settings.
        self._swap_anonymizer: Any = None
        self._build_lock = asyncio.Lock()

    def _build_swap_anonymizer(self) -> Any:
        # Imported here (not at module load) so the ONNX/BlendSwap dependencies are
        # only required when face swap is actually requested.
        from ai_core.face_anonymization.face_anonymizer import FaceAnonymizer
        from ai_core.face_swapping.face_swapper import FaceSwapper

        swapper = FaceSwapper(
            detector=self.video_service.face_detector,
            model_path=self.settings.resolved_blendswap_onnx_path,
        )
        return FaceAnonymizer(face_swapper=swapper)

    async def _ensure_swap_anonymizer(self) -> Any:
        if self._swap_anonymizer is not None:
            return self._swap_anonymizer

        async with self._build_lock:
            if self._swap_anonymizer is None:
                try:
                    self._swap_anonymizer = await run_in_threadpool(
                        self._build_swap_anonymizer
                    )
                except Exception as exc:
                    raise AppException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Face swap engine is unavailable: {exc}",
                    ) from exc

        return self._swap_anonymizer

    def _swap_blocking(
        self,
        *,
        swap_anonymizer: Any,
        input_path: Any,
        output_path: Any,
        target_fps: int | None,
        start_sec: float | None,
        end_sec: float | None,
        codec: str,
        progress_every: int,
        stabilize: bool,
        smooth_min_cutoff: float,
        smooth_beta: float,
        output_smooth: float,
        mask_smooth: float,
        voice_method: str,
        pitch_steps: float,
        formant_shift: float,
        mcadams_alpha: float,
    ) -> Any:
        """Synchronous work for one face-swap run (executed in a worker thread)."""
        from ai_core.video_anonymization import VideoAnonymization

        voice_anonymizer = self.video_service.build_voice_anonymizer(
            voice_method=voice_method,
            pitch_steps=pitch_steps,
            formant_shift=formant_shift,
            mcadams_alpha=mcadams_alpha,
        )
        pipeline = VideoAnonymization(
            self.video_service.video_io,
            self.video_service.face_detector,
            self.video_service.face_tracker,
            swap_anonymizer,
            voice_anonymizer=voice_anonymizer,
        )
        return pipeline.anonymize_video_with_model(
            input_path,
            output_path,
            target_fps=target_fps,
            start_sec=start_sec,
            end_sec=end_sec,
            codec=codec,
            progress_every=progress_every,
            stabilize=stabilize,
            smooth_min_cutoff=smooth_min_cutoff,
            smooth_beta=smooth_beta,
            output_smooth=output_smooth,
            mask_smooth=mask_smooth,
            keep_audio=True,
            anonymize_voice=voice_anonymizer is not None,
            voice_method=voice_method,
        )

    async def run_face_swap(
        self,
        *,
        video_id: str,
        target_fps: int | None,
        start_sec: float | None,
        end_sec: float | None,
        codec: str,
        progress_every: int,
        stabilize: bool,
        smooth_min_cutoff: float,
        smooth_beta: float,
        output_smooth: float,
        mask_smooth: float,
        voice_method: str,
        pitch_steps: float,
        formant_shift: float,
        mcadams_alpha: float,
    ) -> Any:
        swap_anonymizer = await self._ensure_swap_anonymizer()
        input_path = self.video_service.resolve_upload_path(video_id)
        output_path = self.video_service.resolve_output_path(video_id)

        # Share the pipeline lock so a swap never runs concurrently with another
        # swap or a bbox anonymization (single detector/model instance).
        async with self.video_service.process_lock:
            try:
                result = await run_in_threadpool(
                    self._swap_blocking,
                    swap_anonymizer=swap_anonymizer,
                    input_path=input_path,
                    output_path=output_path,
                    target_fps=target_fps,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    codec=codec,
                    progress_every=progress_every,
                    stabilize=stabilize,
                    smooth_min_cutoff=smooth_min_cutoff,
                    smooth_beta=smooth_beta,
                    output_smooth=output_smooth,
                    mask_smooth=mask_smooth,
                    voice_method=voice_method,
                    pitch_steps=pitch_steps,
                    formant_shift=formant_shift,
                    mcadams_alpha=mcadams_alpha,
                )
            except FileNotFoundError as exc:
                raise AppException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(exc),
                ) from exc
            except ValueError as exc:
                raise AppException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
            except AppException:
                raise
            except Exception as exc:
                raise AppException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Face swap failed: {exc}",
                ) from exc

        return result
