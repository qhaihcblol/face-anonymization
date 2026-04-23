from __future__ import annotations

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.responses import FileResponse

from app.api.deps import get_video_service
from app.schemas.video import (
    VideoAnonymizeRequest,
    VideoAnonymizeResponse,
    VideoMetadataPublic,
    VideoUploadResponse,
)
from app.services.video_service import VideoPipelineService


router = APIRouter()


def _to_public_metadata(metadata: object) -> VideoMetadataPublic:
    return VideoMetadataPublic(
        fps=float(getattr(metadata, "fps")),
        frame_count=int(getattr(metadata, "frame_count")),
        duration_sec=float(getattr(metadata, "duration_sec")),
        width=int(getattr(metadata, "width")),
        height=int(getattr(metadata, "height")),
    )


@router.post("/upload", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    video_service: VideoPipelineService = Depends(get_video_service),
) -> VideoUploadResponse:
    stored_video = await video_service.save_upload(file)

    return VideoUploadResponse(
        video_id=stored_video.video_id,
        filename=stored_video.filename,
        size_bytes=stored_video.size_bytes,
        metadata=_to_public_metadata(stored_video.metadata),
        original_video_url=str(
            request.url_for("get_uploaded_original_video", video_id=stored_video.video_id)
        ),
        anonymized_video_url=str(
            request.url_for("get_anonymized_video", video_id=stored_video.video_id)
        ),
    )


@router.post("/{video_id}/anonymize", response_model=VideoAnonymizeResponse)
async def anonymize_video(
    video_id: str,
    payload: VideoAnonymizeRequest,
    request: Request,
    video_service: VideoPipelineService = Depends(get_video_service),
) -> VideoAnonymizeResponse:
    result = await video_service.run_anonymization(
        video_id=video_id,
        method=payload.method,
        detect_interval=payload.detect_interval,
        target_fps=payload.target_fps,
        start_sec=payload.start_sec,
        end_sec=payload.end_sec,
        blur_new=payload.blur_new,
        draw_tracks=payload.draw_tracks,
        codec=payload.codec,
        progress_every=payload.progress_every,
    )

    return VideoAnonymizeResponse(
        video_id=video_id,
        method=payload.method,
        target_fps=payload.target_fps,
        start_sec=payload.start_sec,
        end_sec=payload.end_sec,
        output_video_url=str(request.url_for("get_anonymized_video", video_id=video_id)),
        output_metadata=_to_public_metadata(result.output_metadata),
        elapsed_sec=result.elapsed_sec,
        throughput_fps=result.throughput_fps,
    )


@router.get("/{video_id}/original", name="get_uploaded_original_video")
async def get_uploaded_original_video(
    video_id: str,
    video_service: VideoPipelineService = Depends(get_video_service),
) -> FileResponse:
    original_path = video_service.resolve_upload_path(video_id)
    return FileResponse(
        path=str(original_path),
        filename=original_path.name,
        media_type=video_service.guess_media_type(original_path),
    )


@router.get("/{video_id}/anonymized", name="get_anonymized_video")
async def get_anonymized_video(
    video_id: str,
    video_service: VideoPipelineService = Depends(get_video_service),
) -> FileResponse:
    output_path = video_service.resolve_existing_output_path(video_id)
    return FileResponse(
        path=str(output_path),
        filename=output_path.name,
        media_type=video_service.guess_media_type(output_path),
    )
