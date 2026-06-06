from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db, get_video_service
from app.core.config import settings
from app.models.user import User
from app.schemas.video import (
    PresignedUrlResponse,
    VideoEditCreate,
    VideoEditPublic,
    VideoPublic,
)
from app.services.video_service import VideoService

router = APIRouter()


def _presigned(url: str) -> PresignedUrlResponse:
    return PresignedUrlResponse(url=url, expires_in=settings.r2_presign_expiry_seconds)


@router.post("", response_model=VideoPublic, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: VideoService = Depends(get_video_service),
) -> VideoPublic:
    video = await service.upload_video(db, current_user, file)
    return VideoPublic.model_validate(video)


@router.get("", response_model=list[VideoPublic])
async def list_videos(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: VideoService = Depends(get_video_service),
) -> list[VideoPublic]:
    videos = await service.list_videos(db, current_user)
    return [VideoPublic.model_validate(v) for v in videos]


@router.get("/{video_id}", response_model=VideoPublic)
async def get_video(
    video_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: VideoService = Depends(get_video_service),
) -> VideoPublic:
    video = await service.get_video(db, current_user, video_id)
    return VideoPublic.model_validate(video)


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: VideoService = Depends(get_video_service),
) -> None:
    await service.delete_video(db, current_user, video_id)


@router.get("/{video_id}/download-url", response_model=PresignedUrlResponse)
async def get_video_download_url(
    video_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: VideoService = Depends(get_video_service),
) -> PresignedUrlResponse:
    url = await service.create_source_download_url(db, current_user, video_id)
    return _presigned(url)


@router.post(
    "/{video_id}/edits",
    response_model=VideoEditPublic,
    status_code=status.HTTP_201_CREATED,
)
async def create_edit(
    video_id: int,
    payload: VideoEditCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: VideoService = Depends(get_video_service),
) -> VideoEditPublic:
    edit = await service.create_edit(db, current_user, video_id, payload)
    return VideoEditPublic.model_validate(edit)


@router.get("/{video_id}/edits", response_model=list[VideoEditPublic])
async def list_edits(
    video_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: VideoService = Depends(get_video_service),
) -> list[VideoEditPublic]:
    edits = await service.list_edits(db, current_user, video_id)
    return [VideoEditPublic.model_validate(e) for e in edits]


@router.get("/{video_id}/edits/{edit_id}", response_model=VideoEditPublic)
async def get_edit(
    video_id: int,
    edit_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: VideoService = Depends(get_video_service),
) -> VideoEditPublic:
    edit = await service.get_edit(db, current_user, video_id, edit_id)
    return VideoEditPublic.model_validate(edit)


@router.get(
    "/{video_id}/edits/{edit_id}/download-url",
    response_model=PresignedUrlResponse,
)
async def get_edit_download_url(
    video_id: int,
    edit_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: VideoService = Depends(get_video_service),
) -> PresignedUrlResponse:
    url = await service.create_output_download_url(db, current_user, video_id, edit_id)
    return _presigned(url)
