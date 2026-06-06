from __future__ import annotations

import contextlib
import uuid
from pathlib import PurePosixPath

from fastapi import UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.user import User
from app.models.video import Video, VideoEdit, VideoEditStatus
from app.processing.base import VideoProcessor
from app.repositories.video_edit_repository import VideoEditRepository
from app.repositories.video_repository import VideoRepository
from app.schemas.video import VideoEditCreate
from app.storage.base import Storage, StorageError
from app.utils.exceptions import AppException


class VideoService:
    """Orchestrates video uploads, edits, and presigned downloads.

    Owns no persistence or storage logic itself — it delegates to the repositories,
    the :class:`Storage` backend, and the :class:`VideoProcessor` seam.
    """

    def __init__(self, *, storage: Storage, processor: VideoProcessor) -> None:
        self.storage = storage
        self.processor = processor

    # ------------------------------------------------------------------ #
    # Videos                                                              #
    # ------------------------------------------------------------------ #
    async def upload_video(
        self,
        db: AsyncSession,
        user: User,
        file: UploadFile,
    ) -> Video:
        filename = (file.filename or "").strip() or "video"
        self._validate_upload(file, filename)

        storage_key = self._build_upload_key(user.id, filename)
        await self.storage.upload_fileobj(
            storage_key, file.file, content_type=file.content_type
        )

        return await VideoRepository.create(
            db,
            user_id=user.id,
            original_filename=filename,
            storage_key=storage_key,
            content_type=file.content_type,
            size_bytes=file.size,
        )

    async def list_videos(self, db: AsyncSession, user: User) -> list[Video]:
        return await VideoRepository.list_by_user(db, user.id)

    async def get_video(self, db: AsyncSession, user: User, video_id: int) -> Video:
        video = await VideoRepository.get_by_id(db, video_id)
        return self._ensure_owned(video, user)

    async def delete_video(self, db: AsyncSession, user: User, video_id: int) -> None:
        video = await self.get_video(db, user, video_id)

        # Collect storage keys before the row (and its cascaded edits) are gone.
        edits = await VideoEditRepository.list_by_video(db, video.id)
        keys = [video.storage_key, *(e.output_storage_key for e in edits)]

        await VideoRepository.delete(db, video)

        # Best-effort object cleanup; a storage hiccup must not fail the delete.
        for key in filter(None, keys):
            with contextlib.suppress(StorageError):
                await self.storage.delete(key)

    async def create_source_download_url(
        self,
        db: AsyncSession,
        user: User,
        video_id: int,
    ) -> str:
        video = await self.get_video(db, user, video_id)
        return await self.storage.generate_presigned_get_url(video.storage_key)

    # ------------------------------------------------------------------ #
    # Edits                                                               #
    # ------------------------------------------------------------------ #
    async def create_edit(
        self,
        db: AsyncSession,
        user: User,
        video_id: int,
        payload: VideoEditCreate,
    ) -> VideoEdit:
        video = await self.get_video(db, user, video_id)
        edit = await VideoEditRepository.create(
            db, video_id=video.id, params=payload.model_dump(mode="json")
        )
        # Hand off to the processing seam (stub for now; ai_core wired in later).
        await self.processor.submit(edit)
        return edit

    async def list_edits(
        self,
        db: AsyncSession,
        user: User,
        video_id: int,
    ) -> list[VideoEdit]:
        video = await self.get_video(db, user, video_id)
        return await VideoEditRepository.list_by_video(db, video.id)

    async def get_edit(
        self,
        db: AsyncSession,
        user: User,
        video_id: int,
        edit_id: int,
    ) -> VideoEdit:
        video = await self.get_video(db, user, video_id)
        edit = await VideoEditRepository.get_by_id(db, edit_id)
        if edit is None or edit.video_id != video.id:
            raise AppException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Edit not found."
            )
        return edit

    async def create_output_download_url(
        self,
        db: AsyncSession,
        user: User,
        video_id: int,
        edit_id: int,
    ) -> str:
        edit = await self.get_edit(db, user, video_id, edit_id)
        if edit.status != VideoEditStatus.COMPLETED or not edit.output_storage_key:
            raise AppException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Edit output is not ready yet.",
            )
        return await self.storage.generate_presigned_get_url(edit.output_storage_key)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_owned(video: Video | None, user: User) -> Video:
        # 404 (not 403) so we never reveal that another user's video id exists.
        if video is None or video.user_id != user.id:
            raise AppException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Video not found."
            )
        return video

    @staticmethod
    def _validate_upload(file: UploadFile, filename: str) -> None:
        extension = PurePosixPath(filename).suffix.lower()
        allowed = settings.resolved_video_allowed_extensions
        if extension not in allowed:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported file type '{extension or '?'}'. "
                    f"Allowed: {', '.join(sorted(allowed))}."
                ),
            )
        if file.size is not None and file.size > settings.video_max_upload_bytes:
            raise AppException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max {settings.video_max_upload_mb} MB).",
            )

    @staticmethod
    def _build_upload_key(user_id: int, filename: str) -> str:
        extension = PurePosixPath(filename).suffix.lower()
        return f"uploads/{user_id}/{uuid.uuid4().hex}{extension}"
