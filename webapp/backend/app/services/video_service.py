from __future__ import annotations

import contextlib
import uuid
from pathlib import PurePosixPath

from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.user import User
from app.models.video import Video, VideoEdit, VideoEditStatus
from app.processing.base import VideoProcessor
from app.repositories.video_edit_repository import VideoEditRepository
from app.repositories.video_repository import VideoRepository
from app.schemas.video import (
    VideoEditCreate,
    VideoUploadComplete,
    VideoUploadInit,
    VideoUploadTicket,
)
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
    # Uploads (direct-to-storage, presigned)                              #
    # ------------------------------------------------------------------ #
    async def create_upload_ticket(
        self,
        user: User,
        payload: VideoUploadInit,
    ) -> VideoUploadTicket:
        """Validate the request and hand back a presigned URL to upload to directly.

        No DB row is created yet — the upload is only registered once the client
        confirms it via :meth:`complete_upload`. A key is server-generated under the
        user's prefix so the client can never target another user's namespace.
        """
        filename = payload.filename.strip() or "video"
        self._validate_extension(filename)
        self._validate_size(payload.size_bytes)

        storage_key = self._build_upload_key(user.id, filename)
        upload_url = await self.storage.generate_presigned_put_url(
            storage_key, expires_in=settings.r2_upload_url_expiry_seconds
        )
        headers = (
            {"Content-Type": payload.content_type} if payload.content_type else {}
        )
        return VideoUploadTicket(
            storage_key=storage_key,
            upload_url=upload_url,
            headers=headers,
            expires_in=settings.r2_upload_url_expiry_seconds,
        )

    async def complete_upload(
        self,
        db: AsyncSession,
        user: User,
        payload: VideoUploadComplete,
    ) -> Video:
        """Register an upload the client says it finished (step 3 of the flow).

        The object is HEADed to confirm it actually landed and to read its real size
        (authoritative — the client's declared size is not trusted). Oversized objects
        are deleted and rejected.
        """
        storage_key = payload.storage_key
        if not self._owns_key(storage_key, user.id):
            # The key was server-issued under the user's prefix; anything else is a
            # forged request. 404 keeps the namespace opaque.
            raise AppException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found."
            )

        stat = await self.storage.head(storage_key)
        if stat is None or stat.size_bytes == 0:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload was not found in storage. Please upload the file first.",
            )

        if stat.size_bytes > settings.video_max_upload_bytes:
            with contextlib.suppress(StorageError):
                await self.storage.delete(storage_key)
            raise AppException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max {settings.video_max_upload_mb} MB).",
            )

        return await VideoRepository.create(
            db,
            user_id=user.id,
            original_filename=payload.original_filename.strip() or "video",
            storage_key=storage_key,
            content_type=stat.content_type or payload.content_type,
            size_bytes=stat.size_bytes,
            duration_sec=payload.duration_sec,
            width=payload.width,
            height=payload.height,
        )

    # ------------------------------------------------------------------ #
    # Videos                                                              #
    # ------------------------------------------------------------------ #
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
        return await self.storage.generate_presigned_get_url(
            video.storage_key, download_filename=video.original_filename
        )

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
        video = await self.get_video(db, user, video_id)
        return await self.storage.generate_presigned_get_url(
            edit.output_storage_key,
            download_filename=f"protected-{video.original_filename}",
        )

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
    def _validate_extension(filename: str) -> str:
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
        return extension

    @staticmethod
    def _validate_size(size_bytes: int | None) -> None:
        if size_bytes is not None and size_bytes > settings.video_max_upload_bytes:
            raise AppException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max {settings.video_max_upload_mb} MB).",
            )

    @staticmethod
    def _build_upload_key(user_id: int, filename: str) -> str:
        extension = PurePosixPath(filename).suffix.lower()
        return f"uploads/{user_id}/{uuid.uuid4().hex}{extension}"

    @staticmethod
    def _owns_key(storage_key: str, user_id: int) -> bool:
        # Keys are issued only via _build_upload_key, so a valid one always sits under
        # the user's own prefix. Reject path-traversal tricks defensively.
        prefix = f"uploads/{user_id}/"
        return storage_key.startswith(prefix) and ".." not in storage_key
