from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.video import VideoEdit, VideoEditStatus


class VideoEditRepository:
    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        video_id: int,
        params: dict[str, Any] | None = None,
        status: str = VideoEditStatus.PENDING,
    ) -> VideoEdit:
        edit = VideoEdit(video_id=video_id, params=params, status=status)
        db.add(edit)
        await db.commit()
        await db.refresh(edit)
        return edit

    @staticmethod
    async def get_by_id(db: AsyncSession, edit_id: int) -> VideoEdit | None:
        result = await db.execute(select(VideoEdit).where(VideoEdit.id == edit_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list_by_video(db: AsyncSession, video_id: int) -> list[VideoEdit]:
        result = await db.execute(
            select(VideoEdit)
            .where(VideoEdit.video_id == video_id)
            .order_by(VideoEdit.created_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def set_processing(db: AsyncSession, edit: VideoEdit) -> VideoEdit:
        edit.status = VideoEditStatus.PROCESSING
        edit.error_message = None
        await db.commit()
        await db.refresh(edit)
        return edit

    @staticmethod
    async def mark_completed(
        db: AsyncSession,
        edit: VideoEdit,
        *,
        output_storage_key: str,
    ) -> VideoEdit:
        edit.status = VideoEditStatus.COMPLETED
        edit.output_storage_key = output_storage_key
        edit.error_message = None
        edit.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(edit)
        return edit

    @staticmethod
    async def mark_failed(
        db: AsyncSession,
        edit: VideoEdit,
        *,
        error_message: str,
    ) -> VideoEdit:
        edit.status = VideoEditStatus.FAILED
        edit.error_message = error_message
        edit.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(edit)
        return edit
