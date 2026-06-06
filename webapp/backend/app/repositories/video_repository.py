from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.video import Video


class VideoRepository:
    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        user_id: int,
        original_filename: str,
        storage_key: str,
        content_type: str | None = None,
        size_bytes: int | None = None,
        duration_sec: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Video:
        video = Video(
            user_id=user_id,
            original_filename=original_filename,
            storage_key=storage_key,
            content_type=content_type,
            size_bytes=size_bytes,
            duration_sec=duration_sec,
            width=width,
            height=height,
        )
        db.add(video)
        await db.commit()
        await db.refresh(video)
        return video

    @staticmethod
    async def get_by_id(db: AsyncSession, video_id: int) -> Video | None:
        result = await db.execute(select(Video).where(Video.id == video_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list_by_user(db: AsyncSession, user_id: int) -> list[Video]:
        result = await db.execute(
            select(Video)
            .where(Video.user_id == user_id)
            .order_by(Video.created_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def delete(db: AsyncSession, video: Video) -> None:
        # Related video_edits rows are removed by the ON DELETE CASCADE foreign key.
        await db.delete(video)
        await db.commit()
