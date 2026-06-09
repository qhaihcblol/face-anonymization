from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import anyio

from app.processing.base import VideoProcessor
from app.repositories.video_edit_repository import VideoEditRepository
from app.repositories.video_repository import VideoRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.models.video import VideoEdit
    from app.processing.pipeline import AnonymizationPipeline
    from app.storage.base import Storage

logger = logging.getLogger(__name__)


class LocalVideoProcessor(VideoProcessor):
    """Runs anonymization in-process as a background asyncio task.

    Per edit: download the source from storage, run the (CPU-bound) ai_core pipeline
    in a worker thread, upload the rendered output, and move the edit row through
    ``processing -> completed`` (or ``failed`` with the error). Each task uses its own
    DB session since the request session is already closed by the time it runs.

    Single-instance only. For horizontal scaling, swap in a queue-backed processor
    (Celery / arq / ...) behind the same :class:`VideoProcessor` interface.
    """

    def __init__(
        self,
        *,
        storage: "Storage",
        pipeline: "AnonymizationPipeline",
        session_factory: "async_sessionmaker",
    ) -> None:
        self._storage = storage
        self._pipeline = pipeline
        self._session_factory = session_factory
        # Keep strong references so background tasks are not garbage-collected.
        self._tasks: set[asyncio.Task] = set()

    async def submit(self, edit: "VideoEdit") -> None:
        task = asyncio.create_task(self._run(edit.id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, edit_id: int) -> None:
        async with self._session_factory() as db:
            edit = await VideoEditRepository.get_by_id(db, edit_id)
            if edit is None:
                logger.warning("Edit id=%s vanished before processing.", edit_id)
                return
            video = await VideoRepository.get_by_id(db, edit.video_id)
            if video is None:
                await VideoEditRepository.mark_failed(
                    db, edit, error_message="Source video no longer exists."
                )
                return

            await VideoEditRepository.set_processing(db, edit)
            source_key = video.storage_key
            params = dict(edit.params or {})
            output_key = self._output_key(video.id, edit.id)
            source_suffix = Path(source_key).suffix or ".mp4"

            try:
                with TemporaryDirectory(prefix="anonymize_") as workdir:
                    work = Path(workdir)
                    source_path = work / f"source{source_suffix}"
                    output_path = work / f"{edit.id}.mp4"

                    await self._storage.download_to_path(source_key, str(source_path))

                    # Fetch the selected source face / voice next to the video. Only
                    # for the method that consumes each; a missing key means "use the
                    # engine's bundled default identity".
                    swap_source_path = await self._fetch_optional_asset(
                        params.get("swap_source_key")
                        if params.get("visual_method") == "swap"
                        else None,
                        work / "swap_source",
                    )
                    voice_reference_path = await self._fetch_optional_asset(
                        params.get("voice_reference_key")
                        if params.get("anonymize_voice")
                        and params.get("voice_method") == "convert"
                        else None,
                        work / "voice_reference",
                    )

                    # CPU-bound work off the event loop.
                    await anyio.to_thread.run_sync(
                        partial(
                            self._pipeline.process_file,
                            str(source_path),
                            str(output_path),
                            params,
                            swap_source_path=swap_source_path,
                            voice_reference_path=voice_reference_path,
                        )
                    )
                    with output_path.open("rb") as output_file:
                        await self._storage.upload_fileobj(
                            output_key, output_file, content_type="video/mp4"
                        )

                await VideoEditRepository.mark_completed(
                    db, edit, output_storage_key=output_key
                )
                logger.info("Edit id=%s completed -> %s", edit.id, output_key)
            except Exception as exc:  # noqa: BLE001 - record any failure on the row
                logger.exception("Edit id=%s failed.", edit.id)
                await VideoEditRepository.mark_failed(
                    db, edit, error_message=str(exc)[:1000]
                )

    async def _fetch_optional_asset(
        self, key: str | None, dest_stem: Path
    ) -> str | None:
        """Download ``key`` to ``dest_stem`` (+ the key's extension); ``None`` if unset."""
        if not key:
            return None
        dest = dest_stem.with_suffix(Path(key).suffix)
        await self._storage.download_to_path(key, str(dest))
        return str(dest)

    @staticmethod
    def _output_key(video_id: int, edit_id: int) -> str:
        return f"outputs/{video_id}/{edit_id}.mp4"
