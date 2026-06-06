from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.video import VideoEdit

logger = logging.getLogger(__name__)


class VideoProcessor(ABC):
    """Seam where the anonymization pipeline runs.

    The real implementation will, for a freshly created (pending) edit:
      1. download the source video from storage,
      2. run ai_core ``VideoAnonymization.anonymize_video(...)`` with the edit params,
      3. upload the rendered output and set ``output_storage_key``,
      4. update ``status`` to completed/failed.

    Keeping it behind this interface lets the API/service layer stay independent of
    ai_core and of how/where the work is executed (inline, background task, queue).
    """

    @abstractmethod
    async def submit(self, edit: "VideoEdit") -> None:
        """Begin processing a pending edit (typically schedules background work)."""


class StubVideoProcessor(VideoProcessor):
    """No-op processor: leaves the edit ``pending``.

    Placeholder until the ai_core pipeline is wired in. Useful for developing the
    API/storage layers without loading models or doing heavy CPU work.
    """

    async def submit(self, edit: "VideoEdit") -> None:
        logger.info(
            "VideoProcessor stub: edit id=%s left pending "
            "(anonymization pipeline not wired yet).",
            edit.id,
        )
