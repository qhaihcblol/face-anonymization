from __future__ import annotations

import asyncio
from dataclasses import dataclass
import mimetypes
from pathlib import Path
import re
import sys
from typing import Any
from uuid import uuid4

from fastapi import UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.utils.exceptions import AppException


@dataclass(slots=True)
class StoredVideo:
    video_id: str
    filename: str
    size_bytes: int
    stored_path: Path
    metadata: Any


class VideoPipelineService:
    """Manage upload storage and the AI video anonymization pipeline."""

    _CHUNK_SIZE = 1024 * 1024
    _VIDEO_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

    def __init__(
        self,
        *,
        settings: Settings,
        pipeline: Any,
        video_io: Any,
        upload_dir: Path,
        output_dir: Path,
        allowed_extensions: set[str],
        max_upload_bytes: int,
    ) -> None:
        self.settings = settings
        self.pipeline = pipeline
        self.video_io = video_io
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.allowed_extensions = allowed_extensions
        self.max_upload_bytes = max_upload_bytes
        self._process_lock = asyncio.Lock()

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def process_lock(self) -> asyncio.Lock:
        return self._process_lock

    @classmethod
    def create_from_settings(cls, settings: Settings) -> "VideoPipelineService":
        project_root = settings.project_root
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            from ai_core.face_anonymization.face_anonymizer import FaceAnonymizer
            from ai_core.face_detection.face_detector import FaceDetector
            from ai_core.face_tracking.face_tracker import ByteTracker
            from ai_core.video_anonymization import VideoAnonymization
            from ai_core.video_io.video_io import VideoIO
        except Exception as exc:  # pragma: no cover - startup guard
            raise RuntimeError(
                "Failed to import ai_core modules. "
                "Install AI dependencies and verify PYTHONPATH/project structure."
            ) from exc

        onnx_path = settings.resolved_retinaface_onnx_path
        if not onnx_path.exists():
            raise RuntimeError(f"RetinaFace ONNX model not found: {onnx_path}")

        video_io = VideoIO()
        detector = FaceDetector(onnx_path=onnx_path)
        tracker = ByteTracker()
        anonymizer = FaceAnonymizer()
        pipeline = VideoAnonymization(video_io, detector, tracker, anonymizer)

        return cls(
            settings=settings,
            pipeline=pipeline,
            video_io=video_io,
            upload_dir=settings.resolved_video_upload_dir,
            output_dir=settings.resolved_video_output_dir,
            allowed_extensions=settings.resolved_video_allowed_extensions,
            max_upload_bytes=settings.video_max_upload_bytes,
        )

    def sanitize_filename(self, filename: str) -> str:
        base_name = Path(filename).name.strip()
        if not base_name:
            base_name = "video.mp4"

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name)
        safe_name = safe_name.strip("._")
        if not safe_name:
            safe_name = "video.mp4"

        if not Path(safe_name).suffix:
            safe_name = f"{safe_name}.mp4"

        return safe_name

    def validate_video_id(self, video_id: str) -> str:
        value = (video_id or "").strip().lower()
        if not self._VIDEO_ID_PATTERN.fullmatch(value):
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="video_id must be a 32-character lowercase hex string.",
            )
        return value

    async def save_upload(self, upload_file: UploadFile) -> StoredVideo:
        if upload_file.filename is None:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a filename.",
            )

        safe_filename = self.sanitize_filename(upload_file.filename)
        file_ext = Path(safe_filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            allowed = ", ".join(sorted(self.allowed_extensions))
            raise AppException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported video format '{file_ext}'. Allowed: {allowed}",
            )

        video_id = uuid4().hex
        stored_filename = f"{video_id}_{safe_filename}"
        stored_path = self.upload_dir / stored_filename

        total_bytes = 0
        try:
            await upload_file.seek(0)
            with stored_path.open("wb") as output_file:
                while True:
                    chunk = await upload_file.read(self._CHUNK_SIZE)
                    if not chunk:
                        break

                    total_bytes += len(chunk)
                    if total_bytes > self.max_upload_bytes:
                        raise AppException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=(
                                "Uploaded video exceeds max size "
                                f"({self.settings.video_max_upload_mb} MB)."
                            ),
                        )

                    output_file.write(chunk)
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise
        finally:
            await upload_file.close()

        if total_bytes <= 0:
            stored_path.unlink(missing_ok=True)
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded video is empty.",
            )

        try:
            metadata = await run_in_threadpool(
                self.video_io.get_video_metadata,
                str(stored_path),
            )
        except Exception as exc:
            stored_path.unlink(missing_ok=True)
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid video file: {exc}",
            ) from exc

        return StoredVideo(
            video_id=video_id,
            filename=safe_filename,
            size_bytes=total_bytes,
            stored_path=stored_path,
            metadata=metadata,
        )

    def resolve_upload_path(self, video_id: str) -> Path:
        normalized_id = self.validate_video_id(video_id)
        matches = sorted(
            self.upload_dir.glob(f"{normalized_id}_*"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            raise AppException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Uploaded video not found for id '{normalized_id}'.",
            )
        return matches[0]

    def resolve_output_path(self, video_id: str) -> Path:
        normalized_id = self.validate_video_id(video_id)
        return self.output_dir / f"{normalized_id}_anonymized.mp4"

    def resolve_existing_output_path(self, video_id: str) -> Path:
        output_path = self.resolve_output_path(video_id)
        if not output_path.exists():
            raise AppException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Anonymized output not found for id '{video_id}'.",
            )
        return output_path

    def guess_media_type(self, video_path: Path) -> str:
        guessed, _ = mimetypes.guess_type(str(video_path))
        return guessed or "application/octet-stream"

    async def run_anonymization(
        self,
        *,
        video_id: str,
        method: str,
        detect_interval: int,
        target_fps: int | None,
        start_sec: float | None,
        end_sec: float | None,
        blur_new: bool,
        draw_tracks: bool,
        codec: str,
        progress_every: int,
    ) -> Any:
        input_path = self.resolve_upload_path(video_id)
        output_path = self.resolve_output_path(video_id)

        async with self.process_lock:
            try:
                result = await run_in_threadpool(
                    self.pipeline.anonymize_video_without_model,
                    input_path,
                    output_path,
                    method=method,
                    detect_interval=detect_interval,
                    target_fps=target_fps,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    blur_new=blur_new,
                    draw_tracks=draw_tracks,
                    codec=codec,
                    progress_every=progress_every,
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
                    detail=f"Video anonymization failed: {exc}",
                ) from exc

        return result
