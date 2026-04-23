import cv2
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


class VideoIO:
    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """
        Get metadata of a video file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps <= 0:
                fps = 25  # Default to 25 if fps is not available or invalid

            if frame_count < 0:
                frame_count = 0

            duration_sec = frame_count / fps if fps > 0 else 0.0
            return VideoMetadata(
                fps=fps,
                frame_count=int(frame_count),
                duration_sec=duration_sec,
                width=width,
                height=height,
            )
        finally:
            cap.release()

    def _validate_time_range(
        self,
        duration_sec: float,
        start_sec: float | None,
        end_sec: float | None,
    ) -> tuple[float, float]:

        # Validate duration
        if not isinstance(duration_sec, (int, float)) or duration_sec <= 0:
            raise ValueError(
                f"duration_sec must be a positive number, got {duration_sec}"
            )

        # Type check
        if start_sec is not None and not isinstance(start_sec, (int, float)):
            raise TypeError(
                f"start_sec must be a number or None, got {type(start_sec).__name__}"
            )
        if end_sec is not None and not isinstance(end_sec, (int, float)):
            raise TypeError(
                f"end_sec must be a number or None, got {type(end_sec).__name__}"
            )

        # Normalize
        start_sec = float(start_sec) if start_sec is not None else 0.0
        end_sec = float(end_sec) if end_sec is not None else float(duration_sec)

        # Validate range
        if not (0 <= start_sec < end_sec <= duration_sec):
            raise ValueError(
                f"Invalid time range: start={start_sec}, end={end_sec}, duration={duration_sec}"
            )

        return start_sec, end_sec

    def _compute_frame_range(
        self,
        source_fps: float,
        total_frames: int,
        start_sec: float,
        end_sec: float,
    ) -> tuple[int, int]:

        start_frame = int(start_sec * source_fps)
        end_frame = min(int(end_sec * source_fps), total_frames)

        return start_frame, end_frame

    def _resolve_target_fps(
        self, source_fps: float, target_fps: int | None
    ) -> float | None:

        if target_fps is None:
            return None

        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")

        if target_fps >= source_fps:
            return None

        return float(target_fps)

    def _should_emit_frame(
        self,
        frame_offset: int,
        emitted_count: int,
        source_fps: float,
        target_fps: float,
    ) -> bool:
        # Ratio-based comparison avoids unstable sampling caused by round(.5).
        return (frame_offset * target_fps) + 1e-9 >= (emitted_count * source_fps)

    def _iter_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        source_fps: float,
        target_fps: float | None,
    ) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            emitted_count = 0
            for frame_offset in range(end_frame - start_frame):
                ok, frame = cap.read()
                if not ok:
                    break

                if target_fps is None or self._should_emit_frame(
                    frame_offset, emitted_count, source_fps, target_fps
                ):
                    emitted_count += 1
                    yield frame
        finally:
            cap.release()

    def _validate_output_fps(self, fps: float) -> float:
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise ValueError(f"fps must be a positive number, got {fps}")
        return float(fps)

    def _normalize_frame_for_write(
        self,
        frame: np.ndarray,
        frame_index: int,
        expected_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"frame at index {frame_index} must be numpy.ndarray, "
                f"got {type(frame).__name__}"
            )

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame at index {frame_index} must have shape (H, W, 3), "
                f"got {frame.shape}"
            )

        height, width = frame.shape[:2]
        if expected_size is not None:
            expected_height, expected_width = expected_size
            if (height, width) != (expected_height, expected_width):
                raise ValueError(
                    "all frames must share the same size; "
                    f"frame at index {frame_index} has {(width, height)}, "
                    f"expected {(expected_width, expected_height)}"
                )

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

    def _create_video_writer(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str,
    ) -> cv2.VideoWriter:
        if not isinstance(codec, str) or len(codec) != 4:
            raise ValueError(
                f"codec must be a 4-character string (e.g. 'mp4v'), got {codec}"
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter.fourcc(*codec)
        writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")

        return writer

    def iter_frames(
        self,
        video_path: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
        target_fps: int | None = None,
    ) -> Iterator[np.ndarray]:
        """
        Lazily yield frames from a video file.

        Args:
            video_path: Path to the video file.
            start_sec:  Start time in seconds. Defaults to 0.0.
            end_sec:    End time in seconds. Defaults to end of video.
            target_fps: Target FPS to sample. Defaults to source FPS.

        Yields:
            Frames as numpy arrays (BGR).

        Raises:
            ValueError: If video cannot be opened or time range is invalid.
        """
        meta = self.get_video_metadata(video_path)

        start_sec, end_sec = self._validate_time_range(
            meta.duration_sec, start_sec, end_sec
        )
        start_frame, end_frame = self._compute_frame_range(
            meta.fps, meta.frame_count, start_sec, end_sec
        )
        resolved_target_fps = self._resolve_target_fps(meta.fps, target_fps)

        yield from self._iter_frames(
            video_path, start_frame, end_frame, meta.fps, resolved_target_fps
        )

    def extract_frames(
        self,
        video_path: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
        target_fps: int | None = None,
    ) -> list[np.ndarray]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file.
            start_sec:  Start time in seconds. Defaults to 0.0.
            end_sec:    End time in seconds. Defaults to end of video.
            target_fps: Target FPS to sample. Defaults to source FPS.

        Returns:
            List of frames as numpy arrays (BGR).

        Raises:
            ValueError: If video cannot be opened or time range is invalid.
        """
        return list(self.iter_frames(video_path, start_sec, end_sec, target_fps))

    def write_frames(
        self,
        frames: Iterable[np.ndarray],
        output_path: str,
        fps: float,
        codec: str = "mp4v",
    ) -> VideoMetadata:
        """
        Write frames to a video file.

        Args:
            frames: Iterable of frames as numpy arrays (preferably BGR).
            output_path: Path of the output video file.
            fps: Output video FPS.
            codec: FourCC codec (default: 'mp4v').

        Returns:
            Metadata of the created output video.

        Raises:
            ValueError: If fps/codec/frames are invalid or output cannot be created.
            TypeError: If any frame is not a numpy array.
        """
        fps = self._validate_output_fps(fps)

        frame_iter = iter(frames)
        try:
            first_frame = next(frame_iter)
        except StopIteration as exc:
            raise ValueError("frames is empty; cannot build output video") from exc

        first_frame = self._normalize_frame_for_write(first_frame, frame_index=0)
        height, width = first_frame.shape[:2]

        writer = self._create_video_writer(output_path, fps, width, height, codec)
        frame_count = 0
        try:
            writer.write(first_frame)
            frame_count += 1

            for frame_index, frame in enumerate(frame_iter, start=1):
                normalized = self._normalize_frame_for_write(
                    frame,
                    frame_index=frame_index,
                    expected_size=(height, width),
                )
                writer.write(normalized)
                frame_count += 1
        finally:
            writer.release()

        return VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            duration_sec=frame_count / fps,
            width=width,
            height=height,
        )
