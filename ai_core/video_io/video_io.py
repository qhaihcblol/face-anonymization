import cv2
import json
import shutil
import subprocess
import tempfile
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

    def _open_writer_with_fallback(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str,
    ) -> cv2.VideoWriter:
        """Open a cv2 writer with ``codec``, falling back to 'mp4v' if it can't open.

        Some OpenCV builds ship without an H.264 encoder, so 'H264'/'avc1' fail to
        open a writer. Rather than abort the whole render, retry with the universally
        available 'mp4v' and warn. (The ffmpeg path is preferred and produces real
        H.264; this fallback only runs when ffmpeg is missing.)
        """
        try:
            return self._create_video_writer(output_path, fps, width, height, codec)
        except ValueError:
            if codec.lower() == "mp4v":
                raise
            print(
                f"Warning: OpenCV cannot open a '{codec}' writer in this build; "
                "falling back to 'mp4v'."
            )
            return self._create_video_writer(output_path, fps, width, height, "mp4v")

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

    def has_audio(self, video_path: str) -> bool:
        """True if the file has at least one decodable audio stream."""
        return self._probe_audio_params(str(video_path)) is not None

    def extract_audio(
        self,
        video_path: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Decode the first audio stream into a float32 waveform.

        Mirrors ``iter_frames`` for audio: the optional ``[start_sec, end_sec]``
        window is honoured so the extracted track stays aligned with a trimmed
        render. The source sample rate and channel layout are preserved.

        Args:
            video_path: Path to the media file.
            start_sec:  Start time in seconds. Defaults to 0.0.
            end_sec:    End time in seconds. Defaults to end of stream.

        Returns:
            ``(waveform, sample_rate)`` where ``waveform`` has shape
            ``(n_samples, channels)`` as float32 in roughly [-1, 1] (always 2-D,
            even for mono).

        Raises:
            ValueError: If ffmpeg is unavailable, the file has no audio stream,
                the time range is invalid, or decoding fails.
        """
        if not self._ffmpeg_available():
            raise ValueError(
                "ffmpeg is required to extract audio but was not found on PATH."
            )

        params = self._probe_audio_params(str(video_path))
        if params is None:
            raise ValueError(f"No audio stream found in: {video_path}")
        sample_rate, channels = params

        start = float(start_sec) if start_sec else 0.0
        if start < 0:
            raise ValueError(f"start_sec must be >= 0, got {start_sec}")

        ffmpeg = shutil.which("ffmpeg")
        # -ss before -i = fast, accurate-enough input seek for audio.
        cmd: list[str] = [str(ffmpeg), "-v", "error"]
        if start > 0:
            cmd += ["-ss", f"{start}"]
        if end_sec is not None:
            duration = float(end_sec) - start
            if duration <= 0:
                raise ValueError(
                    f"Invalid time range: start={start}, end={end_sec}"
                )
            cmd += ["-t", f"{duration}"]
        cmd += [
            "-i", str(video_path),
            "-map", "0:a:0",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-",
        ]

        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(
                f"ffmpeg failed extracting audio from {video_path}: "
                f"{stderr or '<no stderr>'}"
            )

        # frombuffer gives a read-only view over ffmpeg's bytes; copy so callers
        # (voice anonymizer) can transform the waveform in place.
        waveform = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        return waveform.reshape(-1, channels), sample_rate

    def write_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        output_path: str,
    ) -> str:
        """Write a float32 waveform to a WAV file via ffmpeg.

        Used to hand a processed audio track back as the ``audio_source`` for
        ``write_frames``. Samples outside [-1, 1] are clipped by ffmpeg during the
        f32 -> pcm_s16le conversion.

        Args:
            waveform: float32 array of shape ``(n_samples,)`` or
                ``(n_samples, channels)``.
            sample_rate: Output sample rate (Hz).
            output_path: Destination WAV path.

        Returns:
            ``output_path`` as a string.

        Raises:
            ValueError: If ffmpeg is unavailable, the waveform shape or sample
                rate is invalid, or encoding fails.
        """
        if not self._ffmpeg_available():
            raise ValueError(
                "ffmpeg is required to write audio but was not found on PATH."
            )
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

        audio = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
        if audio.ndim == 1:
            channels = 1
        elif audio.ndim == 2 and audio.shape[1] >= 1:
            channels = int(audio.shape[1])
        else:
            raise ValueError(
                f"waveform must be 1-D or 2-D (n_samples, channels >= 1), "
                f"got shape {audio.shape}"
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg = shutil.which("ffmpeg")
        cmd = [
            str(ffmpeg),
            "-y",
            "-v", "error",
            "-f", "f32le",
            "-ar", str(int(sample_rate)),
            "-ac", str(channels),
            "-i", "-",
            "-c:a", "pcm_s16le",
            str(out),
        ]

        proc = subprocess.run(cmd, input=audio.tobytes(), capture_output=True)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(
                f"ffmpeg failed writing audio to {output_path}: "
                f"{stderr or '<no stderr>'}"
            )
        return str(out)

    @staticmethod
    def _ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    @staticmethod
    def _source_has_audio(source_path: str) -> bool:
        """True if ``source_path`` has at least one audio stream (via ffprobe)."""
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return False
        try:
            proc = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=index",
                    "-of", "csv=p=0",
                    str(source_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.SubprocessError, OSError):
            return False
        return bool(proc.stdout.strip())

    @staticmethod
    def _probe_audio_params(source_path: str) -> tuple[int, int] | None:
        """Return (sample_rate, channels) of the first audio stream, or None.

        None means ffprobe is missing, the probe failed, or the file carries no
        audio stream. JSON output is parsed so the two fields are unambiguous
        regardless of ffprobe's CSV field ordering.
        """
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return None
        try:
            proc = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=sample_rate,channels",
                    "-of", "json",
                    str(source_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        if proc.returncode != 0:
            return None
        try:
            streams = json.loads(proc.stdout).get("streams", [])
        except (json.JSONDecodeError, AttributeError):
            return None
        if not streams:
            return None
        stream = streams[0]
        try:
            sample_rate = int(stream["sample_rate"])
            channels = int(stream["channels"])
        except (KeyError, ValueError, TypeError):
            return None
        if sample_rate <= 0 or channels <= 0:
            return None
        return sample_rate, channels

    def _write_frames_ffmpeg(
        self,
        frames: Iterable[np.ndarray],
        output_path: str,
        fps: float,
        *,
        audio_source: str | None,
        audio_start_sec: float,
        crf: int,
        preset: str,
    ) -> VideoMetadata:
        """Encode frames via an ffmpeg pipe, muxing audio from ``audio_source``.

        Frames are streamed as raw BGR24 into ffmpeg's stdin and encoded with
        libx264 (yuv420p for broad playback). When ``audio_source`` carries an audio
        stream it is mapped in (seeked to ``audio_start_sec`` to honour a trimmed
        range) and ``-shortest`` keeps it aligned to the rendered video length.
        """
        frame_iter = iter(frames)
        try:
            first_frame = next(frame_iter)
        except StopIteration as exc:
            raise ValueError("frames is empty; cannot build output video") from exc

        first_frame = self._normalize_frame_for_write(first_frame, frame_index=0)
        height, width = first_frame.shape[:2]

        has_audio = audio_source is not None and self._source_has_audio(audio_source)

        ffmpeg = shutil.which("ffmpeg")
        cmd: list[str] = [
            str(ffmpeg),
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps}",
            "-i", "-",
        ]
        if has_audio:
            if audio_start_sec and audio_start_sec > 0:
                cmd += ["-ss", f"{audio_start_sec}"]
            cmd += ["-i", str(audio_source)]

        cmd += ["-map", "0:v:0"]
        if has_audio:
            # '?' makes the audio mapping optional so a stream-less file never aborts.
            cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
        cmd += [
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p",
        ]
        if has_audio:
            cmd += ["-shortest"]
        cmd += [str(output_path)]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # stderr -> temp file (not a pipe) so a chatty ffmpeg can never deadlock the
        # writer by filling an unread pipe buffer mid-stream.
        stderr_buf = tempfile.TemporaryFile()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr_buf)
        assert proc.stdin is not None

        frame_count = 0
        try:
            proc.stdin.write(first_frame.tobytes())
            frame_count += 1
            for frame_index, frame in enumerate(frame_iter, start=1):
                normalized = self._normalize_frame_for_write(
                    frame,
                    frame_index=frame_index,
                    expected_size=(height, width),
                )
                proc.stdin.write(normalized.tobytes())
                frame_count += 1
        except BrokenPipeError:
            # ffmpeg exited early; the return code + stderr below explain why.
            pass
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            returncode = proc.wait()
            stderr_buf.seek(0)
            stderr = stderr_buf.read().decode("utf-8", errors="replace").strip()
            stderr_buf.close()

        if returncode != 0:
            raise ValueError(
                f"ffmpeg failed (exit code {returncode}) writing {output_path}: "
                f"{stderr or '<no stderr>'}"
            )

        return VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            duration_sec=frame_count / fps if fps > 0 else 0.0,
            width=width,
            height=height,
        )

    def write_frames(
        self,
        frames: Iterable[np.ndarray],
        output_path: str,
        fps: float,
        codec: str = "H264",
        *,
        audio_source: str | None = None,
        audio_start_sec: float = 0.0,
        crf: int = 18,
        preset: str = "medium",
    ) -> VideoMetadata:
        """
        Write frames to a video file.

        Encoding prefers ffmpeg (libx264 -> H.264, yuv420p) whenever ffmpeg is on the
        PATH, so the output is real H.264 regardless of audio. ``audio_source`` is
        muxed in on that path. Only when ffmpeg is unavailable does it fall back to the
        OpenCV writer, which is silent and uses the ``codec`` FourCC (auto-falling back
        to 'mp4v' if the requested codec cannot be opened — e.g. OpenCV builds without
        libx264 cannot write H.264).

        Args:
            frames: Iterable of frames as numpy arrays (preferably BGR).
            output_path: Path of the output video file.
            fps: Output video FPS.
            codec: FourCC codec for the OpenCV fallback path only (default: 'H264',
                falls back to 'mp4v' when the build cannot open it).
            audio_source: When set (and ffmpeg is available), this file's audio is
                muxed into the output.
            audio_start_sec: Seek offset applied to the muxed audio so it stays in
                sync with a trimmed (``start_sec``) render.
            crf / preset: libx264 quality knobs for the ffmpeg path.

        Returns:
            Metadata of the created output video.

        Raises:
            ValueError: If fps/codec/frames are invalid or output cannot be created.
            TypeError: If any frame is not a numpy array.
        """
        fps = self._validate_output_fps(fps)

        if self._ffmpeg_available():
            return self._write_frames_ffmpeg(
                frames,
                output_path,
                fps,
                audio_source=audio_source,
                audio_start_sec=audio_start_sec,
                crf=crf,
                preset=preset,
            )

        if audio_source is not None:
            print(
                "Warning: ffmpeg not found on PATH; writing video without audio "
                "via the OpenCV writer."
            )

        frame_iter = iter(frames)
        try:
            first_frame = next(frame_iter)
        except StopIteration as exc:
            raise ValueError("frames is empty; cannot build output video") from exc

        first_frame = self._normalize_frame_for_write(first_frame, frame_index=0)
        height, width = first_frame.shape[:2]

        writer = self._open_writer_with_fallback(
            output_path, fps, width, height, codec
        )
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
