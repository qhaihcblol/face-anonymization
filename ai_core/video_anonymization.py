from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterator

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_anonymization.face_anonymizer import (
    AnonymizationMethod,
    FaceAnonymizer,
)
from ai_core.face_anonymization.face_swap_stabilizer import FaceSwapStabilizer
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_tracking.face_tracker import ByteTracker
from ai_core.video_io.video_io import VideoIO, VideoMetadata


@dataclass(slots=True)
class VideoAnonymizationResult:
    output_path: Path
    output_metadata: VideoMetadata
    elapsed_sec: float
    throughput_fps: float


class VideoAnonymization:
    def __init__(
        self,
        video_io: VideoIO,
        face_detector: FaceDetector,
        face_tracker: ByteTracker,
        face_anonymizer: FaceAnonymizer,
        face_aligner: FaceAligner | None = None,
    ) -> None:
        self.video_io = video_io
        self.face_detector = face_detector
        self.face_tracker = face_tracker
        self.face_anonymizer = face_anonymizer
        # Aligner is only needed for the model-based (face swap) path.
        self.face_aligner = face_aligner

    def _resolve_face_aligner(self) -> FaceAligner:
        # BlendSwap expects the FFHQ template at 256x256; FaceSwapper re-derives the
        # crop from the original 5-point landmarks, so any valid alignment works.
        if self.face_aligner is None:
            self.face_aligner = FaceAligner(output_size=(256, 256), mode=AlignMode.FFHQ)
        return self.face_aligner

    @staticmethod
    def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
        if output_path is not None:
            return output_path
        return Path("outputs") / f"anonymized_{input_path.stem}.mp4"

    @staticmethod
    def _resolve_output_fps(source_fps: float, target_fps: int | None) -> float:
        if target_fps is None:
            return float(source_fps)
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")
        if target_fps >= source_fps:
            return float(source_fps)
        return float(target_fps)

    @staticmethod
    def _build_fresh_tracker(tracker: ByteTracker) -> ByteTracker:
        # Tracking state should be clean for each video run.
        return ByteTracker(
            high_thresh=tracker.high_thresh,
            low_thresh=tracker.low_thresh,
            max_lost=tracker.max_lost,
            min_hits=tracker.min_hits,
            iou_thresh=tracker.iou_thresh,
            iou_thresh_low=tracker.iou_thresh_low,
            gate_mahal=tracker.gate_mahal,
        )

    def _iter_processed_frames(
        self,
        frames: Iterator[np.ndarray],
        tracker: ByteTracker,
        method: AnonymizationMethod,
        detect_interval: int,
        blur_new: bool,
        draw_tracks: bool,
        progress_every: int,
    ) -> Iterator[np.ndarray]:
        frame_idx = 0
        last_detect_ms = 0.0
        tracks: list[dict] = []

        for frame_bgr in frames:
            run_detect = (frame_idx % detect_interval) == 0
            if run_detect:
                t0 = time.perf_counter()
                detections = self.face_detector.detect(frame_bgr)
                last_detect_ms = (time.perf_counter() - t0) * 1000.0
                tracks = tracker.update(detections)
            else:
                tracks = tracker.predict_only()

            if blur_new:
                tracks_for_anonymize = tracks
            else:
                tracks_for_anonymize = [
                    track for track in tracks if track.get("state") == "Tracked"
                ]

            anonymized = self.face_anonymizer.anonymize(
                frame_bgr,
                tracks_for_anonymize,
                method=method,
            )

            if draw_tracks:
                output_frame = tracker.draw(
                    anonymized,
                    tracks,
                    confirmed_only=not blur_new,
                )
            else:
                output_frame = anonymized

            frame_idx += 1
            if progress_every > 0 and frame_idx % progress_every == 0:
                print(
                    f"Processed {frame_idx} frames "
                    f"| detect: {last_detect_ms:5.1f} ms "
                    f"| active tracks: {len(tracks)}"
                )

            yield output_frame

    def anonymize_video_without_model(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        method: AnonymizationMethod | str = AnonymizationMethod.BLUR,
        detect_interval: int = 1,
        target_fps: int | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
        blur_new: bool = False,
        draw_tracks: bool = False,
        codec: str = "H264",
        progress_every: int = 60,
    ) -> VideoAnonymizationResult:
        input_path = Path(input_path)
        output_path = self._resolve_output_path(
            input_path,
            Path(output_path) if output_path is not None else None,
        )
        detect_interval = max(int(detect_interval), 1)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        method_value = method
        if isinstance(method_value, str):
            method_value = AnonymizationMethod(method_value.strip().lower())

        source_meta = self.video_io.get_video_metadata(str(input_path))
        output_fps = self._resolve_output_fps(source_meta.fps, target_fps)
        tracker = self._build_fresh_tracker(self.face_tracker)

        print(f"Input: {input_path}")
        print(
            "Source metadata: "
            f"{source_meta.width}x{source_meta.height}, "
            f"{source_meta.fps:.3f} FPS, {source_meta.frame_count} frames"
        )
        print(f"Output: {output_path}")
        print(f"Anonymization method: {method_value.value}")
        print(f"Detect interval: {detect_interval}")
        print(f"Output FPS: {output_fps:.3f}")

        source_frames = self.video_io.iter_frames(
            str(input_path),
            start_sec=start_sec,
            end_sec=end_sec,
            target_fps=target_fps,
        )
        processed_frames = self._iter_processed_frames(
            frames=source_frames,
            tracker=tracker,
            method=method_value,
            detect_interval=detect_interval,
            blur_new=blur_new,
            draw_tracks=draw_tracks,
            progress_every=progress_every,
        )

        t0 = time.perf_counter()
        output_meta = self.video_io.write_frames(
            frames=processed_frames,
            output_path=str(output_path),
            fps=output_fps,
            codec=codec,
        )
        elapsed = time.perf_counter() - t0
        throughput_fps = output_meta.frame_count / elapsed if elapsed > 0 else 0.0

        print("Done.")
        print(
            "Output metadata: "
            f"{output_meta.width}x{output_meta.height}, "
            f"{output_meta.fps:.3f} FPS, "
            f"{output_meta.frame_count} frames, "
            f"{output_meta.duration_sec:.2f} sec"
        )
        print(
            f"Elapsed: {elapsed:.2f} sec "
            f"| Pipeline throughput: {throughput_fps:.2f} FPS"
        )

        return VideoAnonymizationResult(
            output_path=output_path,
            output_metadata=output_meta,
            elapsed_sec=elapsed,
            throughput_fps=throughput_fps,
        )

    def _iter_swapped_frames(
        self,
        frames: Iterator[np.ndarray],
        aligner: FaceAligner,
        progress_every: int,
        stabilizer: FaceSwapStabilizer | None = None,
    ) -> Iterator[np.ndarray]:
        # Face swap needs fresh 5-point landmarks every frame, so the detector runs
        # on each frame (the Kalman tracker only predicts bboxes, not landmarks).
        frame_idx = 0
        last_face_count = 0

        for frame_bgr in frames:
            t0 = time.perf_counter()

            if stabilizer is not None:
                # Stabilizer tracks faces across frames and smooths the landmarks.
                output_frame = stabilizer.process(frame_bgr)
                last_face_count = stabilizer.last_face_count
            else:
                detections = self.face_detector.detect(frame_bgr)
                last_face_count = len(detections)
                if detections:
                    aligned_faces = aligner.align(detections)
                    # VideoIO yields BGR; BlendSwap operates in RGB.
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    swapped_rgb = self.face_anonymizer.swap_face(frame_rgb, aligned_faces)
                    output_frame = cv2.cvtColor(swapped_rgb, cv2.COLOR_RGB2BGR)
                else:
                    output_frame = frame_bgr

            frame_ms = (time.perf_counter() - t0) * 1000.0
            frame_idx += 1
            if progress_every > 0 and frame_idx % progress_every == 0:
                print(
                    f"Processed {frame_idx} frames "
                    f"| frame: {frame_ms:6.1f} ms "
                    f"| faces: {last_face_count}"
                )

            yield output_frame

    def anonymize_video_with_model(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        target_fps: int | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
        codec: str = "H264",
        progress_every: int = 60,
        stabilize: bool = True,
        smooth_min_cutoff: float = 0.5,
        smooth_beta: float = 0.05,
        output_smooth: float = 0.4,
        mask_smooth: float = 0.5,
    ) -> VideoAnonymizationResult:
        if self.face_anonymizer.face_swapper is None:
            raise RuntimeError(
                "anonymize_video_with_model requires a FaceAnonymizer configured with "
                "a FaceSwapper (face_anonymizer=FaceAnonymizer(face_swapper=...))."
            )

        input_path = Path(input_path)
        if output_path is not None:
            output_path = Path(output_path)
        else:
            output_path = Path("outputs") / f"swapped_{input_path.stem}.mp4"

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        aligner = self._resolve_face_aligner()
        source_meta = self.video_io.get_video_metadata(str(input_path))
        output_fps = self._resolve_output_fps(source_meta.fps, target_fps)

        print(f"Input: {input_path}")
        print(
            "Source metadata: "
            f"{source_meta.width}x{source_meta.height}, "
            f"{source_meta.fps:.3f} FPS, {source_meta.frame_count} frames"
        )
        print(f"Output: {output_path}")
        print("Anonymization method: swap (BlendSwap)")
        print(f"Output FPS: {output_fps:.3f}")
        print(f"Temporal stabilization: {'on' if stabilize else 'off'}")

        # Prepare the source identity once up-front so any source issues surface early.
        self.face_anonymizer.face_swapper.prepare_source()

        stabilizer: FaceSwapStabilizer | None = None
        if stabilize:
            stabilizer = FaceSwapStabilizer(
                detector=self.face_detector,
                swapper=self.face_anonymizer.face_swapper,
                freq=output_fps,
                min_cutoff=smooth_min_cutoff,
                beta=smooth_beta,
                output_smooth=output_smooth,
                mask_smooth=mask_smooth,
            )

        source_frames = self.video_io.iter_frames(
            str(input_path),
            start_sec=start_sec,
            end_sec=end_sec,
            target_fps=target_fps,
        )
        processed_frames = self._iter_swapped_frames(
            frames=source_frames,
            aligner=aligner,
            progress_every=progress_every,
            stabilizer=stabilizer,
        )

        t0 = time.perf_counter()
        output_meta = self.video_io.write_frames(
            frames=processed_frames,
            output_path=str(output_path),
            fps=output_fps,
            codec=codec,
        )
        elapsed = time.perf_counter() - t0
        throughput_fps = output_meta.frame_count / elapsed if elapsed > 0 else 0.0

        print("Done.")
        print(
            "Output metadata: "
            f"{output_meta.width}x{output_meta.height}, "
            f"{output_meta.fps:.3f} FPS, "
            f"{output_meta.frame_count} frames, "
            f"{output_meta.duration_sec:.2f} sec"
        )
        print(
            f"Elapsed: {elapsed:.2f} sec "
            f"| Pipeline throughput: {throughput_fps:.2f} FPS"
        )

        return VideoAnonymizationResult(
            output_path=output_path,
            output_metadata=output_meta,
            elapsed_sec=elapsed,
            throughput_fps=throughput_fps,
        )
