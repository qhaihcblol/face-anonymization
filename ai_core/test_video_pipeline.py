from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

try:
    from ai_core.face_anonymization.face_anonymizer import (
        AnonymizationMethod,
        FaceAnonymizer,
    )
    from ai_core.face_detection.face_detector import FaceDetector
    from ai_core.face_tracking.face_tracker import ByteTracker
    from ai_core.video_io.video_io import VideoIO
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from ai_core.face_anonymization.face_anonymizer import (
        AnonymizationMethod,
        FaceAnonymizer,
    )
    from ai_core.face_detection.face_detector import FaceDetector
    from ai_core.face_tracking.face_tracker import ByteTracker
    from ai_core.video_io.video_io import VideoIO


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline pipeline: VideoIO + FaceDetector + ByteTracker + FaceAnonymizer"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: outputs/anonymized_<input_stem>.mp4)",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).resolve().parent
        / "face_detection"
        / "onnx"
        / "retinaface_best.onnx",
        help="Path to RetinaFace ONNX model",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=[method.value for method in AnonymizationMethod],
        default=AnonymizationMethod.BLUR.value,
        help="Anonymization method",
    )
    parser.add_argument("--blur-strength", type=int, default=31)
    parser.add_argument("--pixelation-level", type=int, default=16)
    parser.add_argument(
        "--mask-color",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=(160, 160, 160),
        help="Mask color in BGR",
    )
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=1,
        help="Run detector every N frames (1 = detect every frame)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help="Optional sampling FPS for processing/output",
    )
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--high-thresh", type=float, default=0.6)
    parser.add_argument("--low-thresh", type=float, default=0.1)
    parser.add_argument("--max-lost", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument(
        "--blur-new",
        action="store_true",
        help="Also anonymize New tracks (default: only confirmed Tracked)",
    )
    parser.add_argument(
        "--draw-tracks",
        action="store_true",
        help="Overlay tracker boxes/ids on output video",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Output video FourCC codec (4 chars)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=60,
        help="Print progress every N frames",
    )
    return parser


def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return Path("outputs") / f"anonymized_{input_path.stem}.mp4"


def _resolve_output_fps(source_fps: float, target_fps: int | None) -> float:
    if target_fps is None:
        return float(source_fps)
    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")
    if target_fps >= source_fps:
        return float(source_fps)
    return float(target_fps)


def _iter_processed_frames(
    frames: Iterator[np.ndarray],
    detector: FaceDetector,
    tracker: ByteTracker,
    anonymizer: FaceAnonymizer,
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
            detections = detector.detect(frame_bgr)
            last_detect_ms = (time.perf_counter() - t0) * 1000.0
            tracks = tracker.update(detections)
        else:
            tracks = tracker.predict_only()

        if blur_new:
            tracks_for_anonymize = tracks
        else:
            tracks_for_anonymize = [t for t in tracks if t.get("state") == "Tracked"]

        anonymized = anonymizer.anonymize_without_model(
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


def main() -> None:
    args = _build_parser().parse_args()

    input_path = args.input
    output_path = _resolve_output_path(input_path, args.output)
    detect_interval = max(int(args.detect_interval), 1)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    method = AnonymizationMethod(args.method)
    video_io = VideoIO()
    meta = video_io.get_video_metadata(str(input_path))
    output_fps = _resolve_output_fps(meta.fps, args.target_fps)

    detector = FaceDetector(onnx_path=args.onnx)
    tracker = ByteTracker(
        high_thresh=args.high_thresh,
        low_thresh=args.low_thresh,
        max_lost=args.max_lost,
        min_hits=args.min_hits,
    )
    anonymizer = FaceAnonymizer(
        blur_strength=args.blur_strength,
        pixelation_level=args.pixelation_level,
        mask_color=tuple(args.mask_color),
    )

    print(f"Input: {input_path}")
    print(
        "Source metadata: "
        f"{meta.width}x{meta.height}, {meta.fps:.3f} FPS, {meta.frame_count} frames"
    )
    print(f"Output: {output_path}")
    print(f"Anonymization method: {method.value}")
    print(f"Detect interval: {detect_interval}")
    print(f"Output FPS: {output_fps:.3f}")

    source_frames = video_io.iter_frames(
        str(input_path),
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        target_fps=args.target_fps,
    )
    processed_frames = _iter_processed_frames(
        frames=source_frames,
        detector=detector,
        tracker=tracker,
        anonymizer=anonymizer,
        method=method,
        detect_interval=detect_interval,
        blur_new=args.blur_new,
        draw_tracks=args.draw_tracks,
        progress_every=args.progress_every,
    )

    t0 = time.perf_counter()
    output_meta = video_io.write_frames(
        frames=processed_frames,
        output_path=str(output_path),
        fps=output_fps,
        codec=args.codec,
    )
    elapsed = time.perf_counter() - t0
    avg_fps = output_meta.frame_count / elapsed if elapsed > 0 else 0.0

    print("Done.")
    print(
        "Output metadata: "
        f"{output_meta.width}x{output_meta.height}, "
        f"{output_meta.fps:.3f} FPS, "
        f"{output_meta.frame_count} frames, "
        f"{output_meta.duration_sec:.2f} sec"
    )
    print(f"Elapsed: {elapsed:.2f} sec | Pipeline throughput: {avg_fps:.2f} FPS")


if __name__ == "__main__":
    main()
