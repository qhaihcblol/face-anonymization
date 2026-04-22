from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

try:
    from ai_core.face_detection.face_detector import FaceDetector
    from ai_core.face_tracking.face_tracker import ByteTracker
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from ai_core.face_detection.face_detector import FaceDetector
    from ai_core.face_tracking.face_tracker import ByteTracker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Webcam realtime FaceDetector + ByteTracker pipeline",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).resolve().parent
        / "face_detection"
        / "onnx"
        / "retinaface_best.onnx",
        help="Path to RetinaFace ONNX model",
    )
    parser.add_argument("--high-thresh", type=float, default=0.6)
    parser.add_argument("--low-thresh", type=float, default=0.1)
    parser.add_argument("--max-lost", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=1,
        help="Run detector every N frames (1 = detect every frame)",
    )
    parser.add_argument("--show-new", action="store_true", help="Draw New tracks too")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    detector = FaceDetector(onnx_path=args.onnx)
    tracker = ByteTracker(
        high_thresh=args.high_thresh,
        low_thresh=args.low_thresh,
        max_lost=args.max_lost,
        min_hits=args.min_hits,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    detect_interval = max(int(args.detect_interval), 1)
    print("Press 'q' or ESC to quit.")
    if detect_interval > 1:
        print(
            f"Detector will run every {detect_interval} frames. "
            f"Tip: keep --max-lost >= {detect_interval} for better track continuity."
        )

    frame_idx = 0
    last_detect_ms = 0.0

    # FPS: count actual rendered frames over a fixed 0.5 s window
    fps_display = 0.0
    fps_win_start = time.perf_counter()
    fps_win_count = 0

    tracks: list[dict] = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("Failed to read frame from webcam. Stopping.")
            break

        # ── Detect + track ────────────────────────────────────────────────
        run_detect = (frame_idx % detect_interval) == 0

        if run_detect:
            t0 = time.perf_counter()
            detections = detector.detect(frame_bgr)
            last_detect_ms = (time.perf_counter() - t0) * 1000.0
            tracks = tracker.update(detections)
        else:
            detections = []
            tracks = tracker.predict_only()

        # ── FPS: count frames inside a 0.5 s window ───────────────────────
        fps_win_count += 1
        elapsed = time.perf_counter() - fps_win_start
        if elapsed >= 0.5:
            fps_display = fps_win_count / elapsed
            fps_win_start = time.perf_counter()
            fps_win_count = 0

        # ── Draw ──────────────────────────────────────────────────────────
        vis = tracker.draw(frame_bgr, tracks, confirmed_only=not args.show_new)

        detect_tag = "RUN" if run_detect else "SKIP"
        cv2.putText(
            vis,
            (
                f"FPS: {fps_display:5.1f}"
                f" | detect[{detect_tag}] every {detect_interval}f"
                f" | last detect: {last_detect_ms:5.1f} ms"
                f" | det: {len(detections)}"
                f" | trk: {len(tracks)}"
            ),
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("FaceDetector + ByteTracker", vis)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
