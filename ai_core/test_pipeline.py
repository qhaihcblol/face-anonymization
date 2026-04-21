import cv2
import os
from .video_io.video_io import VideoIO
from .face_detection.face_detector import FaceDetector

# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_PATH = "test_videos/test1.mp4"
OUTPUT_DIR = "test_output/frames"
ONNX_PATH = "ai_core/face_detection/onnx/retinaface_best.onnx"
TARGET_FPS = 1  # 1 frame/giây để test nhanh
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_io = VideoIO()
face_detector = FaceDetector(onnx_path=ONNX_PATH)

saved = 0
detected_total = 0

for frame_idx, frame_bgr in enumerate(
    video_io.iter_frames(video_path=VIDEO_PATH, target_fps=TARGET_FPS)
):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = face_detector.detect(frame_rgb)  # list[BoundingBox]

    detected_total += len(detections)

    # ── Vẽ bbox lên frame ───────────────────────────────────────────────────
    vis = frame_bgr.copy()
    for det in detections:
        cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)

        if DRAW_CONF:
            label = f"{det.confidence:.2f}"
            cv2.putText(
                vis,
                label,
                (det.x1, det.y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    # ── Lưu frame ───────────────────────────────────────────────────────────
    out_path = os.path.join(
        OUTPUT_DIR, f"frame_{frame_idx:04d}_faces{len(detections)}.jpg"
    )
    cv2.imwrite(out_path, vis)
    saved += 1

    print(f"[frame {frame_idx:04d}] detected: {len(detections)} face(s) → {out_path}")

print(
    f"\nDone. {saved} frames saved to '{OUTPUT_DIR}', total faces detected: {detected_total}"
)
