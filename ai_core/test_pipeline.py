import cv2
import os
import time

from .video_io.video_io import VideoIO
from .face_detection.face_detector import FaceDetector

# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_PATH = "test_videos/test1.mp4"
OUTPUT_DIR = "outputs/frames"
ONNX_PATH = "ai_core/face_detection/onnx/retinaface_best.onnx"
TARGET_FPS = 1
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_io = VideoIO()
face_detector = FaceDetector(onnx_path=ONNX_PATH)

saved = 0
detected_total = 0
frame_count = 0

total_detect_time = 0.0

# ⏱️ Tổng thời gian pipeline
pipeline_start = time.perf_counter()

for frame_idx, frame_bgr in enumerate(
    video_io.iter_frames(video_path=VIDEO_PATH, target_fps=TARGET_FPS)
):
    frame_count += 1

    # ⏱️ đo thời gian detect
    start = time.perf_counter()
    faces = face_detector.detect(frame_bgr)
    detect_time = time.perf_counter() - start

    total_detect_time += detect_time

    print(f"[Frame {frame_idx}] detect_time = {detect_time:.4f}s, faces = {len(faces)}")

    frame_result = face_detector.draw(frame_bgr, faces)
    detected_total += len(faces)

    if len(faces) > 0:
        output_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, frame_result)
        saved += 1

# ⏱️ tổng kết
pipeline_time = time.perf_counter() - pipeline_start

avg_detect_time = total_detect_time / frame_count if frame_count > 0 else 0
fps = frame_count / pipeline_time if pipeline_time > 0 else 0

print("\n===== SUMMARY =====")
print(f"Total frames: {frame_count}")
print(f"Total faces detected: {detected_total}")
print(f"Saved frames: {saved}")
print(f"Total pipeline time: {pipeline_time:.2f}s")
print(f"Avg detect time/frame: {avg_detect_time:.4f}s")
print(f"Effective FPS: {fps:.2f}")
