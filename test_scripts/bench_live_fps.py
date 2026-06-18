"""Ad-hoc benchmark of the live anonymization pipeline (per-stage timings).

Mirrors what the WebSocket server does per frame so we can see exactly where the
real-time budget goes on THIS machine (GPU/EP, model sizes, frame size).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_parsing.face_parser import FaceParser
from ai_core.face_alignment.face_aligner import FaceAligner
from ai_core.face_anonymization.face_anonymizer import (
    FaceAnonymizer,
    AnonymizationMethod,
    ObfuscationParams,
)
from ai_core.face_tracking.face_tracker import ByteTracker
from ai_core.live_anonymization import LiveFaceAnonymizer, LiveVisualConfig

SEND_MAX_WIDTH = 640
JPEG_Q_SEND = 70      # client -> server
JPEG_Q_BACK = 80      # server -> client
ITERS = 60


def now() -> float:
    return time.perf_counter()


def grab_frame() -> np.ndarray:
    cap = cv2.VideoCapture(str(ROOT / "test_videos" / "hai1.mp4"))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read test frame")
    # Mimic the browser: scale to <=640 wide.
    h, w = frame.shape[:2]
    scale = min(1.0, SEND_MAX_WIDTH / w)
    return cv2.resize(frame, (round(w * scale), round(h * scale)))


def bench(label: str, fn, iters: int = ITERS) -> float:
    fn()  # warmup
    fn()
    ts = []
    for _ in range(iters):
        t = now()
        fn()
        ts.append((now() - t) * 1000.0)
    ts.sort()
    med = ts[len(ts) // 2]
    print(f"  {label:<34} median {med:7.2f} ms   (min {ts[0]:6.2f} / max {ts[-1]:6.2f})")
    return med


def main() -> None:
    frame = grab_frame()
    print(f"Frame size sent by browser: {frame.shape[1]}x{frame.shape[0]}\n")

    detector = FaceDetector(onnx_path=str(ROOT / "ai_core/face_detection/onnx/retinaface_best.onnx"))
    parser = FaceParser()
    aligner = FaceAligner(output_size=(256, 256), mode="ffhq")
    anonymizer = FaceAnonymizer(face_swapper=None, face_parser=parser, face_aligner=aligner)

    print("Execution providers actually in use:")
    print("  detector:", detector.session.get_providers())
    print("  parser  :", parser.session.get_providers())
    print(f"  detector input size: {detector.image_size}, parser model size: {parser.model_size}\n")

    # JPEG encode of the frame (so we can time decode of a realistic payload).
    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q_SEND])
    payload = enc.tobytes()
    print(f"JPEG payload (q{JPEG_Q_SEND}): {len(payload)/1024:.1f} KB\n")

    print("=== Per-stage (one frame with faces present) ===")
    bench("JPEG decode (cv2.imdecode)",
          lambda: cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR))

    dets = detector.detect(frame)
    n_faces = len(dets)
    print(f"  -> detector found {n_faces} face(s)")
    bench("RetinaFace detect (GPU)", lambda: detector.detect(frame))

    tracker = ByteTracker()
    tracks = tracker.update(dets)
    bench("Tracker update", lambda: ByteTracker().update(dets))

    # Anonymize (blur) — this is what runs EVERY frame, incl. predict-only frames.
    bench("Anonymize BLUR (mask=BiSeNet/face + composite)",
          lambda: anonymizer.anonymize(frame, tracks, method=AnonymizationMethod.BLUR,
                                       params=ObfuscationParams(blur_strength=31)))

    bench("Anonymize PIXELATE",
          lambda: anonymizer.anonymize(frame, tracks, method=AnonymizationMethod.PIXELATE,
                                       params=ObfuscationParams(pixelation_level=16)))

    out = anonymizer.anonymize(frame, tracks, method=AnonymizationMethod.BLUR)
    bench("JPEG encode back (cv2.imencode q80)",
          lambda: cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q_BACK]))

    print("\n=== Full process_frame via LiveFaceAnonymizer (detect_interval=2) ===")
    live = LiveFaceAnonymizer(
        face_detector=detector, face_tracker=ByteTracker(), face_anonymizer=anonymizer,
        config=LiveVisualConfig(method=AnonymizationMethod.BLUR, detect_interval=2,
                                obfuscation=ObfuscationParams(blur_strength=31)),
    )
    # warm
    for _ in range(4):
        live.process_frame(frame)

    detect_ms, predict_ms = [], []
    for _ in range(ITERS):
        r = live.process_frame(frame)
        (detect_ms if r.detected else predict_ms).append(r.process_ms)
    detect_ms.sort(); predict_ms.sort()
    dm = detect_ms[len(detect_ms)//2]
    pm = predict_ms[len(predict_ms)//2] if predict_ms else float("nan")
    print(f"  detect frame  median {dm:7.2f} ms")
    print(f"  predict frame median {pm:7.2f} ms")
    avg = (dm + pm) / 2.0
    print(f"  avg/frame ~ {avg:.2f} ms  ->  server-only ceiling ~ {1000.0/avg:.1f} FPS")

    # Full server round of work incl. decode + encode (no network).
    def full_cycle():
        f = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        r = live.process_frame(f)
        cv2.imencode(".jpg", r.frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q_BACK])
    print()
    fc = bench("FULL decode+process+encode / frame", full_cycle)
    print(f"\n  => server compute ceiling ~ {1000.0/fc:.1f} FPS (single frame, no network, no concurrency cap)")


if __name__ == "__main__":
    main()
