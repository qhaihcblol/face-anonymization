import sys
from pathlib import Path
try:
    from ai_core.video_io.video_io import VideoIO
    from ai_core.face_detection.face_detector import FaceDetector
    from ai_core.face_tracking.face_tracker import ByteTracker
    from ai_core.face_anonymization.face_anonymizer import FaceAnonymizer, AnonymizationMethod
    from ai_core.video_anonymization import VideoAnonymization
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

try:
    video_io = VideoIO()
    detector = FaceDetector(onnx_path=Path('ai_core/face_detection/onnx/retinaface_best.onnx'))
    tracker = ByteTracker()
    anonymizer = FaceAnonymizer(blur_strength=31)
    pipeline = VideoAnonymization(video_io, detector, tracker, anonymizer)
    result = pipeline.anonymize_video_without_model(
        input_path=Path('test_videos/test1.mp4'),
        output_path=Path('outputs/smoke_video_anonymization_class.mp4'),
        method=AnonymizationMethod.BLUR,
        detect_interval=1,
        progress_every=200,
        draw_tracks=True,
        
    )
    print('SMOKE_OK', result.output_path, result.output_metadata.frame_count)
except Exception as e:
    print(f"Error during execution: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
