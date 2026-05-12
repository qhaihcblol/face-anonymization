import cv2
import numpy as np
from pathlib import Path

from ai_core.face_alignment.face_aligner import FaceAligner
from ai_core.face_detection.face_detector import FaceDetector, FaceDetection


def main():
    detector = FaceDetector(
        onnx_path="ai_core/face_detection/onnx/retinaface_best.onnx"
    )

    aligner = FaceAligner()

    image_path = "test_images/test1.jpg"

    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections = detector.detect(image)

    print(f"Detected faces: {len(detections)}")

    aligned_results = aligner.align_and_warp_batch(image, detections)

    output_dir = Path("test_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # lưu ảnh gốc để visualize bbox
    vis_image = image.copy()

    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection.bbox)

        cv2.rectangle(
            vis_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    for idx, (aligned_face, warped_face) in enumerate(aligned_results):
        # warped_face đang là RGB
        warped_bgr = cv2.cvtColor(warped_face, cv2.COLOR_RGB2BGR)

        output_path = output_dir / f"aligned_face_{idx}.jpg"

        cv2.imwrite(str(output_path), warped_bgr)

        print(f"Saved: {output_path}")

    # save visualization
    vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    vis_path = output_dir / "detections.jpg"

    cv2.imwrite(str(vis_path), vis_bgr)

    print(f"Saved detection visualization: {vis_path}")


if __name__ == "__main__":
    main()
