import cv2
from pathlib import Path

from ai_core.face_alignment.face_aligner import FaceAligner
from ai_core.face_detection.face_detector import FaceDetector


def run_aligner(image, detections, aligner, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = aligner.align_and_warp_batch(image, detections)
    for idx, (_, warped_face) in enumerate(results):
        out_path = output_dir / f"face_{idx}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(warped_face, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_path}")


def main():
    detector = FaceDetector(onnx_path="ai_core/face_detection/onnx/retinaface_best.onnx")

    image_path = "test_images/test5.jpeg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections = detector.detect(image)
    print(f"Detected faces: {len(detections)}")

    run_aligner(image, detections, FaceAligner(output_size=(112, 112), mode="insightface"), Path("test_images/isf"))
    run_aligner(image, detections, FaceAligner(output_size=(256, 256), mode="ffhq"), Path("test_images/ffhq"))


if __name__ == "__main__":
    main()
