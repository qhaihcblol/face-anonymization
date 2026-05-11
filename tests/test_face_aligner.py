from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignedFace, FaceAligner
from ai_core.face_detection.face_detector import FaceDetection, FaceDetector


IMAGE_PATH = Path("test_images/test1.jpg")
MODEL_PATH = Path("ai_core/face_detection/onnx/retinaface_best.onnx")
OUTPUT_FOLDER = Path("test_images/aligned_faces")
OUTPUT_PATH = OUTPUT_FOLDER / "test1_face_alignment_result.jpg"


def _read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def _aligned_face_crop(
    image: np.ndarray,
    aligned_face: AlignedFace,
    output_size: tuple[int, int],
) -> np.ndarray:
    return cv2.warpAffine(
        image,
        aligned_face.matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _draw_original_panel(
    image: np.ndarray,
    detections: list[FaceDetection],
) -> np.ndarray:
    panel = image.copy()
    for det_idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(
            panel,
            f"{det_idx}",
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        for px, py in det.landmarks.as_array():
            cv2.circle(
                panel,
                (int(round(float(px))), int(round(float(py)))),
                2,
                (0, 255, 0),
                -1,
            )

    return panel


def _build_result_canvas(
    original_panel: np.ndarray,
    aligned_crops: list[np.ndarray],
) -> np.ndarray:
    if not aligned_crops:
        return original_panel

    gap = 12
    label_height = 22
    tile_h, tile_w = aligned_crops[0].shape[:2]
    row_w = len(aligned_crops) * tile_w + (len(aligned_crops) - 1) * gap

    canvas_w = max(original_panel.shape[1], row_w)
    canvas_h = original_panel.shape[0] + gap + tile_h + label_height
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    original_x = (canvas_w - original_panel.shape[1]) // 2
    canvas[: original_panel.shape[0], original_x : original_x + original_panel.shape[1]] = (
        original_panel
    )

    row_x = (canvas_w - row_w) // 2
    row_y = original_panel.shape[0] + gap
    for idx, crop in enumerate(aligned_crops, start=1):
        x = row_x + (idx - 1) * (tile_w + gap)
        canvas[row_y : row_y + tile_h, x : x + tile_w] = crop
        cv2.rectangle(
            canvas,
            (x, row_y),
            (x + tile_w - 1, row_y + tile_h - 1),
            (40, 40, 40),
            1,
        )
        cv2.putText(
            canvas,
            f"#{idx}",
            (x + 4, row_y + tile_h + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )

    return canvas


def run_face_alignment_demo(
    image_path: Path = IMAGE_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    detector = FaceDetector(onnx_path=MODEL_PATH)
    aligner = FaceAligner()

    image = _read_image(image_path)
    detections = detector.detect(image)
    if not detections:
        raise AssertionError(f"No faces detected in image: {image_path}")

    aligned_faces = aligner.align(detections)
    aligned_crops = [
        _aligned_face_crop(image, aligned_face, aligner.output_size)
        for aligned_face in aligned_faces
    ]

    original_panel = _draw_original_panel(image, detections)
    result = _build_result_canvas(original_panel, aligned_crops)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), result)
    if not ok:
        raise RuntimeError(f"Cannot save result image: {output_path}")

    return output_path


def test_face_aligner_saves_visual_result() -> None:
    output_path = run_face_alignment_demo()

    assert output_path.exists()
    saved = cv2.imread(str(output_path))
    assert saved is not None
    assert saved.size > 0


def main() -> None:
    output_path = run_face_alignment_demo()
    print(f"Saved face alignment result to: {output_path}")


if __name__ == "__main__":
    main()
