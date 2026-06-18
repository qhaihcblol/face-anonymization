from __future__ import annotations

import numpy as np

from ai_core.face_anonymization.face_anonymizer import FaceAnonymizer


def test_landmark_ellipse_geometry_uses_eye_rotation_and_bbox_clamp() -> None:
    bbox = (100, 80, 220, 240)
    landmarks = np.asarray(
        [
            [130.0, 130.0],  # left eye
            [180.0, 140.0],  # right eye
            [153.0, 163.0],  # nose
            [132.0, 195.0],  # left mouth
            [177.0, 204.0],  # right mouth
        ],
        dtype=np.float32,
    )

    geometry = FaceAnonymizer._landmark_ellipse_geometry(bbox, landmarks)

    assert geometry is not None
    center, axes, angle = geometry
    assert 8.0 < angle < 14.0
    assert abs(center[0] - 160) <= 12
    assert abs(center[1] - 160) <= 16
    assert 48 <= axes[0] <= 60
    assert 74 <= axes[1] <= 80


def test_landmark_ellipse_geometry_rejects_outlier_landmarks() -> None:
    bbox = (100, 80, 220, 240)
    landmarks = np.asarray(
        [
            [630.0, 130.0],
            [680.0, 140.0],
            [653.0, 163.0],
            [632.0, 195.0],
            [677.0, 204.0],
        ],
        dtype=np.float32,
    )

    assert FaceAnonymizer._landmark_ellipse_geometry(bbox, landmarks) is None


def test_bbox_ellipse_fallback_fits_bbox_axes() -> None:
    center, axes, angle = FaceAnonymizer._bbox_ellipse_geometry((10, 20, 110, 220))

    assert center == (60, 120)
    assert axes == (50, 100)
    assert angle == 0.0
