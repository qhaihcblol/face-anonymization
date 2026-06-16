"""ArcFace face-recognition wrapper (InsightFace buffalo_l / w600k_r50).

Used for *evaluation only* — it is not part of the anonymization pipeline. It
turns a face into a 512-d unit embedding so two faces can be compared by cosine
similarity, which is how we measure whether an anonymization method has actually
destroyed identity:

    cosine(original_face, anonymized_face)  -> high = identity still leaks,
                                               low  = identity removed.

For buffalo_l the usual same-identity decision threshold is ~0.28 cosine, so an
anonymized crop scoring well below that is considered de-identified.

The recognition ONNX (``w600k_r50.onnx``) ships inside InsightFace's ``buffalo_l``
pack; it is downloaded on first use to ``~/.insightface/models/buffalo_l/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["ArcFaceRecognizer"]

_DEFAULT_MODEL = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
# buffalo_l w600k_r50 same-identity decision threshold (cosine).
RECOGNITION_THRESHOLD: float = 0.28


class ArcFaceRecognizer:
    """Aligned-face -> unit embedding, with a cosine helper.

    Args:
        model_path: Path to ``w600k_r50.onnx``. If missing, the ``buffalo_l`` pack
            is fetched via InsightFace (needs network on first run).
        providers: ONNX Runtime providers. Defaults to CUDA -> CPU.
        ctx_id: InsightFace device id (>=0 = GPU, -1 = CPU). Auto from providers.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        providers: Sequence[str] | None = None,
        ctx_id: int | None = None,
    ) -> None:
        from insightface.model_zoo import get_model
        from insightface.utils import face_align

        self._face_align = face_align

        path = Path(model_path) if model_path is not None else _DEFAULT_MODEL
        if not path.is_file():
            path = Path(self._ensure_buffalo_l())

        import onnxruntime as ort
        available = set(ort.get_available_providers())
        requested = list(providers) if providers is not None else [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        resolved = [p for p in requested if p in available] or ["CPUExecutionProvider"]

        if ctx_id is None:
            ctx_id = 0 if "CUDAExecutionProvider" in resolved else -1

        self.model = get_model(str(path), providers=resolved)
        self.model.prepare(ctx_id=ctx_id)
        self.model_path = path

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #
    def embed_crop(self, face_bgr: np.ndarray) -> np.ndarray:
        """Unit embedding of a pre-aligned face crop (BGR; resized to 112 inside)."""
        feat = np.asarray(self.model.get_feat(face_bgr), dtype=np.float32).flatten()
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm > 1e-8 else feat

    def embed(self, image_bgr: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
        """Align ``image_bgr`` by 5-point landmarks (ArcFace template), then embed.

        Aligning by the *original* landmarks lets the same geometry be reused on an
        anonymized version of the frame, so the cosine reflects only the appearance
        change, not a re-detection shift.
        """
        kps = np.asarray(landmarks5, dtype=np.float32).reshape(5, 2)
        crop = self._face_align.norm_crop(image_bgr, kps, image_size=112)
        return self.embed_crop(crop)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings (unit or not)."""
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _ensure_buffalo_l() -> str:
        """Download the buffalo_l pack and return the recognition model path."""
        from insightface.utils import storage

        storage.ensure_available("models", "buffalo_l", root="~/.insightface")
        path = _DEFAULT_MODEL
        if not path.is_file():
            raise FileNotFoundError(
                f"buffalo_l downloaded but recognition model not found at {path}"
            )
        return str(path)
