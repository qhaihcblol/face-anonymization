"""Evaluation of the no-model (no-swap) anonymization methods.

Compares blur / pixelate / mask / blackout (and the plain vs. "hardened"
irreversible variants) on two axes:

* **Identity removal (ArcFace).** The headline privacy metric. We embed the
  *original* aligned face and the *anonymized* face with InsightFace ArcFace
  (buffalo_l / w600k_r50) and report cosine similarity. Same-identity threshold
  for this model is ~0.28: a method that leaves the cosine above it has NOT
  de-identified the face. (Aligning both crops by the original landmarks isolates
  the appearance change.)
* **Re-detectability.** Re-run RetinaFace on the anonymized frame and report the
  confidence it still assigns to that face (1.0 = fully detectable).

Figures (under ``outputs/``):
  * ``nonswap_methods.png``   — A. faces x {original, blur, pixelate, mask, blackout}.
  * ``nonswap_hardening.png`` — B. faces x {original, blur plain/hardened, pixelate plain/hardened}.
  * ``nonswap_metrics.png``   — C. ArcFace cosine + re-detection per method (matplotlib).
  * ``nonswap_hardening_metrics.png`` — D. plain vs hardened cosine (matplotlib).

Usage:
    python -m test_scripts.test_anonymize_nonswap_eval
    python -m test_scripts.test_anonymize_nonswap_eval --inputs test_images/male_1.jpeg --max-faces 3
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ai_core.face_anonymization.face_anonymizer import (
    AnonymizationMethod,
    FaceAnonymizer,
    ObfuscationParams,
)
from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_parsing.face_parser import FaceParser
from ai_core.face_recognition.arcface import ArcFaceRecognizer, RECOGNITION_THRESHOLD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = [
    PROJECT_ROOT / "test_images" / "male_1.jpeg",
    PROJECT_ROOT / "test_images" / "female_1.jpeg",
    PROJECT_ROOT / "test_images" / "male_2.jpeg",
    PROJECT_ROOT / "test_images" / "female_2.jpeg",
]

HARD = ObfuscationParams(irreversible=True)
PLAIN = ObfuscationParams(irreversible=False)

# (column label, method, params) for the method-comparison figure (A) + metrics.
METHODS = [
    ("blur", AnonymizationMethod.BLUR, HARD),
    ("pixelate", AnonymizationMethod.PIXELATE, HARD),
    ("mask", AnonymizationMethod.MASK, HARD),
    ("blackout", AnonymizationMethod.BLACKOUT, HARD),
]
# (column label, method, params) for the hardening figure (B) + metrics.
HARDENING = [
    ("blur (plain)", AnonymizationMethod.BLUR, PLAIN),
    ("blur (hardened)", AnonymizationMethod.BLUR, HARD),
    ("pixelate (plain)", AnonymizationMethod.PIXELATE, PLAIN),
    ("pixelate (hardened)", AnonymizationMethod.PIXELATE, HARD),
]


@dataclass
class Face:
    tag: str
    image: np.ndarray
    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    base_emb: np.ndarray


def _anonymize_one(anon: FaceAnonymizer, img, bbox, lms, method, params) -> np.ndarray:
    det = {"bbox": list(bbox), "landmarks": lms.tolist()}
    return anon.anonymize(img, [det], method=method, params=params)


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    ua = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    ub = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    u = ua + ub - inter
    return inter / u if u > 1e-9 else 0.0


def _redetect_score(detector: FaceDetector, anon_img, bbox) -> float:
    """Max RetinaFace confidence on the anonymized frame at the face location."""
    best = 0.0
    for d in detector.detect(anon_img):
        if _iou(d.bbox, bbox) > 0.2:
            best = max(best, float(d.score))
    return best


# --------------------------------------------------------------------------- #
# Layout helpers (compact, A4-friendly).                                       #
# --------------------------------------------------------------------------- #
def _crop_rect(bbox, shape, pad=0.4):
    x1, y1, x2, y2 = bbox
    h, w = shape
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    return (int(max(x1 - pw, 0)), int(max(y1 - ph, 0)),
            int(min(x2 + pw, w)), int(min(y2 + ph, h)))


def _cell(img, rect, size):
    x1, y1, x2, y2 = rect
    crop = img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    return cv2.resize(crop, (size, int(h * size / max(w, 1))), interpolation=cv2.INTER_AREA)


def _pad_to_h(img, h):
    if img.shape[0] == h:
        return img
    return np.vstack([img, np.full((h - img.shape[0], img.shape[1], 3), 255, np.uint8)])


def _hstack_sep(cells, wpx=3):
    h = max(c.shape[0] for c in cells)
    cells = [_pad_to_h(c, h) for c in cells]
    sep = np.full((h, wpx, 3), 255, np.uint8)
    out = []
    for c in cells:
        out += [c, sep]
    return np.hstack(out[:-1])


def _label_left(row, tag, w=120):
    panel = np.full((row.shape[0], w, 3), 250, np.uint8)
    cv2.putText(panel, tag, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return np.hstack([panel, row])


def _header(cols, cell, total_w, label_w=120):
    bar = np.full((30, total_w, 3), 230, np.uint8)
    x = label_w + 4
    for name in cols:
        cv2.putText(bar, name, (x + 4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        x += cell + 3
    return bar


def _build_grid(faces, anon, columns, cell):
    """Rows = faces, columns = [original] + each (label, method, params)."""
    rows = []
    for f in faces:
        rect = _crop_rect(f.bbox, f.image.shape[:2])
        cells = [_cell(f.image, rect, cell)]
        for _label, method, params in columns:
            out = _anonymize_one(anon, f.image, f.bbox, f.landmarks, method, params)
            cells.append(_cell(out, rect, cell))
        rows.append(_label_left(_hstack_sep(cells), f.tag))
    width = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == width else np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    head = _header(["original"] + [c[0] for c in columns], cell, width)
    sep = np.full((4, width, 3), 200, np.uint8)
    stacked = [head]
    for r in rows:
        stacked += [r, sep]
    return np.vstack(stacked[:-1])


# --------------------------------------------------------------------------- #
# Metrics figures (matplotlib).                                                #
# --------------------------------------------------------------------------- #
def _fig_methods_metrics(cos_by_method, det_by_method, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [m[0] for m in METHODS]
    cos = [float(np.mean(cos_by_method[n])) for n in names]
    det = [float(np.mean(det_by_method[n])) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    b1 = ax1.bar(names, cos, color=["#e76f51", "#f4a261", "#2a9d8f", "#264653"])
    ax1.bar_label(b1, fmt="%.2f", padding=3)
    ax1.axhline(RECOGNITION_THRESHOLD, color="red", ls="--", lw=1.2,
                label=f"recognition threshold {RECOGNITION_THRESHOLD}")
    ax1.set_ylabel("ArcFace cosine vs original")
    ax1.set_title("Identity leakage (lower = better)")
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    b2 = ax2.bar(names, det, color=["#e76f51", "#f4a261", "#2a9d8f", "#264653"])
    ax2.bar_label(b2, fmt="%.2f", padding=3)
    ax2.set_ylabel("RetinaFace confidence after anonymize")
    ax2.set_title("Re-detectability (lower = better)")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("No-swap anonymization methods", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_hardening_metrics(cos_by_variant, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ["blur", "pixelate"]
    plain = [float(np.mean(cos_by_variant[f"{g} (plain)"])) for g in groups]
    hard = [float(np.mean(cos_by_variant[f"{g} (hardened)"])) for g in groups]
    x = np.arange(len(groups))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    b1 = ax.bar(x - w / 2, plain, w, label="plain", color="#f4a261")
    b2 = ax.bar(x + w / 2, hard, w, label="hardened (irreversible)", color="#e76f51")
    ax.bar_label(b1, fmt="%.2f", padding=3)
    ax.bar_label(b2, fmt="%.2f", padding=3)
    ax.axhline(RECOGNITION_THRESHOLD, color="red", ls="--", lw=1.2,
               label=f"threshold {RECOGNITION_THRESHOLD}")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("ArcFace cosine vs original")
    ax.set_title("Hardening effect on residual identity (lower = better)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate no-swap anonymization methods.")
    p.add_argument("--inputs", type=Path, nargs="*", default=None)
    p.add_argument("--onnx", type=Path,
                   default=PROJECT_ROOT / "ai_core" / "face_detection" / "onnx" / "retinaface_best.onnx")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--max-faces", type=int, default=4, help="A4-friendly: 3-4 faces.")
    p.add_argument("--cell", type=int, default=200)
    p.add_argument("--no-parser", action="store_true",
                   help="Use the coarse ellipse mask instead of the BiSeNet mask.")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    paths = args.inputs if args.inputs else DEFAULT_IMAGES
    detector = FaceDetector(onnx_path=args.onnx, conf_threshold=args.conf)
    arc = ArcFaceRecognizer()
    if args.no_parser:
        anon = FaceAnonymizer()
    else:
        anon = FaceAnonymizer(face_parser=FaceParser(),
                              face_aligner=FaceAligner(output_size=(512, 512), mode=AlignMode.FFHQ))

    # Collect the largest face from each image (one per image keeps tags clean).
    faces: list[Face] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: cannot read {p}", file=sys.stderr)
            continue
        dets = detector.detect(img)
        if not dets:
            continue
        d = max(dets, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        lms = d.landmarks.as_array()
        faces.append(Face(Path(p).stem, img, d.bbox, lms, arc.embed(img, lms)))
        if len(faces) >= args.max_faces:
            break
    if not faces:
        print("No faces detected.", file=sys.stderr)
        return 1
    print(f"Faces: {[f.tag for f in faces]} (mask: {'ellipse' if args.no_parser else 'BiSeNet'})")

    # Qualitative figures.
    cv2.imwrite(str(args.outdir / "nonswap_methods.png"), _build_grid(faces, anon, METHODS, args.cell))
    cv2.imwrite(str(args.outdir / "nonswap_hardening.png"), _build_grid(faces, anon, HARDENING, args.cell))

    # Metrics.
    cos_method = {m[0]: [] for m in METHODS}
    det_method = {m[0]: [] for m in METHODS}
    cos_hard = {h[0]: [] for h in HARDENING}
    rows = []
    for f in faces:
        row = {"tag": f.tag}
        for label, method, params in METHODS:
            out = _anonymize_one(anon, f.image, f.bbox, f.landmarks, method, params)
            cos = arc.cosine(f.base_emb, arc.embed(out, f.landmarks))
            ds = _redetect_score(detector, out, f.bbox)
            cos_method[label].append(cos); det_method[label].append(ds)
            row[label] = (cos, ds)
        for label, method, params in HARDENING:
            out = _anonymize_one(anon, f.image, f.bbox, f.landmarks, method, params)
            cos_hard[label].append(arc.cosine(f.base_emb, arc.embed(out, f.landmarks)))
        rows.append(row)

    _fig_methods_metrics(cos_method, det_method, args.outdir / "nonswap_metrics.png")
    _fig_hardening_metrics(cos_hard, args.outdir / "nonswap_hardening_metrics.png")

    # Console report.
    print("\nArcFace cosine vs original  (lower = better; >%.2f = still recognizable):" % RECOGNITION_THRESHOLD)
    print(f"{'face':12s} " + " ".join(f"{m[0]:>9s}" for m in METHODS))
    for r in rows:
        print(f"{r['tag']:12s} " + " ".join(f"{r[m[0]][0]:9.3f}" for m in METHODS))
    print(f"{'MEAN':12s} " + " ".join(f"{np.mean(cos_method[m[0]]):9.3f}" for m in METHODS))

    print("\nRetinaFace confidence after anonymize (lower = better):")
    print(f"{'MEAN':12s} " + " ".join(f"{np.mean(det_method[m[0]]):9.3f}" for m in METHODS))

    print("\nHardening (cosine, lower = better):")
    for h in HARDENING:
        print(f"  {h[0]:22s} {np.mean(cos_hard[h[0]]):.3f}")

    print(f"\nSaved -> nonswap_methods.png, nonswap_hardening.png, "
          f"nonswap_metrics.png, nonswap_hardening_metrics.png  (in {args.outdir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
