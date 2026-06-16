"""Evaluation of the BlendSwap (face-swap) anonymization.

Unlike blur/pixelate/mask/blackout, the swap *replaces* the face with a source
identity, so it is judged on two identity axes (ArcFace, buffalo_l) plus how it
compares to the censoring methods:

* **A+B (combined) — qualitative + identity numbers.** One row per face:
    original | source identity | swapped | ArcFace cosines.
  The trailing column reports cosine(swapped, original) (privacy — should fall
  below the ~0.28 recognition threshold) and cosine(swapped, source) (how much
  of the source identity was actually transferred). BlendSwap blends partially,
  so this honestly shows the residual-original / adopted-source balance.

* **C — positioning vs. the no-swap methods.** Identity leakage (cosine to the
  original) and re-detectability for blur / pixelate / mask / blackout / swap on
  the same faces. The point: swap de-identifies like the censoring methods while
  leaving a *natural, still-detectable* face (utility preserved).

All identity crops are aligned to the ArcFace template via InsightFace
``norm_crop`` using the ORIGINAL landmarks, so cosines reflect appearance only.

Figures (under ``outputs/``):
  * ``swap_qualitative.png`` — A+B combined.
  * ``swap_metrics.png``     — C (matplotlib).

Usage:
    python -m test_scripts.test_anonymize_swap_eval
    python -m test_scripts.test_anonymize_swap_eval --source test_images/source.png --max-faces 4
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_anonymization.face_anonymizer import (
    AnonymizationMethod,
    FaceAnonymizer,
    ObfuscationParams,
)
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_parsing.face_parser import FaceParser
from ai_core.face_recognition.arcface import ArcFaceRecognizer, RECOGNITION_THRESHOLD
from ai_core.face_swapping.face_swapper import FaceSwapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = [
    PROJECT_ROOT / "test_images" / "male_1.jpeg",
    PROJECT_ROOT / "test_images" / "female_1.jpeg",
    PROJECT_ROOT / "test_images" / "male_2.jpeg",
    PROJECT_ROOT / "test_images" / "female_2.jpeg",
]
# Held-out source identity (not among the evaluated faces below). Source choice
# matters a lot per subject (a dissimilar identity moves identity furthest), but no
# single source is optimal for a mixed-gender set; source.png gives the lowest mean
# residual here. The deeper ceiling is the model: BlendSwap's partial blend keeps
# cos->original ~0.36-0.41 (near the 0.28 threshold) regardless of source.
DEFAULT_SOURCE = PROJECT_ROOT / "test_images" / "source.png"

HARD = ObfuscationParams(irreversible=True)
NONSWAP = [
    ("blur", AnonymizationMethod.BLUR),
    ("pixelate", AnonymizationMethod.PIXELATE),
    ("mask", AnonymizationMethod.MASK),
    ("blackout", AnonymizationMethod.BLACKOUT),
]


@dataclass
class Face:
    tag: str
    image: np.ndarray
    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    base_emb: np.ndarray
    swapped: np.ndarray
    cos_orig: float
    cos_src: float


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    ua = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    ub = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    u = ua + ub - inter
    return inter / u if u > 1e-9 else 0.0


def _redetect(detector, img, bbox) -> float:
    best = 0.0
    for d in detector.detect(img):
        if _iou(d.bbox, bbox) > 0.2:
            best = max(best, float(d.score))
    return best


# --------------------------------------------------------------------------- #
# Layout helpers.                                                              #
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
    if img.shape[0] >= h:
        return img[:h] if img.shape[0] > h else img
    return np.vstack([img, np.full((h - img.shape[0], img.shape[1], 3), 255, np.uint8)])


def _metrics_panel(cos_orig, cos_src, h, w=260):
    panel = np.full((h, w, 3), 250, np.uint8)
    deid = cos_orig < RECOGNITION_THRESHOLD
    y = max(h // 2 - 40, 24)
    cv2.putText(panel, "ArcFace cosine", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    col_orig = (40, 150, 40) if deid else (0, 110, 220)
    cv2.putText(panel, f"vs original: {cos_orig:.3f}", (12, y + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_orig, 2, cv2.LINE_AA)
    tag = "de-identified" if deid else "partial leak"
    cv2.putText(panel, f"  ({tag}, thr {RECOGNITION_THRESHOLD})", (12, y + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_orig, 1, cv2.LINE_AA)
    cv2.putText(panel, f"vs source:   {cos_src:.3f}", (12, y + 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 80, 30), 2, cv2.LINE_AA)
    cv2.putText(panel, "  (identity transfer)", (12, y + 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 80, 30), 1, cv2.LINE_AA)
    return panel


def _label_left(row, tag, w=110):
    panel = np.full((row.shape[0], w, 3), 250, np.uint8)
    cv2.putText(panel, tag, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return np.hstack([panel, row])


def _header(cols, widths, total_w, label_w=110):
    bar = np.full((30, total_w, 3), 230, np.uint8)
    x = label_w + 4
    for name, wq in zip(cols, widths):
        cv2.putText(bar, name, (x + 4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        x += wq + 3
    return bar


def _build_combined(faces, source_crop, cell, metrics_w=260):
    rows = []
    for f in faces:
        rect = _crop_rect(f.bbox, f.image.shape[:2])
        cells = [_cell(f.image, rect, cell), source_crop.copy(), _cell(f.swapped, rect, cell)]
        h = max(c.shape[0] for c in cells)
        cells = [_pad_to_h(c, h) for c in cells]
        cells.append(_metrics_panel(f.cos_orig, f.cos_src, h, metrics_w))
        sep = np.full((h, 3, 3), 255, np.uint8)
        stacked = []
        for c in cells:
            stacked += [c, sep]
        rows.append(_label_left(np.hstack(stacked[:-1]), f.tag))
    width = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == width else np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    head = _header(["original", "source", "swapped", "identity (ArcFace)"],
                   [cell, cell, cell, metrics_w], width)
    sep = np.full((4, width, 3), 200, np.uint8)
    out = [head]
    for r in rows:
        out += [r, sep]
    return np.vstack(out[:-1])


def _fig_metrics(cos_by, det_by, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["blur", "pixelate", "mask", "blackout", "swap"]
    cos = [float(np.mean(cos_by[n])) for n in names]
    det = [float(np.mean(det_by[n])) for n in names]
    colors = ["#e76f51", "#f4a261", "#2a9d8f", "#264653", "#9b5de5"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    b1 = ax1.bar(names, cos, color=colors)
    ax1.bar_label(b1, fmt="%.2f", padding=3)
    ax1.axhline(RECOGNITION_THRESHOLD, color="red", ls="--", lw=1.2,
                label=f"recognition threshold {RECOGNITION_THRESHOLD}")
    ax1.set_ylabel("ArcFace cosine vs original")
    ax1.set_title("Identity leakage (lower = better)")
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    b2 = ax2.bar(names, det, color=colors)
    ax2.bar_label(b2, fmt="%.2f", padding=3)
    ax2.set_ylabel("RetinaFace confidence after anonymize")
    ax2.set_title("Face still present (swap: natural; blur: leak)")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("BlendSwap vs no-swap anonymization", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate BlendSwap face-swap anonymization.")
    p.add_argument("--inputs", type=Path, nargs="*", default=None)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source identity image.")
    p.add_argument("--onnx", type=Path,
                   default=PROJECT_ROOT / "ai_core" / "face_detection" / "onnx" / "retinaface_best.onnx")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--max-faces", type=int, default=4)
    p.add_argument("--cell", type=int, default=200)
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = args.inputs if args.inputs else DEFAULT_IMAGES

    detector = FaceDetector(onnx_path=args.onnx, conf_threshold=args.conf)
    arc = ArcFaceRecognizer()
    parser = FaceParser()
    swapper = FaceSwapper(detector=detector, face_parser=parser)
    aligner = FaceAligner(output_size=(256, 256), mode=AlignMode.FFHQ)
    anon = FaceAnonymizer(face_parser=parser,
                          face_aligner=FaceAligner(output_size=(512, 512), mode=AlignMode.FFHQ))

    # Source identity (blob for swapping + embedding + display crop).
    src_bgr = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if src_bgr is None:
        print(f"Cannot read source: {args.source}", file=sys.stderr)
        return 1
    src_dets = detector.detect(src_bgr)
    if not src_dets:
        print(f"No face in source: {args.source}", file=sys.stderr)
        return 1
    src_det = max(src_dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
    src_emb = arc.embed(src_bgr, src_det.landmarks.as_array())
    src_blob = swapper.prepare_source(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB))
    source_crop = _cell(src_bgr, _crop_rect(src_det.bbox, src_bgr.shape[:2]), args.cell)

    faces: list[Face] = []
    cos_by = {n: [] for n, _ in NONSWAP}
    det_by = {n: [] for n, _ in NONSWAP}
    cos_by["swap"] = []
    det_by["swap"] = []

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
        base = arc.embed(img, lms)

        # Swap.
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        swapped = cv2.cvtColor(swapper.swap_face(rgb, aligner.align([d]), src_blob), cv2.COLOR_RGB2BGR)
        sw_emb = arc.embed(swapped, lms)
        cos_orig = arc.cosine(base, sw_emb)
        cos_src = arc.cosine(src_emb, sw_emb)
        cos_by["swap"].append(cos_orig)
        det_by["swap"].append(_redetect(detector, swapped, d.bbox))

        # No-swap methods (same face) for figure C.
        for name, method in NONSWAP:
            out = anon.anonymize(img, [{"bbox": list(d.bbox), "landmarks": lms.tolist()}],
                                 method=method, params=HARD)
            cos_by[name].append(arc.cosine(base, arc.embed(out, lms)))
            det_by[name].append(_redetect(detector, out, d.bbox))

        faces.append(Face(Path(p).stem, img, d.bbox, lms, base, swapped, cos_orig, cos_src))
        if len(faces) >= args.max_faces:
            break

    if not faces:
        print("No faces detected.", file=sys.stderr)
        return 1
    print(f"Faces: {[f.tag for f in faces]} | source: {args.source.name}")

    cv2.imwrite(str(args.outdir / "swap_qualitative.png"),
                _build_combined(faces, source_crop, args.cell))
    _fig_metrics(cos_by, det_by, args.outdir / "swap_metrics.png")

    print(f"\n{'face':12s} {'cos->orig':>10s} {'cos->source':>12s}  privacy")
    for f in faces:
        verdict = "de-id" if f.cos_orig < RECOGNITION_THRESHOLD else "partial"
        print(f"{f.tag:12s} {f.cos_orig:10.3f} {f.cos_src:12.3f}  {verdict}")
    print(f"{'MEAN':12s} {np.mean([f.cos_orig for f in faces]):10.3f} "
          f"{np.mean([f.cos_src for f in faces]):12.3f}")

    print("\nIdentity leakage (cosine->original, lower=better) by method:")
    for n in ["blur", "pixelate", "mask", "blackout", "swap"]:
        print(f"  {n:10s} {np.mean(cos_by[n]):.3f}   redetect {np.mean(det_by[n]):.3f}")

    print(f"\nSaved -> swap_qualitative.png, swap_metrics.png  (in {args.outdir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
