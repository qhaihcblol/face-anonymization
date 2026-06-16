"""Face-parsing (BiSeNet) impact figures: ellipse mask vs. semantic mask.

Evaluates what the BiSeNet face parser changes in the no-model anonymization
path. The deployed pipeline (``FaceAnonymizer._region_mask``) uses the parser
mask when landmarks + parser are available — a soft mask that hugs the real face
(skin + features, excluding hair / ears / neck) — and otherwise falls back to a
coarse elliptical bound. This script renders the two side by side so the impact
is visible and measurable.

Outputs (under ``outputs/``):
  * ``parsing_qualitative.png`` — Figure A. One row per face, columns:
      original | ellipse mask | BiSeNet mask | blur(ellipse) | blur(BiSeNet).
  * ``parsing_overlay.png``     — Figure B. Boundary diff per face:
      red  = ellipse covers but it is NOT face (collateral over-blur),
      green= face (per parser) that the ellipse leaves uncovered (potential leak),
      yellow = both agree.
  * ``parsing_metrics.png``     — Figure D. Bar chart of the aggregate ratios,
      plus a printed per-face table.

Metrics honesty note: there is no hand-labelled ground-truth face mask here, so
the BiSeNet mask is taken as the reference "face region". The numbers therefore
quantify how much the ellipse *over-covers* (blurs non-face) and how much face it
*leaves uncovered relative to the parser* — not an absolute segmentation IoU.

Usage:
    # Defaults: a curated set of test_images.
    python -m test_scripts.test_face_parsing_compare

    # Your own images, or frames pulled from a video (for profile / occlusion).
    python -m test_scripts.test_face_parsing_compare --inputs test_images/male_1.jpeg
    python -m test_scripts.test_face_parsing_compare --video test_videos/video_track.mp4 \
        --frames 80,124 --max-faces 8
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
)
from ai_core.face_detection.face_detector import FaceDetection, FaceDetector
from ai_core.face_parsing.face_parser import FaceParser
from ai_core.video_io.video_io import VideoIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMAGES = [
    PROJECT_ROOT / "test_images" / "image.png",
    PROJECT_ROOT / "test_images" / "female_1.jpeg",
    PROJECT_ROOT / "test_images" / "male_1.jpeg",
]


@dataclass
class FaceSample:
    tag: str                       # e.g. "image.png#0"
    image: np.ndarray              # full BGR frame the face lives in
    bbox: tuple[int, int, int, int]  # clipped int bbox
    ellipse: np.ndarray            # soft mask, full-frame, float32 [0,1]
    parser: np.ndarray             # soft mask, full-frame, float32 [0,1]
    parser_ok: bool                # False -> parser failed, fell back to ellipse


def _pad_crop_rect(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    pad: float = 0.45,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    h, w = shape
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    return (
        int(max(x1 - pw, 0)), int(max(y1 - ph, 0)),
        int(min(x2 + pw, w)), int(min(y2 + ph, h)),
    )


def _tint(crop: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45) -> np.ndarray:
    """Overlay a translucent tint where ``mask`` > 0.5 and draw its contour."""
    out = crop.copy()
    binary = (mask > 0.5).astype(np.uint8)
    if binary.any():
        layer = np.zeros_like(out)
        layer[binary > 0] = color
        out = cv2.addWeighted(out, 1.0, layer, alpha, 0.0)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
    return out


def _label_strip(width: int, text: str, color=(40, 40, 40), h: int = 34) -> np.ndarray:
    bar = np.full((h, width, 3), 245, dtype=np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return bar


def _collect_samples(
    images: list[tuple[str, np.ndarray]],
    detector: FaceDetector,
    anonymizer: FaceAnonymizer,
    max_faces: int,
) -> list[FaceSample]:
    """Detect faces and compute both masks (ellipse + parser) per face."""
    samples: list[FaceSample] = []
    for tag, img in images:
        shape = img.shape[:2]
        for i, det in enumerate(detector.detect(img)):
            bbox = anonymizer._valid_bbox(img, {"bbox": list(det.bbox)})
            if bbox is None:
                continue
            ellipse = anonymizer._ellipse_face_mask(bbox, shape, anonymizer.params)
            lms = det.landmarks.as_array()
            parser = anonymizer._parser_face_mask(img, lms, bbox, shape)
            parser_ok = parser is not None
            if parser is None:
                parser = ellipse  # mirror the pipeline's safe fallback
            samples.append(FaceSample(f"{tag}#{i}", img, bbox, ellipse, parser, parser_ok))
    # Keep the largest faces (most legible) up to the cap.
    samples.sort(key=lambda s: (s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1]), reverse=True)
    return samples[:max_faces]


def _blur_one(
    anonymizer: FaceAnonymizer,
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    landmarks: np.ndarray | None,
) -> np.ndarray:
    """Blur a single face (landmarks=None forces the ellipse path)."""
    det = {"bbox": list(bbox), "landmarks": None if landmarks is None else landmarks.tolist()}
    return anonymizer.anonymize(image, [det], method=AnonymizationMethod.BLUR)


def _row_cell(img: np.ndarray, rect, size: int) -> np.ndarray:
    x1, y1, x2, y2 = rect
    crop = img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    scale = size / max(w, 1)
    return cv2.resize(crop, (size, int(h * scale)), interpolation=cv2.INTER_AREA)


def _build_figure_a(
    samples: list[FaceSample],
    detector: FaceDetector,
    anon_ellipse: FaceAnonymizer,
    anon_parser: FaceAnonymizer,
    cell: int,
) -> np.ndarray:
    headers = ["original", "ellipse mask", "BiSeNet mask", "blur(ellipse)", "blur(BiSeNet)"]
    rows: list[np.ndarray] = []
    for s in samples:
        rect = _pad_crop_rect(s.bbox, s.image.shape[:2])
        # Recover this face's landmarks for the parser-path blur.
        lms = None
        for det in detector.detect(s.image):
            if anon_ellipse._valid_bbox(s.image, {"bbox": list(det.bbox)}) == s.bbox:
                lms = det.landmarks.as_array()
                break
        blur_e = _blur_one(anon_ellipse, s.image, s.bbox, None)
        blur_p = _blur_one(anon_parser, s.image, s.bbox, lms)

        cells = [
            _row_cell(s.image, rect, cell),
            _row_cell(_tint(s.image, s.ellipse, (60, 60, 230)), rect, cell),
            _row_cell(_tint(s.image, s.parser, (60, 200, 60)), rect, cell),
            _row_cell(blur_e, rect, cell),
            _row_cell(blur_p, rect, cell),
        ]
        ch = max(c.shape[0] for c in cells)
        cells = [_pad_to_h(c, ch) for c in cells]
        rows.append(_label_left(np.hstack(_with_sep(cells)), s.tag, s.parser_ok))

    header = _grid_header(headers, cell, rows[0].shape[1])
    return np.vstack([header] + _with_vsep(rows))


def _build_figure_b(samples: list[FaceSample], cell: int, ncols: int = 2) -> np.ndarray:
    """Compact grid of boundary-diff overlays (A4-friendly).

    The diff is alpha-blended over the original, so the face still shows through:
      red = ellipse covers but it is NOT face (collateral over-blur),
      green = face (per parser) the ellipse leaves uncovered (potential leak),
      yellow = both agree.
    """
    cells: list[np.ndarray] = []
    for s in samples:
        rect = _pad_crop_rect(s.bbox, s.image.shape[:2])
        e = (s.ellipse > 0.5)
        p = (s.parser > 0.5)
        diff = np.zeros_like(s.image)
        diff[e & ~p] = (60, 60, 230)    # over-cover (blur non-face) -> red
        diff[p & ~e] = (60, 200, 60)    # face left uncovered by ellipse -> green
        diff[e & p] = (60, 200, 230)    # agreement -> yellow
        overlay = cv2.addWeighted(s.image, 0.55, diff, 0.45, 0.0)
        crop = _row_cell(overlay, rect, cell)
        flag = "" if s.parser_ok else " (fallback)"
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, 22), (250, 250, 250), -1)
        cv2.putText(crop, s.tag + flag, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (30, 30, 30), 1, cv2.LINE_AA)
        cells.append(crop)

    grid = _grid(cells, ncols)
    legend = _diff_legend(grid.shape[1])
    return np.vstack([legend, grid])


def _diff_legend(width: int) -> np.ndarray:
    """Compact colour key; spacing follows the real text width so nothing clips."""
    bar = np.full((38, width, 3), 245, dtype=np.uint8)
    items = [("over-blur (non-face)", (60, 60, 230)),
             ("face leak", (60, 200, 60)),
             ("agree", (60, 200, 230))]
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    x = 12
    for name, color in items:
        cv2.rectangle(bar, (x, 11, 20, 16), color, -1)
        cv2.putText(bar, name, (x + 26, 25), font, fs, (30, 30, 30), th, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(name, font, fs, th)
        x += 26 + tw + 26
    return bar


def _grid(cells: list[np.ndarray], ncols: int, sep: int = 4) -> np.ndarray:
    """Pack equal-width cells into an ncols grid with white separators."""
    cw = cells[0].shape[1]
    ch = max(c.shape[0] for c in cells)
    cells = [_pad_to_h(c, ch) for c in cells]
    rows: list[np.ndarray] = []
    for i in range(0, len(cells), ncols):
        chunk = cells[i:i + ncols]
        while len(chunk) < ncols:
            chunk.append(np.full((ch, cw, 3), 255, dtype=np.uint8))
        rows.append(np.hstack(_with_sep(chunk, sep)))
    return np.vstack(_with_vsep(rows, sep))


def _compute_metrics(samples: list[FaceSample]) -> tuple[list[dict], dict]:
    """Per-face + aggregate ratios (binary masks at 0.5)."""
    per_face: list[dict] = []
    for s in samples:
        e = (s.ellipse > 0.5)
        p = (s.parser > 0.5)
        ea, pa = int(e.sum()), int(p.sum())
        over = float((e & ~p).sum()) / ea if ea else 0.0      # blurred non-face
        leak = float((p & ~e).sum()) / pa if pa else 0.0      # face missed by ellipse
        area_reduction = (1.0 - pa / ea) if ea else 0.0       # fewer pixels blurred
        per_face.append({
            "tag": s.tag, "parser_ok": s.parser_ok,
            "ellipse_px": ea, "parser_px": pa,
            "over_cover": over, "leak": leak, "area_reduction": area_reduction,
        })
    valid = [m for m in per_face if m["parser_ok"]] or per_face
    agg = {
        "over_cover": float(np.mean([m["over_cover"] for m in valid])),
        "leak": float(np.mean([m["leak"] for m in valid])),
        "area_reduction": float(np.mean([m["area_reduction"] for m in valid])),
        "n": len(valid),
    }
    return per_face, agg


def _save_figure_d(per_face: list[dict], agg: dict, out_path: Path) -> None:
    """A4-friendly matplotlib chart: aggregate bars + per-face over-blur."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # Left: the three aggregate ratios.
    labels = ["Collateral\nover-blur", "Face leak\nvs parser", "Pixels-blurred\nreduction"]
    vals = [agg["over_cover"] * 100, agg["leak"] * 100, agg["area_reduction"] * 100]
    colors = ["#e63946", "#2a9d8f", "#e76f51"]
    bars = ax1.bar(labels, vals, color=colors, width=0.6)
    ax1.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax1.set_ylabel("% of pixels")
    ax1.set_ylim(0, max(max(vals) * 1.25, 5))
    ax1.set_title(f"BiSeNet vs ellipse mask (mean of {agg['n']} faces)")
    ax1.grid(axis="y", alpha=0.3)

    # Right: per-face collateral over-blur, so spread (not just the mean) is visible.
    tags = [m["tag"] for m in per_face]
    over = [m["over_cover"] * 100 for m in per_face]
    ax2.barh(range(len(tags)), over, color="#e63946")
    ax2.set_yticks(range(len(tags)))
    ax2.set_yticklabels(tags, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Collateral over-blur (%)")
    ax2.set_title("Per-face over-blur")
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Small layout helpers.                                                        #
# --------------------------------------------------------------------------- #
def _pad_to_h(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    pad = np.full((h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
    return np.vstack([img, pad])


def _with_sep(cells: list[np.ndarray], wpx: int = 3) -> list[np.ndarray]:
    h = cells[0].shape[0]
    sep = np.full((h, wpx, 3), 255, dtype=np.uint8)
    out: list[np.ndarray] = []
    for c in cells:
        out += [c, sep]
    return out[:-1]


def _with_vsep(rows: list[np.ndarray], hpx: int = 4) -> list[np.ndarray]:
    w = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == w else np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    sep = np.full((hpx, w, 3), 200, dtype=np.uint8)
    out: list[np.ndarray] = []
    for r in rows:
        out += [r, sep]
    return out[:-1]


def _label_left(row: np.ndarray, tag: str, parser_ok: bool, w: int = 150) -> np.ndarray:
    panel = np.full((row.shape[0], w, 3), 250, dtype=np.uint8)
    cv2.putText(panel, tag, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    if not parser_ok:
        cv2.putText(panel, "(fallback)", (6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
    return np.hstack([panel, row])


def _grid_header(headers: list[str], cell: int, total_w: int) -> np.ndarray:
    bar = np.full((34, total_w, 3), 230, dtype=np.uint8)
    x = 150 + 4
    for name in headers:
        cv2.putText(bar, name, (x + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        x += cell + 3
    return bar


def _load_inputs(args) -> list[tuple[str, np.ndarray]]:
    images: list[tuple[str, np.ndarray]] = []
    if args.video:
        vio = VideoIO()
        frames = {int(x) for x in args.frames.split(",")} if args.frames else {0}
        for idx, frame in enumerate(vio.iter_frames(str(args.video))):
            if idx in frames:
                images.append((f"{Path(args.video).stem}@{idx}", frame))
            if idx > max(frames):
                break
    paths = args.inputs if args.inputs else (DEFAULT_IMAGES if not args.video else [])
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: cannot read {p}", file=sys.stderr)
            continue
        images.append((Path(p).name, img))
    return images


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Figures for BiSeNet face-parsing impact.")
    p.add_argument("--inputs", type=Path, nargs="*", default=None, help="Input image paths.")
    p.add_argument("--video", type=Path, default=None, help="Pull frames from this video instead.")
    p.add_argument("--frames", type=str, default=None, help="Comma frame indices when --video is set.")
    p.add_argument("--onnx", type=Path,
                   default=PROJECT_ROOT / "ai_core" / "face_detection" / "onnx" / "retinaface_best.onnx")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--max-faces", type=int, default=4, help="Max faces in the figures (A4-friendly: 3-4).")
    p.add_argument("--cell", type=int, default=240, help="Per-cell width (px).")
    p.add_argument("--cols", type=int, default=2, help="Columns in the overlay grid (figure B).")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    images = _load_inputs(args)
    if not images:
        print("No input images.", file=sys.stderr)
        return 1

    detector = FaceDetector(onnx_path=args.onnx, conf_threshold=args.conf)
    parser = FaceParser()
    aligner = FaceAligner(output_size=(512, 512), mode=AlignMode.FFHQ)
    anon_parser = FaceAnonymizer(face_parser=parser, face_aligner=aligner)
    anon_ellipse = FaceAnonymizer()  # no parser -> ellipse path

    print(f"Inputs: {[t for t, _ in images]}")
    samples = _collect_samples(images, detector, anon_parser, args.max_faces)
    if not samples:
        print("No faces detected.", file=sys.stderr)
        return 1
    n_fallback = sum(1 for s in samples if not s.parser_ok)
    print(f"Faces used: {len(samples)} (parser fallback to ellipse: {n_fallback})")

    fig_a = _build_figure_a(samples, detector, anon_ellipse, anon_parser, args.cell)
    fig_b = _build_figure_b(samples, args.cell, ncols=args.cols)
    per_face, agg = _compute_metrics(samples)

    cv2.imwrite(str(args.outdir / "parsing_qualitative.png"), fig_a)
    cv2.imwrite(str(args.outdir / "parsing_overlay.png"), fig_b)
    _save_figure_d(per_face, agg, args.outdir / "parsing_metrics.png")

    print("\nPer-face metrics (parser as face reference):")
    print(f"{'face':18s} {'ellipse_px':>11s} {'parser_px':>10s} "
          f"{'over%':>7s} {'leak%':>7s} {'reduce%':>8s}")
    for m in per_face:
        flag = "" if m["parser_ok"] else " *fallback"
        print(f"{m['tag']:18s} {m['ellipse_px']:11d} {m['parser_px']:10d} "
              f"{m['over_cover']*100:7.1f} {m['leak']*100:7.1f} "
              f"{m['area_reduction']*100:8.1f}{flag}")
    print(f"\nMean over {agg['n']} faces -> collateral over-blur: {agg['over_cover']*100:.1f}%, "
          f"face leak vs parser: {agg['leak']*100:.1f}%, "
          f"pixels-blurred reduction: {agg['area_reduction']*100:.1f}%")
    print(f"\nSaved -> {args.outdir/'parsing_qualitative.png'}, "
          f"{args.outdir/'parsing_overlay.png'}, {args.outdir/'parsing_metrics.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
