"""Evaluation of the GFPGAN face-restoration step (a QUALITY step, not anonymization).

Restoration runs *after* the swap, on the already-swapped face (a different
identity). Its job is to harmonize sharpness with the surrounding image — not to
protect identity (the swap does that). This script frames it exactly that way and
produces two figures:

* **Qualitative (headline)** — original | swap (no restore) | swap (restored),
  with a 2x detail inset so the sharpening is obvious. Best shown on a soft /
  low-resolution face (e.g. a webcam clip), where BlendSwap's 256px output is
  upscaled and looks blurry; on high-res stills the swap is already sharp and
  there is little to restore.
* **Quantitative (the "does restoration re-leak identity?" check)** — ArcFace
  cosine to the ORIGINAL for swap-without vs swap-with restoration across several
  frames, plus a blend-factor sweep. Restoration regenerates realistic detail
  that leans slightly back toward the underlying face, so the cosine rises a
  little; the blend factor is the knob that trades sharpness against that
  residual identity.

Figures (under ``outputs/``):
  * ``restore_qualitative.png``
  * ``restore_metrics.png``

Usage:
    python -m test_scripts.test_restoration_eval                       # hai1.mp4
    python -m test_scripts.test_restoration_eval --video test_videos/hai2.mp4
    python -m test_scripts.test_restoration_eval --frame 11 --source test_images/source.png
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_detection.face_detector import FaceDetection, FaceDetector
from ai_core.face_parsing.face_parser import FaceParser
from ai_core.face_recognition.arcface import ArcFaceRecognizer, RECOGNITION_THRESHOLD
from ai_core.face_restoration.face_restorer import FaceRestorer
from ai_core.face_swapping.face_swapper import FaceSwapper
from ai_core.video_io.video_io import VideoIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "test_videos" / "hai1.mp4"
# Pick a source identity dissimilar to the subject (different gender moves identity
# furthest): for the male hai1 subject, female_1 gives the lowest residual cos->original
# (~0.37 vs ~0.61 for a same-gender lookalike source).
DEFAULT_SOURCE = PROJECT_ROOT / "test_images" / "female_1.jpeg"
BLEND_SWEEP = [0.0, 0.3, 0.5, 0.8, 1.0]


@dataclass
class Sample:
    frame_idx: int
    frame: np.ndarray
    det: FaceDetection
    base_emb: np.ndarray


class Swapper:
    """Thin helper: swap one face with restoration on/off (shared sessions)."""

    def __init__(self, detector, parser, source_blob):
        self.detector = detector
        self.swapper = FaceSwapper(detector=detector, face_parser=parser)
        self.aligner = FaceAligner(output_size=(256, 256), mode=AlignMode.FFHQ)
        self.restorer = FaceRestorer(blend=0.8)
        self.source_blob = source_blob

    def swap(self, frame_bgr, det, *, restore: bool, blend: float | None = None) -> np.ndarray:
        if restore:
            if blend is not None:
                self.restorer.blend = float(np.clip(blend, 0.0, 1.0))
            self.swapper.face_restorer = self.restorer
        else:
            self.swapper.face_restorer = None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = self.swapper.swap_face(rgb, self.aligner.align([det]), self.source_blob)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _crop(img, bbox, pad=0.3):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(x1 - pw, 0)), int(max(y1 - ph, 0))
    x2, y2 = int(min(x2 + pw, w)), int(min(y2 + ph, h))
    return img[y1:y2, x1:x2]


def _resize_w(img, width):
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(h * width / max(w, 1))), interpolation=cv2.INTER_CUBIC)


def _with_inset(cell, det, bbox_rect, zoom=2.2, frac=0.42):
    """Paste a magnified detail patch (lower face) into the cell's bottom-right."""
    ch, cw = cell.shape[:2]
    # Detail patch = central-lower region of the cell (mouth/cheek texture).
    pw, ph = int(cw * 0.38), int(ch * 0.30)
    px, py = int(cw * 0.31), int(ch * 0.50)
    patch = cell[py:py + ph, px:px + pw]
    if patch.size == 0:
        return cell
    iw = int(cw * frac)
    inset = cv2.resize(patch, (iw, int(ph * iw / max(pw, 1))), interpolation=cv2.INTER_NEAREST)
    ih = inset.shape[0]
    out = cell.copy()
    cv2.rectangle(out, (px, py), (px + pw, py + ph), (0, 230, 255), 2)          # source box
    y0, x0 = ch - ih - 4, cw - iw - 4
    out[y0:y0 + ih, x0:x0 + iw] = inset
    cv2.rectangle(out, (x0, y0), (x0 + iw, y0 + ih), (0, 230, 255), 2)          # inset box
    return out


def _banner(width, text, color=(70, 70, 70), h=34, fs=0.55):
    bar = np.full((h, width, 3), 245, np.uint8)
    cv2.putText(bar, text, (10, int(h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
    return bar


def _title(width, text, color):
    bar = np.full((40, width, 3), color, np.uint8)
    cv2.putText(bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return bar


def _build_qualitative(hero, swp, cell, out_path):
    bbox = hero.det.bbox
    orig = _resize_w(_crop(hero.frame, bbox), cell)
    no_r = _resize_w(_crop(swp.swap(hero.frame, hero.det, restore=False), bbox), cell)
    res = _resize_w(_crop(swp.swap(hero.frame, hero.det, restore=True, blend=0.8), bbox), cell)
    no_r = _with_inset(no_r, hero.det, bbox)
    res = _with_inset(res, hero.det, bbox)

    h = max(orig.shape[0], no_r.shape[0], res.shape[0])
    cells, titles, colors = [orig, no_r, res], \
        ["original (real)", "swap (no restore)", "swap + restoration"], \
        [(90, 90, 90), (40, 40, 200), (40, 140, 40)]
    cols = []
    for c, t, col in zip(cells, titles, colors):
        c = np.vstack([c, np.full((h - c.shape[0], c.shape[1], 3), 255, np.uint8)]) if c.shape[0] < h else c
        cols.append(np.vstack([_title(c.shape[1], t, col), c]))
    sep = np.full((cols[0].shape[0], 4, 3), 255, np.uint8)
    grid = np.hstack([cols[0], sep, cols[1], sep, cols[2]])
    cap = _banner(grid.shape[1],
                  "Restoration runs AFTER swap, on the swapped identity = a sharpness step, NOT anonymization.",
                  fs=0.46)
    cap2 = _banner(grid.shape[1],
                   "Yellow box = detail region; bottom-right = 2x zoom (sharper skin/lip texture after restoration).",
                   color=(0, 120, 180), fs=0.44)
    cv2.imwrite(str(out_path), np.vstack([grid, cap, cap2]))


def _build_metrics(arc, samples, swp, hero, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cos_no, cos_re = [], []
    for s in samples:
        cos_no.append(arc.cosine(s.base_emb, _emb_after(arc, swp, s, restore=False)))
        cos_re.append(arc.cosine(s.base_emb, _emb_after(arc, swp, s, restore=True, blend=0.8)))
    sweep = [arc.cosine(hero.base_emb, _emb_after(arc, swp, hero, restore=(b > 0), blend=b))
             for b in BLEND_SWEEP]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    x = np.arange(len(samples))
    w = 0.38
    b1 = ax1.bar(x - w / 2, cos_no, w, label="no restore", color="#577590")
    b2 = ax1.bar(x + w / 2, cos_re, w, label="restored (0.8)", color="#43aa8b")
    ax1.bar_label(b1, fmt="%.2f", fontsize=8); ax1.bar_label(b2, fmt="%.2f", fontsize=8)
    ax1.axhline(RECOGNITION_THRESHOLD, color="red", ls="--", lw=1.2, label=f"threshold {RECOGNITION_THRESHOLD}")
    ax1.set_xticks(x); ax1.set_xticklabels([f"f{s.frame_idx}" for s in samples])
    ax1.set_ylabel("ArcFace cosine vs original")
    ax1.set_title("Does restoration re-leak identity?")
    ax1.set_ylim(0, 1.0); ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.plot(BLEND_SWEEP, sweep, "o-", color="#43aa8b", lw=2)
    ax2.axhline(RECOGNITION_THRESHOLD, color="red", ls="--", lw=1.2, label=f"threshold {RECOGNITION_THRESHOLD}")
    for bx, by in zip(BLEND_SWEEP, sweep):
        ax2.annotate(f"{by:.2f}", (bx, by), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax2.set_xlabel("restoration blend factor")
    ax2.set_ylabel("ArcFace cosine vs original")
    ax2.set_title(f"Sharpness vs anonymity trade-off (frame {hero.frame_idx})")
    ax2.set_ylim(0, 1.0); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle("GFPGAN restoration: quality step, identity check", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cos_no, cos_re, sweep


def _emb_after(arc, swp, sample, *, restore, blend=None):
    """Embed the swapped face, aligned by the ORIGINAL landmarks (geometry fixed)."""
    out = swp.swap(sample.frame, sample.det, restore=restore, blend=blend)
    return arc.embed(out, sample.det.landmarks.as_array())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate GFPGAN restoration (quality + identity check).")
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--frame", type=int, default=None, help="Frame for the qualitative figure (default: largest face).")
    p.add_argument("--num-samples", type=int, default=5, help="Frames for the identity-check bars.")
    p.add_argument("--min-face", type=int, default=120, help="Min face width (px) to sample.")
    p.add_argument("--onnx", type=Path,
                   default=PROJECT_ROOT / "ai_core" / "face_detection" / "onnx" / "retinaface_best.onnx")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--cell", type=int, default=300)
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(onnx_path=args.onnx, conf_threshold=args.conf)
    arc = ArcFaceRecognizer()
    parser = FaceParser()

    src = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if src is None:
        print(f"Cannot read source: {args.source}", file=sys.stderr)
        return 1
    sdets = detector.detect(src)
    if not sdets:
        print("No face in source.", file=sys.stderr)
        return 1
    swp = Swapper(detector, parser,
                  FaceSwapper(detector=detector).prepare_source(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)))

    # Collect candidate frames (face >= min-face), keep largest as the hero.
    frames = list(VideoIO().iter_frames(str(args.video)))
    cands: list[Sample] = []
    for i, f in enumerate(frames):
        ds = detector.detect(f)
        if not ds:
            continue
        d = max(ds, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        if (d.bbox[2] - d.bbox[0]) >= args.min_face:
            cands.append(Sample(i, f, d, arc.embed(f, d.landmarks.as_array())))
    if not cands:
        print("No suitable faces found.", file=sys.stderr)
        return 1

    if args.frame is not None:
        hero = min(cands, key=lambda s: abs(s.frame_idx - args.frame))
    else:
        hero = max(cands, key=lambda s: s.det.bbox[2] - s.det.bbox[0])
    step = max(len(cands) // args.num_samples, 1)
    samples = cands[::step][:args.num_samples]
    print(f"Video: {args.video.name} | hero frame {hero.frame_idx} "
          f"(face {int(hero.det.bbox[2]-hero.det.bbox[0])}px) | samples {[s.frame_idx for s in samples]}")

    _build_qualitative(hero, swp, args.cell, args.outdir / "restore_qualitative.png")
    cos_no, cos_re, sweep = _build_metrics(arc, samples, swp, hero, args.outdir / "restore_metrics.png")

    print("\nIdentity check (ArcFace cosine vs original, lower = more anonymous):")
    print(f"{'frame':8s} {'no-restore':>11s} {'restored':>10s} {'delta':>7s}")
    for s, cn, cr in zip(samples, cos_no, cos_re):
        print(f"f{s.frame_idx:<7d} {cn:11.3f} {cr:10.3f} {cr-cn:+7.3f}")
    print(f"{'MEAN':8s} {np.mean(cos_no):11.3f} {np.mean(cos_re):10.3f} {np.mean(cos_re)-np.mean(cos_no):+7.3f}")
    print("\nBlend sweep (frame %d): " % hero.frame_idx
          + ", ".join(f"{b}->{c:.3f}" for b, c in zip(BLEND_SWEEP, sweep)))
    print(f"\nSaved -> restore_qualitative.png, restore_metrics.png  (in {args.outdir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
