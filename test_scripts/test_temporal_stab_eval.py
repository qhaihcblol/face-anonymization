"""Evaluation of the temporal stabilization module (online face-swap smoothing).

A per-frame independent swap flickers: the detector's landmarks jitter frame to
frame, so the swapped crop shimmers in position, colour and texture. The online
:class:`FaceSwapStabilizer` damps this with a causal 1-Euro filter on the
landmarks plus an EMA on the swapped crop and blend mask. This script shows the
flicker reduction three ways (no-stabilization vs online):

* **Jitter curve** — frame-to-frame change of the swapped face. Each output frame
  is registered to the 256px template using that frame's landmarks (so head
  motion is removed), then consecutive aligned crops are differenced (mean
  absolute difference). Lower & flatter = steadier. The mean is the headline number.
* **Temporal slice** — a fixed column through the registered face stacked over
  time into one image (x = time). Flicker shows as horizontal streaks/noise;
  stabilization yields smooth bands. A single static picture of temporal behaviour.
* **Landmark trajectory** — the raw detector landmark (nose x) vs the same signal
  passed through the 1-Euro filter the stabilizer uses. Shows the de-jitter directly.

Restoration is intentionally OFF here (it is a sharpness step and would add its own
per-frame variation); this isolates the stabilizer.

Figures (under ``outputs/``):
  * ``temporal_jitter.png`` · ``temporal_slice.png`` · ``temporal_trajectory.png``

Usage:
    python -m test_scripts.test_temporal_stab_eval
    python -m test_scripts.test_temporal_stab_eval --video test_videos/hai2.mp4 --max-frames 28
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_parsing.face_parser import FaceParser
from ai_core.face_swapping.face_swap_stabilizer import FaceSwapStabilizer, OneEuroFilter
from ai_core.face_swapping.face_swapper import FaceSwapper
from ai_core.video_io.video_io import VideoIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "test_videos" / "hai1.mp4"
DEFAULT_SOURCE = PROJECT_ROOT / "test_images" / "female_1.jpeg"


def _largest(dets):
    return max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))


def _center(det):
    x1, y1, x2, y2 = det.bbox
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])


def _smooth_centers(c, win=7):
    """Low-pass the tracked face-center trajectory so the crop box follows gross
    motion only; the high-frequency jitter the stabilizer removes stays visible
    inside the box (a fixed box would lose a moving face; a raw-centered box would
    cancel the very jitter we measure)."""
    c = np.asarray(c, dtype=float)
    if len(c) < 3:
        return c
    win = win if win % 2 == 1 else win + 1
    pad = win // 2
    k = np.ones(win) / win
    out = np.empty_like(c)
    for d in range(c.shape[1]):
        p = np.pad(c[:, d], pad, mode="edge")
        out[:, d] = np.convolve(p, k, mode="valid")
    return out


def _crop_at(frame_bgr, cx, cy, half, size=256):
    """Fixed-size square crop centered at (cx, cy), clamped inside the frame."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > w:
        x1 -= (x2 - w); x2 = w
    if y2 > h:
        y1 -= (y2 - h); y2 = h
    x1, y1 = max(x1, 0), max(y1, 0)
    return cv2.resize(frame_bgr[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


def _jitter(crops):
    """Temporal high-pass flicker = |c[t] - 2 c[t-1] + c[t-2]| (second difference).

    The second difference cancels smooth motion (a steadily moving mouth/head has
    near-zero acceleration) and keeps the high-frequency shimmer the stabilizer is
    meant to remove, so it is not swamped by genuine talking motion the way a raw
    consecutive-frame difference is.
    """
    out = [0.0, 0.0]
    for i in range(2, len(crops)):
        a, b, c = (crops[i - 2].astype(np.float32), crops[i - 1].astype(np.float32), crops[i].astype(np.float32))
        out.append(float(np.mean(np.abs(c - 2.0 * b + a))))
    return out


def _fig_jitter(j_no, j_on, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    x = range(2, len(j_no))
    ax.plot(x, j_no[2:], color="#e63946", lw=1.8, label=f"no stabilization (mean {np.mean(j_no[2:]):.2f})")
    ax.plot(x, j_on[2:], color="#2a9d8f", lw=1.8, label=f"online stabilization (mean {np.mean(j_on[2:]):.2f})")
    ax.set_xlabel("frame")
    ax.set_ylabel("temporal flicker\n(2nd-difference, fixed face box)")
    ax.set_title("High-frequency flicker of the swapped face (lower = steadier)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_slice(crops_no, crops_on, out_path, col_w=10):
    """Stack a fixed center column of each registered crop over time."""
    def strip(crops):
        cx = crops[0].shape[1] // 2
        cols = [c[:, cx:cx + 1, :] for c in crops]              # (H,1,3) each
        img = np.concatenate(cols, axis=1)                      # (H, T, 3)
        return cv2.resize(img, (img.shape[1] * col_w, img.shape[0]), interpolation=cv2.INTER_NEAREST)

    s_no, s_on = strip(crops_no), strip(crops_on)
    w = s_no.shape[1]

    def band(label, color):
        bar = np.full((30, w, 3), 245, np.uint8)
        cv2.putText(bar, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return bar

    cap = np.full((28, w, 3), 250, np.uint8)
    cv2.putText(cap, "x-axis = time (frames); a fixed vertical line through the face. Streaks/noise = flicker.",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    sep = np.full((4, w, 3), 200, np.uint8)
    out = np.vstack([
        band("no stabilization (note shimmer)", (40, 40, 200)), s_no, sep,
        band("online stabilization (smooth bands)", (40, 140, 40)), s_on, cap,
    ])
    cv2.imwrite(str(out_path), out)


def _fig_trajectory(raw_xy, smooth_xy, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_xy, smooth_xy = np.asarray(raw_xy), np.asarray(smooth_xy)
    n = len(raw_xy)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True)
    for ax, k, name in [(ax1, 0, "nose x"), (ax2, 1, "nose y")]:
        ax.plot(range(n), raw_xy[:, k], color="#e63946", lw=1.5, marker=".", ms=4, label="raw landmark")
        ax.plot(range(n), smooth_xy[:, k], color="#2a9d8f", lw=2.0, label="1-Euro smoothed")
        ax.set_ylabel(f"{name} (px)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax2.set_xlabel("frame")
    rms_raw = np.mean([np.std(np.diff(raw_xy[:, k])) for k in (0, 1)])
    rms_sm = np.mean([np.std(np.diff(smooth_xy[:, k])) for k in (0, 1)])
    fig.suptitle(f"Landmark trajectory: 1-Euro removes jitter "
                 f"(step-to-step std {rms_raw:.2f} -> {rms_sm:.2f} px)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate online temporal stabilization of the face swap.")
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--start", type=int, default=0, help="First frame index to use.")
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--onnx", type=Path,
                   default=PROJECT_ROOT / "ai_core" / "face_detection" / "onnx" / "retinaface_best.onnx")
    p.add_argument("--conf", type=float, default=0.5)
    # 1-Euro / EMA knobs (defaults match SwapOptions online mode).
    p.add_argument("--min-cutoff", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--output-smooth", type=float, default=0.4)
    p.add_argument("--mask-smooth", type=float, default=0.5)
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(onnx_path=args.onnx, conf_threshold=args.conf)
    swapper = FaceSwapper(detector=detector, face_parser=FaceParser())
    aligner = FaceAligner(output_size=(256, 256), mode=AlignMode.FFHQ)

    src = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if src is None:
        print(f"Cannot read source: {args.source}", file=sys.stderr)
        return 1
    sdets = detector.detect(src)
    if not sdets:
        print("No face in source.", file=sys.stderr)
        return 1
    src_blob = swapper.prepare_source(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))

    meta = VideoIO().get_video_metadata(str(args.video))
    frames = list(VideoIO().iter_frames(str(args.video)))[args.start: args.start + args.max_frames]
    print(f"Video: {args.video.name} | {len(frames)} frames @ {meta.fps:.1f} fps | source {args.source.name}")

    # Pass 1: track ONE face across frames (largest first, then nearest-center) so a
    # multi-face / moving clip (video_track) follows a single consistent subject.
    tracked: dict[int, object] = {}
    ref = None
    for i, frame in enumerate(frames):
        dets = detector.detect(frame)
        if not dets:
            continue
        if ref is None:
            det = _largest(dets)
        else:
            det = min(dets, key=lambda d: float(np.linalg.norm(_center(d) - ref)))
            if np.linalg.norm(_center(det) - ref) > (det.bbox[2] - det.bbox[0]):
                continue  # nearest face jumped too far — treat as a miss this frame
        ref = _center(det)
        tracked[i] = det
    if len(tracked) < 3:
        print("Not enough faces across frames.", file=sys.stderr)
        return 1

    order = sorted(tracked)
    centers = np.array([_center(tracked[i]) for i in order])
    sizes = np.array([tracked[i].bbox[2] - tracked[i].bbox[0] for i in order])
    half = int(np.median(sizes) * (0.5 + 0.5))           # box ~2x face width
    sc = _smooth_centers(centers, win=7)
    box_at = {i: sc[k] for k, i in enumerate(order)}

    # Pass 2: swap (no-stab + online), crop at the smoothed following box.
    stab = FaceSwapStabilizer(detector, swapper, freq=meta.fps, min_cutoff=args.min_cutoff,
                              beta=args.beta, output_smooth=args.output_smooth,
                              mask_smooth=args.mask_smooth, source_blob=src_blob)
    euro = OneEuroFilter(meta.fps, min_cutoff=args.min_cutoff, beta=args.beta)

    crops_no, crops_on, raw_xy, smooth_xy = [], [], [], []
    for i, frame in enumerate(frames):
        out_on = stab.process(frame)          # called every frame (causal filter state)
        if i not in tracked:
            continue
        det = tracked[i]
        cx, cy = box_at[i]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_no = cv2.cvtColor(swapper.swap_face(rgb, aligner.align([det]), src_blob), cv2.COLOR_RGB2BGR)

        crops_no.append(_crop_at(out_no, cx, cy, half))
        crops_on.append(_crop_at(out_on, cx, cy, half))

        nose = det.landmarks.as_array()[2]
        sm = euro.filter(det.landmarks.as_array().reshape(-1)).reshape(5, 2)[2]
        raw_xy.append(nose); smooth_xy.append(sm)
        if (i + 1) % 15 == 0:
            print(f"  processed {i + 1}/{len(frames)} frames")

    j_no, j_on = _jitter(crops_no), _jitter(crops_on)
    _fig_jitter(j_no, j_on, args.outdir / "temporal_jitter.png")
    _fig_slice(crops_no, crops_on, args.outdir / "temporal_slice.png")
    _fig_trajectory(raw_xy, smooth_xy, args.outdir / "temporal_trajectory.png")

    m_no, m_on = float(np.mean(j_no[2:])), float(np.mean(j_on[2:]))
    print(f"\nMean temporal flicker (2nd-difference, fixed face box, 0-255):")
    print(f"  no stabilization : {m_no:.2f}")
    print(f"  online           : {m_on:.2f}   ({100*(m_no-m_on)/m_no:+.0f}% vs no-stab)")
    rms_raw = float(np.mean([np.std(np.diff(np.asarray(raw_xy)[:, k])) for k in (0, 1)]))
    rms_sm = float(np.mean([np.std(np.diff(np.asarray(smooth_xy)[:, k])) for k in (0, 1)]))
    print(f"Landmark step-to-step std: raw {rms_raw:.2f}px -> 1-Euro {rms_sm:.2f}px")
    print(f"\nSaved -> temporal_jitter.png, temporal_slice.png, temporal_trajectory.png  (in {args.outdir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
