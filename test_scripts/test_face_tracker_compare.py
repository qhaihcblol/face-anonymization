"""Face-tracker evaluation figure: detection-only vs. tracking.

Builds the side-by-side comparison used in the thesis to justify the face
tracker. The same frames are rendered two ways:

* **Top strip — detection-only (per-frame, no tracking).** The detector runs
  independently on every frame; boxes are kept only above a sensible display
  confidence (``--baseline-conf``, default = the tracker's own high threshold)
  and IDs are assigned by raw detector order. Consequences a reader can see:
  - a hard face the detector finds only at *low confidence* (or misses outright)
    gets no box -> its anonymization (blur) **blinks off and the face is exposed**;
  - when the detection count changes, the index-based IDs **renumber/swap**, so
    the same physical face changes colour/number between frames.

* **Bottom strip — with tracking (ByteTrack).** The tracker is fed the full
  detection set every frame. Its two-stage association recovers the low-score
  box and locks it to a stable ``track_id`` (with Kalman prediction + a lost
  buffer bridging brief gaps), so identities stay consistent and the masked
  region is **continuous**.

This is an honest depiction of the deployed pipeline: only confirmed ``Tracked``
tracks are anonymized (matching ``VideoAnonymization``), and with
``--detect-interval N`` the in-between frames reuse ``predict_only()`` exactly as
the real run does — which makes the "covered even when the detector slips on this
frame" point fully literal.

Outputs (under ``outputs/``):
  * ``tracker_compare_strip.png`` — the 2-row film strip (the thesis figure).
  * ``tracker_compare.mp4``       — full clip, two renders stacked side by side.

Usage:
    # Defaults: test_videos/video_track.mp4, auto-pick the most divergent frames.
    python -m test_scripts.test_face_tracker_compare

    # Pick explicit frames for the strip and a stronger baseline threshold.
    python -m test_scripts.test_face_tracker_compare \
        --frames 80,97,99,102,124,127 --baseline-conf 0.6

    # Demonstrate prediction-based gap filling (detector every 5th frame).
    python -m test_scripts.test_face_tracker_compare --detect-interval 5
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
)
from ai_core.face_detection.face_detector import FaceDetection, FaceDetector
from ai_core.face_tracking.face_tracker import ByteTracker, _track_color
from ai_core.video_io.video_io import VideoIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Per-frame state captured during one forward pass over the clip.             #
# --------------------------------------------------------------------------- #
@dataclass
class FrameRecord:
    index: int
    frame: np.ndarray
    detections: list[FaceDetection]
    # Boxes a detection-only system would act on: (id, bbox, score), id = detector order.
    baseline: list[tuple[int, tuple[float, float, float, float], float]]
    # Confirmed tracker tracks for this frame (the dicts ByteTracker emits).
    tracks: list[dict]


def _det_to_track_dict(det: FaceDetection, track_id: int) -> dict:
    """Wrap a raw detection as a track-like dict the anonymizer understands."""
    return {
        "track_id": track_id,
        "bbox": list(det.bbox),
        "score": float(det.score),
        "state": "Tracked",
        "landmarks": det.landmarks.as_array().tolist(),
    }


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def _draw_boxes(
    canvas: np.ndarray,
    items: list[tuple[int, tuple[float, float, float, float], float]],
    *,
    thickness: int = 3,
) -> np.ndarray:
    """Draw id-coloured boxes with a ``#id`` / score label."""
    for tid, bbox, score in items:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        color = _track_color(tid)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"#{tid} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            canvas, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return canvas


def _render_pair(
    record: FrameRecord,
    anonymizer: FaceAnonymizer,
    method: AnonymizationMethod,
) -> tuple[np.ndarray, np.ndarray]:
    """Render (detection-only, with-tracking) frames for one record."""
    # Detection-only: anonymize the kept baseline boxes, then draw them.
    base_dets = [
        {"bbox": list(bbox), "landmarks": None}
        for _tid, bbox, _score in record.baseline
    ]
    base_img = anonymizer.anonymize(record.frame, base_dets, method=method)
    base_img = _draw_boxes(base_img, record.baseline)

    # With tracking: anonymize + draw the confirmed tracks (stable ids).
    track_img = anonymizer.anonymize(record.frame, record.tracks, method=method)
    track_items = [
        (int(t["track_id"]), tuple(t["bbox"]), float(t["score"]))
        for t in record.tracks
    ]
    track_img = _draw_boxes(track_img, track_items)
    return base_img, track_img


def _banner(width: int, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """A coloured title bar of ``width`` px."""
    bar = np.full((46, width, 3), color, dtype=np.uint8)
    cv2.putText(
        bar, text, (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return bar


def _process(
    video_path: Path,
    detector: FaceDetector,
    tracker: ByteTracker,
    baseline_conf: float,
    detect_interval: int,
) -> tuple[list[FrameRecord], float]:
    """Single pass over the clip building per-frame baseline + tracker state."""
    vio = VideoIO()
    fps = vio.get_video_metadata(str(video_path)).fps
    records: list[FrameRecord] = []

    for idx, frame in enumerate(vio.iter_frames(str(video_path))):
        run_detect = (idx % detect_interval) == 0
        if run_detect:
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
        else:
            # Detector skipped: tracker coasts on Kalman predictions (real pipeline).
            detections = []
            tracks = tracker.predict_only()

        # Detection-only baseline: only frames where the detector actually ran can
        # contribute boxes; keep the confident ones, id by detector order.
        baseline = [
            (i, det.bbox, float(det.score))
            for i, det in enumerate(detections)
            if det.score >= baseline_conf
        ]
        confirmed = [t for t in tracks if t.get("state") == "Tracked"]
        records.append(FrameRecord(idx, frame, detections, baseline, confirmed))

    return records, fps


def _missed_boxes(rec: FrameRecord) -> list[tuple[float, float, float, float]]:
    """Confirmed-track boxes with no matching baseline detection (IoU < 0.3)."""
    return [
        tuple(t["bbox"])
        for t in rec.tracks
        if all(_iou(tuple(t["bbox"]), b) < 0.3 for _i, b, _s in rec.baseline)
    ]


def _auto_select(records: list[FrameRecord], n: int) -> list[int]:
    """Pick the ``n`` frames where tracking covers faces the baseline misses.

    Divergence = number of confirmed tracks with no matching baseline box. The
    most-divergent frames are taken first (thinned so picks spread out); any
    remaining slots are filled with *agreement* frames where the tracker is fully
    warmed up (non-empty confirmed set) so the contrast column looks clean rather
    than catching a tracker start-up frame.
    """
    diverging = sorted(
        (rec.index for rec in records if _missed_boxes(rec)),
        key=lambda i: -len(_missed_boxes(records[i])),
    )
    picked: list[int] = []
    for idx in diverging:
        if all(abs(idx - p) >= 4 for p in picked):
            picked.append(idx)
        if len(picked) >= n:
            break

    # Fill remaining slots with clean, fully-tracked agreement frames (skip the
    # warm-up window where tracks are still unconfirmed).
    if len(picked) < n:
        clean = [
            rec.index
            for rec in records
            if rec.tracks and len(rec.baseline) == len(rec.tracks) and not _missed_boxes(rec)
        ]
        step = max(len(clean) // max(n - len(picked), 1), 1)
        for idx in clean[::step]:
            if idx not in picked:
                picked.append(idx)
            if len(picked) >= n:
                break
    return sorted(picked[:n])


def _common_crop(
    records: list[FrameRecord],
    indices: list[int],
    shape: tuple[int, int],
    pad: float = 1.4,
) -> tuple[int, int, int, int]:
    """A shared zoom rect for every cell.

    Prefer the *missed* face(s) — the privacy leak this figure is about — so the
    strip zooms into where detection-only fails. With generous padding the
    neighbouring faces stay in frame, so the ID story is still visible. Falls back
    to the union of all faces when no frame diverges.
    """
    h, w = shape
    by_index = {r.index: r for r in records}
    focus: list[tuple[float, float, float, float]] = []
    for idx in indices:
        focus += _missed_boxes(by_index[idx])
    if not focus:
        for idx in indices:
            rec = by_index[idx]
            focus += [b for _i, b, _s in rec.baseline]
            focus += [tuple(t["bbox"]) for t in rec.tracks]
    if not focus:
        return 0, 0, w, h

    bx1 = min(b[0] for b in focus); by1 = min(b[1] for b in focus)
    bx2 = max(b[2] for b in focus); by2 = max(b[3] for b in focus)
    pw, ph = (bx2 - bx1) * pad, (by2 - by1) * pad
    return (
        int(max(bx1 - pw, 0)),
        int(max(by1 - ph, 0)),
        int(min(bx2 + pw, w)),
        int(min(by2 + ph, h)),
    )


def _build_strip(
    records: list[FrameRecord],
    indices: list[int],
    anonymizer: FaceAnonymizer,
    method: AnonymizationMethod,
    cell_w: int,
) -> np.ndarray:
    """Compose the 2-row (detection-only / tracking) film strip PNG."""
    by_index = {r.index: r for r in records}
    h, w = records[0].frame.shape[:2]
    cx1, cy1, cx2, cy2 = _common_crop(records, indices, (h, w))
    crop_h, crop_w = cy2 - cy1, cx2 - cx1
    cell_h = int(round(cell_w * crop_h / max(crop_w, 1)))

    top_cells, bot_cells = [], []
    for idx in indices:
        rec = by_index[idx]
        base_img, track_img = _render_pair(rec, anonymizer, method)

        def _cell(img: np.ndarray) -> np.ndarray:
            crop = img[cy1:cy2, cx1:cx2]
            cell = cv2.resize(crop, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            cv2.putText(
                cell, f"frame {idx}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
            )
            return cell

        sep = np.full((cell_h, 3, 3), 255, dtype=np.uint8)
        top_cells += [_cell(base_img), sep]
        bot_cells += [_cell(track_img), sep]

    top_row = np.hstack(top_cells[:-1])
    bot_row = np.hstack(bot_cells[:-1])
    row_w = top_row.shape[1]

    top_banner = _banner(row_w, "Detection-only (per-frame, no tracking)", (40, 40, 200))
    bot_banner = _banner(row_w, "With tracking (ByteTrack): stable IDs + continuous mask", (40, 140, 40))
    gap = np.full((6, row_w, 3), 255, dtype=np.uint8)

    return np.vstack([top_banner, top_row, gap, bot_banner, bot_row])


def _build_video(
    records: list[FrameRecord],
    anonymizer: FaceAnonymizer,
    method: AnonymizationMethod,
    fps: float,
    output_path: Path,
) -> None:
    """Full clip, the two renders stacked side by side with title banners."""
    h, w = records[0].frame.shape[:2]
    scale = 960 / w  # half-width per panel so the pair fits common screens
    pw, ph = int(w * scale), int(h * scale)
    left_banner = _banner(pw, "Detection-only (per-frame)", (40, 40, 200))
    right_banner = _banner(pw, "With tracking (ByteTrack)", (40, 140, 40))
    banner = np.hstack([left_banner, right_banner])

    def _frames():
        for rec in records:
            base_img, track_img = _render_pair(rec, anonymizer, method)
            left = cv2.resize(base_img, (pw, ph), interpolation=cv2.INTER_AREA)
            right = cv2.resize(track_img, (pw, ph), interpolation=cv2.INTER_AREA)
            yield np.vstack([banner, np.hstack([left, right])])

    VideoIO().write_frames(_frames(), str(output_path), fps=fps)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare detection-only vs. tracked face anonymization."
    )
    p.add_argument("--input", type=Path, default=PROJECT_ROOT / "test_videos" / "video_track.mp4")
    p.add_argument("--onnx", type=Path,
                   default=PROJECT_ROOT / "ai_core" / "face_detection" / "onnx" / "retinaface_best.onnx")
    p.add_argument("--conf", type=float, default=0.4,
                   help="Detector confidence threshold (what the tracker is fed).")
    p.add_argument("--baseline-conf", type=float, default=0.6,
                   help="Display threshold a detection-only system would act on (default 0.6).")
    p.add_argument("--detect-interval", type=int, default=1,
                   help="Run the detector every N frames; in-between use tracker prediction.")
    p.add_argument("--frames", type=str, default=None,
                   help="Comma-separated frame indices for the strip (default: auto-pick).")
    p.add_argument("--num-frames", type=int, default=4, help="Strip columns when auto-picking.")
    p.add_argument("--cell-width", type=int, default=520, help="Per-cell width (px) in the strip.")
    p.add_argument("--method", type=str, default="blur",
                   choices=[m.value for m in AnonymizationMethod if m.value != "swap"])
    p.add_argument("--no-video", action="store_true", help="Skip the side-by-side video.")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if not args.input.is_file():
        print(f"Input video not found: {args.input}", file=sys.stderr)
        return 1
    args.outdir.mkdir(parents=True, exist_ok=True)
    method = AnonymizationMethod(args.method)

    detector = FaceDetector(onnx_path=args.onnx, conf_threshold=args.conf)
    tracker = ByteTracker()
    # No parser/aligner: the anonymizer falls back to the soft ellipse mask, which is
    # all this figure needs (and keeps the script dependency-light).
    anonymizer = FaceAnonymizer()

    print(f"Processing {args.input.name} "
          f"(baseline-conf={args.baseline_conf}, detect-interval={args.detect_interval})...")
    records, fps = _process(
        args.input, detector, tracker, args.baseline_conf, max(args.detect_interval, 1)
    )

    # Analysis log: where do the two approaches disagree?
    print("\nframe | baseline(faces) | tracked(faces) | tracker covers extra")
    print("-" * 62)
    for rec in records:
        extra = sum(
            1 for t in rec.tracks
            if all(_iou(tuple(t["bbox"]), b) < 0.3 for _i, b, _s in rec.baseline)
        )
        if extra > 0 or len(rec.baseline) != len(rec.tracks):
            print(f"{rec.index:5d} | {len(rec.baseline):15d} | {len(rec.tracks):14d} | {extra}")

    if args.frames:
        indices = [int(x) for x in args.frames.split(",") if x.strip() != ""]
        indices = [i for i in indices if 0 <= i < len(records)]
    else:
        indices = _auto_select(records, args.num_frames)
    print(f"\nStrip frames: {indices}")

    strip = _build_strip(records, indices, anonymizer, method, args.cell_width)
    strip_path = args.outdir / "tracker_compare_strip.png"
    cv2.imwrite(str(strip_path), strip)
    print(f"Saved film strip -> {strip_path}  ({strip.shape[1]}x{strip.shape[0]})")

    if not args.no_video:
        video_path = args.outdir / "tracker_compare.mp4"
        _build_video(records, anonymizer, method, fps, video_path)
        print(f"Saved comparison video -> {video_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
