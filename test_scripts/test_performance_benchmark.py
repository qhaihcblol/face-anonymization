"""Runtime PERFORMANCE benchmark for the FaceGuard pipeline (speed, not quality).

The ``*_eval.py`` / ``*_compare.py`` scripts measure *how well* the system hides
identity (ArcFace cosine, WER, parsing/tracker comparisons). This script is the
orthogonal axis — *how fast* it runs — covering the three dimensions that matter
for the thesis "hiệu năng" chapter:

1. **Offline pipeline speed.** A per-stage micro-benchmark (detection → tracking →
   parsing/mask → obfuscation → align+swap → restore → voice) over real frames,
   plus the end-to-end ``anonymize_video`` throughput (which also pays decode/
   encode/mux). Tells you the FPS and *where the time goes* (the bottleneck).
2. **Live camera latency.** Pushes frames through :class:`LiveFaceAnonymizer` for a
   sweep of ``detect_interval`` values and reports the per-frame latency
   distribution (p50/p95/p99) and the achievable FPS vs the 30 FPS (33 ms) budget.
3. **Resources.** ONNX Runtime providers actually in use (CPU vs CUDA), process
   CPU utilisation, peak RAM, and — when an NVIDIA GPU is present — sampled GPU
   utilisation and peak VRAM (via ``nvidia-smi``). No psutil dependency.

Everything is measured with warm models (one untimed warm-up pass) so cold-start
ONNX session init never pollutes the numbers.

Figures (under ``outputs/``):
  * ``perf_stage_breakdown.png``  — median ms per pipeline stage + per-method FPS
  * ``perf_live_latency.png``     — latency distribution vs detect_interval
  * ``perf_summary.txt``          — machine-readable summary table

Usage:
    # Full benchmark on the default clip (hai1.mp4), 120 frames.
    python -m test_scripts.test_performance_benchmark

    # Pick a clip / frame budget; skip the heavy swap + voice stages.
    python -m test_scripts.test_performance_benchmark --video test_videos/test.mp4 \
        --frames 200 --no-swap --no-voice

    # Force CPU (to compare against a GPU run) or a specific provider set.
    python -m test_scripts.test_performance_benchmark --providers CPUExecutionProvider
"""
from __future__ import annotations

import argparse
import resource
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_anonymization.face_anonymizer import (
    AnonymizationMethod,
    FaceAnonymizer,
)
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_parsing.face_parser import FaceParser
from ai_core.face_restoration.face_restorer import FaceRestorer
from ai_core.face_swapping.face_swapper import DEFAULT_SOURCE_FACE, FaceSwapper
from ai_core.face_tracking.face_tracker import ByteTracker
from ai_core.live_anonymization import LiveFaceAnonymizer, LiveVisualConfig
from ai_core.video_anonymization import (
    AudioOptions,
    SwapOptions,
    VideoAnonymization,
    VisualOptions,
)
from ai_core.video_io.video_io import VideoIO
from ai_core.voice_anonymization.voice_anonymizer import (
    VoiceAnonymizationMethod,
    VoiceAnonymizer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "test_videos" / "hai1.mp4"
DEFAULT_SOURCE = PROJECT_ROOT / "test_images" / "source.png"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
# Obfuscation methods timed end-to-end (swap is timed separately as it needs the
# aligner + the model path).
OBFUSCATION_METHODS = ["blur", "pixelate", "mask", "blackout"]
# detect_interval values swept for the live-latency dimension.
LIVE_INTERVALS = [1, 2, 3, 5]
# Real-time budgets drawn on the live-latency figure.
FPS_BUDGETS = {"30 FPS (33ms)": 1000.0 / 30.0, "15 FPS (67ms)": 1000.0 / 15.0}


# --------------------------------------------------------------------------- #
# Timing primitives.                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Timing:
    """Latency samples (ms) for one stage, summarised on demand."""

    name: str
    samples: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    @property
    def n(self) -> int:
        return len(self.samples)

    def pct(self, q: float) -> float:
        if not self.samples:
            return float("nan")
        ordered = sorted(self.samples)
        idx = min(int(q / 100.0 * len(ordered)), len(ordered) - 1)
        return ordered[idx]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples) if self.samples else float("nan")

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else float("nan")

    @property
    def fps(self) -> float:
        m = self.mean
        return 1000.0 / m if m and m == m and m > 0 else float("nan")


def _time_call(timing: Timing, fn: Callable[[], object]) -> object:
    """Run ``fn``, record its wall time (ms) into ``timing``, return its result."""
    t0 = time.perf_counter()
    out = fn()
    timing.add((time.perf_counter() - t0) * 1000.0)
    return out


# --------------------------------------------------------------------------- #
# Resource sampling (CPU / RAM / GPU) — no psutil, degrades gracefully.        #
# --------------------------------------------------------------------------- #
class ResourceMonitor:
    """Sample peak RAM + (optionally) GPU util/VRAM around a workload.

    CPU utilisation is derived from process CPU time vs wall time (so >100% means
    the ONNX/OpenCV thread pools are using multiple cores). GPU stats come from
    ``nvidia-smi`` sampled on a background thread; absent on CPU-only boxes.
    """

    def __init__(self) -> None:
        self._has_smi = _have_nvidia_smi()
        self._gpu_util: list[float] = []
        self._gpu_mem: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._wall0 = 0.0
        self._cpu0 = 0.0

    def __enter__(self) -> "ResourceMonitor":
        self._wall0 = time.perf_counter()
        self._cpu0 = _process_cpu_seconds()
        if self._has_smi:
            self._thread = threading.Thread(target=self._sample_gpu, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.wall_sec = time.perf_counter() - self._wall0
        self.cpu_sec = _process_cpu_seconds() - self._cpu0
        # ru_maxrss is KiB on Linux, bytes on macOS.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak_ram_mb = rss / 1024.0 if sys.platform != "darwin" else rss / 1e6

    @property
    def cpu_percent(self) -> float:
        return 100.0 * self.cpu_sec / self.wall_sec if self.wall_sec > 0 else 0.0

    def _sample_gpu(self) -> None:
        while not self._stop.is_set():
            util, mem = _nvidia_smi_query()
            if util is not None:
                self._gpu_util.append(util)
            if mem is not None:
                self._gpu_mem.append(mem)
            self._stop.wait(0.2)

    def summary(self) -> dict[str, float | None]:
        return {
            "cpu_percent": self.cpu_percent,
            "peak_ram_mb": self.peak_ram_mb,
            "gpu_util_mean": statistics.fmean(self._gpu_util) if self._gpu_util else None,
            "gpu_mem_peak_mb": max(self._gpu_mem) if self._gpu_mem else None,
        }


def _process_cpu_seconds() -> float:
    t = resource.getrusage(resource.RUSAGE_SELF)
    return t.ru_utime + t.ru_stime


def _have_nvidia_smi() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def _nvidia_smi_query() -> tuple[float | None, float | None]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().splitlines()
        if not out:
            return None, None
        util_s, mem_s = out[0].split(",")
        return float(util_s), float(mem_s)
    except (subprocess.SubprocessError, ValueError):
        return None, None


# --------------------------------------------------------------------------- #
# Model construction (one shared, warm set).                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Models:
    detector: FaceDetector
    tracker: ByteTracker
    aligner: FaceAligner
    anonymizer: FaceAnonymizer
    voice: VoiceAnonymizer | None
    swap_ready: bool


def build_models(args: argparse.Namespace) -> Models:
    providers = args.providers or None
    detector = FaceDetector(
        onnx_path=PROJECT_ROOT
        / "ai_core"
        / "face_detection"
        / "onnx"
        / "retinaface_best.onnx",
        providers=providers,
    )
    aligner = FaceAligner(output_size=(256, 256), mode=AlignMode.FFHQ)
    parser = FaceParser()

    swapper = None
    swap_ready = False
    if not args.no_swap:
        try:
            restorer = None if args.no_restore else FaceRestorer()
            swapper = FaceSwapper(
                detector=detector,
                source_path=args.source,
                face_parser=parser,
                face_restorer=restorer,
            )
            swap_ready = True
        except Exception as exc:  # missing ONNX weights -> skip the swap dimension
            print(f"[warn] face swap unavailable, skipping swap stage: {exc}")

    anonymizer = FaceAnonymizer(
        face_swapper=swapper, face_parser=parser, face_aligner=aligner
    )

    voice = None
    if not args.no_voice:
        # DSP (mcadams) needs no model weights, so it always works for timing.
        voice = VoiceAnonymizer()

    return Models(
        detector=detector,
        tracker=ByteTracker(),
        aligner=aligner,
        anonymizer=anonymizer,
        voice=voice,
        swap_ready=swap_ready,
    )


def load_frames(video: Path, limit: int) -> list[np.ndarray]:
    io = VideoIO()
    frames: list[np.ndarray] = []
    for frame in io.iter_frames(str(video)):
        frames.append(frame)
        if len(frames) >= limit:
            break
    if not frames:
        raise RuntimeError(f"No frames decoded from {video}")
    return frames


# --------------------------------------------------------------------------- #
# Dimension 1 — offline per-stage micro-benchmark.                             #
# --------------------------------------------------------------------------- #
def benchmark_stages(models: Models, frames: list[np.ndarray]) -> dict[str, Timing]:
    """Time each pipeline stage independently over ``frames`` (warm models)."""
    timings = {
        "detection": Timing("detection"),
        "tracking": Timing("tracking"),
        "obf:blur": Timing("obf:blur"),
        "obf:pixelate": Timing("obf:pixelate"),
        "obf:mask (parse)": Timing("obf:mask (parse)"),
        "obf:blackout": Timing("obf:blackout"),
    }
    if models.swap_ready:
        timings["align"] = Timing("align")
        timings["swap+restore"] = Timing("swap+restore")

    tracker = ByteTracker(
        high_thresh=models.tracker.high_thresh,
        low_thresh=models.tracker.low_thresh,
        max_lost=models.tracker.max_lost,
        min_hits=models.tracker.min_hits,
        iou_thresh=models.tracker.iou_thresh,
        iou_thresh_low=models.tracker.iou_thresh_low,
        gate_mahal=models.tracker.gate_mahal,
    )
    source_blob = (
        models.anonymizer.face_swapper.default_source()
        if models.swap_ready
        else None
    )

    # Untimed warm-up so ONNX graph optimisation / first-call allocs are excluded.
    _ = models.detector.detect(frames[0])

    for frame in frames:
        detections = _time_call(
            timings["detection"], lambda f=frame: models.detector.detect(f)
        )
        tracks = _time_call(
            timings["tracking"], lambda d=detections: tracker.update(d)
        )
        confirmed = [t for t in tracks if t.get("state") == "Tracked"]

        _time_call(
            timings["obf:blur"],
            lambda f=frame, t=confirmed: models.anonymizer.anonymize(
                f, t, method=AnonymizationMethod.BLUR
            ),
        )
        _time_call(
            timings["obf:pixelate"],
            lambda f=frame, t=confirmed: models.anonymizer.anonymize(
                f, t, method=AnonymizationMethod.PIXELATE
            ),
        )
        _time_call(
            timings["obf:mask (parse)"],
            lambda f=frame, t=confirmed: models.anonymizer.anonymize(
                f, t, method=AnonymizationMethod.MASK
            ),
        )
        _time_call(
            timings["obf:blackout"],
            lambda f=frame, t=confirmed: models.anonymizer.anonymize(
                f, t, method=AnonymizationMethod.BLACKOUT
            ),
        )

        if models.swap_ready and detections:
            aligned = _time_call(
                timings["align"], lambda d=detections: models.aligner.align(d)
            )
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _time_call(
                timings["swap+restore"],
                lambda r=frame_rgb, a=aligned: models.anonymizer.swap_face(
                    r, a, source_blob
                ),
            )

    return timings


def method_fps(stages: dict[str, Timing]) -> dict[str, float]:
    """Compose end-to-end FPS for each anonymization method from stage means.

    detect + track + obfuscate (the live/offline obfuscation path); swap is
    detect + align + swap+restore (the swap path runs the detector every frame).
    """
    det = stages["detection"].mean
    trk = stages["tracking"].mean
    out: dict[str, float] = {}
    for m in OBFUSCATION_METHODS:
        key = "obf:mask (parse)" if m == "mask" else f"obf:{m}"
        total = det + trk + stages[key].mean
        out[m] = 1000.0 / total if total > 0 else float("nan")
    if "swap+restore" in stages:
        total = det + stages["align"].mean + stages["swap+restore"].mean
        out["swap"] = 1000.0 / total if total > 0 else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Dimension 1b — end-to-end throughput (includes decode/encode/mux).           #
# --------------------------------------------------------------------------- #
def benchmark_end_to_end(
    models: Models, video: Path, args: argparse.Namespace
) -> dict[str, float]:
    """Run the real ``anonymize_video`` for blur (+swap) to get true wall FPS."""
    va = VideoAnonymization(
        video_io=VideoIO(),
        face_detector=models.detector,
        face_tracker=models.tracker,
        face_anonymizer=models.anonymizer,
        face_aligner=models.aligner,
        voice_anonymizer=models.voice,
    )
    results: dict[str, float] = {}
    out = OUTPUT_DIR / "_perf_blur.mp4"
    res = va.anonymize_video(
        input_path=video,
        output_path=out,
        visual=VisualOptions(method="blur"),
        audio=AudioOptions(keep_audio=False),
        end_sec=args.e2e_sec,
        progress_every=0,
    )
    results["blur (end-to-end)"] = res.throughput_fps
    out.unlink(missing_ok=True)

    if models.swap_ready:
        out = OUTPUT_DIR / "_perf_swap.mp4"
        res = va.anonymize_video(
            input_path=video,
            output_path=out,
            visual=SwapOptions(stabilize=True, smoothing="online"),
            audio=AudioOptions(keep_audio=False),
            end_sec=args.e2e_sec,
            progress_every=0,
        )
        results["swap (end-to-end)"] = res.throughput_fps
        out.unlink(missing_ok=True)
    return results


# --------------------------------------------------------------------------- #
# Dimension 2 — live camera latency sweep.                                     #
# --------------------------------------------------------------------------- #
def benchmark_live(
    models: Models, frames: list[np.ndarray], intervals: Sequence[int]
) -> dict[int, Timing]:
    """For each detect_interval, push every frame and collect process_ms."""
    out: dict[int, Timing] = {}
    for interval in intervals:
        live = LiveFaceAnonymizer(
            face_detector=models.detector,
            face_tracker=models.tracker,
            face_anonymizer=models.anonymizer,
            config=LiveVisualConfig(method="blur", detect_interval=interval),
        )
        live.process_frame(frames[0])  # warm-up (untimed)
        live.reset()
        t = Timing(f"interval={interval}")
        for frame in frames:
            res = live.process_frame(frame)
            t.add(res.process_ms)
        out[interval] = t
    return out


# --------------------------------------------------------------------------- #
# Dimension 3 — voice realtime factor.                                         #
# --------------------------------------------------------------------------- #
def benchmark_voice(
    models: Models, video: Path, args: argparse.Namespace
) -> dict[str, float] | None:
    if models.voice is None:
        return None
    io = VideoIO()
    if not io.has_audio(str(video)):
        print("[warn] input has no audio stream; skipping voice benchmark")
        return None
    waveform, sr = io.extract_audio(str(video), end_sec=args.e2e_sec)
    audio_sec = len(waveform) / sr if sr else 0.0
    out: dict[str, float] = {}
    for method in (VoiceAnonymizationMethod.MCADAMS, VoiceAnonymizationMethod.PITCH):
        t0 = time.perf_counter()
        models.voice.process(waveform, sr, method=method)
        proc = time.perf_counter() - t0
        # xRT > 1 means faster than real time (good for batch / offline).
        out[method.value] = audio_sec / proc if proc > 0 else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Reporting.                                                                   #
# --------------------------------------------------------------------------- #
def print_report(
    *,
    providers: list[str],
    n_frames: int,
    stages: dict[str, Timing],
    method_fps_map: dict[str, float],
    e2e: dict[str, float],
    live: dict[int, Timing],
    voice: dict[str, float] | None,
    resources: dict[str, float | None],
) -> str:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    emit("=" * 68)
    emit("FaceGuard — RUNTIME PERFORMANCE BENCHMARK")
    emit("=" * 68)
    emit(f"ONNX providers in use : {', '.join(providers)}")
    emit(f"Frames measured        : {n_frames}")
    emit("")
    emit("[1] OFFLINE — per-stage latency (warm models)")
    emit(f"  {'stage':<20}{'mean ms':>10}{'median':>10}{'p95':>10}{'~FPS':>10}")
    for name, t in stages.items():
        emit(
            f"  {name:<20}{t.mean:>10.2f}{t.median:>10.2f}"
            f"{t.pct(95):>10.2f}{t.fps:>10.1f}"
        )
    emit("")
    emit("[1] OFFLINE — composite FPS per method (detect+track+method)")
    for m, fps in method_fps_map.items():
        emit(f"  {m:<20}{fps:>8.1f} FPS")
    emit("")
    if e2e:
        emit("[1b] END-TO-END throughput (incl. decode/encode/mux)")
        for label, fps in e2e.items():
            emit(f"  {label:<24}{fps:>8.2f} FPS")
        emit("")
    emit("[2] LIVE — per-frame latency vs detect_interval")
    emit(f"  {'interval':<12}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'~FPS':>10}")
    for interval, t in live.items():
        emit(
            f"  {interval:<12}{t.median:>10.2f}{t.pct(95):>10.2f}"
            f"{t.pct(99):>10.2f}{t.fps:>10.1f}"
        )
    emit("")
    if voice:
        emit("[3] VOICE — realtime factor (xRT; >1 = faster than real time)")
        for method, xrt in voice.items():
            emit(f"  {method:<20}{xrt:>8.2f} xRT")
        emit("")
    emit("[3] RESOURCES")
    emit(f"  CPU utilisation      : {resources['cpu_percent']:.0f}% (of one core)")
    emit(f"  Peak RAM (RSS)       : {resources['peak_ram_mb']:.0f} MB")
    if resources["gpu_util_mean"] is not None:
        emit(f"  GPU utilisation mean : {resources['gpu_util_mean']:.0f}%")
        emit(f"  Peak VRAM            : {resources['gpu_mem_peak_mb']:.0f} MB")
    else:
        emit("  GPU                  : not detected / not used (CPU run)")
    emit("=" * 68)

    text = "\n".join(lines)
    print(text)
    return text


def make_figures(
    stages: dict[str, Timing],
    method_fps_map: dict[str, float],
    live: dict[int, Timing],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: stage breakdown + per-method FPS.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    names = list(stages.keys())
    medians = [stages[n].median for n in names]
    ax1.barh(names, medians, color="#4C72B0")
    ax1.invert_yaxis()
    ax1.set_xlabel("median latency (ms)")
    ax1.set_title("Per-stage latency (offline, warm)")
    for i, v in enumerate(medians):
        ax1.text(v, i, f" {v:.1f}", va="center", fontsize=8)

    methods = list(method_fps_map.keys())
    fpss = [method_fps_map[m] for m in methods]
    bars = ax2.bar(methods, fpss, color="#55A868")
    ax2.set_ylabel("composite FPS")
    ax2.set_title("Throughput per anonymization method")
    ax2.axhline(30, color="red", ls="--", lw=1, label="30 FPS")
    ax2.legend()
    for b, v in zip(bars, fpss):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "perf_stage_breakdown.png", dpi=130)
    plt.close(fig)

    # Figure 2: live latency vs detect_interval.
    fig, ax = plt.subplots(figsize=(8, 5))
    intervals = list(live.keys())
    p50 = [live[i].median for i in intervals]
    p95 = [live[i].pct(95) for i in intervals]
    ax.plot(intervals, p50, "o-", label="p50", color="#4C72B0")
    ax.plot(intervals, p95, "s--", label="p95", color="#C44E52")
    for label, ms in FPS_BUDGETS.items():
        ax.axhline(ms, color="gray", ls=":", lw=1)
        ax.text(intervals[-1], ms, f" {label}", va="bottom", fontsize=8, color="gray")
    ax.set_xlabel("detect_interval (run detector every N frames)")
    ax.set_ylabel("per-frame latency (ms)")
    ax.set_title("Live camera latency vs detect_interval")
    ax.set_xticks(intervals)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "perf_live_latency.png", dpi=130)
    plt.close(fig)

    print(f"\nFigures written to {OUTPUT_DIR}/perf_stage_breakdown.png, perf_live_latency.png")


# --------------------------------------------------------------------------- #
# Entry point.                                                                 #
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FaceGuard runtime performance benchmark.")
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Clip to measure on")
    p.add_argument("--frames", type=int, default=120, help="Frames for the micro-benchmark")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Swap source identity")
    p.add_argument("--e2e-sec", type=float, default=5.0, help="Seconds for end-to-end + voice")
    p.add_argument("--no-swap", action="store_true", help="Skip the face-swap stage")
    p.add_argument("--no-restore", action="store_true", help="Skip GFPGAN restoration in swap")
    p.add_argument("--no-voice", action="store_true", help="Skip the voice stage")
    p.add_argument("--no-e2e", action="store_true", help="Skip the end-to-end video run")
    p.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Force ONNX providers, e.g. CPUExecutionProvider (default: auto)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Input video not found: {args.video}")

    print(f"Loading models + {args.frames} frames from {args.video} ...")
    models = build_models(args)
    frames = load_frames(args.video, args.frames)
    providers = list(models.detector.session.get_providers()) if hasattr(
        models.detector, "session"
    ) else _detector_providers(models.detector)

    with ResourceMonitor() as mon:
        print("[1/3] Offline per-stage micro-benchmark ...")
        stages = benchmark_stages(models, frames)
        e2e: dict[str, float] = {}
        if not args.no_e2e:
            print("[1b] End-to-end throughput ...")
            e2e = benchmark_end_to_end(models, args.video, args)
        print("[2/3] Live latency sweep ...")
        live = benchmark_live(models, frames, LIVE_INTERVALS)
        print("[3/3] Voice realtime factor ...")
        voice = benchmark_voice(models, args.video, args)
    resources = mon.summary()

    method_fps_map = method_fps(stages)
    report = print_report(
        providers=providers,
        n_frames=len(frames),
        stages=stages,
        method_fps_map=method_fps_map,
        e2e=e2e,
        live=live,
        voice=voice,
        resources=resources,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "perf_summary.txt").write_text(report + "\n", encoding="utf-8")
    make_figures(stages, method_fps_map, live)


def _detector_providers(detector: FaceDetector) -> list[str]:
    """Best-effort read of the ONNX providers the detector actually bound to."""
    for attr in ("session", "_session", "ort_session"):
        sess = getattr(detector, attr, None)
        if sess is not None and hasattr(sess, "get_providers"):
            return list(sess.get_providers())
    return ["unknown"]


if __name__ == "__main__":
    main()
