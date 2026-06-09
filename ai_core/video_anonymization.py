"""Top-level video anonymization orchestration.

The business logic is split into three clearly separated layers so the visual
(image) path and the audio path never bleed into each other:

* ``anonymize_video`` — the *master*. It prepares a :class:`_RunContext`, builds
  the visual frame stream, builds the audio source, then writes the result.
* ``_build_visual_pipeline`` — *image only*. Branches between the model-based face
  swap and the no-model obfuscation (blur / pixelate / mask / blackout / none) and
  returns a lazy iterator of output frames.
* ``_build_audio_pipeline`` — *audio only*. Branches between muting, muxing the
  original track, and running the :class:`VoiceAnonymizer` over the trimmed window.
* ``_write_result`` — encodes the frames and muxes the chosen audio.

Callers describe *what* to do with two config objects — :class:`VisualOptions`
(or :class:`SwapOptions`) for the image and :class:`AudioOptions` for the sound —
which keeps the two concerns from getting tangled in a single wide signature.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tempfile
import time
from typing import Iterator

import cv2
import numpy as np

from ai_core.face_alignment.face_aligner import AlignMode, FaceAligner
from ai_core.face_anonymization.face_anonymizer import (
    AnonymizationMethod,
    FaceAnonymizer,
    ObfuscationParams,
)
from ai_core.face_swapping.face_swap_offline import OfflineFaceSwapStabilizer
from ai_core.face_swapping.face_swap_stabilizer import FaceSwapStabilizer
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_tracking.face_tracker import ByteTracker
from ai_core.video_io.video_io import VideoIO, VideoMetadata
from ai_core.voice_anonymization.voice_anonymizer import (
    VoiceAnonymizationMethod,
    VoiceAnonymizer,
    VoiceParams,
)

__all__ = [
    "AudioMode",
    "AudioOptions",
    "ObfuscationParams",
    "SwapOptions",
    "VideoAnonymization",
    "VideoAnonymizationResult",
    "VisualOptions",
    "VoiceParams",
]


# --------------------------------------------------------------------------- #
# Configuration objects: one for the image concern, one for the audio concern. #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class VisualOptions:
    """No-model obfuscation of faces (blur / pixelate / mask / blackout / none).

    This is the *image* concern for the detector + tracker + anonymizer path.
    Use :class:`SwapOptions` instead to drive the model-based face swap.
    """

    method: AnonymizationMethod | str = AnonymizationMethod.BLUR
    # Run the detector every N frames (>= 1); in-between frames reuse tracker
    # predictions. 1 = detect on every frame.
    detect_interval: int = 1
    # Also anonymize "New" (unconfirmed) tracks, not just confirmed "Tracked" ones.
    blur_new: bool = False
    # Overlay tracker boxes/ids on the output (debug visualization).
    draw_tracks: bool = False
    # Per-run obfuscation strengths (blur kernel / pixelation / mask color / hardening).
    # None -> use the FaceAnonymizer instance's own configured defaults.
    obfuscation: ObfuscationParams | None = None

    @property
    def resolved_method(self) -> AnonymizationMethod:
        """Coerce ``method`` to an :class:`AnonymizationMethod`, rejecting SWAP.

        SWAP belongs to the model path; routing it here would be a silent no-op,
        so it is rejected with a pointer to :class:`SwapOptions`.
        """
        method = self.method
        if isinstance(method, str):
            method = AnonymizationMethod(method.strip().lower())
        if not isinstance(method, AnonymizationMethod):
            raise TypeError(
                "VisualOptions.method must be AnonymizationMethod or str, "
                f"got {type(method).__name__}"
            )
        if method is AnonymizationMethod.SWAP:
            raise ValueError(
                "Face swap is driven by SwapOptions, not "
                "VisualOptions(method='swap'). Pass visual=SwapOptions(...)."
            )
        return method

    @property
    def resolved_detect_interval(self) -> int:
        return max(int(self.detect_interval), 1)


@dataclass(slots=True)
class SwapOptions:
    """Model-based face swap (BlendSwap) with temporal stabilization.

    This is the *image* concern for the face-swap path; it requires the
    :class:`FaceAnonymizer` to be configured with a ``FaceSwapper``.
    """

    # Identity to swap onto every detected face. None -> the FaceSwapper's bundled
    # default (source_img.png); a path to an image file swaps a different identity.
    source_face_path: str | Path | None = None
    # Smooth the swap across frames instead of an independent per-frame swap.
    stabilize: bool = True
    # 'online' = causal 1-Euro landmark smoothing; 'offline' = 2-pass zero-phase.
    smoothing: str = "online"
    # One-Euro knobs (online mode only).
    smooth_min_cutoff: float = 0.5
    smooth_beta: float = 0.05
    # EMA weight on the swapped crop (None = per-mode default: 0.4 online / 0.25
    # offline). Higher = less flicker.
    output_smooth: float | None = None
    # EMA weight on the blend mask (steadier edges).
    mask_smooth: float = 0.5

    @property
    def smoothing_mode(self) -> str:
        return self.smoothing.strip().lower()

    @property
    def resolved_output_smooth(self) -> float:
        """Default the crop EMA per smoothing mode when the caller leaves it unset.

        Online's causal 1-Euro leaves residual landmark jitter, so it leans on a
        stronger EMA (0.4) to damp the per-frame restore/color flicker. Offline's
        zero-phase landmarks are already steady, so a light EMA (0.25) suffices; a
        heavier one would reintroduce causal lip-ghosting in pass 2.
        """
        if self.output_smooth is not None:
            return self.output_smooth
        return 0.25 if self.smoothing_mode == "offline" else 0.4


class AudioMode(Enum):
    """How the audio track is handled, derived from :class:`AudioOptions`."""

    MUTE = "mute"  # drop the audio -> silent output
    ORIGINAL = "original"  # mux the source track untouched
    ANONYMIZE = "anonymize"  # run the VoiceAnonymizer, then mux


@dataclass(slots=True)
class AudioOptions:
    """The *audio* concern, fully independent of the image path."""

    keep_audio: bool = True
    anonymize_voice: bool = False
    voice_method: VoiceAnonymizationMethod | str = VoiceAnonymizationMethod.MCADAMS
    # Per-run DSP strengths (mcadams alpha / pitch steps / formant shift / stft).
    # None -> use the VoiceAnonymizer instance's own configured defaults.
    voice: VoiceParams | None = None
    # CONVERT only: reference identity to convert toward. None -> the VoiceConverter's
    # bundled default (reference_voice.wav); a path to a wav uses a different voice.
    voice_reference_path: str | Path | None = None

    @property
    def mode(self) -> AudioMode:
        if not self.keep_audio:
            return AudioMode.MUTE
        if self.anonymize_voice:
            return AudioMode.ANONYMIZE
        return AudioMode.ORIGINAL


@dataclass(slots=True)
class VideoAnonymizationResult:
    output_path: Path
    output_metadata: VideoMetadata
    elapsed_sec: float
    throughput_fps: float


@dataclass(slots=True)
class _RunContext:
    """Resolved, validated state shared by the visual, audio and write layers."""

    input_path: Path
    output_path: Path
    source_meta: VideoMetadata
    output_fps: float
    target_fps: int | None
    start_sec: float | None
    end_sec: float | None
    codec: str
    progress_every: int
    visual: VisualOptions | SwapOptions
    audio: AudioOptions

    @property
    def is_swap(self) -> bool:
        return isinstance(self.visual, SwapOptions)


class VideoAnonymization:
    def __init__(
        self,
        video_io: VideoIO,
        face_detector: FaceDetector,
        face_tracker: ByteTracker,
        face_anonymizer: FaceAnonymizer,
        face_aligner: FaceAligner | None = None,
        voice_anonymizer: VoiceAnonymizer | None = None,
    ) -> None:
        self.video_io = video_io
        self.face_detector = face_detector
        self.face_tracker = face_tracker
        self.face_anonymizer = face_anonymizer
        # Aligner is only needed for the model-based (face swap) path.
        self.face_aligner = face_aligner
        # Voice anonymizer is optional; only needed when AudioOptions.anonymize_voice.
        self.voice_anonymizer = voice_anonymizer

    # --------------------------------------------------------------------- #
    # Master: orchestrate context -> visual -> audio -> write.               #
    # --------------------------------------------------------------------- #
    def anonymize_video(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        visual: VisualOptions | SwapOptions | None = None,
        audio: AudioOptions | None = None,
        target_fps: int | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
        codec: str = "H264",
        progress_every: int = 60,
    ) -> VideoAnonymizationResult:
        """Anonymize ``input_path`` and write the result to ``output_path``.

        The image treatment is selected by the type of ``visual``:

        * :class:`VisualOptions` -> no-model obfuscation (the default).
        * :class:`SwapOptions`   -> model-based face swap (BlendSwap).

        The audio treatment is selected by ``audio`` (:class:`AudioOptions`):
        keep the original track, drop it, or run the :class:`VoiceAnonymizer`.
        """
        context = self._prepare_context(
            input_path,
            output_path,
            visual=visual if visual is not None else VisualOptions(),
            audio=audio if audio is not None else AudioOptions(),
            target_fps=target_fps,
            start_sec=start_sec,
            end_sec=end_sec,
            codec=codec,
            progress_every=progress_every,
        )
        self._log_header(context)

        visual_frames = self._build_visual_pipeline(context)
        with contextlib.ExitStack() as stack:
            audio_source, audio_start = self._build_audio_pipeline(context, stack)
            # Encoding stays inside the stack so a temp voice WAV outlives the write.
            return self._write_result(
                context, visual_frames, audio_source, audio_start
            )

    # --------------------------------------------------------------------- #
    # Context preparation + logging.                                         #
    # --------------------------------------------------------------------- #
    def _prepare_context(
        self,
        input_path: str | Path,
        output_path: str | Path | None,
        *,
        visual: VisualOptions | SwapOptions,
        audio: AudioOptions,
        target_fps: int | None,
        start_sec: float | None,
        end_sec: float | None,
        codec: str,
        progress_every: int,
    ) -> _RunContext:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        is_swap = isinstance(visual, SwapOptions)
        if is_swap:
            # Fail fast on swap prerequisites before any frames are read.
            if self.face_anonymizer.face_swapper is None:
                raise RuntimeError(
                    "SwapOptions requires a FaceAnonymizer configured with a "
                    "FaceSwapper (face_anonymizer=FaceAnonymizer(face_swapper=...))."
                )
            if visual.smoothing_mode not in ("online", "offline"):
                raise ValueError(
                    f"smoothing must be 'online' or 'offline', got {visual.smoothing!r}"
                )
        else:
            # Validate/coerce the obfuscation method up front (rejects SWAP).
            visual.resolved_method

        resolved_output = self._resolve_output_path(
            input_path,
            Path(output_path) if output_path is not None else None,
            swap=is_swap,
        )
        source_meta = self.video_io.get_video_metadata(str(input_path))
        output_fps = self._resolve_output_fps(source_meta.fps, target_fps)

        return _RunContext(
            input_path=input_path,
            output_path=resolved_output,
            source_meta=source_meta,
            output_fps=output_fps,
            target_fps=target_fps,
            start_sec=start_sec,
            end_sec=end_sec,
            codec=codec,
            progress_every=progress_every,
            visual=visual,
            audio=audio,
        )

    def _log_header(self, context: _RunContext) -> None:
        meta = context.source_meta
        print(f"Input: {context.input_path}")
        print(
            "Source metadata: "
            f"{meta.width}x{meta.height}, "
            f"{meta.fps:.3f} FPS, {meta.frame_count} frames"
        )
        print(f"Output: {context.output_path}")

        if isinstance(context.visual, SwapOptions):
            print("Anonymization method: swap (BlendSwap)")
            print(f"Output FPS: {context.output_fps:.3f}")
            mode = context.visual.smoothing_mode if context.visual.stabilize else "off"
            print(f"Temporal stabilization: {mode}")
        else:
            print(f"Anonymization method: {context.visual.resolved_method.value}")
            print(f"Detect interval: {context.visual.resolved_detect_interval}")
            print(f"Output FPS: {context.output_fps:.3f}")

        print(f"Keep audio: {'on' if context.audio.keep_audio else 'off'}")
        if context.audio.mode is AudioMode.ANONYMIZE:
            print(
                "Voice anonymization: on "
                f"({self._voice_label(context.audio.voice_method)})"
            )

    # --------------------------------------------------------------------- #
    # Visual pipeline (image only).                                          #
    # --------------------------------------------------------------------- #
    def _build_visual_pipeline(self, context: _RunContext) -> Iterator[np.ndarray]:
        if isinstance(context.visual, SwapOptions):
            return self._process_visual_with_model(context, context.visual)
        return self._process_visual_without_model(context, context.visual)

    def _process_visual_without_model(
        self,
        context: _RunContext,
        options: VisualOptions,
    ) -> Iterator[np.ndarray]:
        tracker = self._build_fresh_tracker(self.face_tracker)
        source_frames = self.video_io.iter_frames(
            str(context.input_path),
            start_sec=context.start_sec,
            end_sec=context.end_sec,
            target_fps=context.target_fps,
        )
        return self._iter_processed_frames(
            frames=source_frames,
            tracker=tracker,
            method=options.resolved_method,
            detect_interval=options.resolved_detect_interval,
            blur_new=options.blur_new,
            draw_tracks=options.draw_tracks,
            progress_every=context.progress_every,
            obfuscation=options.obfuscation,
        )

    def _process_visual_with_model(
        self,
        context: _RunContext,
        options: SwapOptions,
    ) -> Iterator[np.ndarray]:
        aligner = self._resolve_face_aligner()
        swapper = self.face_anonymizer.face_swapper
        assert swapper is not None  # guaranteed by _prepare_context
        output_smooth = options.resolved_output_smooth

        # Resolve the source identity once up-front (so any source issue surfaces
        # before the first frame) and thread it explicitly through the swap calls. The
        # swapper is shared across runs, so the identity must travel as a value, never
        # as mutable instance state.
        if options.source_face_path is None:
            source_blob = swapper.default_source()
        else:
            source_blob = swapper.prepare_source(options.source_face_path)

        def _make_source_frames() -> Iterator[np.ndarray]:
            return self.video_io.iter_frames(
                str(context.input_path),
                start_sec=context.start_sec,
                end_sec=context.end_sec,
                target_fps=context.target_fps,
            )

        if options.stabilize and options.smoothing_mode == "offline":
            offline = OfflineFaceSwapStabilizer(
                detector=self.face_detector,
                swapper=swapper,
                output_smooth=output_smooth,
                mask_smooth=options.mask_smooth,
                source_blob=source_blob,
            )
            print("Pass 1/2: detecting + tracking faces across the clip...")
            pass1_count = 0
            for frame_bgr in _make_source_frames():
                offline.observe(frame_bgr)
                pass1_count += 1
                if (
                    context.progress_every > 0
                    and pass1_count % context.progress_every == 0
                ):
                    print(
                        f"  pass 1: {pass1_count} frames "
                        f"| faces: {offline.last_face_count}"
                    )
            offline.finalize()
            print(f"Pass 1 done ({pass1_count} frames). Pass 2/2: swapping...")
            return self._iter_offline_rendered_frames(
                _make_source_frames(),
                offline,
                context.progress_every,
            )

        stabilizer: FaceSwapStabilizer | None = None
        if options.stabilize:
            stabilizer = FaceSwapStabilizer(
                detector=self.face_detector,
                swapper=swapper,
                freq=context.output_fps,
                min_cutoff=options.smooth_min_cutoff,
                beta=options.smooth_beta,
                output_smooth=output_smooth,
                mask_smooth=options.mask_smooth,
                source_blob=source_blob,
            )
        return self._iter_swapped_frames(
            frames=_make_source_frames(),
            aligner=aligner,
            progress_every=context.progress_every,
            stabilizer=stabilizer,
            source_blob=source_blob,
        )

    def _iter_processed_frames(
        self,
        frames: Iterator[np.ndarray],
        tracker: ByteTracker,
        method: AnonymizationMethod,
        detect_interval: int,
        blur_new: bool,
        draw_tracks: bool,
        progress_every: int,
        obfuscation: ObfuscationParams | None = None,
    ) -> Iterator[np.ndarray]:
        frame_idx = 0
        last_detect_ms = 0.0
        tracks: list[dict] = []

        for frame_bgr in frames:
            run_detect = (frame_idx % detect_interval) == 0
            if run_detect:
                t0 = time.perf_counter()
                detections = self.face_detector.detect(frame_bgr)
                last_detect_ms = (time.perf_counter() - t0) * 1000.0
                tracks = tracker.update(detections)
            else:
                tracks = tracker.predict_only()

            if blur_new:
                tracks_for_anonymize = tracks
            else:
                tracks_for_anonymize = [
                    track for track in tracks if track.get("state") == "Tracked"
                ]

            anonymized = self.face_anonymizer.anonymize(
                frame_bgr,
                tracks_for_anonymize,
                method=method,
                params=obfuscation,
            )

            if draw_tracks:
                output_frame = tracker.draw(
                    anonymized,
                    tracks,
                    confirmed_only=not blur_new,
                )
            else:
                output_frame = anonymized

            frame_idx += 1
            if progress_every > 0 and frame_idx % progress_every == 0:
                print(
                    f"Processed {frame_idx} frames "
                    f"| detect: {last_detect_ms:5.1f} ms "
                    f"| active tracks: {len(tracks)}"
                )

            yield output_frame

    def _iter_swapped_frames(
        self,
        frames: Iterator[np.ndarray],
        aligner: FaceAligner,
        progress_every: int,
        stabilizer: FaceSwapStabilizer | None = None,
        source_blob: np.ndarray | None = None,
    ) -> Iterator[np.ndarray]:
        # Face swap needs fresh 5-point landmarks every frame, so the detector runs
        # on each frame (the Kalman tracker only predicts bboxes, not landmarks).
        frame_idx = 0
        last_face_count = 0

        for frame_bgr in frames:
            t0 = time.perf_counter()

            if stabilizer is not None:
                # Stabilizer tracks faces across frames and smooths the landmarks.
                output_frame = stabilizer.process(frame_bgr)
                last_face_count = stabilizer.last_face_count
            else:
                detections = self.face_detector.detect(frame_bgr)
                last_face_count = len(detections)
                if detections:
                    aligned_faces = aligner.align(detections)
                    # VideoIO yields BGR; BlendSwap operates in RGB.
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    swapped_rgb = self.face_anonymizer.swap_face(
                        frame_rgb, aligned_faces, source_blob
                    )
                    output_frame = cv2.cvtColor(swapped_rgb, cv2.COLOR_RGB2BGR)
                else:
                    output_frame = frame_bgr

            frame_ms = (time.perf_counter() - t0) * 1000.0
            frame_idx += 1
            if progress_every > 0 and frame_idx % progress_every == 0:
                print(
                    f"Processed {frame_idx} frames "
                    f"| frame: {frame_ms:6.1f} ms "
                    f"| faces: {last_face_count}"
                )

            yield output_frame

    def _iter_offline_rendered_frames(
        self,
        frames: Iterator[np.ndarray],
        offline: OfflineFaceSwapStabilizer,
        progress_every: int,
    ) -> Iterator[np.ndarray]:
        # Pass 2: re-iterate the same frames and swap using the smoothed plan.
        for frame_idx, frame_bgr in enumerate(frames):
            t0 = time.perf_counter()
            output_frame = offline.render(frame_idx, frame_bgr)
            frame_ms = (time.perf_counter() - t0) * 1000.0
            if progress_every > 0 and (frame_idx + 1) % progress_every == 0:
                print(
                    f"  pass 2: {frame_idx + 1} frames "
                    f"| frame: {frame_ms:6.1f} ms "
                    f"| faces: {offline.last_face_count}"
                )
            yield output_frame

    # --------------------------------------------------------------------- #
    # Audio pipeline (audio only).                                           #
    # --------------------------------------------------------------------- #
    def _build_audio_pipeline(
        self,
        context: _RunContext,
        stack: contextlib.ExitStack,
    ) -> tuple[str | None, float]:
        """Resolve the ``(audio_source, audio_start_sec)`` passed to ``write_frames``.

        Branches purely on :attr:`AudioOptions.mode`:

        * ``MUTE``      -> ``(None, 0.0)``: silent output.
        * ``ORIGINAL``  -> ``(input_path, start)``: mux the source track untouched.
        * ``ANONYMIZE`` -> extract the ``[start, end]`` window, run the
          :class:`VoiceAnonymizer`, write a temp WAV (cleaned up via ``stack``).
        """
        mode = context.audio.mode
        if mode is AudioMode.MUTE:
            return None, 0.0
        if mode is AudioMode.ORIGINAL:
            return self._prepare_original_audio(context)
        return self._prepare_anonymized_audio(context, stack)

    def _prepare_original_audio(self, context: _RunContext) -> tuple[str, float]:
        # Seek the muxed audio to match a trimmed (start_sec) render.
        start = float(context.start_sec) if context.start_sec else 0.0
        return str(context.input_path), start

    def _prepare_anonymized_audio(
        self,
        context: _RunContext,
        stack: contextlib.ExitStack,
    ) -> tuple[str | None, float]:
        if self.voice_anonymizer is None:
            raise RuntimeError(
                "AudioOptions(anonymize_voice=True) requires "
                "VideoAnonymization(..., voice_anonymizer=VoiceAnonymizer(...))."
            )

        if not self.video_io.has_audio(str(context.input_path)):
            print(
                "Warning: voice anonymization requested but the input has no audio "
                "stream; the output will be silent."
            )
            return None, 0.0

        waveform, sample_rate = self.video_io.extract_audio(
            str(context.input_path),
            start_sec=context.start_sec,
            end_sec=context.end_sec,
        )
        # Resolve the reference identity for CONVERT once, here, mirroring how the
        # visual path resolves the swap source — the converter is shared, so the
        # matching set travels as a value, not as mutable instance state. Only build
        # it when a custom reference is supplied; otherwise the converter falls back to
        # its cached default (and the DSP methods ignore it entirely).
        matching_set = None
        reference_path = context.audio.voice_reference_path
        if reference_path is not None and self.voice_anonymizer.voice_converter is not None:
            matching_set = self.voice_anonymizer.voice_converter.prepare_reference(
                reference_path
            )
        processed = self.voice_anonymizer.process(
            waveform,
            sample_rate,
            method=context.audio.voice_method,
            params=context.audio.voice,
            matching_set=matching_set,
        )
        work_dir = stack.enter_context(tempfile.TemporaryDirectory())
        wav_path = Path(work_dir) / "voice_anonymized.wav"
        self.video_io.write_audio(processed, sample_rate, str(wav_path))
        # The extracted window is already trimmed, so no further audio seek is needed.
        return str(wav_path), 0.0

    # --------------------------------------------------------------------- #
    # Write + report.                                                        #
    # --------------------------------------------------------------------- #
    def _write_result(
        self,
        context: _RunContext,
        frames: Iterator[np.ndarray],
        audio_source: str | None,
        audio_start_sec: float,
    ) -> VideoAnonymizationResult:
        # Frames are lazy, so the timer spans the actual visual processing too.
        t0 = time.perf_counter()
        output_meta = self.video_io.write_frames(
            frames=frames,
            output_path=str(context.output_path),
            fps=context.output_fps,
            codec=context.codec,
            audio_source=audio_source,
            audio_start_sec=audio_start_sec,
        )
        elapsed = time.perf_counter() - t0
        throughput_fps = output_meta.frame_count / elapsed if elapsed > 0 else 0.0

        print("Done.")
        print(
            "Output metadata: "
            f"{output_meta.width}x{output_meta.height}, "
            f"{output_meta.fps:.3f} FPS, "
            f"{output_meta.frame_count} frames, "
            f"{output_meta.duration_sec:.2f} sec"
        )
        print(
            f"Elapsed: {elapsed:.2f} sec "
            f"| Pipeline throughput: {throughput_fps:.2f} FPS"
        )

        return VideoAnonymizationResult(
            output_path=context.output_path,
            output_metadata=output_meta,
            elapsed_sec=elapsed,
            throughput_fps=throughput_fps,
        )

    # --------------------------------------------------------------------- #
    # Small shared helpers.                                                  #
    # --------------------------------------------------------------------- #
    def _resolve_face_aligner(self) -> FaceAligner:
        # BlendSwap expects the FFHQ template at 256x256; FaceSwapper re-derives the
        # crop from the original 5-point landmarks, so any valid alignment works.
        if self.face_aligner is None:
            self.face_aligner = FaceAligner(output_size=(256, 256), mode=AlignMode.FFHQ)
        return self.face_aligner

    @staticmethod
    def _resolve_output_path(
        input_path: Path,
        output_path: Path | None,
        *,
        swap: bool,
    ) -> Path:
        if output_path is not None:
            return output_path
        prefix = "swapped" if swap else "anonymized"
        return Path("outputs") / f"{prefix}_{input_path.stem}.mp4"

    @staticmethod
    def _resolve_output_fps(source_fps: float, target_fps: int | None) -> float:
        if target_fps is None:
            return float(source_fps)
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")
        if target_fps >= source_fps:
            return float(source_fps)
        return float(target_fps)

    @staticmethod
    def _build_fresh_tracker(tracker: ByteTracker) -> ByteTracker:
        # Tracking state should be clean for each video run.
        return ByteTracker(
            high_thresh=tracker.high_thresh,
            low_thresh=tracker.low_thresh,
            max_lost=tracker.max_lost,
            min_hits=tracker.min_hits,
            iou_thresh=tracker.iou_thresh,
            iou_thresh_low=tracker.iou_thresh_low,
            gate_mahal=tracker.gate_mahal,
        )

    @staticmethod
    def _voice_label(voice_method: VoiceAnonymizationMethod | str) -> str:
        if isinstance(voice_method, VoiceAnonymizationMethod):
            return voice_method.value
        return str(voice_method)
