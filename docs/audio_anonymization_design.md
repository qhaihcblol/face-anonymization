# Audio (Voice) Anonymization — Design Doc

Status: **proposal / not yet implemented**
Scope: add speaker-identity anonymization to the audio track, as the audio counterpart
of the existing face-swap video pipeline.

---

## 1. Goals & non-goals

**Goals**
- Hide the **speaker identity / voiceprint** so the talker cannot be re-identified by
  automatic speaker verification (ASV) or by a human who knows the voice.
- **Preserve the linguistic content** (what is said) — keep speech intelligible.
- **Preserve duration / timing** so the audio stays in lip-sync with the swapped video.
- **Consistent pseudo-identity**: the same real speaker maps to one stable fake voice
  for the whole clip (mirrors "one `source_img.png` per video" in face swap).
- Tiered strength (light DSP → neural voice conversion → full re-synthesis), selectable
  like the existing `AnonymizationMethod` enum.

**Non-goals (initially)**
- Real-time on the live-cam path (Tier 1 DSP can do it; neural tiers are offline-first).
- Perfect naturalness / studio quality.
- Emotion/prosody transfer (only the strongest re-synthesis tier intentionally drops it).

---

## 2. Threat model — what "anonymized" means here

We defeat three linkage attacks, in increasing order of importance:

1. **Automatic speaker verification (ASV)** — an attacker has a voiceprint (x-vector /
   ECAPA embedding) of the target and tries to match it. Success metric: the ASV
   **equal-error-rate (EER) rises toward 50%** and embedding cosine similarity between
   original and anonymized drops below the verification threshold.
2. **Human recognition** — someone who knows the voice. Requires audible timbre change
   (formant/pitch shift or conversion), not just metadata stripping.
3. **Trivial inversion** — the transform must not be cheaply reversible (a fixed
   pitch-shift is invertible; McAdams and neural VC are effectively not). This is the
   audio analog of `FaceAnonymizer._destroy`.

Utility constraint: **WER (word error rate)** of an ASR run on the anonymized audio must
stay close to the original — content must survive.

---

## 3. Design principles (mirror the face pipeline)

| Face pipeline | Audio pipeline analog |
|---|---|
| detect faces (RetinaFace) | VAD + (optional) **diarization** — find speech, who speaks when |
| IoU track → track_id | cluster segments → stable **speaker_id** across the clip |
| swap → one source identity | **voice conversion** → one fixed pseudo-voice per speaker |
| parser mask (face-only, skip occluders) | **source separation** — convert vocals only, keep music/SFX |
| OneEuro / 2-pass landmark smoothing | consistent pseudo-speaker embedding (no drift) |
| `_destroy` (non-recoverable) | choose non-invertible method (McAdams / VC) |
| eval: flicker / sharpness / temporal-std | eval: **ASV-EER↑ (privacy)** + **WER (utility)** + MOS |
| `source_img.png` | `reference_voice.wav` (Tier 2 target identity) |
| `FaceAligner` templates | fixed sample rate (16 kHz internal), framing config |

**Hard constraint unique to a *video* system:** keep duration constant (no time-stretch)
so audio never desyncs from the mouth. This rules out WSOLA/tempo methods and favors
McAdams + frame-synchronous VC (both preserve length).

---

## 4. Architecture & data flow

New package mirroring `ai_core/face_anonymization/`:

```
ai_core/audio_anonymization/
  audio_io.py          # AudioIO: extract / load / resample / save / mux via ffmpeg
  audio_anonymizer.py  # AudioAnonymizer + AudioAnonymizationMethod enum; McAdams + pitch DSP
  voice_converter.py   # VoiceConverter (kNN-VC, ONNX)            [Tier 2]
  diarizer.py          # SpeakerDiarizer (VAD + clustering)       [Tier 2/3, optional]
  separator.py         # SourceSeparator (vocals vs background)   [Tier 3, optional]
  voice_eval.py        # ASV-EER + WER evaluation harness
  reference_voice.wav  # target pseudo-voice (analog of source_img.png) [Tier 2]
```

Each ONNX-backed class follows the existing house style already used by `FaceSwapper`,
`FaceParser`, `FaceRestorer`:
- `importlib`-lazy `onnxruntime`, `_resolve_model_path` via `huggingface_hub`,
  `_create_session` with CUDA→CPU provider resolution, `intra_op_num_threads`.
- `from __future__ import annotations`, dataclasses, internal validators.
- One canonical internal representation: **mono float32 PCM at 16 kHz** (resample on
  the edges), the audio analog of "pipeline works in RGB".

### End-to-end flow (offline / upload path)

```
input.mp4
  ├─ video → existing face-swap pipeline → frames (BGR)
  └─ audio → AudioIO.extract(segment[start,end])           # wav, 16k mono
              → [SourceSeparator] vocals / background        # Tier 3
              → [SpeakerDiarizer] segments + speaker_id       # Tier 2/3
              → AudioAnonymizer.anonymize(method, per speaker)
                  ├─ MCADAMS / PITCH  (DSP)                  # Tier 1
                  └─ VOICE_CONVERSION (VoiceConverter)        # Tier 2
              → [remux vocals + background]                   # Tier 3
              → anon_audio.wav
VideoIO.write_frames(frames, ..., audio_source=anon_audio.wav, audio_start_sec=0)
```

**Plumbing reuse:** `VideoIO.write_frames(audio_source=...)` already muxes any audio
file via ffmpeg. We simply pass the **anonymized wav** instead of the source. Because we
extract the `[start,end]` segment up front, `audio_start_sec=0` for the muxed track.

---

## 5. Methods (tiers)

### Tier 1 — McAdams coefficient transform (DSP) · analog of BLUR
The VoicePrivacy Challenge DSP baseline. Per short frame:
1. LPC analysis (order ≈ `sr/1000 + 2`), get the pole roots of the LPC polynomial.
2. For each complex pole with `Im > 0`, raise its **angle** to the McAdams power `α`
   (`new_angle = angle**α`, angle ∈ (0, π)), keep magnitude; rebuild conjugate pairs.
3. Reconstruct LPC from the modified poles.
4. Inverse-filter the frame with the *original* LPC to get the excitation/residual,
   then re-filter the residual with the *modified* LPC; overlap-add.

Properties: shifts **formants** (timbre/identity) while leaving **pitch and duration**
untouched → lip-safe, non-invertible, language-independent. Pure `numpy/scipy`, no model,
fast enough for real-time. Pick `α` **per speaker** (constant across the clip for
consistency; randomized in e.g. `[0.5, 0.9]` across runs for unlinkability).

Optional companion DSP: small **formant + pitch shift** (vocal-tract-length perturbation)
for extra timbre change.

### Tier 2 — Neural voice conversion · analog of SWAP
Disentangle *content* from *identity*, resynthesize with a fixed target voice.
Recommended: **kNN-VC** (simple, strong, single reference voice = `reference_voice.wav`):
1. Extract self-supervised features (WavLM-Large) for the source utterance.
2. Build a "matching set" of features from the **reference voice**.
3. For each source frame, average its k nearest neighbors (cosine) in the matching set →
   converted feature sequence (frame-synchronous → duration preserved).
4. **HiFi-GAN** (prematched) vocoder → waveform.

Alternative: **VoicePrivacy B1** (ASR-bottleneck content + x-vector replaced by a
*pseudo-speaker* = average of N far x-vectors from a pool + NSF-HiFiGAN). Stronger,
ships with a standard eval recipe.

Identity consistency: one fixed reference (kNN-VC) or one pseudo-speaker embedding
(B1) per `speaker_id` for the whole clip — the audio analog of preparing the source
identity once.

### Tier 3 — ASR → TTS re-synthesis · analog of BLACKOUT/MASK
Transcribe (ASR) then speak with a synthetic TTS voice. Strongest anonymity (removes all
speaker + emotion cues) but heaviest, drops prosody, and inherits ASR errors. Defer.

### Auxiliary (Tier 2/3)
- **Diarization** → distinct pseudo-voice per speaker (reuse the track_id idea).
- **Source separation** (Demucs) → anonymize vocals only, keep music/SFX (parser-mask
  analog). Both heavier (torch) and gated behind a flag; pipeline falls back to
  whole-track processing when absent (same fallback discipline as parser→ellipse).

---

## 6. Integration surface

### Orchestrator (`VideoAnonymization`)
Add to `anonymize_video_with_model` (and the bbox path):
- `audio_method: str = "original"` — one of `silent | original | mcadams | pitch |
  voice_conversion`.
- Keep `keep_audio` for back-compat: `keep_audio=False` ⇒ `silent`; `True` +
  `audio_method="original"` ⇒ current behavior (mux source); other values ⇒ anonymize.

Internally: build anon wav before/around frame writing, pass it to
`write_frames(audio_source=...)`. (Frame generation and audio anon are independent → can
overlap, but keep sequential first for simplicity.)

### Webapp
- `video.py` schema: add `audio_method` field; thread through `FaceSwapService`.
- `AudioAnonymizer`/`VoiceConverter` built **lazily** like the swap engine
  (`run_in_threadpool`, shared `process_lock`), so startup stays fast and torch/ONNX
  audio deps load only when used.

### CLI
`test_face_swap_video_pipeline.py`: add `--audio-method`, `--reference-voice`,
`--no-diarize`, `--no-separate`.

---

## 7. Evaluation harness (`voice_eval.py`) — analog of the flicker eval

- **Privacy (ASV):** ECAPA-TDNN / x-vector embeddings (cosine).
  - `linkability`: cosine(original, anonymized) of the same utterance → should drop.
  - `EER`: enroll original, trial anonymized → should climb toward 50%.
- **Utility (ASR):** Whisper / wav2vec2 WER of anonymized vs reference transcript →
  should stay near the original WER.
- **Naturalness:** proxy MOS (DNSMOS / NISQA) if available; else informal listening.
- **Sanity (DSP, model-free):** duration identical, F0 contour preserved (pitch
  unchanged for McAdams), measurable formant shift.

---

## 8. Dependencies & model sources

- Already present: `ffmpeg`/`ffprobe`, `numpy`, `scipy`, `onnxruntime`, `huggingface-hub`.
- Tier 1: **none new** (pure DSP).
- Tier 2: WavLM + HiFi-GAN (kNN-VC) or NSF-HiFiGAN + x-vector — ONNX from HF hub,
  downloaded lazily like BlendSwap/GFPGAN/BiSeNet.
- Eval: ECAPA-TDNN + an ASR model (ONNX or torch), download-on-demand.
- Optional aux: `pyannote.audio` (diarization), `demucs` (separation) — torch, gated.

---

## 9. Risks & edge cases

- **No audio stream / silent video** → skip gracefully, emit a note (like
  `_source_has_audio` already does).
- **Multi-speaker without diarization** → all voices converted to one pseudo-voice
  (acceptable default; diarization makes them distinct).
- **Background music** → without separation it gets distorted by VC; McAdams degrades it
  less. Separation flag mitigates.
- **Duration drift** → forbid any tempo change; assert `len(anon) == len(src)` (±1 frame).
- **Sample-rate / channels** → normalize to 16 kHz mono internally, restore on mux.
- **Inversion resistance** → randomize McAdams `α` per run; never store the parameter
  with the output (mirrors the unseeded-noise rule in `_destroy`).
- **Lip-sync** → muxing the equal-length anon wav at `audio_start_sec=0` keeps sync.

---

## 10. Phased roadmap

- **Phase 0 — scaffolding (no model).** `AudioIO` (extract/resample/save; reuse ffmpeg),
  `AudioAnonymizationMethod` enum, wire `audio_method` through the orchestrator + CLI +
  webapp, `voice_eval.py` skeleton. Fully testable with synthetic wavs.
- **Phase 1 — Tier 1 DSP.** McAdams (+ optional pitch/formant). Verifiable offline with
  pure DSP metrics (duration, F0, formant shift). **Ship-able milestone.**
- **Phase 2 — real eval.** ECAPA + Whisper harness; quantify EER↑ / WER on a small set.
- **Phase 3 — Tier 2 VC.** kNN-VC with a single `reference_voice.wav`, consistent
  pseudo-speaker; ONNX export + lazy download.
- **Phase 4 — aux.** Diarization (distinct voices) + source separation (keep background).
- **Phase 5 — optional.** ASR→TTS strongest tier; adversarial ASV-perturbation mode
  (fool the voiceprint while sounding unchanged to humans).

---

## 11. Verification strategy (given large models aren't in the dev env)

Same discipline as the face work (`verify-with-mocked-onnx`):
- Tier 1 (McAdams/pitch) and all I/O: **verified end-to-end with synthetic wavs**
  (sine/chirp/noise + ffmpeg-generated clips) — no model needed. Check duration equality,
  F0 preservation, formant shift, lossless mux/extract roundtrip.
- Tier 2 VC and eval models: **mock the ONNX sessions** (synthetic feature/waveform
  outputs) to test the conversion/eval plumbing; the user runs the real models and
  shares outputs for quality judgment.
```
