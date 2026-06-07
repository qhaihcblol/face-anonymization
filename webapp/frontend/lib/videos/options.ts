/**
 * UI metadata + the form model for the Upload Video panel.
 *
 * This module is the single source of truth for the selectable options, their
 * descriptions, and the mapping from the form state to the backend
 * `VideoEditCreate` payload. Components stay presentational by importing from here
 * instead of hard-coding choices, so adding a method/knob is a one-line change.
 */

import type {
  VideoEditCreate,
  VideoEditStatus,
  VisualMethod,
  VoiceMethod,
} from '@/lib/videos/types'

export type SelectOption<TValue extends string> = {
  value: TValue
  label: string
  description: string
}

// --- Visual (face) --------------------------------------------------------- //

export const visualMethodOptions: SelectOption<VisualMethod>[] = [
  { value: 'blur', label: 'Blur', description: 'Soften each detected face with a Gaussian blur.' },
  { value: 'pixelate', label: 'Pixelate', description: 'Replace each face with coarse mosaic blocks.' },
  { value: 'mask', label: 'Mask', description: 'Cover the precise face region with a solid colour.' },
  { value: 'blackout', label: 'Blackout', description: 'Fill the detected face region with solid black.' },
  { value: 'swap', label: 'Face Swap', description: 'Replace every face with the bundled source identity (BlendSwap).' },
  { value: 'none', label: 'None', description: 'Leave faces untouched — only audio / range settings run.' },
]

// --- Audio (voice) --------------------------------------------------------- //

/** A friendlier framing of the three backend audio states. */
export type AudioMode = 'keep' | 'anonymize' | 'remove'

export const audioModeOptions: SelectOption<AudioMode>[] = [
  { value: 'keep', label: 'Keep original', description: 'Pass the original audio track through unchanged.' },
  { value: 'anonymize', label: 'Anonymize voice', description: 'Disguise the speaker’s voice while keeping speech intelligible.' },
  { value: 'remove', label: 'Remove audio', description: 'Strip the audio track from the output entirely.' },
]

/** Voice methods exclude 'none' — "keep original" is expressed via {@link AudioMode}. */
export type DspVoiceMethod = Exclude<VoiceMethod, 'none'>

export const voiceMethodOptions: SelectOption<DspVoiceMethod>[] = [
  { value: 'mcadams', label: 'McAdams (recommended)', description: 'Formant warp that preserves pitch and timing.' },
  { value: 'pitch', label: 'Pitch shift', description: 'Shift the pitch up or down (semitones).' },
  { value: 'formant', label: 'Formant shift', description: 'Reshape the vocal-tract formants.' },
  { value: 'pitch_formant', label: 'Pitch + Formant', description: 'Combine pitch and formant shifting.' },
  { value: 'convert', label: 'Voice conversion (AI)', description: 'kNN-VC toward the bundled reference voice (falls back to DSP if unavailable).' },
]

export function usesPitch(method: DspVoiceMethod): boolean {
  return method === 'pitch' || method === 'pitch_formant'
}

export function usesFormant(method: DspVoiceMethod): boolean {
  return method === 'formant' || method === 'pitch_formant'
}

// --- Status display (reusable by History later) ---------------------------- //

export const editStatusLabel: Record<VideoEditStatus, string> = {
  pending: 'Queued',
  processing: 'Processing',
  completed: 'Completed',
  failed: 'Failed',
}

export const editStatusBadgeClass: Record<VideoEditStatus, string> = {
  pending: 'bg-amber-500/20 text-amber-700 dark:text-amber-300',
  processing: 'bg-cyan-500/20 text-cyan-700 dark:text-cyan-100',
  completed: 'bg-emerald-500/20 text-emerald-700 dark:text-emerald-300',
  failed: 'bg-rose-500/20 text-rose-700 dark:text-rose-300',
}

// --- Form model + form -> API mapping -------------------------------------- //

/** Controlled state for the protection-settings form. Numbers stay as input
 * strings so the fields can be cleared while typing; they are parsed on submit. */
export type ProtectionForm = {
  visualMethod: VisualMethod
  blurStrength: string
  pixelationLevel: string
  maskColor: string
  drawBoxes: boolean
  audioMode: AudioMode
  voiceMethod: DspVoiceMethod
  mcadamsAlpha: string
  pitchSteps: string
  formantShift: string
  targetFps: string
  startSec: string
  endSec: string
}

export const defaultProtectionForm: ProtectionForm = {
  visualMethod: 'blur',
  blurStrength: '31',
  pixelationLevel: '16',
  maskColor: '#A0A0A0',
  drawBoxes: false,
  audioMode: 'keep',
  voiceMethod: 'mcadams',
  mcadamsAlpha: '0.8',
  pitchSteps: '-4',
  formantShift: '1.2',
  targetFps: '',
  startSec: '',
  endSec: '',
}

function parseNumber(input: string, fallback: number): number {
  const value = Number(input.trim())
  return Number.isFinite(value) ? value : fallback
}

function parseOptional(input: string): number | null {
  const trimmed = input.trim()
  if (!trimmed) {
    return null
  }
  const value = Number(trimmed)
  return Number.isFinite(value) ? value : null
}

/** Map {@link AudioMode} (+ chosen voice method) onto the backend audio fields. */
function audioParams(
  mode: AudioMode,
  voiceMethod: DspVoiceMethod,
): Pick<VideoEditCreate, 'keep_audio' | 'anonymize_voice' | 'voice_method'> {
  if (mode === 'remove') {
    return { keep_audio: false, anonymize_voice: false, voice_method: 'none' }
  }
  if (mode === 'anonymize') {
    return { keep_audio: true, anonymize_voice: true, voice_method: voiceMethod }
  }
  return { keep_audio: true, anonymize_voice: false, voice_method: 'none' }
}

/**
 * Validate the form and build the `VideoEditCreate` payload. Returns the payload
 * on success, or a human-readable `error` (mirrors the backend's own checks so the
 * user gets feedback before the request is made). The backend re-validates.
 */
export function buildEditPayload(
  form: ProtectionForm,
): { payload: VideoEditCreate; error: null } | { payload: null; error: string } {
  const startSec = parseOptional(form.startSec)
  const endSec = parseOptional(form.endSec)
  if (startSec !== null && startSec < 0) {
    return { payload: null, error: 'Start time must be zero or positive.' }
  }
  if (endSec !== null && endSec <= 0) {
    return { payload: null, error: 'End time must be greater than 0.' }
  }
  if (startSec !== null && endSec !== null && endSec <= startSec) {
    return { payload: null, error: 'End time must be greater than start time.' }
  }

  const payload: VideoEditCreate = {
    visual_method: form.visualMethod,
    blur_strength: parseNumber(form.blurStrength, 31),
    pixelation_level: parseNumber(form.pixelationLevel, 16),
    mask_color: form.maskColor,
    draw_boxes: form.drawBoxes,
    ...audioParams(form.audioMode, form.voiceMethod),
    mcadams_alpha: parseNumber(form.mcadamsAlpha, 0.8),
    pitch_steps: parseNumber(form.pitchSteps, -4),
    formant_shift: parseNumber(form.formantShift, 1.2),
    target_fps: parseOptional(form.targetFps),
    start_sec: startSec,
    end_sec: endSec,
  }

  return { payload, error: null }
}
