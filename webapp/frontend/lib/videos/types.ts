/**
 * Wire types for the video API, mirroring the backend Pydantic schemas in
 * `webapp/backend/app/schemas/video.py`. Kept in sync by hand and exchanged with
 * FastAPI through the same-origin `/api/videos` proxy.
 */

export type VisualMethod =
  | 'none'
  | 'blur'
  | 'pixelate'
  | 'mask'
  | 'blackout'
  | 'swap'

export type VoiceMethod =
  | 'none'
  | 'mcadams'
  | 'pitch'
  | 'formant'
  | 'pitch_formant'
  | 'convert'

export type VideoEditStatus = 'pending' | 'processing' | 'completed' | 'failed'

/** Request body for `POST /api/videos/{id}/edits` — one anonymization run. */
export type VideoEditCreate = {
  // Visual (face)
  visual_method: VisualMethod
  blur_strength: number
  pixelation_level: number
  mask_color: string // #RRGGBB
  draw_boxes: boolean
  // Audio (voice)
  keep_audio: boolean
  anonymize_voice: boolean
  voice_method: VoiceMethod
  mcadams_alpha: number
  pitch_steps: number
  formant_shift: number
  // Processing range
  target_fps: number | null
  start_sec: number | null
  end_sec: number | null
}

/** A source video the user uploaded. */
export type VideoPublic = {
  id: number
  original_filename: string
  content_type: string | null
  size_bytes: number | null
  duration_sec: number | null
  width: number | null
  height: number | null
  created_at: string
}

/** One anonymization run on a video, with its status + result. */
export type VideoEditPublic = {
  id: number
  video_id: number
  status: VideoEditStatus
  params: Record<string, unknown> | null
  error_message: string | null
  created_at: string
  updated_at: string
  completed_at: string | null
}

/** A short-lived presigned URL (e.g. an R2 GET link) for direct download/playback. */
export type PresignedUrlResponse = {
  url: string
  expires_in: number
}

/** Request body for `POST /api/videos/upload-url` — start a direct upload. */
export type VideoUploadInit = {
  filename: string
  content_type: string | null
  size_bytes: number | null
}

/** A presigned upload target: PUT the file straight to `upload_url`. */
export type VideoUploadTicket = {
  storage_key: string
  upload_url: string
  method: string
  headers: Record<string, string>
  expires_in: number
}

/** Request body for `POST /api/videos` — confirm a direct upload finished. */
export type VideoUploadComplete = {
  storage_key: string
  original_filename: string
  content_type: string | null
}
