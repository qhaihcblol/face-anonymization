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
  /** Curated source-face key to swap onto every face (SWAP only); null = default. */
  swap_source_key: string | null
  // Audio (voice)
  keep_audio: boolean
  anonymize_voice: boolean
  voice_method: VoiceMethod
  mcadams_alpha: number
  pitch_steps: number
  formant_shift: number
  /** Curated source-voice key to convert toward (CONVERT only); null = default. */
  voice_reference_key: string | null
  // Processing range
  target_fps: number | null
  start_sec: number | null
  end_sec: number | null
}

/** Which catalog a {@link SourceAsset} belongs to. */
export type SourceAssetKind = 'face' | 'voice'

/**
 * One curated, selectable asset (a face image or a voice clip), from
 * `GET /api/sources/faces` or `/voices`. `key` is the stable id passed back into a
 * `VideoEditCreate`; `url` is a short-lived presigned link used only to preview it.
 */
export type SourceAsset = {
  kind: SourceAssetKind
  key: string
  name: string
  gender: string | null
  url: string
  content_type: string | null
  size_bytes: number
  expires_in: number
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

/** Request body for `POST /api/videos` — confirm a direct upload finished.
 * Duration/dimensions are probed client-side and optional (a browser that can't
 * read them still completes the upload). */
export type VideoUploadComplete = {
  storage_key: string
  original_filename: string
  content_type: string | null
  duration_sec: number | null
  width: number | null
  height: number | null
}
