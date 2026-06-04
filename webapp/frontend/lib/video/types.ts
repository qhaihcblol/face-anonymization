export type AnonymizeMethod = 'none' | 'blur' | 'pixelate' | 'mask' | 'blackout'

export type VoiceMethod =
  | 'none'
  | 'mcadams'
  | 'pitch'
  | 'formant'
  | 'pitch_formant'
  | 'convert'

// Voice anonymization controls. Sent alongside the face/image request so audio is
// processed in the same run by the backend's VoiceAnonymizer.
export type VoiceParams = {
  voice_method: VoiceMethod
  pitch_steps?: number
  formant_shift?: number
  mcadams_alpha?: number
}

export type VideoMetadata = {
  fps: number
  frame_count: number
  duration_sec: number
  width: number
  height: number
}

export type VideoUploadResponse = {
  video_id: string
  filename: string
  size_bytes: number
  metadata: VideoMetadata
  original_video_url: string
  anonymized_video_url: string
}

export type AnonymizeRequest = {
  method: AnonymizeMethod
  target_fps?: number
  start_sec?: number
  end_sec?: number
  draw_tracks: boolean
  // Method-specific appearance controls: blur kernel size, pixelation block count,
  // and the solid mask fill color [r,g,b] (RGB; the backend converts to BGR).
  blur_strength?: number
  pixelation_level?: number
  mask_color?: [number, number, number]
} & VoiceParams

export type AnonymizeResponse = {
  video_id: string
  method: AnonymizeMethod
  voice_method: VoiceMethod
  target_fps: number | null
  start_sec: number | null
  end_sec: number | null
  output_video_url: string
  output_metadata: VideoMetadata
  elapsed_sec: number
  throughput_fps: number
}

export type FaceSwapRequest = {
  target_fps?: number
  start_sec?: number
  end_sec?: number
  // Temporal stabilization is always on (better output), so it is not exposed in the
  // UI; the field stays optional for callers that still want to set it explicitly.
  stabilize?: boolean
} & VoiceParams

export type FaceSwapResponse = {
  video_id: string
  method: 'swap'
  voice_method: VoiceMethod
  target_fps: number | null
  start_sec: number | null
  end_sec: number | null
  stabilize: boolean
  output_video_url: string
  output_metadata: VideoMetadata
  elapsed_sec: number
  throughput_fps: number
}
