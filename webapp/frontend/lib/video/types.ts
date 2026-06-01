export type AnonymizeMethod = 'none' | 'blur' | 'pixelate' | 'mask' | 'blackout'

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
}

export type AnonymizeResponse = {
  video_id: string
  method: AnonymizeMethod
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
  stabilize: boolean
}

export type FaceSwapResponse = {
  video_id: string
  method: 'swap'
  target_fps: number | null
  start_sec: number | null
  end_sec: number | null
  stabilize: boolean
  output_video_url: string
  output_metadata: VideoMetadata
  elapsed_sec: number
  throughput_fps: number
}
