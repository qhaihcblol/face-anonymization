/**
 * Wire protocol for the live-camera WebSocket, mirroring the backend in
 * `webapp/backend/app/api/endpoints/live.py` and `app/schemas/live.py`.
 *
 * Per connection: the client sends a JSON config message (on connect and on every
 * change) plus binary JPEG frames; the server replies with a processed binary JPEG
 * frame immediately followed by its {@link LiveFrameMeta}. Every binary frame yields
 * exactly one terminal response (`frame` or `frame_error`), which the transport uses
 * for backpressure.
 */

import type { LiveFilterForm } from '@/lib/videos/options'

/** One tracked face in a processed frame, in the frame's pixel coordinates. */
export type LiveFaceBox = {
  id: number
  bbox: [number, number, number, number] // [x1, y1, x2, y2]
  score: number
}

/** Metadata accompanying each processed frame (the JSON that follows the binary). */
export type LiveFrameMeta = {
  type: 'frame'
  width: number
  height: number
  detected: boolean
  detect_ms: number
  process_ms: number
  faces: LiveFaceBox[]
}

/** Every text message the server can send. */
export type LiveServerMessage =
  | { type: 'ready' }
  | { type: 'config_ack' }
  | { type: 'error'; detail: string }
  | { type: 'frame_error'; detail: string }
  | LiveFrameMeta

/**
 * Build the JSON config message from the UI filter form. The backend ignores
 * unknown keys and re-validates ranges, so out-of-range input is rejected server-side
 * (surfaced as an `error` message) rather than silently applied.
 *
 * `draw_boxes` stays `false`: the overlay is drawn client-side from each frame's
 * `faces` so it can be toggled without reconfiguring the stream.
 */
export function buildLiveConfigMessage(filter: LiveFilterForm) {
  return {
    type: 'config',
    visual_method: filter.method,
    blur_strength: Number(filter.blurStrength) || 31,
    pixelation_level: Number(filter.pixelationLevel) || 16,
    mask_color: filter.maskColor,
    draw_boxes: false,
  }
}
