'use client'

/**
 * Streams the live camera through the backend anonymizer and paints the result.
 *
 *   raw <video>  ──capture JPEG──▶  WebSocket  ──processed JPEG──▶  <canvas>
 *
 * Connects only while `enabled` (the workspace gates this on "streaming AND a filter
 * is selected", so nothing leaves the device for a raw preview). The frame pump is
 * throttled to `targetFps` and runs with backpressure — at most one frame is in
 * flight, so a slow round-trip drops frames instead of piling up latency. The
 * processed frame and its detected-face overlay are drawn onto the same canvas, so
 * boxes stay aligned regardless of CSS scaling.
 */

import { useEffect, useRef, useState, type RefObject } from 'react'

import {
  buildLiveConfigMessage,
  type LiveFaceBox,
  type LiveFrameMeta,
  type LiveServerMessage,
} from '@/lib/videos/live-protocol'
import type { LiveFilterForm } from '@/lib/videos/options'

export type LiveProcessingStatus = 'idle' | 'connecting' | 'live' | 'error'

export type LiveProcessingStats = {
  fps: number
  processMs: number | null
  detectMs: number | null
  faces: number
}

const TICKET_ENDPOINT = '/api/live/ticket'
// Upper bound on how many frames/sec we *send*. This is only a ceiling: the real
// rate self-regulates to the server's render speed via `maxInFlight` backpressure
// (the server renders one frame per connection serially, so a single stream tops
// out near 1000/process_ms ≈ 20-33 FPS at ~30-50ms/frame). Keep this at or above
// that ceiling so the client never throttles below what the engine can deliver;
// 30 is the sweet spot for smooth-to-the-eye output without spending GPU/bandwidth
// chasing frames past the refresh humans notice.
const DEFAULT_TARGET_FPS = 30
const DEFAULT_SEND_MAX_WIDTH = 640
const DEFAULT_JPEG_QUALITY = 0.7
// How many frames may be "in flight" (sent, awaiting their response) at once.
// 1 = strict request/response: throughput collapses to 1000/round-trip, so on a
// high-latency link (deployed backend reached over the internet, ~190ms round-trip)
// it caps at ~5 FPS even though the GPU renders in ~40ms. Allowing several
// overlapping frames hides that latency: throughput -> min(targetFps, depth/RTT,
// server rate). At ~190ms round-trip, depth 8 covers the 30 FPS ceiling
// (30 * 0.19s ≈ 6 frames need to be in flight), with headroom for RTT jitter so a
// network hiccup doesn't stall the stream. The targetFps throttle still caps the
// send rate, so depth is an upper bound — in-flight settles near rate*RTT and the
// preview lags real time by ~that many frames (≈190ms here), not the full depth.
// The server renders serially in order, so binary/JSON pairs stay interleaved.
const DEFAULT_MAX_IN_FLIGHT = 2

const EMPTY_STATS: LiveProcessingStats = {
  fps: 0,
  processMs: null,
  detectMs: null,
  faces: 0,
}

type Params = {
  enabled: boolean
  sourceVideoRef: RefObject<HTMLVideoElement | null>
  outputCanvasRef: RefObject<HTMLCanvasElement | null>
  filter: LiveFilterForm
  showBoundingBox: boolean
  showConfidence: boolean
  targetFps?: number
  sendMaxWidth?: number
  jpegQuality?: number
  maxInFlight?: number
}

export type LiveProcessing = {
  status: LiveProcessingStatus
  error: string | null
  /** True once at least one processed frame has been painted (show the canvas). */
  hasFrame: boolean
  stats: LiveProcessingStats
}

function drawFaceBoxes(
  ctx: CanvasRenderingContext2D,
  faces: LiveFaceBox[],
  showConfidence: boolean,
): void {
  ctx.lineWidth = 2
  ctx.strokeStyle = 'rgba(34, 211, 238, 0.95)' // cyan-400
  ctx.font = '14px ui-sans-serif, system-ui, sans-serif'
  ctx.textBaseline = 'alphabetic'

  for (const face of faces) {
    const [x1, y1, x2, y2] = face.bbox
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

    if (showConfidence) {
      const label = `#${face.id}  ${(face.score * 100).toFixed(0)}%`
      const width = ctx.measureText(label).width
      ctx.fillStyle = 'rgba(2, 6, 23, 0.7)' // slate-950/70
      ctx.fillRect(x1, Math.max(y1 - 18, 0), width + 8, 18)
      ctx.fillStyle = 'rgba(207, 250, 254, 1)' // cyan-100
      ctx.fillText(label, x1 + 4, Math.max(y1 - 4, 14))
    }
  }
}

async function paintFrame(
  canvas: HTMLCanvasElement,
  blob: Blob,
  meta: LiveFrameMeta,
  showBoundingBox: boolean,
  showConfidence: boolean,
): Promise<void> {
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return
  }

  let bitmap: ImageBitmap
  try {
    bitmap = await createImageBitmap(blob)
  } catch {
    return
  }

  if (canvas.width !== meta.width) {
    canvas.width = meta.width
  }
  if (canvas.height !== meta.height) {
    canvas.height = meta.height
  }

  ctx.drawImage(bitmap, 0, 0, meta.width, meta.height)
  bitmap.close()

  if (showBoundingBox && meta.faces.length > 0) {
    drawFaceBoxes(ctx, meta.faces, showConfidence)
  }
}

export function useLiveProcessing({
  enabled,
  sourceVideoRef,
  outputCanvasRef,
  filter,
  showBoundingBox,
  showConfidence,
  targetFps = DEFAULT_TARGET_FPS,
  sendMaxWidth = DEFAULT_SEND_MAX_WIDTH,
  jpegQuality = DEFAULT_JPEG_QUALITY,
  maxInFlight = DEFAULT_MAX_IN_FLIGHT,
}: Params): LiveProcessing {
  const [status, setStatus] = useState<LiveProcessingStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [hasFrame, setHasFrame] = useState(false)
  const [stats, setStats] = useState<LiveProcessingStats>(EMPTY_STATS)

  // Imperative state kept off the render path.
  const socketRef = useRef<WebSocket | null>(null)
  // Number of frames sent but not yet answered (frame / frame_error). Gates the
  // pump so at most `maxInFlight` overlap — this is the depth that hides link latency.
  const inFlightRef = useRef(0)
  const pendingFrameRef = useRef<Blob | null>(null)
  const rafRef = useRef<number | null>(null)
  const lastSentRef = useRef(0)
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const framesRef = useRef(0)
  const lastMetaRef = useRef<LiveFrameMeta | null>(null)

  // Latest values read by long-lived handlers without re-subscribing.
  const overlayRef = useRef({ box: showBoundingBox, confidence: showConfidence })
  const optsRef = useRef({ targetFps, sendMaxWidth, jpegQuality, maxInFlight })

  useEffect(() => {
    overlayRef.current = { box: showBoundingBox, confidence: showConfidence }
  }, [showBoundingBox, showConfidence])

  useEffect(() => {
    optsRef.current = { targetFps, sendMaxWidth, jpegQuality, maxInFlight }
  }, [targetFps, sendMaxWidth, jpegQuality, maxInFlight])

  // Push filter changes to the live socket without reconnecting.
  useEffect(() => {
    if (status !== 'live') {
      return
    }
    const socket = socketRef.current
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(buildLiveConfigMessage(filter)))
    }
  }, [filter, status])

  // Connect / tear down the whole pipeline whenever `enabled` flips.
  useEffect(() => {
    if (!enabled) {
      return
    }

    let disposed = false
    setStatus('connecting')
    setError(null)
    setHasFrame(false)
    setStats(EMPTY_STATS)
    framesRef.current = 0
    inFlightRef.current = 0
    pendingFrameRef.current = null
    lastMetaRef.current = null

    const pump = (now: number) => {
      rafRef.current = requestAnimationFrame(pump)

      const { targetFps: fps, sendMaxWidth: maxWidth, jpegQuality: quality, maxInFlight: depth } =
        optsRef.current
      const socket = socketRef.current
      const video = sourceVideoRef.current
      if (
        inFlightRef.current >= depth ||
        !socket ||
        socket.readyState !== WebSocket.OPEN ||
        !video ||
        video.readyState < 2 ||
        video.videoWidth === 0
      ) {
        return
      }

      if (now - lastSentRef.current < 1000 / fps) {
        return
      }

      const capture =
        captureCanvasRef.current ??
        (captureCanvasRef.current = document.createElement('canvas'))
      const scale = Math.min(1, maxWidth / video.videoWidth)
      const width = Math.round(video.videoWidth * scale)
      const height = Math.round(video.videoHeight * scale)
      if (capture.width !== width) {
        capture.width = width
      }
      if (capture.height !== height) {
        capture.height = height
      }

      const captureCtx = capture.getContext('2d')
      if (!captureCtx) {
        return
      }
      captureCtx.drawImage(video, 0, 0, width, height)

      lastSentRef.current = now
      inFlightRef.current += 1 // reserve a slot; released on send-failure or response
      capture.toBlob(
        (blob) => {
          const active = socketRef.current
          if (blob && active && active.readyState === WebSocket.OPEN) {
            active.send(blob)
          } else {
            inFlightRef.current = Math.max(0, inFlightRef.current - 1)
          }
        },
        'image/jpeg',
        quality,
      )
    }

    const handleMessage = (event: MessageEvent) => {
      // Binary = a processed frame; it always precedes its JSON metadata.
      if (typeof event.data !== 'string') {
        pendingFrameRef.current = event.data as Blob
        return
      }

      let message: LiveServerMessage
      try {
        message = JSON.parse(event.data) as LiveServerMessage
      } catch {
        return
      }

      switch (message.type) {
        case 'ready':
          if (!disposed) {
            setStatus('live')
          }
          break
        case 'config_ack':
          break
        case 'error':
          if (!disposed) {
            setError(message.detail)
          }
          break
        case 'frame_error':
          inFlightRef.current = Math.max(0, inFlightRef.current - 1)
          break
        case 'frame': {
          inFlightRef.current = Math.max(0, inFlightRef.current - 1)
          framesRef.current += 1
          lastMetaRef.current = message
          if (!disposed) {
            setHasFrame(true)
          }
          const blob = pendingFrameRef.current
          pendingFrameRef.current = null
          const canvas = outputCanvasRef.current
          if (blob && canvas) {
            void paintFrame(
              canvas,
              blob,
              message,
              overlayRef.current.box,
              overlayRef.current.confidence,
            )
          }
          break
        }
      }
    }

    const statsTimer = window.setInterval(() => {
      if (disposed) {
        return
      }
      const meta = lastMetaRef.current
      setStats({
        fps: framesRef.current,
        processMs: meta ? meta.process_ms : null,
        detectMs: meta ? meta.detect_ms : null,
        faces: meta ? meta.faces.length : 0,
      })
      framesRef.current = 0
    }, 1000)

    const connect = async () => {
      let url: string
      try {
        const response = await fetch(TICKET_ENDPOINT, { method: 'GET' })
        if (!response.ok) {
          throw new Error('ticket')
        }
        const body = (await response.json()) as { url: string }
        url = body.url
      } catch {
        if (!disposed) {
          setStatus('error')
          setError('Could not start live protection. Please sign in and retry.')
        }
        return
      }

      if (disposed) {
        return
      }

      const socket = new WebSocket(url)
      socket.binaryType = 'blob'
      socketRef.current = socket

      socket.onopen = () => {
        socket.send(JSON.stringify(buildLiveConfigMessage(filter)))
        rafRef.current = requestAnimationFrame(pump)
      }
      socket.onmessage = handleMessage
      socket.onerror = () => {
        if (!disposed) {
          setError('Live connection error.')
        }
      }
      socket.onclose = () => {
        if (!disposed) {
          setStatus((current) => (current === 'connecting' ? 'error' : current))
        }
      }
    }

    void connect()

    return () => {
      disposed = true
      window.clearInterval(statsTimer)
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
      const socket = socketRef.current
      socketRef.current = null
      if (socket) {
        socket.onopen = null
        socket.onmessage = null
        socket.onerror = null
        socket.onclose = null
        if (
          socket.readyState === WebSocket.OPEN ||
          socket.readyState === WebSocket.CONNECTING
        ) {
          socket.close()
        }
      }
      inFlightRef.current = 0
      pendingFrameRef.current = null
      setStatus('idle')
      setHasFrame(false)
      setStats(EMPTY_STATS)
    }
    // `filter` is intentionally omitted: changes are pushed by the effect above
    // without reconnecting. The connect closure reads the initial value on open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, sourceVideoRef, outputCanvasRef])

  return { status, error, hasFrame, stats }
}
