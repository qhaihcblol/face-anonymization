'use client'

/**
 * Owns the live-camera device lifecycle so the workspace components stay
 * declarative:
 *
 *   enumerate devices → open stream → (optionally) record → tear down cleanly
 *
 * The heavy imperative bits — `getUserMedia`, `MediaRecorder`, the FPS monitor and
 * all the ref/cleanup bookkeeping — live here behind a small, stable API. This
 * mirrors how the Upload panel leans on {@link useVideoAnonymization}.
 */

import { useEffect, useRef, useState, type RefObject } from 'react'

export type ResolutionPreset = '480p' | '720p' | '1080p'

export type CameraDevice = {
  id: string
  label: string
}

export type RecordedClip = {
  id: string
  url: string
  createdAt: string
  sizeMB: string
  durationSec: string
}

export type ActiveStreamProfile = {
  width?: number
  height?: number
  frameRate?: number
}

export const resolutionConfig: Record<
  ResolutionPreset,
  { width: number; height: number; label: string }
> = {
  '480p': { width: 854, height: 480, label: '480p (16:9)' },
  '720p': { width: 1280, height: 720, label: '720p (HD, 16:9)' },
  '1080p': { width: 1920, height: 1080, label: '1080p (Full HD, 16:9)' },
}

/** Everything the workspace needs to render and drive the camera. */
export type LiveCameraController = {
  videoRef: RefObject<HTMLVideoElement | null>
  cameraDevices: CameraDevice[]
  selectedCamera: string
  setSelectedCamera: (id: string) => void
  resolution: ResolutionPreset
  setResolution: (resolution: ResolutionPreset) => void
  isStreaming: boolean
  isRecording: boolean
  fps: number
  latencyMs: number | null
  activeProfile: ActiveStreamProfile | null
  statusMessage: string
  errorMessage: string | null
  recordedClips: RecordedClip[]
  startStream: () => Promise<void>
  stopStream: () => void
  startRecording: () => void
  stopRecording: () => void
  removeClip: (clipId: string) => void
}

function getRecorderMimeType() {
  if (typeof MediaRecorder === 'undefined') {
    return undefined
  }

  const candidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ]

  return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType))
}

function mapCameraDevices(devices: MediaDeviceInfo[]): CameraDevice[] {
  return devices
    .filter(
      (device) =>
        device.kind === 'videoinput' && device.deviceId.trim().length > 0,
    )
    .map((device, index) => ({
      id: device.deviceId,
      label: device.label || `Camera ${index + 1}`,
    }))
}

export function useLiveCamera(): LiveCameraController {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const startedAtRef = useRef<number>(0)
  const fpsIntervalRef = useRef<number | null>(null)
  const rafRef = useRef<number | null>(null)
  const frameCounterRef = useRef(0)
  const clipsRef = useRef<RecordedClip[]>([])
  const isMountedRef = useRef(true)

  const [cameraDevices, setCameraDevices] = useState<CameraDevice[]>([])
  const [selectedCamera, setSelectedCamera] = useState('default')
  const [resolution, setResolution] = useState<ResolutionPreset>('720p')
  const [isStreaming, setIsStreaming] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [fps, setFps] = useState(0)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [activeProfile, setActiveProfile] = useState<ActiveStreamProfile | null>(null)
  const [statusMessage, setStatusMessage] = useState('Idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [recordedClips, setRecordedClips] = useState<RecordedClip[]>([])

  useEffect(() => {
    clipsRef.current = recordedClips
  }, [recordedClips])

  const stopFpsMonitor = () => {
    if (rafRef.current !== null) {
      window.cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }

    if (fpsIntervalRef.current !== null) {
      window.clearInterval(fpsIntervalRef.current)
      fpsIntervalRef.current = null
    }

    frameCounterRef.current = 0
    setFps(0)
    setLatencyMs(null)
  }

  const startFpsMonitor = () => {
    stopFpsMonitor()

    const tick = () => {
      frameCounterRef.current += 1
      rafRef.current = window.requestAnimationFrame(tick)
    }

    rafRef.current = window.requestAnimationFrame(tick)
    fpsIntervalRef.current = window.setInterval(() => {
      const currentFps = frameCounterRef.current
      setFps(currentFps)
      setLatencyMs(currentFps > 0 ? Math.round(1000 / currentFps) : null)
      frameCounterRef.current = 0
    }, 1000)
  }

  const refreshCameraDevices = async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) {
        return
      }

      const devices = await navigator.mediaDevices.enumerateDevices()
      const cameras = mapCameraDevices(devices)

      setCameraDevices(cameras)

      if (
        selectedCamera !== 'default' &&
        !cameras.some((camera) => camera.id === selectedCamera)
      ) {
        setSelectedCamera('default')
      }
    } catch {
      if (isMountedRef.current) {
        setErrorMessage('Unable to list camera devices.')
      }
    }
  }

  const stopRecording = () => {
    const recorder = recorderRef.current
    if (!recorder || recorder.state === 'inactive') {
      return
    }

    recorder.stop()
  }

  const stopStream = (updateStatus = true) => {
    stopRecording()
    stopFpsMonitor()

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    if (isMountedRef.current) {
      setIsStreaming(false)
      setActiveProfile(null)
    }

    if (updateStatus && isMountedRef.current) {
      setStatusMessage('Camera stream stopped')
    }
  }

  const startStream = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMessage('Your browser does not support camera streaming.')
      return
    }

    setErrorMessage(null)

    try {
      stopStream(false)

      const { width, height, label } = resolutionConfig[resolution]
      const preferredVideoConstraints: MediaTrackConstraints = {
        width: { ideal: width },
        height: { ideal: height },
        aspectRatio: { ideal: 16 / 9 },
        ...(selectedCamera !== 'default'
          ? { deviceId: { exact: selectedCamera } }
          : {}),
      }

      const fallbackVideoConstraints: MediaTrackConstraints = {
        ...(selectedCamera !== 'default'
          ? { deviceId: { exact: selectedCamera } }
          : {}),
      }

      const stream = await navigator.mediaDevices
        .getUserMedia({
          video: preferredVideoConstraints,
          audio: false,
        })
        .catch(async () =>
          navigator.mediaDevices.getUserMedia({
            video: fallbackVideoConstraints,
            audio: false,
          }),
        )

      streamRef.current = stream

      const [videoTrack] = stream.getVideoTracks()
      const trackSettings = videoTrack?.getSettings()
      setActiveProfile({
        width: trackSettings?.width,
        height: trackSettings?.height,
        frameRate: trackSettings?.frameRate,
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      await refreshCameraDevices()
      startFpsMonitor()
      if (isMountedRef.current) {
        setIsStreaming(true)
        const resolvedWidth = trackSettings?.width
        const resolvedHeight = trackSettings?.height
        if (resolvedWidth && resolvedHeight) {
          setStatusMessage(`Streaming at ${label}`)
        } else {
          setStatusMessage(`Streaming with ${label} preference`)
        }
      }
    } catch {
      if (isMountedRef.current) {
        setErrorMessage(
          'Failed to access camera stream. Please allow camera permission and retry.',
        )
        setStatusMessage('Camera unavailable')
      }
    }
  }

  const startRecording = () => {
    if (!streamRef.current) {
      setErrorMessage('Start the camera stream before recording.')
      return
    }

    if (typeof MediaRecorder === 'undefined') {
      setErrorMessage('MediaRecorder is not supported in this browser.')
      return
    }

    const mimeType = getRecorderMimeType()
    const recorder = mimeType
      ? new MediaRecorder(streamRef.current, { mimeType })
      : new MediaRecorder(streamRef.current)

    chunksRef.current = []
    startedAtRef.current = Date.now()

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data)
      }
    }

    recorder.onstop = () => {
      const clipBlob = new Blob(chunksRef.current, {
        type: recorder.mimeType || 'video/webm',
      })

      if (clipBlob.size === 0) {
        if (isMountedRef.current) {
          setIsRecording(false)
        }
        return
      }

      const clipUrl = URL.createObjectURL(clipBlob)
      const clip: RecordedClip = {
        id: `${Date.now()}`,
        url: clipUrl,
        createdAt: new Date().toLocaleString(),
        sizeMB: (clipBlob.size / (1024 * 1024)).toFixed(2),
        durationSec: ((Date.now() - startedAtRef.current) / 1000).toFixed(1),
      }

      if (isMountedRef.current) {
        setRecordedClips((prev) => [clip, ...prev])
        setIsRecording(false)
        setStatusMessage('Recording completed')
      } else {
        URL.revokeObjectURL(clipUrl)
      }
    }

    recorder.onerror = () => {
      if (isMountedRef.current) {
        setErrorMessage('Recording failed. Please retry.')
        setIsRecording(false)
      }
    }

    recorderRef.current = recorder
    recorder.start(500)
    setIsRecording(true)
    setErrorMessage(null)
    setStatusMessage('Recording in progress')
  }

  const removeClip = (clipId: string) => {
    setRecordedClips((prev) => {
      const targetClip = prev.find((clip) => clip.id === clipId)
      if (targetClip) {
        URL.revokeObjectURL(targetClip.url)
      }
      return prev.filter((clip) => clip.id !== clipId)
    })
  }

  useEffect(() => {
    isMountedRef.current = true

    const refreshDevicesOnMount = async () => {
      try {
        if (!navigator.mediaDevices?.enumerateDevices) {
          return
        }

        const devices = await navigator.mediaDevices.enumerateDevices()
        const cameras = mapCameraDevices(devices)

        setCameraDevices(cameras)
        setSelectedCamera((current) =>
          current !== 'default' &&
          !cameras.some((camera) => camera.id === current)
            ? 'default'
            : current,
        )
      } catch {
        if (isMountedRef.current) {
          setErrorMessage('Unable to list camera devices.')
        }
      }
    }

    void refreshDevicesOnMount()

    const handleDeviceChange = () => {
      void refreshDevicesOnMount()
    }

    navigator.mediaDevices?.addEventListener?.('devicechange', handleDeviceChange)
    const videoEl = videoRef.current // copy ref for cleanup
    return () => {
      isMountedRef.current = false

      navigator.mediaDevices?.removeEventListener?.(
        'devicechange',
        handleDeviceChange,
      )

      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current)
      }

      if (fpsIntervalRef.current !== null) {
        window.clearInterval(fpsIntervalRef.current)
      }

      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        recorderRef.current.stop()
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
      }

      if (videoEl) {
        videoEl.srcObject = null
      }

      clipsRef.current.forEach((clip) => URL.revokeObjectURL(clip.url))
    }
  }, [])

  return {
    videoRef,
    cameraDevices,
    selectedCamera,
    setSelectedCamera,
    resolution,
    setResolution,
    isStreaming,
    isRecording,
    fps,
    latencyMs,
    activeProfile,
    statusMessage,
    errorMessage,
    recordedClips,
    startStream,
    stopStream: () => stopStream(),
    startRecording,
    stopRecording,
    removeClip,
  }
}
