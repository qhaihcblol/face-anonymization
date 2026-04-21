'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Circle,
  Clapperboard,
  Download,
  Monitor,
  RefreshCcw,
  Trash2,
  Video,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'

type ResolutionPreset = '480p' | '720p' | '1080p'
type FilterPreset = 'none' | 'blur' | 'pixelate' | 'mask' | 'blackout'

type CameraDevice = {
  id: string
  label: string
}

type RecordedClip = {
  id: string
  url: string
  createdAt: string
  sizeMB: string
  durationSec: string
}

type ActiveStreamProfile = {
  width?: number
  height?: number
  frameRate?: number
}

const resolutionConfig: Record<
  ResolutionPreset,
  { width: number; height: number; label: string }
> = {
  '480p': { width: 854, height: 480, label: '480p (16:9)' },
  '720p': { width: 1280, height: 720, label: '720p (HD, 16:9)' },
  '1080p': { width: 1920, height: 1080, label: '1080p (Full HD, 16:9)' },
}

const filterOptions: Array<{ value: FilterPreset; label: string }> = [
  { value: 'none', label: 'None' },
  { value: 'blur', label: 'Blur' },
  { value: 'pixelate', label: 'Pixelate' },
  { value: 'mask', label: 'Mask' },
  { value: 'blackout', label: 'Blackout' },
]

const streamFilterMap: Record<FilterPreset, string> = {
  none: 'none',
  blur: 'none',
  pixelate: 'none',
  mask: 'none',
  blackout: 'none',
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

export function LiveCameraWorkspace() {
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
  const [filter, setFilter] = useState<FilterPreset>('none')
  const [showBoundingBox, setShowBoundingBox] = useState(true)
  const [showConfidence, setShowConfidence] = useState(true)
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

  const filterStyle = useMemo(
    () => ({ filter: streamFilterMap[filter] }),
    [filter],
  )

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
          setStatusMessage(
            `Streaming at ${label}`,
          )
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
    const videoEl = videoRef.current // copy ref
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

  return (
    <div className="grid gap-6 xl:grid-cols-[1.45fr_1fr]">
      <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle className="text-xl tracking-tight">Webcam Stream</CardTitle>
              <CardDescription>
                Real-time input stream for privacy protection and identity masking.
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Badge className="bg-cyan-500/20 text-cyan-700 dark:text-cyan-100">
                {isStreaming ? 'Live' : 'Offline'}
              </Badge>
              <Badge className="bg-cyan-500/15 text-cyan-700 dark:text-cyan-100">
                FPS {fps}
              </Badge>
              <Badge className="bg-cyan-500/15 text-cyan-700 dark:text-cyan-100">
                Latency {latencyMs !== null ? `${latencyMs} ms` : '--'}
              </Badge>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-5">
          <div className="relative aspect-video overflow-hidden rounded-xl border border-cyan-300/25 bg-slate-900/90">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="h-full w-full object-contain"
              style={filterStyle}
            />

            {!isStreaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/75 text-cyan-100">
                <Monitor className="size-10 text-cyan-300/80" />
                <p className="text-sm text-cyan-100/90">
                  Camera is idle. Press Start Stream to initialize webcam.
                </p>
              </div>
            )}

            {showBoundingBox && isStreaming && (
              <>
                <div className="pointer-events-none absolute top-[18%] left-[18%] h-24 w-20 rounded-md border-2 border-cyan-300/90 shadow-[0_0_20px_-6px_rgba(34,211,238,0.95)]" />
                <div className="pointer-events-none absolute top-[34%] right-[24%] h-28 w-24 rounded-md border-2 border-cyan-300/90 shadow-[0_0_20px_-6px_rgba(34,211,238,0.95)]" />
                {showConfidence && (
                  <p className="pointer-events-none absolute right-3 bottom-3 rounded-md border border-cyan-300/35 bg-slate-950/70 px-2 py-1 text-xs text-cyan-100">
                    Face confidence: 98.1%
                  </p>
                )}
              </>
            )}

            {isRecording && (
              <div className="absolute top-3 left-3 flex items-center gap-2 rounded-full border border-rose-400/35 bg-rose-500/15 px-3 py-1 text-xs font-medium text-rose-200">
                <Circle className="size-2.5 fill-rose-300 text-rose-300 animate-pulse" />
                Recording
              </div>
            )}
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            <Button
              onClick={startStream}
              className="bg-cyan-400 text-cyan-950 hover:bg-cyan-300"
            >
              <Video className="size-4" />
              Start Stream
            </Button>
            <Button
              onClick={() => stopStream()}
              variant="outline"
              className="border-cyan-300/35 bg-cyan-500/5 hover:bg-cyan-500/15"
            >
              <RefreshCcw className="size-4" />
              Stop Stream
            </Button>
            {isRecording ? (
              <Button
                onClick={stopRecording}
                variant="outline"
                className="border-rose-300/40 bg-rose-500/10 text-rose-200 hover:bg-rose-500/20"
              >
                <Circle className="size-4 fill-rose-300 text-rose-300" />
                Stop Recording
              </Button>
            ) : (
              <Button
                onClick={startRecording}
                disabled={!isStreaming}
                className="bg-emerald-400 text-emerald-950 hover:bg-emerald-300"
              >
                <Clapperboard className="size-4" />
                Record
              </Button>
            )}
          </div>

          <div className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-4 py-2 text-sm text-muted-foreground">
            <p className="font-medium text-foreground">Status: {statusMessage}</p>
            {errorMessage && <p className="mt-1 text-rose-300">{errorMessage}</p>}
          </div>
        </CardContent>
      </Card>

      <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-xl tracking-tight">Control Panel</CardTitle>
          <CardDescription>
            Tune camera and privacy controls before sending stream to detection.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6">
          <section className="space-y-4">
            <h3 className="text-xs font-semibold tracking-[0.14em] text-cyan-700 uppercase dark:text-cyan-200">
              Settings
            </h3>

            <div className="space-y-2">
              <Label htmlFor="camera-select">Camera</Label>
              <Select value={selectedCamera} onValueChange={setSelectedCamera}>
                <SelectTrigger id="camera-select" className="w-full border-cyan-300/35">
                  <SelectValue placeholder="Select camera" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Default Camera</SelectItem>
                  {cameraDevices.map((camera) => (
                    <SelectItem key={camera.id} value={camera.id}>
                      {camera.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="resolution-select">Capture Resolution</Label>
              <Select
                value={resolution}
                onValueChange={(value) => setResolution(value as ResolutionPreset)}
              >
                <SelectTrigger
                  id="resolution-select"
                  className="w-full border-cyan-300/35"
                >
                  <SelectValue placeholder="Select resolution" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(resolutionConfig).map(([value, config]) => (
                    <SelectItem key={value} value={value}>
                      {config.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Button
              onClick={startStream}
              variant="outline"
              className="w-full border-cyan-300/35 bg-cyan-500/5 hover:bg-cyan-500/15"
            >
              Apply Settings
            </Button>
          </section>

          <section className="space-y-4">
            <h3 className="text-xs font-semibold tracking-[0.14em] text-cyan-700 uppercase dark:text-cyan-200">
              Filter & Overlay
            </h3>

            <div className="space-y-2">
              <Label htmlFor="filter-select">Privacy Filter</Label>
              <Select
                value={filter}
                onValueChange={(value) => setFilter(value as FilterPreset)}
              >
                <SelectTrigger id="filter-select" className="w-full border-cyan-300/35">
                  <SelectValue placeholder="Select filter" />
                </SelectTrigger>
                <SelectContent>
                  {filterOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-3 py-2">
              <Label htmlFor="bounding-box" className="text-sm">
                Show Bounding Box
              </Label>
              <Switch
                id="bounding-box"
                checked={showBoundingBox}
                onCheckedChange={setShowBoundingBox}
              />
            </div>

            <div className="flex items-center justify-between rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-3 py-2">
              <Label htmlFor="confidence-badge" className="text-sm">
                Confidence Tag
              </Label>
              <Switch
                id="confidence-badge"
                checked={showConfidence}
                onCheckedChange={setShowConfidence}
              />
            </div>
          </section>

          <section className="space-y-3">
            <h3 className="text-xs font-semibold tracking-[0.14em] text-cyan-700 uppercase dark:text-cyan-200">
              Output
            </h3>

            {recordedClips.length === 0 ? (
              <div className="rounded-lg border border-dashed border-cyan-300/30 px-4 py-6 text-center text-sm text-muted-foreground">
                No recordings yet. Start stream and press Record.
              </div>
            ) : (
              <div className="space-y-2">
                {recordedClips.map((clip) => (
                  <article
                    key={clip.id}
                    className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 p-3"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="text-sm font-semibold text-foreground">
                          Recorded Clip
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {clip.createdAt}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {clip.durationSec}s • {clip.sizeMB} MB
                        </p>
                      </div>
                      <div className="flex items-center gap-1">
                        <Button
                          asChild
                          size="icon-sm"
                          variant="ghost"
                          className="text-cyan-700 hover:bg-cyan-500/20 dark:text-cyan-300"
                        >
                          <a href={clip.url} download={`recording-${clip.id}.webm`}>
                            <Download className="size-4" />
                          </a>
                        </Button>
                        <Button
                          size="icon-sm"
                          variant="ghost"
                          className="text-rose-300 hover:bg-rose-500/20"
                          onClick={() => removeClip(clip.id)}
                        >
                          <Trash2 className="size-4" />
                        </Button>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>

          <div className="rounded-lg border border-cyan-300/20 bg-linear-to-r from-cyan-500/16 via-cyan-400/10 to-transparent px-4 py-3">
            <p className="text-sm font-semibold text-foreground">
              Active Pipeline: Webcam → Face Detector → Privacy Filter → Output
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Mode: {filterOptions.find((item) => item.value === filter)?.label} /
              Current {fps} FPS • Latency {latencyMs !== null ? `${latencyMs} ms` : '--'}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Capture:{' '}
              {activeProfile?.width && activeProfile?.height
                ? `${activeProfile.width}x${activeProfile.height}`
                : 'Not active'}
              {' '}• Detector input: 640x640
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
