'use client'

import { useEffect, useMemo, useRef, useState } from 'react'

import { CameraControlsCard } from '@/components/faceguard/dashboard/live/camera-controls-card'
import { CameraPreviewCard } from '@/components/faceguard/dashboard/live/camera-preview-card'
import {
  defaultLiveFilterForm,
  summarizeLiveFilter,
  type LiveFilterForm,
} from '@/lib/videos/options'
import { useLiveCamera } from '@/lib/videos/use-live-camera'
import { useLiveProcessing } from '@/lib/videos/use-live-processing'

/**
 * Orchestrates the Live Camera workspace. Owns the privacy-filter form and the
 * preview overlay toggles; delegates the camera device lifecycle to
 * {@link useLiveCamera} and the real-time anonymization stream to
 * {@link useLiveProcessing}. Protection runs only while streaming with a filter
 * selected, so a raw preview never leaves the device. The two cards stay purely
 * presentational — mirroring the Upload Video panel's structure.
 */
export function LiveCameraWorkspace() {
  const outputCanvasRef = useRef<HTMLCanvasElement | null>(null)
  // Set during render below once protection state is known; the camera hook reads
  // it lazily at record-start to choose the anonymized canvas over the raw feed.
  // A ref breaks the camera → processing → camera value cycle.
  const recordProcessedRef = useRef(false)

  const camera = useLiveCamera({
    processedCanvasRef: outputCanvasRef,
    recordProcessedRef,
  })

  const [filter, setFilter] = useState<LiveFilterForm>(defaultLiveFilterForm)
  const [showBoundingBox, setShowBoundingBox] = useState(true)
  const [showConfidence, setShowConfidence] = useState(true)

  // A filter of "none" is a pure local preview: keep the socket closed so nothing
  // leaves the device until the user actually picks a protection filter.
  const protectionEnabled = camera.isStreaming && filter.method !== 'none'

  const processing = useLiveProcessing({
    enabled: protectionEnabled,
    sourceVideoRef: camera.videoRef,
    outputCanvasRef,
    filter,
    showBoundingBox,
    showConfidence,
  })

  // Protection is on AND at least one anonymized frame has painted — i.e. the
  // canvas is showing protected output and is safe to record.
  const showProcessed = protectionEnabled && processing.hasFrame

  // Mirror into the ref the camera hook reads at record-start time.
  useEffect(() => {
    recordProcessedRef.current = showProcessed
  }, [showProcessed])

  const filterSummary = useMemo(() => summarizeLiveFilter(filter), [filter])

  const updateFilter = (patch: Partial<LiveFilterForm>) => {
    setFilter((previous) => ({ ...previous, ...patch }))
  }

  // Surface the processed-stream numbers when protection is live, else the raw
  // camera monitor's.
  const isLive = processing.status === 'live'
  const fps = isLive ? processing.stats.fps : camera.fps
  const latencyMs = isLive
    ? processing.stats.processMs !== null
      ? Math.round(processing.stats.processMs)
      : null
    : camera.latencyMs

  return (
    <div className="grid gap-6 xl:grid-cols-[1.45fr_1fr]">
      <CameraPreviewCard
        videoRef={camera.videoRef}
        outputCanvasRef={outputCanvasRef}
        isStreaming={camera.isStreaming}
        isRecording={camera.isRecording}
        showProcessed={showProcessed}
        processingStatus={processing.status}
        processingError={processing.error}
        fps={fps}
        latencyMs={latencyMs}
        statusMessage={camera.statusMessage}
        errorMessage={camera.errorMessage}
        filterSummary={filterSummary}
        onStartStream={() => void camera.startStream()}
        onStopStream={camera.stopStream}
        onStartRecording={camera.startRecording}
        onStopRecording={camera.stopRecording}
      />

      <CameraControlsCard
        cameraDevices={camera.cameraDevices}
        selectedCamera={camera.selectedCamera}
        onSelectCamera={camera.setSelectedCamera}
        resolution={camera.resolution}
        onSelectResolution={camera.setResolution}
        onApplySettings={() => void camera.startStream()}
        filter={filter}
        onFilterChange={updateFilter}
        showBoundingBox={showBoundingBox}
        onToggleBoundingBox={setShowBoundingBox}
        showConfidence={showConfidence}
        onToggleConfidence={setShowConfidence}
        recordedClips={camera.recordedClips}
        onRemoveClip={camera.removeClip}
      />
    </div>
  )
}
