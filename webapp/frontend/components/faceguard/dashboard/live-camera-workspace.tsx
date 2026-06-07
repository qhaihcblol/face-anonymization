'use client'

import { useMemo, useState } from 'react'

import { CameraControlsCard } from '@/components/faceguard/dashboard/live/camera-controls-card'
import { CameraPreviewCard } from '@/components/faceguard/dashboard/live/camera-preview-card'
import {
  defaultLiveFilterForm,
  summarizeLiveFilter,
  type LiveFilterForm,
} from '@/lib/videos/options'
import { useLiveCamera } from '@/lib/videos/use-live-camera'

/**
 * Orchestrates the Live Camera workspace. Owns the privacy-filter form and the
 * preview overlay toggles, and delegates the camera device lifecycle to
 * {@link useLiveCamera}. The two cards stay purely presentational — mirroring the
 * Upload Video panel's structure.
 */
export function LiveCameraWorkspace() {
  const camera = useLiveCamera()

  const [filter, setFilter] = useState<LiveFilterForm>(defaultLiveFilterForm)
  const [showBoundingBox, setShowBoundingBox] = useState(true)
  const [showConfidence, setShowConfidence] = useState(true)

  const filterSummary = useMemo(() => summarizeLiveFilter(filter), [filter])

  const updateFilter = (patch: Partial<LiveFilterForm>) => {
    setFilter((previous) => ({ ...previous, ...patch }))
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[1.45fr_1fr]">
      <CameraPreviewCard
        videoRef={camera.videoRef}
        isStreaming={camera.isStreaming}
        isRecording={camera.isRecording}
        fps={camera.fps}
        latencyMs={camera.latencyMs}
        statusMessage={camera.statusMessage}
        errorMessage={camera.errorMessage}
        showBoundingBox={showBoundingBox}
        showConfidence={showConfidence}
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
