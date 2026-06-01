'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  CheckCircle2,
  FileVideo,
  FolderUp,
  Loader2,
  ShieldCheck,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import {
  anonymizeVideo,
  faceSwapVideo,
  uploadVideo,
  VideoApiError,
} from '@/lib/video/client'
import type { AnonymizeMethod } from '@/lib/video/types'

type FilterPreset = AnonymizeMethod | 'swap'

type ProcessingStatus = 'uploading' | 'processing' | null

type LastRun = {
  filter: FilterPreset
  targetFps: number
  startSec?: number
  endSec?: number
  drawTracks: boolean
  stabilize: boolean
  elapsedSec: number
}

type ProcessingParams = {
  targetFps: number
  startSec?: number
  endSec?: number
}

const filterOptions: Array<{ value: FilterPreset; label: string }> = [
  { value: 'none', label: 'None' },
  { value: 'blur', label: 'Blur' },
  { value: 'pixelate', label: 'Pixelate' },
  { value: 'mask', label: 'Mask' },
  { value: 'blackout', label: 'Blackout' },
  { value: 'swap', label: 'Face Swap' },
]

function getFilterLabel(value: FilterPreset): string {
  return filterOptions.find((option) => option.value === value)?.label ?? value
}

function parseOptionalSecond(value: string, fieldName: string): number | undefined {
  const trimmed = value.trim()
  if (trimmed.length === 0) {
    return undefined
  }

  const parsed = Number(trimmed)
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${fieldName} must be a number >= 0.`)
  }

  return parsed
}

function parseTargetFps(value: string): number {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error('Target FPS must be an integer greater than 0.')
  }

  return parsed
}

export function UploadVideoPanel() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [originalVideoUrl, setOriginalVideoUrl] = useState<string | null>(null)
  const [resultVideoUrl, setResultVideoUrl] = useState<string | null>(null)
  const [filter, setFilter] = useState<FilterPreset>('blur')
  const [targetFpsInput, setTargetFpsInput] = useState('30')
  const [startSecInput, setStartSecInput] = useState('')
  const [endSecInput, setEndSecInput] = useState('')
  const [showBoundingBox, setShowBoundingBox] = useState(true)
  const [stabilize, setStabilize] = useState(true)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus>(null)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [formError, setFormError] = useState<string | null>(null)
  const [lastRun, setLastRun] = useState<LastRun | null>(null)

  const previewUrlRef = useRef<string | null>(null)
  const progressIntervalRef = useRef<number | null>(null)
  // The uploaded video is reused across runs so changing only the filter does not
  // re-upload the file; reset whenever a new file is selected.
  const uploadedVideoIdRef = useRef<string | null>(null)
  // Cache-buster: the backend overwrites the same output path on every run.
  const runCounterRef = useRef(0)

  const isFaceSwap = filter === 'swap'

  const selectedFileSizeMB = useMemo(() => {
    if (!selectedFile) {
      return '0.00'
    }

    return (selectedFile.size / (1024 * 1024)).toFixed(2)
  }, [selectedFile])

  const queueProgress = selectedFile ? 100 : 0

  const stopProgressAnimation = () => {
    if (progressIntervalRef.current !== null) {
      window.clearInterval(progressIntervalRef.current)
      progressIntervalRef.current = null
    }
  }

  const startProgressAnimation = () => {
    stopProgressAnimation()
    // Processing is a single blocking request, so animate toward 92% and let the
    // completion handler snap to 100% when the backend responds.
    progressIntervalRef.current = window.setInterval(() => {
      setProcessingProgress((current) => Math.min(92, current + 4))
    }, 240)
  }

  const handleUploadChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null
    setSelectedFile(nextFile)

    stopProgressAnimation()
    uploadedVideoIdRef.current = null
    setIsProcessing(false)
    setProcessingStatus(null)
    setProcessingProgress(0)
    setResultVideoUrl(null)
    setLastRun(null)
    setFormError(null)

    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = null
    }

    if (!nextFile) {
      setOriginalVideoUrl(null)
      return
    }

    const nextPreviewUrl = URL.createObjectURL(nextFile)
    previewUrlRef.current = nextPreviewUrl
    setOriginalVideoUrl(nextPreviewUrl)
  }

  const buildProcessingParams = (): ProcessingParams => {
    const targetFps = parseTargetFps(targetFpsInput)
    const startSec = parseOptionalSecond(startSecInput, 'Start sec')
    const endSec = parseOptionalSecond(endSecInput, 'End sec')

    if (startSec !== undefined && endSec !== undefined && endSec <= startSec) {
      throw new Error('End sec must be greater than Start sec.')
    }

    return { targetFps, startSec, endSec }
  }

  const activateGuard = async () => {
    if (!selectedFile || isProcessing) {
      return
    }

    let params: ProcessingParams
    try {
      params = buildProcessingParams()
    } catch (error) {
      setFormError(
        error instanceof Error ? error.message : 'Invalid processing parameters.',
      )
      return
    }

    setFormError(null)
    setResultVideoUrl(null)
    setIsProcessing(true)
    setProcessingProgress(6)
    startProgressAnimation()

    try {
      setProcessingStatus('uploading')
      let videoId = uploadedVideoIdRef.current
      if (!videoId) {
        const uploaded = await uploadVideo(selectedFile)
        videoId = uploaded.video_id
        uploadedVideoIdRef.current = videoId
      }

      setProcessingStatus('processing')
      const result = isFaceSwap
        ? await faceSwapVideo(videoId, {
            target_fps: params.targetFps,
            start_sec: params.startSec,
            end_sec: params.endSec,
            stabilize,
          })
        : await anonymizeVideo(videoId, {
            method: filter,
            target_fps: params.targetFps,
            start_sec: params.startSec,
            end_sec: params.endSec,
            draw_tracks: showBoundingBox,
          })

      runCounterRef.current += 1
      // The backend overwrites the same output file each run, so cache-bust the URL.
      const separator = result.output_video_url.includes('?') ? '&' : '?'
      const bustedUrl = `${result.output_video_url}${separator}v=${runCounterRef.current}`
      const elapsedSec = result.elapsed_sec
      stopProgressAnimation()
      setProcessingProgress(100)
      setResultVideoUrl(bustedUrl)
      setLastRun({
        filter,
        targetFps: params.targetFps,
        startSec: params.startSec,
        endSec: params.endSec,
        drawTracks: showBoundingBox,
        stabilize,
        elapsedSec,
      })
    } catch (error) {
      stopProgressAnimation()
      setProcessingProgress(0)
      setFormError(
        error instanceof VideoApiError
          ? error.message
          : 'Processing failed. Please try again.',
      )
    } finally {
      setProcessingStatus(null)
      setIsProcessing(false)
    }
  }

  useEffect(() => {
    return () => {
      stopProgressAnimation()
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current)
      }
    }
  }, [])

  return (
    <div className="grid gap-6 xl:grid-cols-[1.05fr_0.85fr_1.05fr]">
      <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-xl tracking-tight">Original Upload</CardTitle>
          <CardDescription>
            Upload one video file and preview the raw source before processing.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <label
            htmlFor="video-upload"
            className="group grid cursor-pointer gap-3 rounded-xl border border-dashed border-cyan-300/35 bg-cyan-500/10 p-4 transition-colors hover:border-cyan-300/60 hover:bg-cyan-500/15 sm:grid-cols-[auto_1fr_auto] sm:items-center"
          >
            <FolderUp className="mx-auto size-7 text-cyan-700 transition-transform group-hover:scale-105 dark:text-cyan-300 sm:mx-0" />
            <div className="text-center sm:text-left">
              <p className="text-sm font-semibold text-foreground">
                Upload one video file
              </p>
              <p className="text-xs text-muted-foreground">
                Supports .mp4, .mov, .webm (up to 2GB)
              </p>
            </div>
            <Input
              id="video-upload"
              type="file"
              accept="video/*"
              className="sr-only"
              onChange={handleUploadChange}
            />
            <span className="inline-flex items-center justify-center rounded-md border border-cyan-300/45 bg-cyan-500/15 px-3 py-1.5 text-xs font-medium text-cyan-100 transition-colors group-hover:bg-cyan-500/25">
              Choose file
            </span>
          </label>

          <div className="relative aspect-video overflow-hidden rounded-xl border border-cyan-300/25 bg-slate-900/90">
            {originalVideoUrl ? (
              <video
                key={originalVideoUrl}
                src={originalVideoUrl}
                controls
                playsInline
                className="h-full w-full object-contain"
              />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/75 text-cyan-100">
                <FileVideo className="size-10 text-cyan-300/80" />
                <p className="text-sm text-cyan-100/90">
                  Upload a video to preview the original source.
                </p>
              </div>
            )}
          </div>

          <section className="rounded-xl border border-cyan-300/20 bg-cyan-500/10 p-4">
            <div className="mb-2 flex items-center justify-between gap-2">
              <p className="text-sm font-semibold text-foreground">Upload Status</p>
              <Badge className="bg-cyan-500/20 text-cyan-700 dark:text-cyan-100">
                {selectedFile ? '1 file selected' : 'No file'}
              </Badge>
            </div>
            <Progress value={queueProgress} className="h-2.5" />
            <p className="mt-2 text-xs text-muted-foreground">
              {selectedFile
                ? `${selectedFile.name} (${selectedFileSizeMB} MB)`
                : 'Select a video file to begin.'}
            </p>
          </section>
        </CardContent>
      </Card>

      <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-xl tracking-tight">Filter and Overlay</CardTitle>
          <CardDescription>
            Configure processing options and run protection for the selected video.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="space-y-2">
            <Label htmlFor="filter-select">Privacy Filter</Label>
            <Select
              value={filter}
              onValueChange={(value) => {
                setFilter(value as FilterPreset)
                setFormError(null)
              }}
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
            {isFaceSwap && (
              <p className="text-xs text-muted-foreground">
                Replaces every detected face with the bundled source identity using
                the BlendSwap model.
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="target-fps">Target FPS</Label>
            <Input
              id="target-fps"
              type="number"
              min={1}
              step={1}
              inputMode="numeric"
              value={targetFpsInput}
              onChange={(event) => {
                setTargetFpsInput(event.target.value)
                setFormError(null)
              }}
              className="border-cyan-300/35"
            />
            <p className="text-xs text-muted-foreground">
              This value is sent as `target_fps` in the processing request.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="start-sec">Start sec</Label>
              <Input
                id="start-sec"
                type="number"
                min={0}
                step={0.1}
                inputMode="decimal"
                placeholder="Auto"
                value={startSecInput}
                onChange={(event) => {
                  setStartSecInput(event.target.value)
                  setFormError(null)
                }}
                className="border-cyan-300/35"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="end-sec">End sec</Label>
              <Input
                id="end-sec"
                type="number"
                min={0}
                step={0.1}
                inputMode="decimal"
                placeholder="Auto"
                value={endSecInput}
                onChange={(event) => {
                  setEndSecInput(event.target.value)
                  setFormError(null)
                }}
                className="border-cyan-300/35"
              />
            </div>
          </div>

          {isFaceSwap ? (
            <div className="flex items-center justify-between rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-3 py-2">
              <Label htmlFor="stabilize" className="text-sm">
                Temporal Stabilization
              </Label>
              <Switch
                id="stabilize"
                checked={stabilize}
                onCheckedChange={(checked) => {
                  setStabilize(checked)
                  setFormError(null)
                }}
              />
            </div>
          ) : (
            <div className="flex items-center justify-between rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-3 py-2">
              <Label htmlFor="show-bounding-box" className="text-sm">
                Show Bounding Box
              </Label>
              <Switch
                id="show-bounding-box"
                checked={showBoundingBox}
                onCheckedChange={(checked) => {
                  setShowBoundingBox(checked)
                  setFormError(null)
                }}
              />
            </div>
          )}

          <div className="rounded-lg border border-cyan-300/20 bg-linear-to-r from-cyan-500/14 via-cyan-400/8 to-transparent px-4 py-3">
            <p className="text-sm font-semibold text-foreground">Current Settings</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Filter {getFilterLabel(filter)} | FPS {targetFpsInput.trim() || '--'} |
              {' '}
              Range {startSecInput.trim() || 'auto'}s to {endSecInput.trim() || 'auto'}s |
              {' '}
              {isFaceSwap
                ? `Stabilization ${stabilize ? 'On' : 'Off'}`
                : `Bounding Box ${showBoundingBox ? 'On' : 'Off'}`}
            </p>
          </div>

          {formError && (
            <div className="rounded-lg border border-red-400/35 bg-red-500/10 px-3 py-2 text-xs text-red-100">
              {formError}
            </div>
          )}

          <Button
            onClick={activateGuard}
            disabled={!originalVideoUrl || !selectedFile || isProcessing}
            className="w-full bg-cyan-400 text-cyan-950 hover:bg-cyan-300"
          >
            {isProcessing ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                {processingStatus === 'uploading' ? 'Uploading...' : 'Processing...'}
              </>
            ) : (
              <>
                <ShieldCheck className="size-4" />
                Activate Guard
              </>
            )}
          </Button>

          {(isProcessing || processingProgress === 100) && (
            <section className="rounded-xl border border-cyan-300/20 bg-cyan-500/10 p-4">
              <div className="mb-2 flex items-center justify-between gap-2">
                <p className="text-sm font-semibold text-foreground">Processing Status</p>
                <Badge className="bg-cyan-500/20 text-cyan-700 dark:text-cyan-100">
                  {isProcessing
                    ? processingStatus === 'uploading'
                      ? 'Uploading'
                      : 'Running'
                    : 'Completed'}
                </Badge>
              </div>
              <Progress value={processingProgress} className="h-2.5" />
              <p className="mt-2 text-xs text-muted-foreground">
                {isProcessing
                  ? processingStatus === 'uploading'
                    ? 'Uploading the source video to the processing backend.'
                    : 'Guard is processing this video with selected settings.'
                  : 'Guard completed. Review the protected output on the right panel.'}
              </p>
            </section>
          )}
        </CardContent>
      </Card>

      <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-xl tracking-tight">Processed Result</CardTitle>
          <CardDescription>
            Preview protected output and verify overlays before export.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="relative aspect-video overflow-hidden rounded-xl border border-cyan-300/25 bg-slate-900/90">
            {(isProcessing || !resultVideoUrl) && (
              <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 bg-slate-950/75 text-cyan-100">
                {isProcessing ? (
                  <>
                    <Loader2 className="size-10 animate-spin text-cyan-300/90" />
                    <p className="text-sm text-cyan-100/90">
                      {processingStatus === 'uploading'
                        ? 'Uploading source video...'
                        : `Processing secure output... ${processingProgress}%`}
                    </p>
                  </>
                ) : (
                  <>
                    <ShieldCheck className="size-10 text-cyan-300/70" />
                    <p className="text-sm text-cyan-100/90">
                      Activate Guard to generate processed preview.
                    </p>
                  </>
                )}
              </div>
            )}

            {resultVideoUrl && (
              <video
                key={resultVideoUrl}
                src={resultVideoUrl}
                controls
                playsInline
                className="h-full w-full object-contain"
              />
            )}
          </div>

          <div className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-4 py-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="size-4 text-cyan-700 dark:text-cyan-300" />
              <p className="text-sm font-semibold text-foreground">Output Summary</p>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {lastRun ? (
                <>
                  Mode: {getFilterLabel(lastRun.filter)} | Target {lastRun.targetFps} FPS |
                  {' '}
                  Range {lastRun.startSec ?? 'auto'}s to {lastRun.endSec ?? 'auto'}s |
                  {' '}
                  {lastRun.filter === 'swap'
                    ? `Stabilization ${lastRun.stabilize ? 'On' : 'Off'}`
                    : `Bounding Box ${lastRun.drawTracks ? 'On' : 'Off'}`}
                  {' '}
                  | {lastRun.elapsedSec.toFixed(1)}s
                </>
              ) : (
                'Run Activate Guard to generate a protected output.'
              )}
            </p>
          </div>

          {resultVideoUrl && (
            <Button
              asChild
              variant="outline"
              className="w-full border-cyan-300/35 bg-cyan-500/5 hover:bg-cyan-500/15"
            >
              <a href={resultVideoUrl} download>
                <FileVideo className="size-4" />
                Download Result
              </a>
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
