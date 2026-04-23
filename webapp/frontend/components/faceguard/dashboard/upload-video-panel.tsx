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

type FilterPreset = 'none' | 'blur' | 'pixelate' | 'mask' | 'blackout'

type ProcessingRequest = {
  method: FilterPreset
  target_fps: number
  start_sec?: number
  end_sec?: number
  draw_tracks: boolean
}

const filterOptions: Array<{ value: FilterPreset; label: string }> = [
  { value: 'none', label: 'None' },
  { value: 'blur', label: 'Blur' },
  { value: 'pixelate', label: 'Pixelate' },
  { value: 'mask', label: 'Mask' },
  { value: 'blackout', label: 'Blackout' },
]

const resultFilterMap: Record<FilterPreset, string> = {
  none: 'none',
  blur: 'blur(14px)',
  pixelate: 'contrast(1.06) saturate(0.65)',
  mask: 'none',
  blackout: 'brightness(0.14) contrast(1.25)',
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
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [formError, setFormError] = useState<string | null>(null)
  const [lastRequest, setLastRequest] = useState<ProcessingRequest | null>(null)

  const previewUrlRef = useRef<string | null>(null)
  const progressIntervalRef = useRef<number | null>(null)
  const completionTimeoutRef = useRef<number | null>(null)

  const selectedFileSizeMB = useMemo(() => {
    if (!selectedFile) {
      return '0.00'
    }

    return (selectedFile.size / (1024 * 1024)).toFixed(2)
  }, [selectedFile])

  const queueProgress = selectedFile ? 100 : 0

  const clearProcessingTimers = () => {
    if (progressIntervalRef.current !== null) {
      window.clearInterval(progressIntervalRef.current)
      progressIntervalRef.current = null
    }

    if (completionTimeoutRef.current !== null) {
      window.clearTimeout(completionTimeoutRef.current)
      completionTimeoutRef.current = null
    }
  }

  const handleUploadChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null
    setSelectedFile(nextFile)

    clearProcessingTimers()
    setIsProcessing(false)
    setProcessingProgress(0)
    setResultVideoUrl(null)
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

  const buildProcessingRequest = (): ProcessingRequest => {
    const targetFps = parseTargetFps(targetFpsInput)
    const startSec = parseOptionalSecond(startSecInput, 'Start sec')
    const endSec = parseOptionalSecond(endSecInput, 'End sec')

    if (
      startSec !== undefined &&
      endSec !== undefined &&
      endSec <= startSec
    ) {
      throw new Error('End sec must be greater than Start sec.')
    }

    return {
      method: filter,
      target_fps: targetFps,
      start_sec: startSec,
      end_sec: endSec,
      draw_tracks: showBoundingBox,
    }
  }

  const activateGuard = () => {
    if (!originalVideoUrl || !selectedFile) {
      return
    }

    let requestPayload: ProcessingRequest
    try {
      requestPayload = buildProcessingRequest()
    } catch (error) {
      setFormError(
        error instanceof Error ? error.message : 'Invalid processing parameters.',
      )
      return
    }

    setFormError(null)
    setLastRequest(requestPayload)

    clearProcessingTimers()
    setIsProcessing(true)
    setProcessingProgress(8)
    setResultVideoUrl(null)

    // Placeholder processing simulation. Replace with API call when backend endpoint is ready.
    progressIntervalRef.current = window.setInterval(() => {
      setProcessingProgress((current) => Math.min(94, current + 8))
    }, 160)

    completionTimeoutRef.current = window.setTimeout(() => {
      clearProcessingTimers()
      setProcessingProgress(100)
      setResultVideoUrl(originalVideoUrl)
      setIsProcessing(false)
    }, 2600)
  }

  useEffect(() => {
    return () => {
      clearProcessingTimers()
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

          <div className="rounded-lg border border-cyan-300/20 bg-linear-to-r from-cyan-500/14 via-cyan-400/8 to-transparent px-4 py-3">
            <p className="text-sm font-semibold text-foreground">Current Settings</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Filter {filterOptions.find((option) => option.value === filter)?.label} | FPS
              {' '}
              {targetFpsInput.trim() || '--'} | Range {startSecInput.trim() || 'auto'}s to
              {' '}
              {endSecInput.trim() || 'auto'}s
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
                Processing...
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
                  {isProcessing ? 'Running' : 'Completed'}
                </Badge>
              </div>
              <Progress value={processingProgress} className="h-2.5" />
              <p className="mt-2 text-xs text-muted-foreground">
                {isProcessing
                  ? 'Guard is processing this video with selected settings.'
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
                      Processing secure output... {processingProgress}%
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
              <>
                <video
                  key={`${resultVideoUrl}-${filter}`}
                  src={resultVideoUrl}
                  controls
                  playsInline
                  className="h-full w-full object-contain"
                  style={{ filter: resultFilterMap[filter] }}
                />

                {filter === 'pixelate' && (
                  <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(transparent_94%,rgba(8,145,178,0.5)_95%),linear-gradient(90deg,transparent_94%,rgba(8,145,178,0.5)_95%)] bg-size-[14px_14px] opacity-30" />
                )}

                {showBoundingBox && (
                  <>
                    <div className="pointer-events-none absolute top-[19%] left-[21%] h-24 w-20 rounded-md border-2 border-cyan-300/90 shadow-[0_0_18px_-6px_rgba(34,211,238,0.95)]" />
                    <div className="pointer-events-none absolute top-[36%] right-[23%] h-28 w-24 rounded-md border-2 border-cyan-300/90 shadow-[0_0_18px_-6px_rgba(34,211,238,0.95)]" />
                  </>
                )}

                {filter === 'mask' && (
                  <>
                    <div className="pointer-events-none absolute top-[19%] left-[21%] h-24 w-20 rounded-full bg-slate-950/80" />
                    <div className="pointer-events-none absolute top-[36%] right-[23%] h-28 w-24 rounded-full bg-slate-950/80" />
                  </>
                )}
              </>
            )}
          </div>

          <div className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-4 py-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="size-4 text-cyan-700 dark:text-cyan-300" />
              <p className="text-sm font-semibold text-foreground">Output Summary</p>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Mode: {filterOptions.find((option) => option.value === filter)?.label} | Target
              {' '}
              {lastRequest?.target_fps ?? '--'} FPS | Range {lastRequest?.start_sec ?? 'auto'}s
              {' '}
              to {lastRequest?.end_sec ?? 'auto'}s | Bounding Box
              {' '}
              {lastRequest?.draw_tracks ? 'On' : 'Off'}
            </p>
          </div>

          {selectedFile && resultVideoUrl && (
            <Button
              asChild
              variant="outline"
              className="w-full border-cyan-300/35 bg-cyan-500/5 hover:bg-cyan-500/15"
            >
              <a href={resultVideoUrl} download={`guarded-preview-${selectedFile.name}`}>
                <FileVideo className="size-4" />
                Download Preview
              </a>
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
