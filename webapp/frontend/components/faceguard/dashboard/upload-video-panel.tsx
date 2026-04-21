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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
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

type FilterPreset = 'none' | 'blur' | 'pixelate' | 'mask' | 'blackout' | 'faceswap'

const filterOptions: Array<{ value: FilterPreset; label: string }> = [
  { value: 'none', label: 'None' },
  { value: 'blur', label: 'Blur' },
  { value: 'pixelate', label: 'Pixelate' },
  { value: 'mask', label: 'Mask' },
  { value: 'blackout', label: 'Blackout' },
  { value: 'faceswap', label: 'FaceSwap' },
]

const resultFilterMap: Record<FilterPreset, string> = {
  none: 'none',
  blur: 'blur(14px)',
  pixelate: 'contrast(1.06) saturate(0.65)',
  mask: 'none',
  blackout: 'brightness(0.14) contrast(1.25)',
  faceswap: 'hue-rotate(128deg) saturate(1.35) contrast(1.05)',
}

export function UploadVideoPanel() {
  const [files, setFiles] = useState<File[]>([])
  const [originalVideoUrl, setOriginalVideoUrl] = useState<string | null>(null)
  const [resultVideoUrl, setResultVideoUrl] = useState<string | null>(null)
  const [filter, setFilter] = useState<FilterPreset>('blur')
  const [targetFps, setTargetFps] = useState('30')
  const [showBoundingBox, setShowBoundingBox] = useState(true)
  const [showConfidence, setShowConfidence] = useState(true)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)

  const previewUrlRef = useRef<string | null>(null)
  const progressIntervalRef = useRef<number | null>(null)
  const completionTimeoutRef = useRef<number | null>(null)

  const totalSizeMB = useMemo(
    () =>
      files.reduce((total, file) => total + file.size / (1024 * 1024), 0).toFixed(2),
    [files],
  )

  const queueProgress = files.length === 0 ? 0 : Math.min(100, files.length * 34)

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
    const selectedFiles = Array.from(event.target.files ?? [])
    setFiles(selectedFiles)

    clearProcessingTimers()
    setIsProcessing(false)
    setProcessingProgress(0)
    setResultVideoUrl(null)

    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = null
    }

    if (selectedFiles.length === 0) {
      setOriginalVideoUrl(null)
      return
    }

    const nextPreviewUrl = URL.createObjectURL(selectedFiles[0])
    previewUrlRef.current = nextPreviewUrl
    setOriginalVideoUrl(nextPreviewUrl)
  }

  const activateGuard = () => {
    if (!originalVideoUrl || files.length === 0) {
      return
    }

    clearProcessingTimers()
    setIsProcessing(true)
    setProcessingProgress(8)
    setResultVideoUrl(null)

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
            Upload your video and preview the raw source before enabling protection.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <label
            htmlFor="video-upload"
            className="group block cursor-pointer rounded-2xl border border-dashed border-cyan-300/35 bg-cyan-500/10 p-8 text-center transition-colors hover:border-cyan-300/60 hover:bg-cyan-500/15"
          >
            <FolderUp className="mx-auto mb-3 size-10 text-cyan-700 transition-transform group-hover:scale-105 dark:text-cyan-300" />
            <p className="text-base font-semibold text-foreground">
              Drop video files here or click to browse
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              Supports .mp4, .mov, .webm (up to 2GB per file)
            </p>
            <Input
              id="video-upload"
              type="file"
              accept="video/*"
              multiple
              className="mt-4 border-cyan-300/35"
              onChange={handleUploadChange}
            />
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
              <p className="text-sm font-semibold text-foreground">Queue Status</p>
              <Badge className="bg-cyan-500/20 text-cyan-700 dark:text-cyan-100">
                {files.length} file(s)
              </Badge>
            </div>
            <Progress value={queueProgress} className="h-2.5" />
            <p className="mt-2 text-xs text-muted-foreground">
              Total size: {totalSizeMB} MB
            </p>
          </section>
        </CardContent>
      </Card>

      <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-xl tracking-tight">Filter and Overlay</CardTitle>
          <CardDescription>
            Configure identity protection before processing your uploaded file.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
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

          <div className="space-y-2">
            <Label htmlFor="target-fps">Target FPS</Label>
            <Select value={targetFps} onValueChange={setTargetFps}>
              <SelectTrigger id="target-fps" className="w-full border-cyan-300/35">
                <SelectValue placeholder="Select FPS" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="15">15 FPS</SelectItem>
                <SelectItem value="24">24 FPS</SelectItem>
                <SelectItem value="30">30 FPS</SelectItem>
                <SelectItem value="60">60 FPS</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center justify-between rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-3 py-2">
            <Label htmlFor="show-bounding-box" className="text-sm">
              Show Bounding Box
            </Label>
            <Switch
              id="show-bounding-box"
              checked={showBoundingBox}
              onCheckedChange={setShowBoundingBox}
            />
          </div>

          <div className="flex items-center justify-between rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-3 py-2">
            <Label htmlFor="show-confidence" className="text-sm">
              Confidence Tag
            </Label>
            <Switch
              id="show-confidence"
              checked={showConfidence}
              onCheckedChange={setShowConfidence}
            />
          </div>

          <div className="rounded-lg border border-cyan-300/20 bg-linear-to-r from-cyan-500/16 via-cyan-400/10 to-transparent px-4 py-3">
            <p className="text-sm font-semibold text-foreground">
              Pipeline: Upload Video -&gt; Face Detection -&gt; Privacy Filter -&gt; Result
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Active preset: {filterOptions.find((option) => option.value === filter)?.label} |
              Target {targetFps} FPS
            </p>
          </div>

          <Button
            onClick={activateGuard}
            disabled={!originalVideoUrl || isProcessing}
            className="w-full bg-cyan-400 text-cyan-950 hover:bg-cyan-300"
          >
            {isProcessing ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                Activating...
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
                  ? 'Guard is processing this video with selected privacy controls.'
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
                  <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(transparent_94%,rgba(8,145,178,0.5)_95%),linear-gradient(90deg,transparent_94%,rgba(8,145,178,0.5)_95%)] bg-[size:14px_14px] opacity-30" />
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

                {showConfidence && (
                  <p className="pointer-events-none absolute right-3 bottom-3 rounded-md border border-cyan-300/35 bg-slate-950/70 px-2 py-1 text-xs text-cyan-100">
                    Face confidence: 98.4%
                  </p>
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
              {targetFps} FPS | Bounding Box {showBoundingBox ? 'On' : 'Off'} | Confidence
              {' '}
              {showConfidence ? 'On' : 'Off'}
            </p>
          </div>

          {files[0] && resultVideoUrl && (
            <Button
              asChild
              variant="outline"
              className="w-full border-cyan-300/35 bg-cyan-500/5 hover:bg-cyan-500/15"
            >
              <a href={resultVideoUrl} download={`guarded-preview-${files[0].name}`}>
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
