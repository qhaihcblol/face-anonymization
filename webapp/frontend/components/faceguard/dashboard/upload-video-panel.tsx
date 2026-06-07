'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AudioLines,
  CheckCircle2,
  FileVideo,
  FolderUp,
  ScanFace,
  ShieldCheck,
  SlidersHorizontal,
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

// UI-only shell: the backend video API integration was removed, so the form
// controls below are purely presentational local state and "Activate Guard"
// does not process anything.
type FilterPreset = 'none' | 'blur' | 'pixelate' | 'mask' | 'blackout' | 'swap'

type VoiceMethod =
  | 'none'
  | 'mcadams'
  | 'pitch'
  | 'formant'
  | 'pitch_formant'
  | 'convert'

const filterOptions: Array<{ value: FilterPreset; label: string }> = [
  { value: 'none', label: 'None' },
  { value: 'blur', label: 'Blur' },
  { value: 'pixelate', label: 'Pixelate' },
  { value: 'mask', label: 'Mask' },
  { value: 'blackout', label: 'Blackout' },
  { value: 'swap', label: 'Face Swap' },
]

const voiceOptions: Array<{ value: VoiceMethod; label: string }> = [
  { value: 'none', label: 'Keep original' },
  { value: 'mcadams', label: 'McAdams (recommended)' },
  { value: 'pitch', label: 'Pitch shift' },
  { value: 'formant', label: 'Formant shift' },
  { value: 'pitch_formant', label: 'Pitch + Formant' },
  { value: 'convert', label: 'Voice conversion (AI)' },
]

function usesPitch(method: VoiceMethod): boolean {
  return method === 'pitch' || method === 'pitch_formant'
}

function usesFormant(method: VoiceMethod): boolean {
  return method === 'formant' || method === 'pitch_formant'
}

export function UploadVideoPanel() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [originalVideoUrl, setOriginalVideoUrl] = useState<string | null>(null)
  const [filter, setFilter] = useState<FilterPreset>('blur')
  const [targetFpsInput, setTargetFpsInput] = useState('30')
  const [startSecInput, setStartSecInput] = useState('')
  const [endSecInput, setEndSecInput] = useState('')
  const [showBoundingBox, setShowBoundingBox] = useState(true)
  const [blurStrengthInput, setBlurStrengthInput] = useState('31')
  const [pixelationLevelInput, setPixelationLevelInput] = useState('16')
  const [maskColor, setMaskColor] = useState('#a0a0a0')
  const [voiceMethod, setVoiceMethod] = useState<VoiceMethod>('mcadams')
  const [pitchStepsInput, setPitchStepsInput] = useState('-4')
  const [formantShiftInput, setFormantShiftInput] = useState('1.2')
  const [mcadamsAlphaInput, setMcadamsAlphaInput] = useState('0.8')

  const previewUrlRef = useRef<string | null>(null)

  const isFaceSwap = filter === 'swap'

  const selectedFileSizeMB = useMemo(() => {
    if (!selectedFile) {
      return '0.00'
    }

    return (selectedFile.size / (1024 * 1024)).toFixed(2)
  }, [selectedFile])

  const queueProgress = selectedFile ? 100 : 0

  const handleUploadChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null
    setSelectedFile(nextFile)

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

  useEffect(() => {
    return () => {
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
          <CardTitle className="text-xl tracking-tight">Protection Settings</CardTitle>
          <CardDescription>
            Configure face and voice protection, then run it on the selected video.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Face privacy */}
          <section className="space-y-4">
            <div className="flex items-center gap-2 text-cyan-700 dark:text-cyan-200">
              <ScanFace className="size-4" />
              <h3 className="text-xs font-semibold tracking-[0.14em] uppercase">
                Face Privacy
              </h3>
            </div>

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
              {isFaceSwap && (
                <p className="text-xs text-muted-foreground">
                  Replaces every detected face with the bundled source identity using
                  the BlendSwap model.
                </p>
              )}
            </div>

            {filter === 'blur' && (
              <div className="space-y-2">
                <Label htmlFor="blur-strength">Blur strength</Label>
                <Input
                  id="blur-strength"
                  type="number"
                  min={3}
                  step={2}
                  inputMode="numeric"
                  value={blurStrengthInput}
                  onChange={(event) => setBlurStrengthInput(event.target.value)}
                  className="border-cyan-300/35"
                />
                <p className="text-xs text-muted-foreground">
                  Gaussian kernel size — higher is blurrier (rounded up to an odd
                  number).
                </p>
              </div>
            )}

            {filter === 'pixelate' && (
              <div className="space-y-2">
                <Label htmlFor="pixelation-level">Pixelation level</Label>
                <Input
                  id="pixelation-level"
                  type="number"
                  min={4}
                  step={1}
                  inputMode="numeric"
                  value={pixelationLevelInput}
                  onChange={(event) => setPixelationLevelInput(event.target.value)}
                  className="border-cyan-300/35"
                />
                <p className="text-xs text-muted-foreground">
                  Lower means chunkier blocks (more obscured).
                </p>
              </div>
            )}

            {filter === 'mask' && (
              <div className="space-y-2">
                <Label htmlFor="mask-color">Mask color</Label>
                <div className="flex items-center gap-3">
                  <input
                    id="mask-color"
                    type="color"
                    value={maskColor}
                    onChange={(event) => setMaskColor(event.target.value)}
                    className="h-9 w-14 cursor-pointer rounded-md border border-cyan-300/35 bg-transparent p-1"
                  />
                  <span className="text-xs text-muted-foreground uppercase">
                    {maskColor}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Solid fill drawn over the detected face region.
                </p>
              </div>
            )}

            {filter === 'blackout' && (
              <p className="text-xs text-muted-foreground">
                Fills the detected face region with solid black.
              </p>
            )}

            {filter === 'none' && (
              <p className="text-xs text-muted-foreground">
                No face filter is applied — only voice and range settings run.
              </p>
            )}

            {!isFaceSwap && (
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
            )}
          </section>

          {/* Voice privacy */}
          <section className="space-y-4 border-t border-cyan-300/15 pt-5">
            <div className="flex items-center gap-2 text-cyan-700 dark:text-cyan-200">
              <AudioLines className="size-4" />
              <h3 className="text-xs font-semibold tracking-[0.14em] uppercase">
                Voice Privacy
              </h3>
            </div>

            <div className="space-y-2">
              <Label htmlFor="voice-select">Voice Method</Label>
              <Select
                value={voiceMethod}
                onValueChange={(value) => setVoiceMethod(value as VoiceMethod)}
              >
                <SelectTrigger id="voice-select" className="w-full border-cyan-300/35">
                  <SelectValue placeholder="Select voice method" />
                </SelectTrigger>
                <SelectContent>
                  {voiceOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {(usesPitch(voiceMethod) || usesFormant(voiceMethod)) && (
              <div
                className={`grid gap-3 ${
                  usesPitch(voiceMethod) && usesFormant(voiceMethod)
                    ? 'sm:grid-cols-2'
                    : 'grid-cols-1'
                }`}
              >
                {usesPitch(voiceMethod) && (
                  <div className="space-y-2">
                    <Label htmlFor="pitch-steps">Pitch shift (semitones)</Label>
                    <Input
                      id="pitch-steps"
                      type="number"
                      step={1}
                      inputMode="decimal"
                      value={pitchStepsInput}
                      onChange={(event) => setPitchStepsInput(event.target.value)}
                      className="border-cyan-300/35"
                    />
                  </div>
                )}

                {usesFormant(voiceMethod) && (
                  <div className="space-y-2">
                    <Label htmlFor="formant-shift">Formant shift</Label>
                    <Input
                      id="formant-shift"
                      type="number"
                      step={0.05}
                      min={0.5}
                      inputMode="decimal"
                      value={formantShiftInput}
                      onChange={(event) => setFormantShiftInput(event.target.value)}
                      className="border-cyan-300/35"
                    />
                  </div>
                )}
              </div>
            )}

            {voiceMethod === 'mcadams' && (
              <div className="space-y-2">
                <Label htmlFor="mcadams-alpha">Warp strength</Label>
                <Input
                  id="mcadams-alpha"
                  type="number"
                  step={0.05}
                  min={0.5}
                  inputMode="decimal"
                  value={mcadamsAlphaInput}
                  onChange={(event) => setMcadamsAlphaInput(event.target.value)}
                  className="border-cyan-300/35"
                />
                <p className="text-xs text-muted-foreground">
                  Formant warp that keeps pitch and timing — values further from 1.0 are
                  stronger.
                </p>
              </div>
            )}

            {voiceMethod === 'convert' && (
              <p className="text-xs text-muted-foreground">
                Model-based voice conversion toward the bundled reference voice.
              </p>
            )}

            {voiceMethod === 'none' && (
              <p className="text-xs text-muted-foreground">
                Keeps the original audio track unchanged.
              </p>
            )}
          </section>

          {/* Processing range */}
          <section className="space-y-4 border-t border-cyan-300/15 pt-5">
            <div className="flex items-center gap-2 text-cyan-700 dark:text-cyan-200">
              <SlidersHorizontal className="size-4" />
              <h3 className="text-xs font-semibold tracking-[0.14em] uppercase">
                Processing Range
              </h3>
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
                onChange={(event) => setTargetFpsInput(event.target.value)}
                className="border-cyan-300/35"
              />
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
                  onChange={(event) => setStartSecInput(event.target.value)}
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
                  onChange={(event) => setEndSecInput(event.target.value)}
                  className="border-cyan-300/35"
                />
              </div>
            </div>
          </section>

          {/* Inert in the UI-only shell: no processing backend is wired up. */}
          <Button
            disabled={!selectedFile}
            className="w-full bg-cyan-400 text-cyan-950 hover:bg-cyan-300"
          >
            <ShieldCheck className="size-4" />
            Activate Guard
          </Button>
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
            <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 bg-slate-950/75 text-cyan-100">
              <ShieldCheck className="size-10 text-cyan-300/70" />
              <p className="text-sm text-cyan-100/90">
                Activate Guard to generate processed preview.
              </p>
            </div>
          </div>

          <div className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-4 py-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="size-4 text-cyan-700 dark:text-cyan-300" />
              <p className="text-sm font-semibold text-foreground">Output Summary</p>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Run Activate Guard to generate a protected output.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
