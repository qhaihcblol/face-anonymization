'use client'

import { Camera, Download, Film, ScanFace, SquareDashed, Trash2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import {
  ColorField,
  NumberField,
  OptionSelect,
  SettingsSection,
} from '@/components/faceguard/dashboard/upload/settings-section'
import { liveFilterOptions, type LiveFilterForm } from '@/lib/videos/options'
import {
  resolutionConfig,
  type CameraDevice,
  type RecordedClip,
  type ResolutionPreset,
} from '@/lib/videos/use-live-camera'

type FilterPatch = Partial<LiveFilterForm>

/**
 * Right card: tune the capture device and the privacy filter, toggle the preview
 * overlays, and manage recordings. Purely presentational — all state lives in the
 * workspace orchestrator.
 */
export function CameraControlsCard({
  cameraDevices,
  selectedCamera,
  onSelectCamera,
  resolution,
  onSelectResolution,
  onApplySettings,
  filter,
  onFilterChange,
  showBoundingBox,
  onToggleBoundingBox,
  showConfidence,
  onToggleConfidence,
  recordedClips,
  onRemoveClip,
}: {
  cameraDevices: CameraDevice[]
  selectedCamera: string
  onSelectCamera: (id: string) => void
  resolution: ResolutionPreset
  onSelectResolution: (resolution: ResolutionPreset) => void
  onApplySettings: () => void
  filter: LiveFilterForm
  onFilterChange: (patch: FilterPatch) => void
  showBoundingBox: boolean
  onToggleBoundingBox: (value: boolean) => void
  showConfidence: boolean
  onToggleConfidence: (value: boolean) => void
  recordedClips: RecordedClip[]
  onRemoveClip: (clipId: string) => void
}) {
  return (
    <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-xl tracking-tight">Control Panel</CardTitle>
        <CardDescription>
          Tune camera and privacy controls before sending the stream to detection.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Capture device */}
        <SettingsSection icon={Camera} title="Camera">
          <div className="space-y-2">
            <Label htmlFor="camera-select">Source</Label>
            <Select value={selectedCamera} onValueChange={onSelectCamera}>
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
              onValueChange={(value) => onSelectResolution(value as ResolutionPreset)}
            >
              <SelectTrigger id="resolution-select" className="w-full border-cyan-300/35">
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
            onClick={onApplySettings}
            variant="outline"
            className="w-full border-cyan-300/35 bg-cyan-500/5 hover:bg-cyan-500/15"
          >
            Apply Settings
          </Button>
        </SettingsSection>

        {/* Privacy filter */}
        <SettingsSection
          icon={ScanFace}
          title="Privacy Filter"
          className="border-t border-cyan-300/15 pt-5"
        >
          <OptionSelect
            id="live-filter"
            label="Filter"
            value={filter.method}
            options={liveFilterOptions}
            onValueChange={(method) => onFilterChange({ method })}
          />

          {filter.method === 'blur' && (
            <NumberField
              id="live-blur-strength"
              label="Blur strength"
              value={filter.blurStrength}
              onChange={(blurStrength) => onFilterChange({ blurStrength })}
              min={3}
              step={2}
              inputMode="numeric"
              hint="Gaussian kernel size — higher is blurrier (rounded up to an odd number)."
            />
          )}

          {filter.method === 'pixelate' && (
            <NumberField
              id="live-pixelation-level"
              label="Pixelation level"
              value={filter.pixelationLevel}
              onChange={(pixelationLevel) => onFilterChange({ pixelationLevel })}
              min={4}
              step={1}
              inputMode="numeric"
              hint="Lower means chunkier blocks (more obscured)."
            />
          )}

          {filter.method === 'mask' && (
            <ColorField
              id="live-mask-color"
              label="Mask color"
              value={filter.maskColor}
              onChange={(maskColor) => onFilterChange({ maskColor })}
              hint="Solid fill drawn over the detected face region."
            />
          )}
        </SettingsSection>

        {/* Preview overlays */}
        <SettingsSection
          icon={SquareDashed}
          title="Overlay"
          className="border-t border-cyan-300/15 pt-5"
        >
          <div className="flex items-center justify-between rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-3 py-2">
            <Label htmlFor="bounding-box" className="text-sm">
              Show Bounding Box
            </Label>
            <Switch
              id="bounding-box"
              checked={showBoundingBox}
              onCheckedChange={onToggleBoundingBox}
            />
          </div>

          <div className="flex items-center justify-between rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-3 py-2">
            <Label htmlFor="confidence-badge" className="text-sm">
              Confidence Tag
            </Label>
            <Switch
              id="confidence-badge"
              checked={showConfidence}
              disabled={!showBoundingBox}
              onCheckedChange={onToggleConfidence}
            />
          </div>
        </SettingsSection>

        {/* Recordings */}
        <SettingsSection
          icon={Film}
          title="Recordings"
          className="border-t border-cyan-300/15 pt-5"
        >
          {recordedClips.length === 0 ? (
            <div className="rounded-lg border border-dashed border-cyan-300/30 px-4 py-6 text-center text-sm text-muted-foreground">
              No recordings yet. Start the stream and press Record.
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
                      <p className="text-xs text-muted-foreground">{clip.createdAt}</p>
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
                        onClick={() => onRemoveClip(clip.id)}
                      >
                        <Trash2 className="size-4" />
                      </Button>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </SettingsSection>
      </CardContent>
    </Card>
  )
}
