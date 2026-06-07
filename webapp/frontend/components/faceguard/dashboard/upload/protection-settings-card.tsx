'use client'

import { AudioLines, Loader2, ScanFace, ShieldCheck, SlidersHorizontal, X } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import {
  ColorField,
  NumberField,
  OptionSelect,
  SettingsSection,
} from '@/components/faceguard/dashboard/upload/settings-section'
import {
  audioModeOptions,
  usesFormant,
  usesPitch,
  visualMethodOptions,
  voiceMethodOptions,
  type ProtectionForm,
} from '@/lib/videos/options'

type FormPatch = Partial<ProtectionForm>

/** Middle card: configure face + voice protection and run it on the source video. */
export function ProtectionSettingsCard({
  form,
  onChange,
  rangeError,
  isRunning,
  canSubmit,
  submitLabel,
  onSubmit,
  onCancel,
}: {
  form: ProtectionForm
  onChange: (patch: FormPatch) => void
  rangeError: string | null
  isRunning: boolean
  canSubmit: boolean
  submitLabel: string
  onSubmit: () => void
  onCancel: () => void
}) {
  const isSwap = form.visualMethod === 'swap'
  const isAnonymizeVoice = form.audioMode === 'anonymize'

  return (
    <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-xl tracking-tight">Protection Settings</CardTitle>
        <CardDescription>
          Configure face and voice protection, then run it on the selected video.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Face privacy */}
        <SettingsSection icon={ScanFace} title="Face Privacy">
          <OptionSelect
            id="visual-method"
            label="Privacy Filter"
            value={form.visualMethod}
            options={visualMethodOptions}
            onValueChange={(visualMethod) => onChange({ visualMethod })}
          />

          {form.visualMethod === 'blur' && (
            <NumberField
              id="blur-strength"
              label="Blur strength"
              value={form.blurStrength}
              onChange={(blurStrength) => onChange({ blurStrength })}
              min={3}
              step={2}
              inputMode="numeric"
              hint="Gaussian kernel size — higher is blurrier (rounded up to an odd number)."
            />
          )}

          {form.visualMethod === 'pixelate' && (
            <NumberField
              id="pixelation-level"
              label="Pixelation level"
              value={form.pixelationLevel}
              onChange={(pixelationLevel) => onChange({ pixelationLevel })}
              min={4}
              step={1}
              inputMode="numeric"
              hint="Lower means chunkier blocks (more obscured)."
            />
          )}

          {form.visualMethod === 'mask' && (
            <ColorField
              id="mask-color"
              label="Mask color"
              value={form.maskColor}
              onChange={(maskColor) => onChange({ maskColor })}
              hint="Solid fill drawn over the detected face region."
            />
          )}

          {!isSwap && (
            <div className="flex items-center justify-between rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-3 py-2">
              <Label htmlFor="draw-boxes" className="text-sm">
                Show Bounding Box
              </Label>
              <Switch
                id="draw-boxes"
                checked={form.drawBoxes}
                onCheckedChange={(drawBoxes) => onChange({ drawBoxes })}
              />
            </div>
          )}
        </SettingsSection>

        {/* Voice privacy */}
        <SettingsSection
          icon={AudioLines}
          title="Voice Privacy"
          className="border-t border-cyan-300/15 pt-5"
        >
          <OptionSelect
            id="audio-mode"
            label="Audio Handling"
            value={form.audioMode}
            options={audioModeOptions}
            onValueChange={(audioMode) => onChange({ audioMode })}
          />

          {isAnonymizeVoice && (
            <OptionSelect
              id="voice-method"
              label="Voice Method"
              value={form.voiceMethod}
              options={voiceMethodOptions}
              onValueChange={(voiceMethod) => onChange({ voiceMethod })}
            />
          )}

          {isAnonymizeVoice && form.voiceMethod === 'mcadams' && (
            <NumberField
              id="mcadams-alpha"
              label="Warp strength"
              value={form.mcadamsAlpha}
              onChange={(mcadamsAlpha) => onChange({ mcadamsAlpha })}
              min={0.5}
              step={0.05}
              inputMode="decimal"
              hint="Formant warp that keeps pitch and timing — values further from 1.0 are stronger."
            />
          )}

          {isAnonymizeVoice && (usesPitch(form.voiceMethod) || usesFormant(form.voiceMethod)) && (
            <div
              className={
                usesPitch(form.voiceMethod) && usesFormant(form.voiceMethod)
                  ? 'grid gap-3 sm:grid-cols-2'
                  : 'grid gap-3 grid-cols-1'
              }
            >
              {usesPitch(form.voiceMethod) && (
                <NumberField
                  id="pitch-steps"
                  label="Pitch shift (semitones)"
                  value={form.pitchSteps}
                  onChange={(pitchSteps) => onChange({ pitchSteps })}
                  step={1}
                  inputMode="decimal"
                />
              )}
              {usesFormant(form.voiceMethod) && (
                <NumberField
                  id="formant-shift"
                  label="Formant shift"
                  value={form.formantShift}
                  onChange={(formantShift) => onChange({ formantShift })}
                  min={0.5}
                  step={0.05}
                  inputMode="decimal"
                />
              )}
            </div>
          )}
        </SettingsSection>

        {/* Processing range */}
        <SettingsSection
          icon={SlidersHorizontal}
          title="Processing Range"
          className="border-t border-cyan-300/15 pt-5"
        >
          <NumberField
            id="target-fps"
            label="Target FPS"
            value={form.targetFps}
            onChange={(targetFps) => onChange({ targetFps })}
            min={1}
            step={1}
            inputMode="numeric"
            placeholder="Source rate"
            hint="Downsamples only — leave blank to keep the source frame rate."
          />

          <div className="grid gap-3 sm:grid-cols-2">
            <NumberField
              id="start-sec"
              label="Start sec"
              value={form.startSec}
              onChange={(startSec) => onChange({ startSec })}
              min={0}
              step={0.1}
              inputMode="decimal"
              placeholder="Auto"
            />
            <NumberField
              id="end-sec"
              label="End sec"
              value={form.endSec}
              onChange={(endSec) => onChange({ endSec })}
              min={0}
              step={0.1}
              inputMode="decimal"
              placeholder="Auto"
            />
          </div>

          {rangeError ? (
            <p className="text-xs text-rose-500 dark:text-rose-300">{rangeError}</p>
          ) : null}
        </SettingsSection>

        <div className="flex gap-3">
          <Button
            onClick={onSubmit}
            disabled={!canSubmit}
            className="flex-1 bg-cyan-400 text-cyan-950 hover:bg-cyan-300"
          >
            {isRunning ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <ShieldCheck className="size-4" />
            )}
            {submitLabel}
          </Button>
          {isRunning && (
            <Button
              onClick={onCancel}
              variant="outline"
              className="border-rose-300/40 bg-rose-500/10 text-rose-600 hover:bg-rose-500/20 dark:text-rose-200"
            >
              <X className="size-4" />
              Cancel
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
