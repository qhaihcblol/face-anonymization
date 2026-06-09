'use client'

/**
 * Visual pickers for the curated source faces / voices.
 *
 *  - {@link SourceFacePicker}  — a grid of face thumbnails (shown when the face
 *    filter is Face Swap). Click a tile to select it; click the expand badge to
 *    enlarge it in a dialog.
 *  - {@link SourceVoicePicker} — a list of voice clips (shown when Voice Conversion
 *    is chosen). Click a row to select it; click the play button to audition it.
 *
 * Selection is optional in both: `null` keeps the engine's bundled default identity.
 * Both stay presentational — the catalog is loaded by {@link useSourceAssets} and
 * passed in as `state`, and selection is lifted to the protection form.
 */

import { useRef, useState } from 'react'
import {
  AlertCircle,
  Check,
  Loader2,
  Maximize2,
  Pause,
  Play,
  RotateCw,
} from 'lucide-react'

import { cn } from '@/lib/utils'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import type { SourceAsset } from '@/lib/videos/types'
import type { SourceAssetsState } from '@/lib/videos/use-source-assets'

// --- Shared scaffolding ---------------------------------------------------- //

function GenderChip({ gender }: { gender: string | null }) {
  if (!gender) return null
  return (
    <span className="rounded-full bg-cyan-500/15 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-cyan-700 dark:text-cyan-200">
      {gender}
    </span>
  )
}

/** Header (label + clear) + the loading/error/empty states shared by both pickers. */
function PickerFrame({
  label,
  hint,
  selectedName,
  onClear,
  state,
  emptyText,
  children,
}: {
  label: string
  hint: string
  selectedName: string | null
  onClear: () => void
  state: SourceAssetsState
  emptyText: string
  children: React.ReactNode
}) {
  const isLoading = state.status === 'idle' || state.status === 'loading'
  const isReady = state.status === 'ready' && state.assets.length > 0

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <Label>{label}</Label>
        {selectedName ? (
          <button
            type="button"
            onClick={onClear}
            className="text-xs font-medium text-cyan-700 transition-colors hover:text-cyan-500 dark:text-cyan-300"
          >
            Use default
          </button>
        ) : null}
      </div>

      <div className="rounded-xl border border-cyan-300/25 bg-cyan-500/5 p-2.5">
        {isLoading ? (
          <div className="flex items-center justify-center gap-2 py-6 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin text-cyan-500" />
            Loading…
          </div>
        ) : state.status === 'error' ? (
          <div className="flex flex-col items-center gap-2 py-5 text-center">
            <AlertCircle className="size-5 text-rose-400" />
            <p className="text-xs text-rose-500 dark:text-rose-300">{state.error}</p>
            <button
              type="button"
              onClick={state.reload}
              className="inline-flex items-center gap-1 rounded-md border border-cyan-300/40 bg-cyan-500/10 px-2.5 py-1 text-xs font-medium text-cyan-700 transition-colors hover:bg-cyan-500/20 dark:text-cyan-200"
            >
              <RotateCw className="size-3" />
              Retry
            </button>
          </div>
        ) : !isReady ? (
          <p className="py-5 text-center text-xs text-muted-foreground">{emptyText}</p>
        ) : (
          children
        )}
      </div>

      <p className="text-xs text-muted-foreground">
        {selectedName ? `Selected: ${selectedName}` : hint}
      </p>
    </div>
  )
}

// --- Faces ----------------------------------------------------------------- //

export function SourceFacePicker({
  state,
  selectedKey,
  onSelect,
}: {
  state: SourceAssetsState
  selectedKey: string | null
  onSelect: (key: string | null) => void
}) {
  const [zoom, setZoom] = useState<SourceAsset | null>(null)
  const selectedName =
    state.assets.find((asset) => asset.key === selectedKey)?.name ?? null

  return (
    <>
      <PickerFrame
        label="Source face"
        hint="Optional — pick a face to wear, or keep the default identity."
        selectedName={selectedName}
        onClear={() => onSelect(null)}
        state={state}
        emptyText="No source faces are available yet."
      >
        <div className="grid max-h-64 grid-cols-3 gap-2 overflow-y-auto pr-0.5">
          {state.assets.map((asset) => {
            const isSelected = asset.key === selectedKey
            return (
              <div key={asset.key} className="group relative">
                <button
                  type="button"
                  aria-pressed={isSelected}
                  aria-label={`Select ${asset.name}`}
                  onClick={() => onSelect(isSelected ? null : asset.key)}
                  className={cn(
                    'block w-full overflow-hidden rounded-lg border bg-slate-900/80 transition-all',
                    isSelected
                      ? 'border-cyan-400 ring-2 ring-cyan-400/70'
                      : 'border-cyan-300/25 hover:border-cyan-300/60',
                  )}
                >
                  <span className="block aspect-square">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={asset.url}
                      alt={asset.name}
                      loading="lazy"
                      className="size-full object-cover"
                    />
                  </span>
                  <span className="flex items-center justify-between gap-1 px-1.5 py-1">
                    <span className="truncate text-[11px] font-medium text-cyan-50">
                      {asset.name}
                    </span>
                    <GenderChip gender={asset.gender} />
                  </span>
                </button>

                {isSelected ? (
                  <span className="pointer-events-none absolute left-1 top-1 grid size-5 place-items-center rounded-full bg-cyan-400 text-cyan-950 shadow">
                    <Check className="size-3.5" />
                  </span>
                ) : null}

                <button
                  type="button"
                  aria-label={`Enlarge ${asset.name}`}
                  onClick={() => setZoom(asset)}
                  className="absolute right-1 top-1 grid size-6 place-items-center rounded-md bg-slate-950/70 text-cyan-100 opacity-0 transition-opacity hover:bg-slate-950/90 focus-visible:opacity-100 group-hover:opacity-100"
                >
                  <Maximize2 className="size-3.5" />
                </button>
              </div>
            )
          })}
        </div>
      </PickerFrame>

      <Dialog open={zoom !== null} onOpenChange={(open) => !open && setZoom(null)}>
        <DialogContent className="max-w-md border-cyan-300/30">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {zoom?.name}
              <GenderChip gender={zoom?.gender ?? null} />
            </DialogTitle>
          </DialogHeader>
          <div className="overflow-hidden rounded-lg border border-cyan-300/25 bg-slate-950">
            {zoom ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={zoom.url}
                alt={zoom.name}
                className="max-h-[70vh] w-full object-contain"
              />
            ) : null}
          </div>
          {zoom ? (
            <button
              type="button"
              onClick={() => {
                onSelect(zoom.key)
                setZoom(null)
              }}
              className="inline-flex items-center justify-center gap-2 rounded-md bg-cyan-400 px-3 py-2 text-sm font-medium text-cyan-950 transition-colors hover:bg-cyan-300"
            >
              <Check className="size-4" />
              Use this face
            </button>
          ) : null}
        </DialogContent>
      </Dialog>
    </>
  )
}

// --- Voices ---------------------------------------------------------------- //

export function SourceVoicePicker({
  state,
  selectedKey,
  onSelect,
}: {
  state: SourceAssetsState
  selectedKey: string | null
  onSelect: (key: string | null) => void
}) {
  // A single <audio> element drives playback (so only one clip plays at a time); it
  // lives in the DOM, so React tears it down — stopping audio — when this picker
  // unmounts (e.g. switching away from Voice Conversion).
  const audioRef = useRef<HTMLAudioElement>(null)
  const [playingKey, setPlayingKey] = useState<string | null>(null)
  const selectedName =
    state.assets.find((asset) => asset.key === selectedKey)?.name ?? null

  const togglePlay = (asset: SourceAsset) => {
    const audio = audioRef.current
    if (!audio) return
    if (playingKey === asset.key) {
      audio.pause()
      setPlayingKey(null)
      return
    }
    audio.src = asset.url
    audio.currentTime = 0
    void audio
      .play()
      .then(() => setPlayingKey(asset.key))
      .catch(() => setPlayingKey(null))
  }

  return (
    <PickerFrame
      label="Target voice"
      hint="Optional — pick a target voice, or keep the default reference."
      selectedName={selectedName}
      onClear={() => onSelect(null)}
      state={state}
      emptyText="No source voices are available yet."
    >
      <audio ref={audioRef} onEnded={() => setPlayingKey(null)} className="hidden" />
      <div className="grid max-h-64 gap-2 overflow-y-auto pr-0.5">
        {state.assets.map((asset) => {
          const isSelected = asset.key === selectedKey
          const isPlaying = playingKey === asset.key
          return (
            <div
              key={asset.key}
              className={cn(
                'flex items-center gap-2 rounded-lg border bg-slate-900/40 p-1.5 transition-all',
                isSelected
                  ? 'border-cyan-400 ring-2 ring-cyan-400/70'
                  : 'border-cyan-300/25 hover:border-cyan-300/60',
              )}
            >
              <button
                type="button"
                aria-label={isPlaying ? `Pause ${asset.name}` : `Play ${asset.name}`}
                onClick={() => togglePlay(asset)}
                className="grid size-9 shrink-0 place-items-center rounded-md bg-cyan-400 text-cyan-950 transition-colors hover:bg-cyan-300"
              >
                {isPlaying ? <Pause className="size-4" /> : <Play className="size-4" />}
              </button>

              <button
                type="button"
                aria-pressed={isSelected}
                aria-label={`Select ${asset.name}`}
                onClick={() => onSelect(isSelected ? null : asset.key)}
                className="flex flex-1 items-center justify-between gap-2 text-left"
              >
                <span className="flex items-center gap-2">
                  <span className="text-sm font-medium text-foreground">{asset.name}</span>
                  <GenderChip gender={asset.gender} />
                </span>
                {isSelected ? (
                  <span className="grid size-5 shrink-0 place-items-center rounded-full bg-cyan-400 text-cyan-950">
                    <Check className="size-3.5" />
                  </span>
                ) : (
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    {isPlaying ? 'Playing' : 'Tap to select'}
                  </span>
                )}
              </button>
            </div>
          )
        })}
      </div>
    </PickerFrame>
  )
}
