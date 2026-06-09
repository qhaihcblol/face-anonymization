'use client'

import type { LucideIcon } from 'lucide-react'
import {
  AudioLines,
  Clock,
  Gauge,
  PlayCircle,
  ScanFace,
  Scissors,
  SquareDashed,
} from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { DownloadButton } from '@/components/faceguard/dashboard/history/download-button'
import { VideoPreviewDialog } from '@/components/faceguard/dashboard/history/video-preview-dialog'
import { getEditDownloadUrl } from '@/lib/videos/client'
import { editStatusBadgeClass, editStatusLabel, summarizeEdit } from '@/lib/videos/options'
import { formatDateTime, formatElapsed } from '@/lib/videos/format'
import type { VideoEditPublic, VideoPublic } from '@/lib/videos/types'

/** A single labelled fact in the edit's detail grid. */
function DetailItem({
  icon: Icon,
  label,
  value,
  swatch,
}: {
  icon: LucideIcon
  label: string
  value: string
  swatch?: string | null
}) {
  return (
    <div className="flex items-start gap-2">
      <Icon className="mt-0.5 size-4 shrink-0 text-cyan-700 dark:text-cyan-300" />
      <div className="min-w-0">
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </p>
        <p className="flex items-center gap-1.5 truncate text-sm text-foreground">
          {swatch ? (
            <span
              className="inline-block size-3 shrink-0 rounded-full border border-border"
              style={{ backgroundColor: swatch }}
            />
          ) : null}
          {value}
        </p>
      </div>
    </div>
  )
}

/** One protection run: its status, the exact image/audio settings used, timing,
 * and (once completed) inline preview + download of the result. */
export function EditCard({
  video,
  edit,
}: {
  video: VideoPublic
  edit: VideoEditPublic
}) {
  const summary = summarizeEdit(edit.params)
  const completed = edit.status === 'completed'
  const elapsed = formatElapsed(edit.created_at, edit.completed_at)
  const getUrl = () => getEditDownloadUrl(video.id, edit.id)
  const outputName = `protected-${video.original_filename}`

  return (
    <article className="rounded-lg border border-cyan-300/20 bg-cyan-500/[0.07] p-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Badge className={editStatusBadgeClass[edit.status]}>
            {editStatusLabel[edit.status]}
          </Badge>
          <span className="text-xs text-muted-foreground">Run #{edit.id}</span>
        </div>

        {completed ? (
          <div className="flex items-center gap-1">
            <VideoPreviewDialog
              getUrl={getUrl}
              title="Protected output"
              description={outputName}
            >
              <Button
                type="button"
                size="sm"
                variant="ghost"
                className="text-cyan-700 hover:bg-cyan-500/20 dark:text-cyan-300"
              >
                <PlayCircle className="size-4" />
                Preview
              </Button>
            </VideoPreviewDialog>
            <DownloadButton getUrl={getUrl} label="Download processed video" />
          </div>
        ) : null}
      </header>

      {edit.status === 'failed' && edit.error_message ? (
        <p className="mt-3 rounded-md bg-rose-500/10 px-3 py-2 text-xs text-rose-600 dark:text-rose-300">
          {edit.error_message}
        </p>
      ) : (
        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
          <DetailItem
            icon={ScanFace}
            label="Image"
            value={
              summary.image.detail
                ? `${summary.image.method} · ${summary.image.detail}`
                : summary.image.method
            }
            swatch={summary.image.color}
          />
          <DetailItem
            icon={AudioLines}
            label="Audio"
            value={
              summary.audio.detail
                ? `${summary.audio.mode} · ${summary.audio.detail}`
                : summary.audio.mode
            }
          />
          {summary.range ? (
            <DetailItem icon={Scissors} label="Range" value={summary.range} />
          ) : null}
          {summary.fps ? (
            <DetailItem icon={Gauge} label="Frame rate" value={summary.fps} />
          ) : null}
        </div>
      )}

      <footer className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-cyan-300/15 pt-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <Clock className="size-3.5" />
          {formatDateTime(edit.created_at)}
        </span>
        {elapsed ? <span>Took {elapsed}</span> : null}
        {summary.drawBoxes ? (
          <span className="flex items-center gap-1.5">
            <SquareDashed className="size-3.5" />
            Detection boxes
          </span>
        ) : null}
      </footer>
    </article>
  )
}
