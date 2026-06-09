'use client'

import { useState } from 'react'
import type { LucideIcon } from 'lucide-react'
import {
  AlertCircle,
  CalendarClock,
  ChevronDown,
  Clock,
  FileVideo,
  HardDrive,
  Loader2,
  MonitorPlay,
  PlayCircle,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { DeleteVideoButton } from '@/components/faceguard/dashboard/history/delete-video-button'
import { DownloadButton } from '@/components/faceguard/dashboard/history/download-button'
import { EditCard } from '@/components/faceguard/dashboard/history/edit-card'
import { VideoPreviewDialog } from '@/components/faceguard/dashboard/history/video-preview-dialog'
import {
  VideoApiError,
  getVideoDownloadUrl,
  listEdits,
} from '@/lib/videos/client'
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatResolution,
} from '@/lib/videos/format'
import type { VideoEditPublic, VideoPublic } from '@/lib/videos/types'

/** A compact icon + text fact in the source video's metadata row. */
function MetaChip({ icon: Icon, children }: { icon: LucideIcon; children: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md bg-cyan-500/10 px-2 py-1 text-xs text-muted-foreground">
      <Icon className="size-3.5 text-cyan-700 dark:text-cyan-300" />
      {children}
    </span>
  )
}

/**
 * One uploaded video: its metadata and source preview/download in the header, and
 * its protection runs in the body. Edits are fetched lazily the first time the card
 * is expanded, so the History list stays a single request on load.
 */
export function HistoryVideoCard({
  video,
  onDeleted,
}: {
  video: VideoPublic
  onDeleted: (videoId: number) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const [edits, setEdits] = useState<VideoEditPublic[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadEdits = async () => {
    setLoading(true)
    setError(null)
    try {
      setEdits(await listEdits(video.id))
    } catch (err) {
      setError(
        err instanceof VideoApiError || err instanceof Error
          ? err.message
          : 'Could not load edits.',
      )
    } finally {
      setLoading(false)
    }
  }

  const toggle = () => {
    const next = !expanded
    setExpanded(next)
    if (next && edits === null && !loading) {
      void loadEdits()
    }
  }

  const resolution = formatResolution(video.width, video.height)

  return (
    <div className="overflow-hidden rounded-xl border border-cyan-300/30 bg-background/60">
      <div className="flex items-start justify-between gap-3 p-4">
        <button
          type="button"
          onClick={toggle}
          aria-expanded={expanded}
          className="flex min-w-0 flex-1 items-start gap-3 text-left"
        >
          <ChevronDown
            className={cn(
              'mt-1 size-4 shrink-0 text-cyan-700 transition-transform dark:text-cyan-300',
              expanded && 'rotate-180',
            )}
          />
          <span className="mt-0.5 flex size-10 shrink-0 items-center justify-center rounded-lg bg-cyan-500/15 text-cyan-700 dark:text-cyan-300">
            <FileVideo className="size-5" />
          </span>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-foreground">
              {video.original_filename}
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <MetaChip icon={HardDrive}>{formatBytes(video.size_bytes)}</MetaChip>
              <MetaChip icon={Clock}>{formatDuration(video.duration_sec)}</MetaChip>
              {resolution ? <MetaChip icon={MonitorPlay}>{resolution}</MetaChip> : null}
              <MetaChip icon={CalendarClock}>{formatDateTime(video.created_at)}</MetaChip>
            </div>
          </div>
        </button>

        <div className="flex shrink-0 items-center gap-1">
          <VideoPreviewDialog
            getUrl={() => getVideoDownloadUrl(video.id)}
            title={video.original_filename}
            description="Original source video"
          >
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              aria-label="Preview original video"
              title="Preview original video"
              className="text-cyan-700 hover:bg-cyan-500/20 dark:text-cyan-300"
            >
              <PlayCircle className="size-4" />
            </Button>
          </VideoPreviewDialog>
          <DownloadButton
            getUrl={() => getVideoDownloadUrl(video.id)}
            label="Download original video"
          />
          <DeleteVideoButton
            videoId={video.id}
            videoName={video.original_filename}
            onDeleted={onDeleted}
          />
        </div>
      </div>

      {expanded && (
        <div className="border-t border-cyan-300/15 bg-cyan-500/[0.03] px-4 py-4">
          {loading ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Loading protection runs…
            </div>
          ) : error ? (
            <div className="flex items-center justify-between gap-3 text-sm text-rose-500 dark:text-rose-300">
              <span className="flex items-center gap-2">
                <AlertCircle className="size-4" />
                {error}
              </span>
              <Button
                size="sm"
                variant="outline"
                onClick={loadEdits}
                className="border-cyan-300/35"
              >
                Retry
              </Button>
            </div>
          ) : edits && edits.length > 0 ? (
            <div className="space-y-3">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                {edits.length} protection {edits.length === 1 ? 'run' : 'runs'}
              </p>
              {edits.map((edit) => (
                <EditCard key={edit.id} video={video} edit={edit} />
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              No protection runs yet for this video.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
