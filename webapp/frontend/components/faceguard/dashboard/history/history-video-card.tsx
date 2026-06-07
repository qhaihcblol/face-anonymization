'use client'

import { useState } from 'react'
import { AlertCircle, ChevronDown, FileVideo, Loader2 } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { DeleteVideoButton } from '@/components/faceguard/dashboard/history/delete-video-button'
import { DownloadButton } from '@/components/faceguard/dashboard/history/download-button'
import {
  VideoApiError,
  getEditDownloadUrl,
  getVideoDownloadUrl,
  listEdits,
} from '@/lib/videos/client'
import {
  editStatusBadgeClass,
  editStatusLabel,
  summarizeEditParams,
} from '@/lib/videos/options'
import { formatBytes, formatDateTime, formatDuration } from '@/lib/videos/format'
import type { VideoEditPublic, VideoPublic } from '@/lib/videos/types'

/**
 * One uploaded video and its protection runs. Edits are fetched lazily the first
 * time the card is expanded, so the History list stays one request on load.
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

  const meta = [
    formatBytes(video.size_bytes),
    formatDuration(video.duration_sec),
    formatDateTime(video.created_at),
  ].join(' • ')

  return (
    <div className="rounded-xl border border-cyan-300/30 bg-background/60">
      <div className="flex items-center justify-between gap-3 p-4">
        <button
          type="button"
          onClick={toggle}
          aria-expanded={expanded}
          className="flex min-w-0 flex-1 items-center gap-3 text-left"
        >
          <ChevronDown
            className={cn(
              'size-4 shrink-0 text-cyan-700 transition-transform dark:text-cyan-300',
              expanded && 'rotate-180',
            )}
          />
          <FileVideo className="size-5 shrink-0 text-cyan-700 dark:text-cyan-300" />
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-foreground">
              {video.original_filename}
            </p>
            <p className="truncate text-xs text-muted-foreground">{meta}</p>
          </div>
        </button>

        <div className="flex shrink-0 items-center gap-1">
          <DownloadButton
            getUrl={() => getVideoDownloadUrl(video.id)}
            filename={video.original_filename}
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
        <div className="border-t border-cyan-300/15 px-4 py-3">
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
            <ul className="space-y-2">
              {edits.map((edit) => (
                <li
                  key={edit.id}
                  className="flex items-center justify-between gap-3 rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-3 py-2"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <Badge className={editStatusBadgeClass[edit.status]}>
                        {editStatusLabel[edit.status]}
                      </Badge>
                      <span className="text-xs text-muted-foreground">#{edit.id}</span>
                    </div>
                    <p className="mt-1 truncate text-sm text-foreground">
                      {summarizeEditParams(edit.params)}
                    </p>
                    {edit.status === 'failed' && edit.error_message ? (
                      <p className="truncate text-xs text-rose-500 dark:text-rose-300">
                        {edit.error_message}
                      </p>
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        {formatDateTime(edit.created_at)}
                      </p>
                    )}
                  </div>

                  <DownloadButton
                    getUrl={() => getEditDownloadUrl(video.id, edit.id)}
                    filename={`protected-${video.original_filename}`}
                    label="Download processed video"
                    disabled={edit.status !== 'completed'}
                    disabledHint={
                      edit.status === 'failed'
                        ? 'This run failed — nothing to download'
                        : 'Available once processing completes'
                    }
                  />
                </li>
              ))}
            </ul>
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
