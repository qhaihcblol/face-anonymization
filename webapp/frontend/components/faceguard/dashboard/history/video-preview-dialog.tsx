'use client'

import { useState, type ReactNode } from 'react'
import { AlertCircle, Loader2 } from 'lucide-react'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { VideoApiError } from '@/lib/videos/client'
import type { PresignedUrlResponse } from '@/lib/videos/types'

type PreviewStatus = 'idle' | 'loading' | 'ready' | 'error'

/**
 * Plays a stored video inline in a dialog. The short-lived presigned URL is
 * resolved only when the dialog opens (links expire, so we never prefetch) and
 * dropped when it closes. `<video>` ignores the `Content-Disposition: attachment`
 * the download endpoint signs, so the same URL streams fine for playback.
 */
export function VideoPreviewDialog({
  getUrl,
  title,
  description,
  children,
}: {
  getUrl: () => Promise<PresignedUrlResponse>
  title: string
  description?: string
  children: ReactNode
}) {
  const [status, setStatus] = useState<PreviewStatus>('idle')
  const [url, setUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleOpenChange = async (open: boolean) => {
    if (!open) {
      // Drop the (possibly expired-by-next-open) URL so reopening re-resolves it.
      setStatus('idle')
      setUrl(null)
      setError(null)
      return
    }

    setStatus('loading')
    setError(null)
    try {
      const resolved = await getUrl()
      setUrl(resolved.url)
      setStatus('ready')
    } catch (err) {
      setError(
        err instanceof VideoApiError || err instanceof Error
          ? err.message
          : 'Could not load the video.',
      )
      setStatus('error')
    }
  }

  return (
    <Dialog onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="max-w-3xl border-cyan-300/30">
        <DialogHeader>
          <DialogTitle className="truncate">{title}</DialogTitle>
          {description ? <DialogDescription>{description}</DialogDescription> : null}
        </DialogHeader>

        <div className="relative aspect-video overflow-hidden rounded-lg border border-cyan-300/25 bg-slate-950">
          {status === 'ready' && url ? (
            <video
              key={url}
              src={url}
              controls
              autoPlay
              playsInline
              className="h-full w-full object-contain"
            />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center text-cyan-100">
              {status === 'error' ? (
                <>
                  <AlertCircle className="size-9 text-rose-400" />
                  <p className="text-sm text-rose-200">{error}</p>
                </>
              ) : (
                <>
                  <Loader2 className="size-9 animate-spin text-cyan-300/80" />
                  <p className="text-sm text-cyan-100/80">Loading video…</p>
                </>
              )}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
