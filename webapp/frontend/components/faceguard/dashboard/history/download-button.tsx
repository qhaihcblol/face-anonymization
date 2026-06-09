'use client'

import { useState } from 'react'
import { AlertCircle, Download, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { VideoApiError } from '@/lib/videos/client'
import type { PresignedUrlResponse } from '@/lib/videos/types'

/** Navigate to a presigned URL to save the file. The backend signs the URL with
 * `Content-Disposition: attachment` (and the real filename), so the browser
 * downloads it rather than opening it — even though R2 is a different origin. */
function startDownload(url: string): void {
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.rel = 'noreferrer'
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
}

type DownloadStatus = 'idle' | 'loading' | 'error'

/**
 * Icon button that resolves a short-lived presigned URL only when clicked (links
 * expire, so we never prefetch). Reused for the source video and every processed
 * output. The hover hint uses the native `title` so it needs no ref plumbing.
 */
export function DownloadButton({
  getUrl,
  label,
  disabled = false,
  disabledHint,
}: {
  getUrl: () => Promise<PresignedUrlResponse>
  label: string
  disabled?: boolean
  disabledHint?: string
}) {
  const [status, setStatus] = useState<DownloadStatus>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const handleClick = async () => {
    if (status === 'loading') {
      return
    }
    setStatus('loading')
    setErrorMessage(null)
    try {
      const { url } = await getUrl()
      startDownload(url)
      setStatus('idle')
    } catch (error) {
      setErrorMessage(
        error instanceof VideoApiError || error instanceof Error
          ? error.message
          : 'Download failed.',
      )
      setStatus('error')
    }
  }

  const icon =
    status === 'loading' ? (
      <Loader2 className="size-4 animate-spin" />
    ) : status === 'error' ? (
      <AlertCircle className="size-4 text-rose-500 dark:text-rose-300" />
    ) : (
      <Download className="size-4" />
    )

  return (
    <Button
      type="button"
      size="icon-sm"
      variant="ghost"
      aria-label={label}
      title={disabled ? disabledHint ?? label : errorMessage ?? label}
      disabled={disabled || status === 'loading'}
      onClick={handleClick}
      className="text-cyan-700 hover:bg-cyan-500/20 disabled:opacity-40 dark:text-cyan-300"
    >
      {icon}
    </Button>
  )
}
