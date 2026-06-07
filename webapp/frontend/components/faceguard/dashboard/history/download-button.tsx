'use client'

import { useState } from 'react'
import { AlertCircle, Download, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { VideoApiError } from '@/lib/videos/client'
import type { PresignedUrlResponse } from '@/lib/videos/types'

/** Open a presigned URL as a download. Cross-origin (R2) links fall back to a new
 * tab when the browser ignores the `download` attribute — the user can still save. */
function triggerDownload(url: string, filename: string): void {
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.target = '_blank'
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
  filename,
  label,
  disabled = false,
  disabledHint,
}: {
  getUrl: () => Promise<PresignedUrlResponse>
  filename: string
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
      triggerDownload(url, filename)
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
