'use client'

/**
 * Orchestrates one anonymization run end-to-end so components stay declarative:
 *
 *   upload source → create edit → poll until settled → fetch presigned result URL
 *
 * The uploaded source is cached per `File`, so re-running with the same file but
 * different settings skips the (potentially large) re-upload. A single
 * `AbortController` cancels the in-flight upload and stops polling.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import {
  VideoApiError,
  createEdit,
  getEdit,
  getEditDownloadUrl,
  uploadVideo,
} from '@/lib/videos/client'
import type {
  VideoEditCreate,
  VideoEditPublic,
  VideoPublic,
} from '@/lib/videos/types'

export type AnonymizationPhase =
  | 'idle'
  | 'uploading'
  | 'processing'
  | 'completed'
  | 'failed'

export type AnonymizationState = {
  phase: AnonymizationPhase
  uploadPercent: number
  edit: VideoEditPublic | null
  resultUrl: string | null
  error: string | null
  isRunning: boolean
}

const POLL_INTERVAL_MS = 2000
const POLL_TIMEOUT_MS = 10 * 60 * 1000 // give long renders 10 minutes before bailing

function abortError(): DOMException {
  return new DOMException('Aborted', 'AbortError')
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortError())
      return
    }
    const timer = setTimeout(resolve, ms)
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(timer)
        reject(abortError())
      },
      { once: true },
    )
  })
}

async function pollUntilSettled(
  videoId: number,
  editId: number,
  signal: AbortSignal,
  onTick: (edit: VideoEditPublic) => void,
): Promise<VideoEditPublic> {
  const deadline = Date.now() + POLL_TIMEOUT_MS
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const edit = await getEdit(videoId, editId)
    onTick(edit)
    if (edit.status === 'completed' || edit.status === 'failed') {
      return edit
    }
    if (Date.now() > deadline) {
      throw new Error('Processing timed out. Please try again.')
    }
    await sleep(POLL_INTERVAL_MS, signal)
  }
}

export function useVideoAnonymization() {
  const [phase, setPhase] = useState<AnonymizationPhase>('idle')
  const [uploadPercent, setUploadPercent] = useState(0)
  const [edit, setEdit] = useState<VideoEditPublic | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  // Caches the uploaded source so re-runs of the same file skip the upload.
  const uploadedRef = useRef<{ file: File; video: VideoPublic } | null>(null)

  const cancel = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
  }, [])

  const reset = useCallback(() => {
    cancel()
    uploadedRef.current = null
    setPhase('idle')
    setUploadPercent(0)
    setEdit(null)
    setResultUrl(null)
    setError(null)
  }, [cancel])

  const run = useCallback(
    async (file: File, payload: VideoEditCreate) => {
      cancel()
      const controller = new AbortController()
      abortRef.current = controller
      const { signal } = controller

      setError(null)
      setResultUrl(null)
      setEdit(null)

      try {
        // 1. Upload the source once per file; reuse it across re-runs.
        let source = uploadedRef.current?.file === file ? uploadedRef.current.video : null
        if (!source) {
          setPhase('uploading')
          setUploadPercent(0)
          source = await uploadVideo(file, {
            signal,
            onProgress: ({ percent }) => setUploadPercent(percent),
          })
          uploadedRef.current = { file, video: source }
        }
        if (signal.aborted) return

        // 2. Create the edit and poll until it settles.
        setPhase('processing')
        const created = await createEdit(source.id, payload)
        setEdit(created)
        const settled = await pollUntilSettled(source.id, created.id, signal, setEdit)
        if (signal.aborted) return

        if (settled.status === 'failed') {
          setError(settled.error_message || 'Processing failed.')
          setPhase('failed')
          return
        }

        // 3. Resolve the presigned URL for the rendered output.
        const { url } = await getEditDownloadUrl(source.id, settled.id)
        if (signal.aborted) return
        setResultUrl(url)
        setPhase('completed')
      } catch (err) {
        if (isAbort(err) || signal.aborted) return
        setError(
          err instanceof VideoApiError || err instanceof Error
            ? err.message
            : 'Something went wrong. Please try again.',
        )
        setPhase('failed')
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null
        }
      }
    },
    [cancel],
  )

  // Stop any in-flight work if the component unmounts mid-run.
  useEffect(() => cancel, [cancel])

  const state: AnonymizationState = {
    phase,
    uploadPercent,
    edit,
    resultUrl,
    error,
    isRunning: phase === 'uploading' || phase === 'processing',
  }

  return { ...state, run, reset, cancel }
}
