/**
 * Browser-side client for the video API. Every call targets the same-origin
 * `/api/videos` proxy (see `app/api/videos/[[...segments]]/route.ts`), which
 * attaches the httpOnly session token server-side — so nothing here deals with
 * auth. JSON calls use `fetch`; the upload uses `XMLHttpRequest` for progress.
 */

import type {
  PresignedUrlResponse,
  VideoEditCreate,
  VideoEditPublic,
  VideoPublic,
} from '@/lib/videos/types'

const BASE_PATH = '/api/videos'

export class VideoApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'VideoApiError'
    this.status = status
  }
}

/** Pull a useful message out of a FastAPI error body (`{ detail }`). */
function messageFromBody(body: unknown, status: number): string {
  const fallback = `Request failed (${status}).`
  if (!body || typeof body !== 'object' || !('detail' in body)) {
    return fallback
  }
  const detail = (body as { detail: unknown }).detail
  if (typeof detail === 'string') {
    return detail
  }
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0]
    if (first && typeof first === 'object' && 'msg' in first) {
      return String((first as { msg: unknown }).msg)
    }
  }
  return fallback
}

async function readJson(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') || ''
  if (!contentType.includes('application/json')) {
    return null
  }
  try {
    return await response.json()
  } catch {
    return null
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE_PATH}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
    })
  } catch {
    throw new VideoApiError(0, 'Could not reach the server. Check your connection.')
  }

  const body = await readJson(response)
  if (!response.ok) {
    throw new VideoApiError(response.status, messageFromBody(body, response.status))
  }
  return body as T
}

// --- Edits + downloads (JSON) ---------------------------------------------- //

export function listVideos(): Promise<VideoPublic[]> {
  return requestJson<VideoPublic[]>('')
}

export function deleteVideo(videoId: number): Promise<void> {
  return requestJson<void>(`/${videoId}`, { method: 'DELETE' })
}

export function getVideoDownloadUrl(videoId: number): Promise<PresignedUrlResponse> {
  return requestJson<PresignedUrlResponse>(`/${videoId}/download-url`)
}

export function createEdit(
  videoId: number,
  payload: VideoEditCreate,
): Promise<VideoEditPublic> {
  return requestJson<VideoEditPublic>(`/${videoId}/edits`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function listEdits(videoId: number): Promise<VideoEditPublic[]> {
  return requestJson<VideoEditPublic[]>(`/${videoId}/edits`)
}

export function getEdit(
  videoId: number,
  editId: number,
): Promise<VideoEditPublic> {
  return requestJson<VideoEditPublic>(`/${videoId}/edits/${editId}`)
}

export function getEditDownloadUrl(
  videoId: number,
  editId: number,
): Promise<PresignedUrlResponse> {
  return requestJson<PresignedUrlResponse>(`/${videoId}/edits/${editId}/download-url`)
}

// --- Upload (XHR, for progress) -------------------------------------------- //

export type UploadProgress = { loaded: number; total: number; percent: number }

export type UploadOptions = {
  onProgress?: (progress: UploadProgress) => void
  signal?: AbortSignal
}

/** Upload one video file. `fetch` can't report upload progress, so we use XHR. */
export function uploadVideo(
  file: File,
  options: UploadOptions = {},
): Promise<VideoPublic> {
  const { onProgress, signal } = options

  return new Promise<VideoPublic>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new VideoApiError(0, 'Upload cancelled.'))
      return
    }

    const formData = new FormData()
    formData.append('file', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', BASE_PATH)
    xhr.responseType = 'json'

    const onAbort = () => xhr.abort()
    signal?.addEventListener('abort', onAbort, { once: true })
    const cleanup = () => signal?.removeEventListener('abort', onAbort)

    if (onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          onProgress({
            loaded: event.loaded,
            total: event.total,
            percent: Math.round((event.loaded / event.total) * 100),
          })
        }
      }
    }

    xhr.onload = () => {
      cleanup()
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as VideoPublic)
      } else {
        reject(new VideoApiError(xhr.status, messageFromBody(xhr.response, xhr.status)))
      }
    }
    xhr.onerror = () => {
      cleanup()
      reject(new VideoApiError(0, 'Network error during upload.'))
    }
    xhr.onabort = () => {
      cleanup()
      reject(new VideoApiError(0, 'Upload cancelled.'))
    }

    xhr.send(formData)
  })
}
