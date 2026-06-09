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
  VideoUploadComplete,
  VideoUploadInit,
  VideoUploadTicket,
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

// --- Upload (presigned, direct-to-storage) --------------------------------- //

export type UploadProgress = { loaded: number; total: number; percent: number }

export type UploadOptions = {
  onProgress?: (progress: UploadProgress) => void
  signal?: AbortSignal
}

type VideoMetadata = {
  duration_sec: number | null
  width: number | null
  height: number | null
}

const EMPTY_METADATA: VideoMetadata = { duration_sec: null, width: null, height: null }

/**
 * Read duration + dimensions from a video file via a throwaway `<video>` element.
 * The backend never sees the bytes (direct-to-storage uploads), so this is the only
 * place that metadata can be captured. Best-effort: any failure resolves to nulls so
 * it can never block an upload.
 */
function probeVideoMetadata(file: File): Promise<VideoMetadata> {
  if (typeof document === 'undefined') {
    return Promise.resolve(EMPTY_METADATA)
  }
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const video = document.createElement('video')
    video.preload = 'metadata'

    const finish = (metadata: VideoMetadata) => {
      URL.revokeObjectURL(url)
      resolve(metadata)
    }

    video.onloadedmetadata = () => {
      finish({
        duration_sec: Number.isFinite(video.duration) ? video.duration : null,
        width: video.videoWidth || null,
        height: video.videoHeight || null,
      })
    }
    video.onerror = () => finish(EMPTY_METADATA)
    video.src = url
  })
}

/** Step 1: ask the backend for a presigned URL to upload the file to. */
function createUploadTicket(init: VideoUploadInit): Promise<VideoUploadTicket> {
  return requestJson<VideoUploadTicket>('/upload-url', {
    method: 'POST',
    body: JSON.stringify(init),
  })
}

/** Step 3: tell the backend the upload finished so it registers the video. */
function completeUpload(payload: VideoUploadComplete): Promise<VideoPublic> {
  return requestJson<VideoPublic>('', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

/**
 * Step 2: PUT the file straight to object storage (R2). `fetch` can't report
 * upload progress, so we use XHR. This call is cross-origin to the storage host,
 * which must allow PUT via its CORS policy.
 */
function putToStorage(
  ticket: VideoUploadTicket,
  file: File,
  options: UploadOptions,
): Promise<void> {
  const { onProgress, signal } = options

  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new VideoApiError(0, 'Upload cancelled.'))
      return
    }

    const xhr = new XMLHttpRequest()
    xhr.open(ticket.method || 'PUT', ticket.upload_url)
    for (const [name, value] of Object.entries(ticket.headers)) {
      xhr.setRequestHeader(name, value)
    }

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
        resolve()
      } else {
        reject(new VideoApiError(xhr.status, `Upload to storage failed (${xhr.status}).`))
      }
    }
    xhr.onerror = () => {
      cleanup()
      reject(
        new VideoApiError(
          0,
          'Network error during upload. The storage CORS policy may be blocking it.',
        ),
      )
    }
    xhr.onabort = () => {
      cleanup()
      reject(new VideoApiError(0, 'Upload cancelled.'))
    }

    xhr.send(file)
  })
}

/**
 * Upload one video file directly to object storage, then register it with the
 * backend. Three steps: presign → PUT to storage → confirm. The file bytes never
 * pass through the app servers, so uploads aren't bound by request-body limits.
 */
export async function uploadVideo(
  file: File,
  options: UploadOptions = {},
): Promise<VideoPublic> {
  if (options.signal?.aborted) {
    throw new VideoApiError(0, 'Upload cancelled.')
  }

  const contentType = file.type || null
  // Probe metadata while the presign round-trip is in flight — it adds no latency.
  const metadataPromise = probeVideoMetadata(file)
  const ticket = await createUploadTicket({
    filename: file.name,
    content_type: contentType,
    size_bytes: file.size,
  })

  await putToStorage(ticket, file, options)

  return completeUpload({
    storage_key: ticket.storage_key,
    original_filename: file.name,
    content_type: contentType,
    ...(await metadataPromise),
  })
}
