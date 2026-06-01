import { DEFAULT_BACKEND_API_BASE_URL } from '@/lib/auth/constants'
import type {
  AnonymizeRequest,
  AnonymizeResponse,
  FaceSwapRequest,
  FaceSwapResponse,
  VideoUploadResponse,
} from '@/lib/video/types'

// Browser-side client: the /videos endpoints are public and CORS-enabled for the
// frontend origin, so the upload (up to 2GB) and processing calls go straight to
// the backend instead of being proxied through a Next server action (which caps
// request bodies). Override the host with NEXT_PUBLIC_BACKEND_API_BASE_URL.
const backendApiBaseUrl = (
  process.env.NEXT_PUBLIC_BACKEND_API_BASE_URL?.trim() ||
  DEFAULT_BACKEND_API_BASE_URL
).replace(/\/$/, '')

export class VideoApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'VideoApiError'
    this.status = status
  }
}

type ValidationDetail = { msg: string }

function extractErrorDetail(payload: unknown, fallbackMessage: string): string {
  if (!payload || typeof payload !== 'object' || !('detail' in payload)) {
    return fallbackMessage
  }

  const detail = (payload as { detail: unknown }).detail
  if (typeof detail === 'string') {
    return detail
  }

  if (
    Array.isArray(detail) &&
    detail.length > 0 &&
    typeof detail[0] === 'object' &&
    detail[0] !== null &&
    'msg' in detail[0]
  ) {
    return (detail[0] as ValidationDetail).msg || fallbackMessage
  }

  return fallbackMessage
}

async function readJsonPayload(response: Response): Promise<unknown> {
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

async function videoRequest<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${backendApiBaseUrl}${path}`, {
      ...init,
      cache: 'no-store',
      headers: {
        Accept: 'application/json',
        ...init.headers,
      },
    })
  } catch {
    throw new VideoApiError(503, 'Could not connect to the processing backend.')
  }

  const payload = await readJsonPayload(response)
  if (!response.ok) {
    throw new VideoApiError(
      response.status,
      extractErrorDetail(payload, `Request failed with status ${response.status}.`),
    )
  }

  return payload as T
}

export async function uploadVideo(file: File): Promise<VideoUploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  // No Content-Type header: the browser sets the multipart boundary itself.
  return videoRequest<VideoUploadResponse>('/videos/upload', {
    method: 'POST',
    body: formData,
  })
}

export async function anonymizeVideo(
  videoId: string,
  payload: AnonymizeRequest,
): Promise<AnonymizeResponse> {
  return videoRequest<AnonymizeResponse>(`/videos/${videoId}/anonymize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function faceSwapVideo(
  videoId: string,
  payload: FaceSwapRequest,
): Promise<FaceSwapResponse> {
  return videoRequest<FaceSwapResponse>(`/videos/${videoId}/face-swap`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}
