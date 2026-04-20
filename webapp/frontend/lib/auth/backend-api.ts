import 'server-only'

import {
  DEFAULT_BACKEND_API_BASE_URL,
} from '@/lib/auth/constants'
import type {
  AuthResponse,
  BackendErrorResponse,
  LoginPayload,
  RegisterPayload,
  UserPublic,
} from '@/lib/auth/types'

type RequestConfig = Omit<RequestInit, 'cache'>

const configuredBaseUrl = process.env.BACKEND_API_BASE_URL?.trim()

const backendApiBaseUrl = (
  configuredBaseUrl || DEFAULT_BACKEND_API_BASE_URL
).replace(/\/$/, '')

function isValidationDetail(
  detail: unknown,
): detail is BackendErrorResponse['detail'] & Array<{ msg: string }> {
  return (
    Array.isArray(detail) &&
    detail.length > 0 &&
    typeof detail[0] === 'object' &&
    detail[0] !== null &&
    'msg' in detail[0]
  )
}

function extractErrorDetail(payload: unknown, fallbackMessage: string): string {
  if (!payload || typeof payload !== 'object' || !("detail" in payload)) {
    return fallbackMessage
  }

  const detail = (payload as BackendErrorResponse).detail
  if (typeof detail === 'string') {
    return detail
  }

  if (isValidationDetail(detail)) {
    return detail[0]?.msg ?? fallbackMessage
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

export class BackendApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'BackendApiError'
    this.status = status
  }
}

async function backendRequest<T>(
  path: string,
  config: RequestConfig,
): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${backendApiBaseUrl}${path}`, {
      ...config,
      cache: 'no-store',
      headers: {
        Accept: 'application/json',
        ...(config.body ? { 'Content-Type': 'application/json' } : {}),
        ...config.headers,
      },
    })
  } catch {
    throw new BackendApiError(503, 'Could not connect to backend API.')
  }

  const payload = await readJsonPayload(response)
  if (!response.ok) {
    throw new BackendApiError(
      response.status,
      extractErrorDetail(payload, `Request failed with status ${response.status}.`),
    )
  }

  return payload as T
}

export async function loginToBackend(payload: LoginPayload): Promise<AuthResponse> {
  return backendRequest<AuthResponse>('/auth/login', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function registerToBackend(
  payload: RegisterPayload,
): Promise<AuthResponse> {
  return backendRequest<AuthResponse>('/auth/register', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function getMeFromBackend(accessToken: string): Promise<UserPublic> {
  return backendRequest<UserPublic>('/users/me', {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  })
}
