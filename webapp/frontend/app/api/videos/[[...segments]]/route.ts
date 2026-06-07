/**
 * Same-origin proxy for the backend video API.
 *
 * The session token lives in an httpOnly cookie that browser JS can't read, so the
 * client calls these same-origin routes and we forward to FastAPI with the bearer
 * token attached server-side. One optional catch-all handles every `/api/videos`
 * sub-path and method, so new backend endpoints need no new route files.
 *
 * Request bodies (incl. the multipart upload) are streamed through rather than
 * buffered, so large videos don't sit in memory.
 */

import type { NextRequest } from 'next/server'

import { DEFAULT_BACKEND_API_BASE_URL } from '@/lib/auth/constants'
import { getSessionToken } from '@/lib/auth/server'

export const dynamic = 'force-dynamic'

const BACKEND_BASE_URL = (
  process.env.BACKEND_API_BASE_URL?.trim() || DEFAULT_BACKEND_API_BASE_URL
).replace(/\/$/, '')

type RouteContext = { params: Promise<{ segments?: string[] }> }

// `RequestInit` doesn't yet type `duplex`, required by undici to stream a body.
type StreamingRequestInit = RequestInit & { duplex?: 'half' }

function jsonError(detail: string, status: number): Response {
  return Response.json({ detail }, { status })
}

async function proxy(request: NextRequest, context: RouteContext): Promise<Response> {
  const token = await getSessionToken()
  if (!token) {
    return jsonError('Not authenticated.', 401)
  }

  const { segments } = await context.params
  const subPath = segments?.length ? `/${segments.join('/')}` : ''
  const targetUrl = `${BACKEND_BASE_URL}/videos${subPath}${request.nextUrl.search}`

  const headers = new Headers()
  headers.set('Authorization', `Bearer ${token}`)
  headers.set('Accept', 'application/json')
  // Preserve the multipart boundary / JSON content type from the client.
  const contentType = request.headers.get('content-type')
  if (contentType) {
    headers.set('content-type', contentType)
  }

  const init: StreamingRequestInit = { method: request.method, headers }
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    init.body = request.body
    init.duplex = 'half'
  }

  let upstream: Response
  try {
    upstream = await fetch(targetUrl, init)
  } catch {
    return jsonError('Could not reach the video service.', 502)
  }

  // Pass the backend status + body straight through (incl. its error JSON).
  const responseHeaders = new Headers()
  const upstreamContentType = upstream.headers.get('content-type')
  if (upstreamContentType) {
    responseHeaders.set('content-type', upstreamContentType)
  }
  return new Response(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  })
}

export const GET = proxy
export const POST = proxy
export const PUT = proxy
export const PATCH = proxy
export const DELETE = proxy
