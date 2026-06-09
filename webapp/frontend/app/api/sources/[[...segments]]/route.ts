/**
 * Same-origin proxy for the backend source-asset catalogs (`/api/sources/*`).
 *
 * Mirrors the videos proxy: the httpOnly session token is attached server-side and
 * the request is forwarded to FastAPI's `/sources` endpoints. These catalogs are
 * read-only, so only GET is exposed.
 */

import type { NextRequest } from 'next/server'

import { DEFAULT_BACKEND_API_BASE_URL } from '@/lib/auth/constants'
import { getSessionToken } from '@/lib/auth/server'

export const dynamic = 'force-dynamic'

const BACKEND_BASE_URL = (
  process.env.BACKEND_API_BASE_URL?.trim() || DEFAULT_BACKEND_API_BASE_URL
).replace(/\/$/, '')

type RouteContext = { params: Promise<{ segments?: string[] }> }

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
  const targetUrl = `${BACKEND_BASE_URL}/sources${subPath}${request.nextUrl.search}`

  const headers = new Headers()
  headers.set('Authorization', `Bearer ${token}`)
  headers.set('Accept', 'application/json')

  let upstream: Response
  try {
    upstream = await fetch(targetUrl, { method: 'GET', headers })
  } catch {
    return jsonError('Could not reach the source-asset service.', 502)
  }

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
