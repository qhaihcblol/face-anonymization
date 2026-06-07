/**
 * Issues a one-shot WebSocket URL for the live-camera socket.
 *
 * The backend authenticates the live socket with a `?token=` query parameter (a
 * browser can't set an `Authorization` header on a WebSocket). The session token
 * lives in an httpOnly cookie that client JS can't read, so this same-origin route
 * reads it server-side and hands back the fully-formed `ws(s)://…/live/ws?token=…`
 * URL — the backend base URL and the token never need to be exposed as build-time
 * env to the browser. Mirrors the auth model of the `/api/videos` proxy.
 */

import { DEFAULT_BACKEND_API_BASE_URL } from '@/lib/auth/constants'
import { getSessionToken } from '@/lib/auth/server'

export const dynamic = 'force-dynamic'

const BACKEND_BASE_URL = (
  process.env.BACKEND_API_BASE_URL?.trim() || DEFAULT_BACKEND_API_BASE_URL
).replace(/\/$/, '')

/** `http(s)://host/api` -> `ws(s)://host/api` (https -> wss, http -> ws). */
function toWebSocketBase(httpBase: string): string {
  return httpBase.replace(/^http/, 'ws')
}

export async function GET(): Promise<Response> {
  const token = await getSessionToken()
  if (!token) {
    return Response.json({ detail: 'Not authenticated.' }, { status: 401 })
  }

  const url = `${toWebSocketBase(BACKEND_BASE_URL)}/live/ws?token=${encodeURIComponent(token)}`
  return Response.json({ url })
}
