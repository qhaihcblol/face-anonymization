import 'server-only'

import { cache } from 'react'
import { cookies } from 'next/headers'
import { redirect } from 'next/navigation'

import {
  AUTH_SESSION_COOKIE_NAME,
  DEFAULT_POST_AUTH_REDIRECT,
} from '@/lib/auth/constants'
import { BackendApiError, getMeFromBackend } from '@/lib/auth/backend-api'
import type { UserPublic } from '@/lib/auth/types'

function sanitizeNextPath(nextPath: string): string {
  if (!nextPath.startsWith('/')) {
    return DEFAULT_POST_AUTH_REDIRECT
  }

  if (nextPath.startsWith('//')) {
    return DEFAULT_POST_AUTH_REDIRECT
  }

  return nextPath
}

export async function setSessionToken(accessToken: string): Promise<void> {
  const cookieStore = await cookies()
  cookieStore.set(AUTH_SESSION_COOKIE_NAME, accessToken, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
  })
}

export async function clearSessionToken(): Promise<void> {
  const cookieStore = await cookies()
  cookieStore.delete(AUTH_SESSION_COOKIE_NAME)
}

export async function getSessionToken(): Promise<string | null> {
  const cookieStore = await cookies()
  return cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value ?? null
}

export const getCurrentUser = cache(async (): Promise<UserPublic | null> => {
  const accessToken = await getSessionToken()
  if (!accessToken) {
    return null
  }

  try {
    return await getMeFromBackend(accessToken)
  } catch (error) {
    if (error instanceof BackendApiError) {
      if (error.status === 401 || error.status === 403) {
        return null
      }
    }
    return null
  }
})

export async function redirectIfAuthenticated(
  destination: string = DEFAULT_POST_AUTH_REDIRECT,
): Promise<void> {
  const currentUser = await getCurrentUser()
  if (currentUser) {
    redirect(sanitizeNextPath(destination))
  }
}

export async function requireAuthenticatedUser(
  redirectAfterLogin: string = DEFAULT_POST_AUTH_REDIRECT,
): Promise<UserPublic> {
  const currentUser = await getCurrentUser()
  if (!currentUser) {
    const safeRedirect = sanitizeNextPath(redirectAfterLogin)
    redirect(`/auth/login?next=${encodeURIComponent(safeRedirect)}`)
  }

  return currentUser
}
