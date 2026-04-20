'use server'

import { redirect } from 'next/navigation'
import { z } from 'zod'

import {
  DEFAULT_POST_AUTH_REDIRECT,
  MAX_BCRYPT_PASSWORD_BYTES,
} from '@/lib/auth/constants'
import {
  BackendApiError,
  loginToBackend,
  registerToBackend,
} from '@/lib/auth/backend-api'
import { clearSessionToken, setSessionToken } from '@/lib/auth/server'

type AuthActionState = {
  error: string | null
}

const textEncoder = new TextEncoder()

function isWithinBcryptByteLimit(value: string): boolean {
  return textEncoder.encode(value).length <= MAX_BCRYPT_PASSWORD_BYTES
}

const loginSchema = z.object({
  email: z
    .string()
    .trim()
    .email('Please enter a valid email address.'),
  password: z
    .string()
    .min(8, 'Password must be at least 8 characters.')
    .max(128, 'Password must not exceed 128 characters.')
    .refine(isWithinBcryptByteLimit, {
      message: 'Password cannot be longer than 72 bytes.',
    }),
  next: z.string().optional(),
})

const registerSchema = z
  .object({
    fullName: z
      .string()
      .trim()
      .min(2, 'Full name must have at least 2 characters.')
      .max(120, 'Full name must not exceed 120 characters.'),
    email: z
      .string()
      .trim()
      .email('Please enter a valid email address.'),
    password: z
      .string()
      .min(8, 'Password must be at least 8 characters.')
      .max(128, 'Password must not exceed 128 characters.')
      .refine(isWithinBcryptByteLimit, {
        message: 'Password cannot be longer than 72 bytes.',
      }),
    confirmPassword: z
      .string()
      .min(8, 'Confirm password must be at least 8 characters.')
      .max(128, 'Confirm password must not exceed 128 characters.')
      .refine(isWithinBcryptByteLimit, {
        message: 'Password cannot be longer than 72 bytes.',
      }),
    termsAccepted: z.literal(true, {
      errorMap: () => ({
        message: 'Please accept the Terms of Service and Privacy Policy.',
      }),
    }),
    next: z.string().optional(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: 'Password and confirm password do not match.',
    path: ['confirmPassword'],
  })

function getFormValue(formData: FormData, fieldName: string): string {
  const value = formData.get(fieldName)
  return typeof value === 'string' ? value : ''
}

function getFirstValidationMessage(error: z.ZodError): string {
  const firstIssue = error.issues[0]
  return firstIssue?.message || 'Please check the form values and try again.'
}

function sanitizeNextPath(nextPath: string | undefined): string {
  if (!nextPath) {
    return DEFAULT_POST_AUTH_REDIRECT
  }

  if (!nextPath.startsWith('/') || nextPath.startsWith('//')) {
    return DEFAULT_POST_AUTH_REDIRECT
  }

  return nextPath
}

function mapAuthError(error: unknown, fallbackMessage: string): string {
  if (error instanceof BackendApiError) {
    return error.message
  }

  return fallbackMessage
}

export async function loginAction(
  _: AuthActionState,
  formData: FormData,
): Promise<AuthActionState> {
  const parsed = loginSchema.safeParse({
    email: getFormValue(formData, 'email'),
    password: getFormValue(formData, 'password'),
    next: getFormValue(formData, 'next') || undefined,
  })

  if (!parsed.success) {
    return { error: getFirstValidationMessage(parsed.error) }
  }

  try {
    const authResponse = await loginToBackend({
      email: parsed.data.email,
      password: parsed.data.password,
    })
    await setSessionToken(authResponse.access_token)
  } catch (error) {
    return {
      error: mapAuthError(
        error,
        'Unable to sign in right now. Please try again.',
      ),
    }
  }

  redirect(sanitizeNextPath(parsed.data.next))
}

export async function registerAction(
  _: AuthActionState,
  formData: FormData,
): Promise<AuthActionState> {
  const parsed = registerSchema.safeParse({
    fullName: getFormValue(formData, 'fullName'),
    email: getFormValue(formData, 'email'),
    password: getFormValue(formData, 'password'),
    confirmPassword: getFormValue(formData, 'confirmPassword'),
    termsAccepted: getFormValue(formData, 'terms') === 'on',
    next: getFormValue(formData, 'next') || undefined,
  })

  if (!parsed.success) {
    return { error: getFirstValidationMessage(parsed.error) }
  }

  try {
    const authResponse = await registerToBackend({
      fullName: parsed.data.fullName,
      email: parsed.data.email,
      password: parsed.data.password,
      confirmPassword: parsed.data.confirmPassword,
    })
    await setSessionToken(authResponse.access_token)
  } catch (error) {
    return {
      error: mapAuthError(
        error,
        'Unable to create account right now. Please try again.',
      ),
    }
  }

  redirect(sanitizeNextPath(parsed.data.next))
}

export async function logoutAction(): Promise<void> {
  await clearSessionToken()
  redirect('/auth/login')
}