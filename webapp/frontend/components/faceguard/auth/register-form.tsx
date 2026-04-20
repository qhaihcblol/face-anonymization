"use client"

import { useActionState } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { useFormStatus } from 'react-dom'
import { AlertCircle } from 'lucide-react'
import { registerAction } from '@/app/auth/actions'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

const initialState = {
  error: null as string | null,
}

function RegisterSubmitButton() {
  const { pending } = useFormStatus()

  return (
    <Button
      type="submit"
      disabled={pending}
      className="h-10 w-full bg-cyan-400 text-cyan-950 hover:bg-cyan-300"
    >
      {pending ? 'Creating account...' : 'Create Account'}
    </Button>
  )
}

export function RegisterForm() {
  const [state, formAction] = useActionState(registerAction, initialState)
  const searchParams = useSearchParams()
  const nextPath = searchParams.get('next') ?? '/dashboard/live'
  const loginHref = `/auth/login?next=${encodeURIComponent(nextPath)}`

  return (
    <Card className="border-cyan-300/30 bg-background/75 shadow-[0_0_40px_-20px_rgba(34,211,238,0.75)] backdrop-blur-md">
      <CardHeader className="space-y-2">
        <CardTitle className="text-2xl tracking-tight">Create Account</CardTitle>
        <CardDescription>
          Register your team identity to start protecting faces in live and recorded video.
        </CardDescription>
      </CardHeader>

      <CardContent>
        <form className="space-y-5" action={formAction}>
          <input type="hidden" name="next" value={nextPath} />

          <div className="space-y-2">
            <Label htmlFor="full-name">Full Name</Label>
            <Input
              id="full-name"
              name="fullName"
              type="text"
              autoComplete="name"
              placeholder="Jane Doe"
              required
              className="border-cyan-300/35"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              name="email"
              type="email"
              autoComplete="email"
              placeholder="you@company.com"
              required
              className="border-cyan-300/35"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              name="password"
              type="password"
              autoComplete="new-password"
              placeholder="At least 8 characters"
              minLength={8}
              required
              className="border-cyan-300/35"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="confirm-password">Confirm Password</Label>
            <Input
              id="confirm-password"
              name="confirmPassword"
              type="password"
              autoComplete="new-password"
              placeholder="Repeat your password"
              minLength={8}
              required
              className="border-cyan-300/35"
            />
          </div>

          <div className="flex items-start gap-2">
            <Checkbox id="terms" name="terms" required className="mt-1" />
            <Label htmlFor="terms" className="text-sm leading-relaxed text-muted-foreground">
              I agree to the Terms of Service and Privacy Policy for using FaceGuard AI.
            </Label>
          </div>

          {state.error ? (
            <Alert variant="destructive" className="border-destructive/30">
              <AlertCircle className="size-4" />
              <AlertDescription>{state.error}</AlertDescription>
            </Alert>
          ) : null}

          <RegisterSubmitButton />

          <p className="text-center text-sm text-muted-foreground">
            Already have an account?{' '}
            <Link
              href={loginHref}
              className="font-medium text-cyan-700 hover:underline dark:text-cyan-300"
            >
              Sign in now
            </Link>
          </p>
        </form>
      </CardContent>
    </Card>
  )
}
