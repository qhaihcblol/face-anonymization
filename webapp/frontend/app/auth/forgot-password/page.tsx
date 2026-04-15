import Link from 'next/link'
import { AuthShell } from '@/components/faceguard/auth/auth-shell'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

export default function ForgotPasswordPage() {
  return (
    <AuthShell
      title="Recover account access"
      description="Reset your password securely to continue managing identity protection workflows in FaceGuard AI."
    >
      <Card className="border-cyan-300/30 bg-background/75 shadow-[0_0_40px_-20px_rgba(34,211,238,0.75)] backdrop-blur-md">
        <CardHeader className="space-y-2">
          <CardTitle className="text-2xl tracking-tight">Forgot Password</CardTitle>
          <CardDescription>
            Enter your account email and we will send a secure reset link.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-5" method="post">
            <div className="space-y-2">
              <Label htmlFor="recovery-email">Email</Label>
              <Input
                id="recovery-email"
                name="email"
                type="email"
                autoComplete="email"
                placeholder="you@company.com"
                required
                className="border-cyan-300/35"
              />
            </div>

            <Button type="submit" className="h-10 w-full bg-cyan-400 text-cyan-950 hover:bg-cyan-300">
              Send Reset Link
            </Button>

            <p className="text-center text-sm text-muted-foreground">
              Remember your password?{' '}
              <Link
                href="/auth/login"
                className="font-medium text-cyan-700 hover:underline dark:text-cyan-300"
              >
                Back to sign in
              </Link>
            </p>
          </form>
        </CardContent>
      </Card>
    </AuthShell>
  )
}
