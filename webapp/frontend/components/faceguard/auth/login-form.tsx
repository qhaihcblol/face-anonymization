import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

export function LoginForm() {
  return (
    <Card className="border-cyan-300/30 bg-background/75 shadow-[0_0_40px_-20px_rgba(34,211,238,0.75)] backdrop-blur-md">
      <CardHeader className="space-y-2">
        <CardTitle className="text-2xl tracking-tight">Sign In</CardTitle>
        <CardDescription>
          Access your FaceGuard AI workspace with your email and password.
        </CardDescription>
      </CardHeader>

      <CardContent>
        <form className="space-y-5" method="post">
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
            <div className="flex items-center justify-between gap-3">
              <Label htmlFor="password">Password</Label>
              <Link
                href="/auth/forgot-password"
                className="text-xs text-cyan-700 hover:underline dark:text-cyan-300"
              >
                Forgot password?
              </Link>
            </div>
            <Input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              placeholder="Enter your password"
              required
              className="border-cyan-300/35"
            />
          </div>

          <div className="flex items-center gap-2">
            <Checkbox id="remember" name="remember" />
            <Label htmlFor="remember" className="text-sm text-muted-foreground">
              Remember me for 30 days
            </Label>
          </div>

          <Button type="submit" className="h-10 w-full bg-cyan-400 text-cyan-950 hover:bg-cyan-300">
            Sign In
          </Button>

          <p className="text-center text-sm text-muted-foreground">
            New to FaceGuard AI?{' '}
            <Link
              href="/auth/register"
              className="font-medium text-cyan-700 hover:underline dark:text-cyan-300"
            >
              Create an account
            </Link>
          </p>
        </form>
      </CardContent>
    </Card>
  )
}
