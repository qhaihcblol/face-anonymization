import type { ReactNode } from 'react'
import Link from 'next/link'
import { cookies } from 'next/headers'
import { LockKeyhole } from 'lucide-react'
import { DashboardTabs } from '@/components/faceguard/dashboard/dashboard-tabs'
import { SiteHeader } from '@/components/faceguard/site-header'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

type DashboardLayoutProps = {
  children: ReactNode
}

export default async function DashboardLayout({
  children,
}: DashboardLayoutProps) {
  const cookieStore = await cookies()
  const hasSessionCookie = Boolean(cookieStore.get('faceguard_session')?.value)

  // Keep dashboard explorable during local UI development without auth backend.
  const isAuthenticated = hasSessionCookie || process.env.NODE_ENV !== 'production'

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-0 cyber-grid opacity-30" />
      <SiteHeader />

      <main className="relative z-10 mx-auto w-full max-w-7xl space-y-6 px-4 py-10 sm:px-6 lg:px-8 lg:py-12">
        <header className="rounded-2xl border border-cyan-300/25 bg-gradient-to-r from-cyan-500/14 via-cyan-400/10 to-transparent p-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs tracking-[0.14em] text-cyan-700 uppercase dark:text-cyan-200">
                FaceGuard Operations
              </p>
              <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
                Dashboard Control Center
              </h1>
              <p className="mt-2 max-w-3xl text-sm text-muted-foreground sm:text-base">
                Monitor live camera streams, process uploaded videos, and review
                identity-protection history from one cyber vision workspace.
              </p>
            </div>
            <Badge className="border border-cyan-300/35 bg-cyan-500/15 px-3 py-1.5 text-cyan-700 dark:text-cyan-100">
              {isAuthenticated ? 'Authenticated Session' : 'Guest Mode'}
            </Badge>
          </div>
        </header>

        {isAuthenticated ? (
          <>
            <DashboardTabs />
            {children}
          </>
        ) : (
          <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-xl tracking-tight">
                <LockKeyhole className="size-5 text-cyan-700 dark:text-cyan-200" />
                Sign in required
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Dashboard tabs are available after authentication. Sign in to access
                Live Camera, Upload Video, and History modules.
              </p>
              <Button
                asChild
                className="bg-cyan-400 text-cyan-950 hover:bg-cyan-300"
              >
                <Link href="/auth/login">Go to Sign In</Link>
              </Button>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  )
}
