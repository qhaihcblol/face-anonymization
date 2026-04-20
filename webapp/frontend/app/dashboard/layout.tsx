import type { ReactNode } from 'react'
import { DashboardTabs } from '@/components/faceguard/dashboard/dashboard-tabs'
import { SiteHeader } from '@/components/faceguard/site-header'
import { Badge } from '@/components/ui/badge'
import { requireAuthenticatedUser } from '@/lib/auth/server'

type DashboardLayoutProps = {
  children: ReactNode
}

export default async function DashboardLayout({
  children,
}: DashboardLayoutProps) {
  const currentUser = await requireAuthenticatedUser('/dashboard/live')

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
              Signed in: {currentUser.full_name}
            </Badge>
          </div>
        </header>

        <DashboardTabs />
        {children}
      </main>
    </div>
  )
}
