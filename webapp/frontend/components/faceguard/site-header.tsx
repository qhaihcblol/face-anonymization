import Link from 'next/link'
import { ShieldCheck } from 'lucide-react'
import { logoutAction } from '@/app/auth/actions'
import { Button } from '@/components/ui/button'
import { ThemeToggle } from '@/components/faceguard/theme-toggle'
import { getCurrentUser } from '@/lib/auth/server'

export async function SiteHeader() {
  const currentUser = await getCurrentUser()

  return (
    <header className="sticky top-0 z-40 border-b border-cyan-400/20 bg-background/70 backdrop-blur-xl">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        <Link href="/" className="group flex items-center gap-3">
          <span className="flex size-10 items-center justify-center rounded-xl border border-cyan-300/40 bg-cyan-500/15 transition-colors group-hover:border-cyan-200/70 group-hover:bg-cyan-400/25">
            <ShieldCheck className="size-5 text-cyan-700 dark:text-cyan-300" />
          </span>
          <span className="leading-tight">
            <span className="block text-xs uppercase tracking-[0.22em] text-cyan-700/80 dark:text-cyan-300/80">
              AI Identity Protection
            </span>
            <span className="block text-base font-semibold tracking-wide text-foreground sm:text-lg">
              FaceGuard AI
            </span>
          </span>
        </Link>

        <div className="flex items-center gap-2 sm:gap-3">
          <ThemeToggle />
          {currentUser ? (
            <>
              <p className="hidden text-xs text-cyan-900/80 sm:block dark:text-cyan-100/85">
                {currentUser.full_name}
              </p>
              <Button
                asChild
                variant="ghost"
                className="text-cyan-900 hover:bg-cyan-500/10 hover:text-cyan-950 dark:text-cyan-100 dark:hover:bg-cyan-500/15 dark:hover:text-cyan-50"
              >
                <Link href="/dashboard/live">Dashboard</Link>
              </Button>
              <form action={logoutAction}>
                <Button
                  type="submit"
                  variant="ghost"
                  className="text-cyan-900 hover:bg-cyan-500/10 hover:text-cyan-950 dark:text-cyan-100 dark:hover:bg-cyan-500/15 dark:hover:text-cyan-50"
                >
                  Sign Out
                </Button>
              </form>
            </>
          ) : (
            <>
              <Button
                asChild
                variant="ghost"
                className="text-cyan-900 hover:bg-cyan-500/10 hover:text-cyan-950 dark:text-cyan-100 dark:hover:bg-cyan-500/15 dark:hover:text-cyan-50"
              >
                <Link href="/auth/login">Sign In</Link>
              </Button>
              <Button
                asChild
                className="bg-cyan-400 text-cyan-950 shadow-[0_0_28px_-10px_rgba(34,211,238,0.9)] hover:bg-cyan-300"
              >
                <Link href="/auth/register">Register</Link>
              </Button>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
