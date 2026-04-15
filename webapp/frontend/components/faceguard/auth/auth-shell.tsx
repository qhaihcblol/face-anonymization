import type { ReactNode } from 'react'
import { LockKeyhole, ShieldCheck, Zap } from 'lucide-react'
import { SiteHeader } from '@/components/faceguard/site-header'

type AuthShellProps = {
  title: string
  description: string
  children: ReactNode
}

const trustSignals = [
  {
    icon: ShieldCheck,
    title: 'Identity-first processing',
    description:
      'Built to anonymize faces before downstream analytics and storage.',
  },
  {
    icon: Zap,
    title: 'Low-latency response',
    description:
      'Designed for continuous live streams with real-time privacy controls.',
  },
  {
    icon: LockKeyhole,
    title: 'Secure access model',
    description:
      'Structured account permissions for teams managing sensitive footage.',
  },
]

export function AuthShell({ title, description, children }: AuthShellProps) {
  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-0 cyber-grid opacity-30" />
      <SiteHeader />

      <main className="relative z-10 mx-auto grid w-full max-w-7xl gap-8 px-4 py-10 sm:px-6 lg:grid-cols-[1fr_460px] lg:px-8 lg:py-14">
        <section className="hidden rounded-2xl border border-cyan-300/20 bg-cyan-500/10 p-7 backdrop-blur-sm lg:block">
          <p className="inline-flex rounded-full border border-cyan-300/45 bg-cyan-400/10 px-4 py-1 text-xs tracking-[0.14em] text-cyan-700 uppercase dark:text-cyan-200">
            FaceGuard AI Access
          </p>
          <h1 className="mt-4 text-4xl leading-tight font-semibold tracking-tight text-foreground">
            {title}
          </h1>
          <p className="mt-4 max-w-xl text-base leading-relaxed text-muted-foreground">
            {description}
          </p>

          <div className="mt-8 space-y-4">
            {trustSignals.map(({ icon: Icon, title: signalTitle, description }) => (
              <article
                key={signalTitle}
                className="rounded-xl border border-cyan-300/20 bg-background/70 p-4"
              >
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 flex size-9 items-center justify-center rounded-lg bg-cyan-500/15">
                    <Icon className="size-4 text-cyan-700 dark:text-cyan-300" />
                  </span>
                  <div>
                    <h2 className="text-sm font-semibold text-foreground">
                      {signalTitle}
                    </h2>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {description}
                    </p>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="self-center">{children}</section>
      </main>
    </div>
  )
}
