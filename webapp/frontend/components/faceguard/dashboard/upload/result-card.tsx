'use client'

import { AlertCircle, CheckCircle2, Download, Loader2, ShieldCheck } from 'lucide-react'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { editStatusBadgeClass, editStatusLabel } from '@/lib/videos/options'
import type { AnonymizationState } from '@/lib/videos/use-video-anonymization'

const overlayMessage: Record<AnonymizationState['phase'], string> = {
  idle: 'Activate Guard to generate the protected output.',
  uploading: 'Uploading source video…',
  processing: 'Protecting your video — this can take a while.',
  completed: '',
  failed: 'Processing failed. Adjust the settings and try again.',
}

/** Right card: preview the protected output, follow live status, download result. */
export function ResultCard({
  state,
}: {
  state: AnonymizationState
}) {
  const { phase, uploadPercent, edit, resultUrl, error } = state

  return (
    <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle className="text-xl tracking-tight">Processed Result</CardTitle>
            <CardDescription>Preview and download the protected output.</CardDescription>
          </div>
          {edit ? (
            <Badge className={editStatusBadgeClass[edit.status]}>
              {editStatusLabel[edit.status]}
            </Badge>
          ) : null}
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="relative aspect-video overflow-hidden rounded-xl border border-cyan-300/25 bg-slate-900/90">
          {resultUrl ? (
            <video
              key={resultUrl}
              src={resultUrl}
              controls
              playsInline
              className="h-full w-full object-contain"
            />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/75 px-6 text-center text-cyan-100">
              {phase === 'uploading' || phase === 'processing' ? (
                <Loader2 className="size-10 animate-spin text-cyan-300/80" />
              ) : (
                <ShieldCheck className="size-10 text-cyan-300/70" />
              )}
              <p className="text-sm text-cyan-100/90">{overlayMessage[phase]}</p>
            </div>
          )}
        </div>

        {phase === 'uploading' && (
          <div className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-4 py-3">
            <p className="mb-2 text-sm font-medium text-foreground">
              Uploading source… {uploadPercent}%
            </p>
            <Progress value={uploadPercent} className="h-2" />
          </div>
        )}

        {phase === 'failed' && error && (
          <Alert variant="destructive" className="border-destructive/30">
            <AlertCircle className="size-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-4 py-3">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="size-4 text-cyan-700 dark:text-cyan-300" />
            <p className="text-sm font-semibold text-foreground">Output Summary</p>
          </div>
          {phase === 'completed' && resultUrl ? (
            <div className="mt-3 space-y-3">
              <p className="text-xs text-muted-foreground">
                Protected video ready{edit ? ` (edit #${edit.id})` : ''}. The link is a
                short-lived presigned URL.
              </p>
              <Button
                asChild
                className="w-full bg-cyan-400 text-cyan-950 hover:bg-cyan-300"
              >
                <a href={resultUrl} rel="noreferrer">
                  <Download className="size-4" />
                  Download protected video
                </a>
              </Button>
            </div>
          ) : (
            <p className="mt-1 text-xs text-muted-foreground">
              Run Activate Guard to generate a protected output.
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
