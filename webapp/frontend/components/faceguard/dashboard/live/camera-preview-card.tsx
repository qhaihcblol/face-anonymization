'use client'

import type { RefObject } from 'react'
import {
  Circle,
  Clapperboard,
  Loader2,
  Monitor,
  RefreshCcw,
  ShieldCheck,
  Video,
} from 'lucide-react'

import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import type { LiveProcessingStatus } from '@/lib/videos/use-live-processing'

/**
 * Left card: the camera preview and the stream/record controls.
 *
 * The raw local stream renders into the `<video>`; once protection is active the
 * processed `<canvas>` (frame + detected-face overlay, drawn by
 * {@link useLiveProcessing}) fades in over it. Nothing here runs detection — it only
 * displays what the camera/processing hooks produce.
 */
export function CameraPreviewCard({
  videoRef,
  outputCanvasRef,
  isStreaming,
  isRecording,
  showProcessed,
  processingStatus,
  processingError,
  fps,
  latencyMs,
  statusMessage,
  errorMessage,
  filterSummary,
  onStartStream,
  onStopStream,
  onStartRecording,
  onStopRecording,
}: {
  videoRef: RefObject<HTMLVideoElement | null>
  outputCanvasRef: RefObject<HTMLCanvasElement | null>
  isStreaming: boolean
  isRecording: boolean
  showProcessed: boolean
  processingStatus: LiveProcessingStatus
  processingError: string | null
  fps: number
  latencyMs: number | null
  statusMessage: string
  errorMessage: string | null
  filterSummary: string | null
  onStartStream: () => void
  onStopStream: () => void
  onStartRecording: () => void
  onStopRecording: () => void
}) {
  const isProtected = processingStatus === 'live' && showProcessed
  const isConnecting = isStreaming && processingStatus === 'connecting'

  return (
    <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle className="text-xl tracking-tight">Webcam Stream</CardTitle>
            <CardDescription>
              Real-time input stream for privacy protection and identity masking.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge
              className={
                isProtected
                  ? 'bg-emerald-500/20 text-emerald-700 dark:text-emerald-200'
                  : 'bg-cyan-500/20 text-cyan-700 dark:text-cyan-100'
              }
            >
              {isProtected ? 'Protected' : isStreaming ? 'Live' : 'Offline'}
            </Badge>
            <Badge className="bg-cyan-500/15 text-cyan-700 dark:text-cyan-100">
              FPS {fps}
            </Badge>
            <Badge className="bg-cyan-500/15 text-cyan-700 dark:text-cyan-100">
              Latency {latencyMs !== null ? `${latencyMs} ms` : '--'}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        <div className="relative aspect-video overflow-hidden rounded-xl border border-cyan-300/25 bg-slate-900/90">
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="absolute inset-0 h-full w-full -scale-x-100 object-contain"
          />
          <canvas
            ref={outputCanvasRef}
            className={cn(
              'pointer-events-none absolute inset-0 h-full w-full -scale-x-100 object-contain transition-opacity duration-200',
              showProcessed ? 'opacity-100' : 'opacity-0',
            )}
          />

          {!isStreaming && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/75 text-cyan-100">
              <Monitor className="size-10 text-cyan-300/80" />
              <p className="text-sm text-cyan-100/90">
                Camera is idle. Press Start Stream to initialize webcam.
              </p>
            </div>
          )}

          {isConnecting && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/55 text-cyan-100">
              <Loader2 className="size-8 animate-spin text-cyan-300/90" />
              <p className="text-sm text-cyan-100/90">Connecting to protection…</p>
            </div>
          )}

          {isStreaming && filterSummary && (
            <div className="pointer-events-none absolute bottom-3 left-3 flex items-center gap-1.5 rounded-full border border-cyan-300/35 bg-slate-950/70 px-3 py-1 text-xs font-medium text-cyan-100">
              <ShieldCheck className="size-3.5" />
              {filterSummary}
            </div>
          )}

          {isRecording && (
            <div className="absolute top-3 left-3 flex items-center gap-2 rounded-full border border-rose-400/35 bg-rose-500/15 px-3 py-1 text-xs font-medium text-rose-200">
              <Circle className="size-2.5 fill-rose-300 text-rose-300 animate-pulse" />
              Recording
            </div>
          )}
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <Button
            onClick={onStartStream}
            className="bg-cyan-400 text-cyan-950 hover:bg-cyan-300"
          >
            <Video className="size-4" />
            Start Stream
          </Button>
          <Button
            onClick={onStopStream}
            variant="outline"
            className="border-cyan-300/35 bg-cyan-500/5 hover:bg-cyan-500/15"
          >
            <RefreshCcw className="size-4" />
            Stop Stream
          </Button>
          {isRecording ? (
            <Button
              onClick={onStopRecording}
              variant="outline"
              className="border-rose-300/40 bg-rose-500/10 text-rose-200 hover:bg-rose-500/20"
            >
              <Circle className="size-4 fill-rose-300 text-rose-300" />
              Stop Recording
            </Button>
          ) : (
            <Button
              onClick={onStartRecording}
              disabled={!isStreaming}
              className="bg-emerald-400 text-emerald-950 hover:bg-emerald-300"
            >
              <Clapperboard className="size-4" />
              Record
            </Button>
          )}
        </div>

        <div className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 px-4 py-2 text-sm text-muted-foreground">
          <p className="font-medium text-foreground">Status: {statusMessage}</p>
          {errorMessage && <p className="mt-1 text-rose-300">{errorMessage}</p>}
          {processingError && <p className="mt-1 text-rose-300">{processingError}</p>}
        </div>
      </CardContent>
    </Card>
  )
}
