'use client'

import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, FileVideo, RefreshCcw } from 'lucide-react'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from '@/components/ui/empty'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { HistoryVideoCard } from '@/components/faceguard/dashboard/history/history-video-card'
import { VideoApiError, listVideos } from '@/lib/videos/client'
import type { VideoPublic } from '@/lib/videos/types'

type LoadState = 'loading' | 'error' | 'ready'

export function HistoryPanel() {
  const [videos, setVideos] = useState<VideoPublic[]>([])
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoadState('loading')
    setError(null)
    try {
      setVideos(await listVideos())
      setLoadState('ready')
    } catch (err) {
      setError(
        err instanceof VideoApiError || err instanceof Error
          ? err.message
          : 'Could not load your videos.',
      )
      setLoadState('error')
    }
  }, [])

  const handleDeleted = useCallback((videoId: number) => {
    setVideos((current) => current.filter((video) => video.id !== videoId))
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  return (
    <div className="grid gap-6">
      <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle className="text-xl tracking-tight">Processing History</CardTitle>
              <CardDescription>
                Your uploaded videos and every identity-protection run on them.
              </CardDescription>
            </div>
            <Button
              variant="outline"
              onClick={() => void load()}
              disabled={loadState === 'loading'}
              className="border-cyan-300/35 bg-cyan-500/5 hover:bg-cyan-500/15"
            >
              <RefreshCcw
                className={cn('size-4', loadState === 'loading' && 'animate-spin')}
              />
              Refresh
            </Button>
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          {loadState === 'loading' && (
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, index) => (
                <Skeleton key={index} className="h-16 w-full rounded-xl" />
              ))}
            </div>
          )}

          {loadState === 'error' && (
            <Alert variant="destructive" className="border-destructive/30">
              <AlertCircle className="size-4" />
              <AlertDescription className="flex items-center justify-between gap-3">
                <span>{error}</span>
                <Button size="sm" variant="outline" onClick={() => void load()}>
                  Retry
                </Button>
              </AlertDescription>
            </Alert>
          )}

          {loadState === 'ready' && videos.length === 0 && (
            <Empty className="border border-cyan-300/20 bg-cyan-500/5">
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <FileVideo />
                </EmptyMedia>
                <EmptyTitle>No videos yet</EmptyTitle>
                <EmptyDescription>
                  Upload a video from the Upload tab to see its protection history here.
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}

          {loadState === 'ready' &&
            videos.map((video) => (
              <HistoryVideoCard
                key={video.id}
                video={video}
                onDeleted={handleDeleted}
              />
            ))}
        </CardContent>
      </Card>
    </div>
  )
}
