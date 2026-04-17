'use client'

import { useMemo, useState } from 'react'
import { FileVideo, FolderUp, ShieldCheck, Sparkles } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { Textarea } from '@/components/ui/textarea'

export function UploadVideoPanel() {
  const [files, setFiles] = useState<File[]>([])

  const totalSizeMB = useMemo(
    () =>
      files.reduce((total, file) => total + file.size / (1024 * 1024), 0).toFixed(2),
    [files],
  )

  const queueProgress = files.length === 0 ? 0 : Math.min(90, files.length * 25)

  return (
    <div className="grid gap-6 xl:grid-cols-[1.45fr_1fr]">
      <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-xl tracking-tight">Upload Video</CardTitle>
          <CardDescription>
            Send pre-recorded footage for identity masking and compliance-safe export.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <label
            htmlFor="video-upload"
            className="group block cursor-pointer rounded-2xl border border-dashed border-cyan-300/35 bg-cyan-500/10 p-8 text-center transition-colors hover:border-cyan-300/60 hover:bg-cyan-500/15"
          >
            <FolderUp className="mx-auto mb-3 size-10 text-cyan-700 transition-transform group-hover:scale-105 dark:text-cyan-300" />
            <p className="text-base font-semibold text-foreground">
              Drop video files here or click to browse
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              Supports `.mp4`, `.mov`, `.webm` (up to 2GB per file)
            </p>
            <Input
              id="video-upload"
              type="file"
              accept="video/*"
              multiple
              className="mt-4 border-cyan-300/35"
              onChange={(event) =>
                setFiles(Array.from(event.target.files ?? []))
              }
            />
          </label>

          <section className="rounded-xl border border-cyan-300/20 bg-cyan-500/10 p-4">
            <div className="mb-2 flex items-center justify-between gap-2">
              <p className="text-sm font-semibold text-foreground">Queue Status</p>
              <Badge className="bg-cyan-500/20 text-cyan-700 dark:text-cyan-100">
                {files.length} file(s)
              </Badge>
            </div>
            <Progress value={queueProgress} className="h-2.5" />
            <p className="mt-2 text-xs text-muted-foreground">
              Total size: {totalSizeMB} MB
            </p>
          </section>

          <div className="space-y-2">
            <Label htmlFor="processing-notes">Processing Notes</Label>
            <Textarea
              id="processing-notes"
              placeholder="Optional notes for this upload batch..."
              className="min-h-24 border-cyan-300/35"
            />
          </div>

          <div className="flex flex-wrap gap-3">
            <Button className="bg-cyan-400 text-cyan-950 hover:bg-cyan-300">
              <FileVideo className="size-4" />
              Start Processing Queue
            </Button>
            <Button
              variant="outline"
              className="border-cyan-300/35 bg-cyan-500/5 hover:bg-cyan-500/15"
            >
              Save Draft Batch
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-xl tracking-tight">Processing Profile</CardTitle>
          <CardDescription>
            Default profile tuned for privacy-first video pipelines.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <article className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 p-3">
            <div className="flex items-start gap-3">
              <ShieldCheck className="mt-0.5 size-4 text-cyan-700 dark:text-cyan-300" />
              <div>
                <p className="text-sm font-semibold">Identity Protection</p>
                <p className="text-xs text-muted-foreground">
                  Smart blur with temporal smoothing and confidence threshold.
                </p>
              </div>
            </div>
          </article>

          <article className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 p-3">
            <div className="flex items-start gap-3">
              <Sparkles className="mt-0.5 size-4 text-cyan-700 dark:text-cyan-300" />
              <div>
                <p className="text-sm font-semibold">Output Preset</p>
                <p className="text-xs text-muted-foreground">
                  H.264, 30 FPS, encrypted archive packaging.
                </p>
              </div>
            </div>
          </article>

          <div className="rounded-lg border border-cyan-300/20 bg-gradient-to-r from-cyan-500/16 via-cyan-400/10 to-transparent p-3">
            <p className="text-sm font-semibold">Next step after upload</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Processed outputs will be available in History with logs and export links.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
