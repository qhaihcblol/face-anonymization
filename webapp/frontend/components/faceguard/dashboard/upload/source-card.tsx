'use client'

import { FileVideo, FolderUp } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'

function formatMb(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(2)
}

/** Left card: pick one video file, preview the raw source, show upload progress. */
export function SourceCard({
  previewUrl,
  selectedFile,
  isUploading,
  uploadPercent,
  onFileSelected,
}: {
  previewUrl: string | null
  selectedFile: File | null
  isUploading: boolean
  uploadPercent: number
  onFileSelected: (file: File | null) => void
}) {
  return (
    <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-xl tracking-tight">Original Upload</CardTitle>
        <CardDescription>
          Upload one video file and preview the raw source before processing.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <label
          htmlFor="video-upload"
          className="group grid cursor-pointer gap-3 rounded-xl border border-dashed border-cyan-300/35 bg-cyan-500/10 p-4 transition-colors hover:border-cyan-300/60 hover:bg-cyan-500/15 sm:grid-cols-[auto_1fr_auto] sm:items-center"
        >
          <FolderUp className="mx-auto size-7 text-cyan-700 transition-transform group-hover:scale-105 dark:text-cyan-300 sm:mx-0" />
          <div className="text-center sm:text-left">
            <p className="text-sm font-semibold text-foreground">Upload one video file</p>
            <p className="text-xs text-muted-foreground">
              Supports .mp4, .mov, .webm, .mkv, .avi (up to 2GB)
            </p>
          </div>
          <Input
            id="video-upload"
            type="file"
            accept="video/*"
            className="sr-only"
            onChange={(event) => onFileSelected(event.target.files?.[0] ?? null)}
          />
          <span className="inline-flex items-center justify-center rounded-md border border-cyan-300/45 bg-cyan-500/15 px-3 py-1.5 text-xs font-medium text-cyan-700 transition-colors group-hover:bg-cyan-500/25 dark:text-cyan-100">
            Choose file
          </span>
        </label>

        <div className="relative aspect-video overflow-hidden rounded-xl border border-cyan-300/25 bg-slate-900/90">
          {previewUrl ? (
            <video
              key={previewUrl}
              src={previewUrl}
              controls
              playsInline
              className="h-full w-full object-contain"
            />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/75 text-cyan-100">
              <FileVideo className="size-10 text-cyan-300/80" />
              <p className="text-sm text-cyan-100/90">
                Upload a video to preview the original source.
              </p>
            </div>
          )}
        </div>

        <section className="rounded-xl border border-cyan-300/20 bg-cyan-500/10 p-4">
          <div className="mb-2 flex items-center justify-between gap-2">
            <p className="text-sm font-semibold text-foreground">
              {isUploading ? 'Uploading…' : 'Source File'}
            </p>
            <Badge className="bg-cyan-500/20 text-cyan-700 dark:text-cyan-100">
              {selectedFile ? '1 file selected' : 'No file'}
            </Badge>
          </div>
          <Progress
            value={isUploading ? uploadPercent : selectedFile ? 100 : 0}
            className="h-2.5"
          />
          <p className="mt-2 text-xs text-muted-foreground">
            {selectedFile
              ? `${selectedFile.name} (${formatMb(selectedFile.size)} MB)${
                  isUploading ? ` • ${uploadPercent}%` : ''
                }`
              : 'Select a video file to begin.'}
          </p>
        </section>
      </CardContent>
    </Card>
  )
}
