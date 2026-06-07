'use client'

import { useState } from 'react'
import { AlertCircle, Loader2, Trash2 } from 'lucide-react'

import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { VideoApiError, deleteVideo } from '@/lib/videos/client'

/**
 * Trash button + confirm dialog that deletes a video (and, on the backend, all its
 * edits and stored files). The dialog is controlled so it stays open while the
 * request is in flight and surfaces any error; on success the parent drops the row.
 */
export function DeleteVideoButton({
  videoId,
  videoName,
  onDeleted,
}: {
  videoId: number
  videoName: string
  onDeleted: (videoId: number) => void
}) {
  const [open, setOpen] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleConfirm = async () => {
    setDeleting(true)
    setError(null)
    try {
      await deleteVideo(videoId)
      onDeleted(videoId) // parent unmounts this card
    } catch (err) {
      setError(
        err instanceof VideoApiError || err instanceof Error
          ? err.message
          : 'Could not delete this video.',
      )
      setDeleting(false)
    }
  }

  return (
    <>
      <Button
        type="button"
        size="icon-sm"
        variant="ghost"
        aria-label="Delete video"
        title="Delete video"
        onClick={() => {
          setError(null)
          setOpen(true)
        }}
        className="text-rose-600 hover:bg-rose-500/15 dark:text-rose-300"
      >
        <Trash2 className="size-4" />
      </Button>

      <AlertDialog
        open={open}
        onOpenChange={(next) => {
          // Don't let the dialog close mid-delete (Esc / overlay click).
          if (!deleting) {
            setOpen(next)
          }
        }}
      >
        <AlertDialogContent className="border-cyan-300/30">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this video?</AlertDialogTitle>
            <AlertDialogDescription>
              “{videoName}” and all of its protection runs will be permanently
              removed, including the stored files. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>

          {error ? (
            <p className="flex items-center gap-2 text-sm text-rose-500 dark:text-rose-300">
              <AlertCircle className="size-4 shrink-0" />
              {error}
            </p>
          ) : null}

          <AlertDialogFooter>
            <Button
              type="button"
              variant="outline"
              disabled={deleting}
              onClick={() => setOpen(false)}
              className="border-cyan-300/35"
            >
              Cancel
            </Button>
            <Button
              type="button"
              disabled={deleting}
              onClick={handleConfirm}
              className="bg-rose-500 text-white hover:bg-rose-600"
            >
              {deleting ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Trash2 className="size-4" />
              )}
              Delete
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
