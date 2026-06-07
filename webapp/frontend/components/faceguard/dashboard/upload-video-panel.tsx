'use client'

import { useEffect, useMemo, useRef, useState } from 'react'

import { ResultCard } from '@/components/faceguard/dashboard/upload/result-card'
import { ProtectionSettingsCard } from '@/components/faceguard/dashboard/upload/protection-settings-card'
import { SourceCard } from '@/components/faceguard/dashboard/upload/source-card'
import {
  buildEditPayload,
  defaultProtectionForm,
  type ProtectionForm,
} from '@/lib/videos/options'
import { useVideoAnonymization } from '@/lib/videos/use-video-anonymization'

/**
 * Orchestrates the Upload Video workflow. Owns the selected file, its local preview
 * URL, and the protection form; delegates the upload → edit → poll → download flow
 * to {@link useVideoAnonymization}. The three cards stay purely presentational.
 */
export function UploadVideoPanel() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [form, setForm] = useState<ProtectionForm>(defaultProtectionForm)

  const previewUrlRef = useRef<string | null>(null)
  const anonymization = useVideoAnonymization()
  const { phase, uploadPercent, isRunning, run, reset } = anonymization

  const { payload, error: rangeError } = useMemo(
    () => buildEditPayload(form),
    [form],
  )

  const updateForm = (patch: Partial<ProtectionForm>) => {
    setForm((previous) => ({ ...previous, ...patch }))
  }

  const handleFileSelected = (file: File | null) => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = null
    }

    reset() // clear any previous run's result/status
    setSelectedFile(file)

    if (!file) {
      setPreviewUrl(null)
      return
    }

    const nextUrl = URL.createObjectURL(file)
    previewUrlRef.current = nextUrl
    setPreviewUrl(nextUrl)
  }

  const handleSubmit = () => {
    if (selectedFile && payload) {
      void run(selectedFile, payload)
    }
  }

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current)
      }
    }
  }, [])

  const submitLabel =
    phase === 'uploading'
      ? `Uploading ${uploadPercent}%`
      : phase === 'processing'
        ? 'Processing…'
        : phase === 'completed'
          ? 'Re-run protection'
          : 'Activate Guard'

  return (
    <div className="grid gap-6 xl:grid-cols-[1.05fr_0.85fr_1.05fr]">
      <SourceCard
        previewUrl={previewUrl}
        selectedFile={selectedFile}
        isUploading={phase === 'uploading'}
        uploadPercent={uploadPercent}
        onFileSelected={handleFileSelected}
      />

      <ProtectionSettingsCard
        form={form}
        onChange={updateForm}
        rangeError={rangeError}
        isRunning={isRunning}
        canSubmit={Boolean(selectedFile) && payload !== null && !isRunning}
        submitLabel={submitLabel}
        onSubmit={handleSubmit}
        onCancel={anonymization.cancel}
      />

      <ResultCard
        state={anonymization}
        downloadName={
          selectedFile ? `protected-${selectedFile.name}` : 'protected-video.mp4'
        }
      />
    </div>
  )
}
