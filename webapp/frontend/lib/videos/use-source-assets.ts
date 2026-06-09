'use client'

/**
 * Lazily load a curated source-asset catalog (faces or voices) for the picker.
 *
 * The fetch is deferred until `enabled` first becomes true — i.e. until the user
 * actually selects Face Swap / Voice Conversion — so users who never pick those
 * methods never hit the catalog. It loads once and is kept across method toggles
 * (the owning component stays mounted); `reload` re-arms it after an error.
 */

import { useCallback, useEffect, useState } from 'react'

import { VideoApiError, listSourceFaces, listSourceVoices } from '@/lib/videos/client'
import type { SourceAsset, SourceAssetKind } from '@/lib/videos/types'

export type SourceAssetsStatus = 'idle' | 'loading' | 'ready' | 'error'

export type SourceAssetsState = {
  assets: SourceAsset[]
  status: SourceAssetsStatus
  error: string | null
  reload: () => void
}

const fetchers: Record<SourceAssetKind, () => Promise<SourceAsset[]>> = {
  face: listSourceFaces,
  voice: listSourceVoices,
}

export function useSourceAssets(
  kind: SourceAssetKind,
  enabled: boolean,
): SourceAssetsState {
  const [assets, setAssets] = useState<SourceAsset[]>([])
  const [status, setStatus] = useState<SourceAssetsStatus>('idle')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Fetch exactly once: only while idle and enabled. After ready/error it stays put
    // until `reload()` sets it back to idle. State is set only in the async callbacks
    // (never synchronously in the effect body); 'idle' while enabled reads as loading.
    if (!enabled || status !== 'idle') {
      return
    }

    let cancelled = false
    fetchers[kind]()
      .then((result) => {
        if (cancelled) return
        setAssets(result)
        setStatus('ready')
      })
      .catch((err) => {
        if (cancelled) return
        setError(
          err instanceof VideoApiError || err instanceof Error
            ? err.message
            : 'Could not load the catalog.',
        )
        setStatus('error')
      })

    return () => {
      cancelled = true
    }
  }, [enabled, status, kind])

  const reload = useCallback(() => setStatus('idle'), [])

  return { assets, status, error, reload }
}
