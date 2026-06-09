'use client'

import { createContext, useContext, type ReactNode } from 'react'

import { useVideoAnonymization } from '@/lib/videos/use-video-anonymization'

type AnonymizationContextValue = ReturnType<typeof useVideoAnonymization>

const AnonymizationContext = createContext<AnonymizationContextValue | null>(null)

/**
 * Hosts the single anonymization run for the whole dashboard. Mounted in the
 * dashboard layout (which Next.js keeps alive across sibling-route navigation), so
 * the upload → edit → poll flow and its state survive switching between the
 * Upload / Live / History tabs. Without this, the panel would unmount on a tab
 * switch and abort an in-flight upload before the edit job was ever submitted.
 */
export function AnonymizationProvider({ children }: { children: ReactNode }) {
  const anonymization = useVideoAnonymization()
  return (
    <AnonymizationContext.Provider value={anonymization}>
      {children}
    </AnonymizationContext.Provider>
  )
}

export function useAnonymization(): AnonymizationContextValue {
  const value = useContext(AnonymizationContext)
  if (value === null) {
    throw new Error('useAnonymization must be used within an AnonymizationProvider')
  }
  return value
}
