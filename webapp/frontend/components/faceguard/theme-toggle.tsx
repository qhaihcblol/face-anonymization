'use client'

import { useEffect, useState } from 'react'
import { Moon, SunMedium } from 'lucide-react'
import { useTheme } from 'next-themes'
import { Button } from '@/components/ui/button'

export function ThemeToggle() {
  const [mounted, setMounted] = useState(false)
  const { resolvedTheme, setTheme } = useTheme()

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <Button
        variant="outline"
        size="icon"
        className="border-cyan-400/40 bg-cyan-500/10"
        aria-label="Đổi giao diện"
      >
        <SunMedium className="size-4 text-cyan-200" />
      </Button>
    )
  }

  const isDark = resolvedTheme === 'dark'

  return (
    <Button
      variant="outline"
      size="icon"
      className="border-cyan-400/40 bg-cyan-500/10 hover:bg-cyan-500/20"
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      aria-label={isDark ? 'Chuyển sang giao diện sáng' : 'Chuyển sang giao diện tối'}
    >
      {isDark ? (
        <SunMedium className="size-4 text-cyan-200" />
      ) : (
        <Moon className="size-4 text-cyan-700" />
      )}
    </Button>
  )
}
