/** Small display formatters for video metadata. Pure + UI-agnostic. */

export function formatBytes(bytes: number | null): string {
  if (bytes === null || !Number.isFinite(bytes)) {
    return '—'
  }
  if (bytes < 1024) {
    return `${bytes} B`
  }
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) {
    return '—'
  }
  const total = Math.round(seconds)
  const minutes = Math.floor(total / 60)
  const secs = total % 60
  return minutes > 0 ? `${minutes}m ${secs}s` : `${secs}s`
}

export function formatDateTime(iso: string): string {
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString()
}

export function formatResolution(
  width: number | null,
  height: number | null,
): string | null {
  if (!width || !height) {
    return null
  }
  return `${width}×${height}`
}

/** Wall-clock time between two ISO timestamps (e.g. how long a run took). */
export function formatElapsed(
  fromIso: string,
  toIso: string | null,
): string | null {
  if (!toIso) {
    return null
  }
  const from = new Date(fromIso).getTime()
  const to = new Date(toIso).getTime()
  if (Number.isNaN(from) || Number.isNaN(to) || to < from) {
    return null
  }
  return formatDuration((to - from) / 1000)
}
