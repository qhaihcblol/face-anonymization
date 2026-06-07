'use client'

import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'

import { cn } from '@/lib/utils'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { SelectOption } from '@/lib/videos/options'

/**
 * One titled block of controls (icon + uppercase tracked heading). Shared by the
 * Face / Voice / Range sections so they all read the same — the "đồng bộ style"
 * the visual and voice settings should have.
 */
export function SettingsSection({
  icon: Icon,
  title,
  className,
  children,
}: {
  icon: LucideIcon
  title: string
  className?: string
  children: ReactNode
}) {
  return (
    <section className={cn('space-y-4', className)}>
      <div className="flex items-center gap-2 text-cyan-700 dark:text-cyan-200">
        <Icon className="size-4" />
        <h3 className="text-xs font-semibold tracking-[0.14em] uppercase">{title}</h3>
      </div>
      {children}
    </section>
  )
}

/**
 * A labelled `Select` driven by a {@link SelectOption} list, with the selected
 * option's description shown beneath it. Generic over the value type so visual
 * methods, audio modes and voice methods all render identically.
 */
export function OptionSelect<TValue extends string>({
  id,
  label,
  value,
  options,
  onValueChange,
}: {
  id: string
  label: string
  value: TValue
  options: ReadonlyArray<SelectOption<TValue>>
  onValueChange: (value: TValue) => void
}) {
  const description = options.find((option) => option.value === value)?.description

  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      <Select value={value} onValueChange={(next) => onValueChange(next as TValue)}>
        <SelectTrigger id={id} className="w-full border-cyan-300/35">
          <SelectValue placeholder={`Select ${label.toLowerCase()}`} />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {description ? (
        <p className="text-xs text-muted-foreground">{description}</p>
      ) : null}
    </div>
  )
}
