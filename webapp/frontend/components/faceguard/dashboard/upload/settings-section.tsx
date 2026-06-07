'use client'

import type { ComponentProps, ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'

import { cn } from '@/lib/utils'
import { Input } from '@/components/ui/input'
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

/**
 * A labelled numeric input. Keeps the repetitive field markup in one place so the
 * Upload and Live panels render their tuning knobs (blur strength, pixelation
 * level, …) identically.
 */
export function NumberField({
  id,
  label,
  value,
  onChange,
  hint,
  ...inputProps
}: {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  hint?: string
} & Pick<ComponentProps<'input'>, 'min' | 'max' | 'step' | 'placeholder' | 'inputMode'>) {
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      <Input
        id={id}
        type="number"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="border-cyan-300/35"
        {...inputProps}
      />
      {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  )
}

/** A labelled colour picker with the chosen hex shown beside the swatch. */
export function ColorField({
  id,
  label,
  value,
  onChange,
  hint,
}: {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  hint?: string
}) {
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      <div className="flex items-center gap-3">
        <input
          id={id}
          type="color"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="h-9 w-14 cursor-pointer rounded-md border border-cyan-300/35 bg-transparent p-1"
        />
        <span className="text-xs text-muted-foreground uppercase">{value}</span>
      </div>
      {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  )
}
