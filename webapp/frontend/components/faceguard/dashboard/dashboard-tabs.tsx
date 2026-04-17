'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Camera, History, Upload } from 'lucide-react'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

const dashboardTabs = [
  {
    value: 'live',
    href: '/dashboard/live',
    label: 'Live Camera',
    icon: Camera,
  },
  {
    value: 'upload',
    href: '/dashboard/upload',
    label: 'Upload Video',
    icon: Upload,
  },
  {
    value: 'history',
    href: '/dashboard/history',
    label: 'History',
    icon: History,
  },
]

function getActiveTab(pathname: string) {
  if (pathname.startsWith('/dashboard/upload')) {
    return 'upload'
  }

  if (pathname.startsWith('/dashboard/history')) {
    return 'history'
  }

  return 'live'
}

export function DashboardTabs() {
  const pathname = usePathname()
  const activeTab = getActiveTab(pathname)

  return (
    <Tabs value={activeTab} className="w-full">
      <TabsList className="h-11 w-full max-w-2xl rounded-xl border border-cyan-300/20 bg-cyan-500/10 p-1">
        {dashboardTabs.map(({ value, href, label, icon: Icon }) => (
          <TabsTrigger
            key={value}
            value={value}
            asChild
            className="h-full rounded-lg text-cyan-900 data-[state=active]:border-cyan-300/30 data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-950 dark:text-cyan-100 dark:data-[state=active]:bg-cyan-500/25 dark:data-[state=active]:text-cyan-50"
          >
            <Link href={href}>
              <Icon className="size-4" />
              {label}
            </Link>
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  )
}
