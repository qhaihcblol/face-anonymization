import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

const historyRows = [
  {
    id: 'FG-2419',
    source: 'Webcam Live Session',
    mode: 'Smart Blur',
    status: 'Completed',
    duration: '11m 32s',
    createdAt: '2026-04-16 08:24',
  },
  {
    id: 'FG-2418',
    source: 'upload_ops_review.mp4',
    mode: 'Privacy Mask',
    status: 'Processing',
    duration: '03m 10s',
    createdAt: '2026-04-16 08:03',
  },
  {
    id: 'FG-2417',
    source: 'LobbyCam Feed',
    mode: 'Edge Boost',
    status: 'Completed',
    duration: '27m 44s',
    createdAt: '2026-04-16 07:42',
  },
  {
    id: 'FG-2416',
    source: 'warehouse_shift.mov',
    mode: 'Smart Blur',
    status: 'Flagged',
    duration: '15m 08s',
    createdAt: '2026-04-16 07:20',
  },
]

function statusClass(status: string) {
  if (status === 'Completed') {
    return 'bg-emerald-500/20 text-emerald-700 dark:text-emerald-300'
  }

  if (status === 'Processing') {
    return 'bg-cyan-500/20 text-cyan-700 dark:text-cyan-100'
  }

  return 'bg-amber-500/20 text-amber-700 dark:text-amber-300'
}

export function HistoryPanel() {
  return (
    <div className="grid gap-6">
      <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-xl tracking-tight">Processing History</CardTitle>
          <CardDescription>
            Audit and track every processed stream and uploaded video batch.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Job ID</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Filter</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead>Created At</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {historyRows.map((row) => (
                <TableRow key={row.id}>
                  <TableCell className="font-semibold">{row.id}</TableCell>
                  <TableCell>{row.source}</TableCell>
                  <TableCell>{row.mode}</TableCell>
                  <TableCell>
                    <Badge className={statusClass(row.status)}>{row.status}</Badge>
                  </TableCell>
                  <TableCell>{row.duration}</TableCell>
                  <TableCell className="text-muted-foreground">{row.createdAt}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card className="border-cyan-300/30 bg-background/75 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-lg tracking-tight">Retention & Export</CardTitle>
          <CardDescription>
            Policy snapshot for archived identity-protected outputs.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-3">
          <article className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 p-3">
            <p className="text-xs tracking-[0.12em] text-cyan-700 uppercase dark:text-cyan-200">
              Archive Window
            </p>
            <p className="mt-1 text-base font-semibold">30 Days</p>
          </article>
          <article className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 p-3">
            <p className="text-xs tracking-[0.12em] text-cyan-700 uppercase dark:text-cyan-200">
              Encryption
            </p>
            <p className="mt-1 text-base font-semibold">AES-256</p>
          </article>
          <article className="rounded-lg border border-cyan-300/20 bg-cyan-500/10 p-3">
            <p className="text-xs tracking-[0.12em] text-cyan-700 uppercase dark:text-cyan-200">
              Last Export
            </p>
            <p className="mt-1 text-base font-semibold">2026-04-16 08:30</p>
          </article>
        </CardContent>
      </Card>
    </div>
  )
}
