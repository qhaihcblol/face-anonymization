import Link from 'next/link'
import {
  ArrowRight,
  Camera,
  EyeOff,
  Shield,
  Radar,
  FileCheck2,
  ScanFace,
  Workflow,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { SiteHeader } from '@/components/faceguard/site-header'

const features = [
  {
    icon: EyeOff,
    title: 'Ẩn danh tính thông minh',
    description:
      'Tự động làm mờ hoặc che mặt theo từng khung hình, giữ độ ổn định cao khi đối tượng di chuyển nhanh.',
  },
  {
    icon: Radar,
    title: 'Real-time Monitoring',
    description:
      'Theo dõi luồng camera trực tiếp với độ trễ thấp, phát hiện khuôn mặt liên tục và phản hồi tức thời.',
  },
  {
    icon: FileCheck2,
    title: 'Chuẩn hóa bảo mật',
    description:
      'Lưu dấu vết xử lý rõ ràng cho từng phiên video, hỗ trợ kiểm tra nội bộ và yêu cầu tuân thủ.',
  },
]

const pipeline = [
  { icon: Camera, title: 'Input Stream', value: 'Video / Camera' },
  { icon: ScanFace, title: 'Face Detection', value: 'Computer Vision' },
  { icon: Shield, title: 'Identity Shield', value: 'Mask / Blur / Encrypt' },
  { icon: Workflow, title: 'Secure Output', value: 'Archive / Live Feed' },
]

export function HomePage() {
  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-0 cyber-grid opacity-30" />
      <SiteHeader />

      <main className="relative z-10 mx-auto flex w-full max-w-7xl flex-col gap-16 px-4 py-10 sm:px-6 lg:px-8 lg:py-14">
        <section className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">
          <div className="space-y-6">
            <p className="inline-flex items-center gap-2 rounded-full border border-cyan-300/40 bg-cyan-500/10 px-4 py-1.5 text-xs tracking-[0.16em] text-cyan-800 uppercase dark:text-cyan-200">
              Cyber Vision Security
            </p>
            <h1 className="text-balance text-4xl leading-tight font-semibold tracking-tight sm:text-5xl lg:text-6xl">
              FaceGuard AI bảo vệ danh tính con người trong video và luồng
              real-time
            </h1>
            <p className="max-w-2xl text-pretty text-base text-muted-foreground sm:text-lg">
              Hệ thống tập trung vào quyền riêng tư hình ảnh: phát hiện khuôn
              mặt theo thời gian thực, che danh tính linh hoạt và đảm bảo dữ
              liệu đầu ra sẵn sàng cho vận hành doanh nghiệp.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <Button
                asChild
                size="lg"
                className="bg-cyan-400 text-cyan-950 hover:bg-cyan-300"
              >
                <Link href="/auth/register">
                  Khởi tạo tài khoản
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
              <Button
                asChild
                variant="outline"
                size="lg"
                className="border-cyan-300/50 bg-cyan-500/5 text-cyan-900 hover:bg-cyan-500/12 dark:text-cyan-100 dark:hover:bg-cyan-500/15"
              >
                <Link href="/dashboard">Mở Dashboard</Link>
              </Button>
            </div>
          </div>

          <div className="rounded-2xl border border-cyan-300/30 bg-cyan-500/10 p-5 shadow-[0_0_40px_-18px_rgba(34,211,238,0.9)] backdrop-blur-sm">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-sm font-medium tracking-[0.12em] text-cyan-900 uppercase dark:text-cyan-100">
                Vision Pipeline
              </h2>
              <span className="rounded-full border border-emerald-400/40 bg-emerald-500/15 px-2.5 py-1 text-xs text-emerald-300">
                Online
              </span>
            </div>
            <div className="space-y-3">
              {pipeline.map(({ icon: Icon, title, value }) => (
                <article
                  key={title}
                  className="rounded-xl border border-cyan-300/20 bg-background/70 p-3"
                >
                  <div className="flex items-center gap-3">
                    <span className="flex size-9 items-center justify-center rounded-lg bg-cyan-500/15">
                      <Icon className="size-4 text-cyan-700 dark:text-cyan-300" />
                    </span>
                    <div>
                      <p className="text-sm font-medium">{title}</p>
                      <p className="text-xs text-muted-foreground">{value}</p>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="grid gap-4 md:grid-cols-3">
          {features.map(({ icon: Icon, title, description }) => (
            <article
              key={title}
              className="rounded-2xl border border-cyan-300/20 bg-background/60 p-5 backdrop-blur-sm transition-colors hover:border-cyan-300/50"
            >
              <span className="mb-4 flex size-10 items-center justify-center rounded-xl bg-cyan-500/15">
                <Icon className="size-5 text-cyan-700 dark:text-cyan-300" />
              </span>
              <h3 className="mb-2 text-lg font-semibold">{title}</h3>
              <p className="text-sm leading-relaxed text-muted-foreground">
                {description}
              </p>
            </article>
          ))}
        </section>

        <section className="rounded-2xl border border-cyan-300/25 bg-gradient-to-r from-cyan-500/15 via-cyan-400/10 to-transparent p-6 sm:p-8">
          <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Luồng xử lý minh bạch, dễ kiểm soát cho đội vận hành
          </h2>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted-foreground sm:text-base">
            FaceGuard AI tách rõ từng lớp xử lý: nhận video, nhận diện khuôn
            mặt, áp chính sách bảo vệ danh tính và xuất dữ liệu bảo mật. Kiến
            trúc này giúp team kỹ thuật dễ mở rộng thành dashboard nghiệp vụ ở
            các bước tiếp theo.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Button
              asChild
              className="bg-cyan-300 text-cyan-950 hover:bg-cyan-200"
            >
              <Link href="/auth/register">Đăng ký dùng thử</Link>
            </Button>
            <Button
              asChild
              variant="ghost"
              className="text-cyan-900 hover:bg-cyan-500/12 dark:text-cyan-100 dark:hover:bg-cyan-500/20"
            >
              <Link href="/dashboard">Đi đến trung tâm giám sát</Link>
            </Button>
          </div>
        </section>
      </main>
    </div>
  )
}
