import { AuthShell } from '@/components/faceguard/auth/auth-shell'
import { LoginForm } from '@/components/faceguard/auth/login-form'
import { redirectIfAuthenticated } from '@/lib/auth/server'

export default async function LoginPage() {
  await redirectIfAuthenticated('/dashboard/live')

  return (
    <AuthShell
      title="Secure access for privacy operations"
      description="Sign in to manage identity-protected streams, review anonymization history, and monitor computer vision pipelines from one security dashboard."
    >
      <LoginForm />
    </AuthShell>
  )
}
