import { AuthShell } from '@/components/faceguard/auth/auth-shell'
import { RegisterForm } from '@/components/faceguard/auth/register-form'
import { redirectIfAuthenticated } from '@/lib/auth/server'

export default async function RegisterPage() {
  await redirectIfAuthenticated('/dashboard/live')

  return (
    <AuthShell
      title="Create your FaceGuard AI account"
      description="Set up your identity-protection workspace for real-time and recorded video. Start with core credentials and expand with team roles later."
    >
      <RegisterForm />
    </AuthShell>
  )
}
