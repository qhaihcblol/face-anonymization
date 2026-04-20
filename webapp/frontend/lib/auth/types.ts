export type UserPublic = {
  id: number
  full_name: string
  email: string
  is_active: boolean
  created_at: string
}

export type AuthResponse = {
  access_token: string
  token_type: string
  user: UserPublic
}

export type LoginPayload = {
  email: string
  password: string
}

export type RegisterPayload = {
  fullName: string
  email: string
  password: string
  confirmPassword: string
}

export type FastApiValidationDetail = {
  loc: Array<string | number>
  msg: string
  type: string
}

export type BackendErrorResponse = {
  detail: string | FastApiValidationDetail[]
}
