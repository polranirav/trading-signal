export interface User {
  id: string
  email: string
  full_name?: string
  email_verified: boolean
  is_admin?: boolean
  is_active?: boolean
  tier?: string
  limits?: SubscriptionLimits
  created_at?: string
  last_login?: string
}

export interface SubscriptionLimits {
  tier: string
  max_signals_per_day: number
  max_api_calls_per_day: number
  features: Record<string, any>
  price_monthly: number
  price_yearly: number
}

export interface Signal {
  id: string
  symbol: string
  signal_type: string
  confluence_score: number | null
  technical_score?: number | null
  sentiment_score?: number | null
  ml_score?: number | null
  price_at_signal?: number | null
  risk_reward_ratio?: number | null
  var_95?: number | null
  cvar_95?: number | null
  max_drawdown?: number | null
  sharpe_ratio?: number | null
  suggested_position_size?: number | null
  created_at: string
  technical_rationale?: string | null
  sentiment_rationale?: string | null
  risk_warning?: string | null
  is_executed?: boolean
  execution_price?: number | null
  realized_pnl_pct?: number | null
}

export interface ApiKey {
  id: string
  key_prefix: string
  name?: string
  last_used?: string
  expires_at?: string
  created_at: string
}

export interface Subscription {
  tier: string
  limits: SubscriptionLimits
  status: string
  current_period_start?: string
  current_period_end?: string
}

export interface ApiResponse<T> {
  success: boolean
  message: string
  data?: T
  errors?: Record<string, string[]>
  timestamp: string
}
