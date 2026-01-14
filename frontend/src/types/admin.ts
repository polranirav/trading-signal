/**
 * Admin Types
 * 
 * Type definitions for admin features.
 */

import type { User } from './index'

export interface AdminApiKey {
  id: string
  key_prefix: string
  name?: string
  last_used?: string
  expires_at?: string
  created_at: string
}

export interface AdminUser extends User {
  subscription_tier?: string
  subscription_status?: string
  signal_count?: number
  subscriptions?: AdminSubscription[]
  api_keys?: AdminApiKey[]
}

export interface AdminSubscription {
  id: string
  user_id: string
  user_email?: string
  tier: string
  status: string
  current_period_start?: string
  current_period_end?: string
  stripe_subscription_id?: string
  stripe_customer_id?: string
  created_at: string
}

export interface AdminSignal {
  id: string
  user_id?: string
  symbol: string
  signal_type: string
  confluence_score?: number
  technical_score?: number
  sentiment_score?: number
  ml_score?: number
  price_at_signal?: number
  risk_reward_ratio?: number
  var_95?: number
  suggested_position_size?: number
  created_at: string
  is_executed: boolean
}

export interface AdminDashboardStats {
  users: {
    total: number
    active: number
    admins: number
    signups_last_30_days: number
    signups_by_day: Array<{ date: string; count: number }>
  }
  subscriptions: {
    total: number
    active: number
    tier_breakdown: Record<string, number>
  }
  signals: {
    total: number
    today: number
    last_7_days: number
  }
}

export interface SystemSetting {
  key: string
  value: any
  value_type: string
  description?: string
  category?: string
  is_public: boolean
  updated_at?: string
}

export interface AuditLog {
  id: string
  admin_id?: string
  admin_email: string
  action: string
  resource_type: string
  resource_id?: string
  ip_address?: string
  request_path?: string
  request_method?: string
  changes?: Record<string, any>
  metadata?: Record<string, any>
  success: boolean
  error_message?: string
  created_at: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  pages: number
}

export interface AdminUsersResponse extends PaginatedResponse<AdminUser> { }
export interface AdminSignalsResponse extends PaginatedResponse<AdminSignal> { }
export interface AdminSubscriptionsResponse extends PaginatedResponse<AdminSubscription> { }
export interface AdminAuditLogsResponse extends PaginatedResponse<AuditLog> { }
