/**
 * Admin Service
 * 
 * API calls for admin operations.
 */

import apiClient from './api';
import type { ApiResponse } from '../types';
import type {
  AdminUser,
  AdminDashboardStats,
  AdminSignal,
  AdminSubscription,
  SystemSetting,
  AuditLog,
  AdminUsersResponse,
  AdminSignalsResponse,
  AdminSubscriptionsResponse,
  AdminAuditLogsResponse,
} from '../types/admin';

export const adminService = {
  /**
   * Get dashboard statistics
   */
  async getDashboardStats(): Promise<ApiResponse<AdminDashboardStats>> {
    const response = await apiClient.get<ApiResponse<AdminDashboardStats>>('/admin/analytics/dashboard');
    return response.data;
  },

  /**
   * List users
   */
  async listUsers(params: {
    page?: number
    page_size?: number
    search?: string
    is_active?: boolean
    tier?: string
    created_after?: string
    created_before?: string
  } = {}): Promise<ApiResponse<AdminUsersResponse>> {
    const response = await apiClient.get<ApiResponse<AdminUsersResponse>>('/admin/users', { params });
    return response.data;
  },

  /**
   * Get user details
   */
  async getUser(userId: string): Promise<ApiResponse<AdminUser>> {
    const response = await apiClient.get<ApiResponse<AdminUser>>(`/admin/users/${userId}`);
    return response.data;
  },

  /**
   * Update user
   */
  async updateUser(
    userId: string,
    data: {
      full_name?: string
      is_active?: boolean
      email_verified?: boolean
    }
  ): Promise<ApiResponse<AdminUser>> {
    const response = await apiClient.patch<ApiResponse<AdminUser>>(`/admin/users/${userId}`, data);
    return response.data;
  },

  /**
   * Delete user
   */
  async deleteUser(userId: string): Promise<ApiResponse<void>> {
    const response = await apiClient.delete<ApiResponse<void>>(`/admin/users/${userId}`);
    return response.data;
  },

  /**
   * List signals
   */
  async listSignals(params: {
    page?: number
    page_size?: number
    user_id?: string
    symbol?: string
    signal_type?: string
    min_confidence?: number
    created_after?: string
    created_before?: string
  } = {}): Promise<ApiResponse<AdminSignalsResponse>> {
    const response = await apiClient.get<ApiResponse<AdminSignalsResponse>>('/admin/signals', { params });
    return response.data;
  },

  /**
   * Delete signal
   */
  async deleteSignal(signalId: string): Promise<ApiResponse<void>> {
    const response = await apiClient.delete<ApiResponse<void>>(`/admin/signals/${signalId}`);
    return response.data;
  },

  /**
   * List subscriptions
   */
  async listSubscriptions(params: {
    page?: number
    page_size?: number
    user_id?: string
    tier?: string
    status?: string
  } = {}): Promise<ApiResponse<AdminSubscriptionsResponse>> {
    const response = await apiClient.get<ApiResponse<AdminSubscriptionsResponse>>('/admin/subscriptions', { params });
    return response.data;
  },

  /**
   * Update subscription
   */
  async updateSubscription(
    subscriptionId: string,
    data: {
      tier?: string
      status?: string
    }
  ): Promise<ApiResponse<AdminSubscription>> {
    const response = await apiClient.patch<ApiResponse<AdminSubscription>>(`/admin/subscriptions/${subscriptionId}`, data);
    return response.data;
  },

  /**
   * List system settings
   */
  async listSettings(category?: string): Promise<ApiResponse<{ settings: SystemSetting[] }>> {
    const params = category ? { category } : undefined;
    const response = await apiClient.get<ApiResponse<{ settings: SystemSetting[] }>>('/admin/settings', { params });
    return response.data;
  },

  /**
   * Get system setting
   */
  async getSetting(key: string): Promise<ApiResponse<SystemSetting>> {
    const response = await apiClient.get<ApiResponse<SystemSetting>>(`/admin/settings/${key}`);
    return response.data;
  },

  /**
   * Set/update system setting
   */
  async setSetting(
    key: string,
    data: {
      value: any
      value_type: string
      description?: string
      category?: string
      is_public?: boolean
    }
  ): Promise<ApiResponse<SystemSetting>> {
    const response = await apiClient.put<ApiResponse<SystemSetting>>(`/admin/settings/${key}`, data);
    return response.data;
  },

  /**
   * List audit logs
   */
  async listAuditLogs(params: {
    page?: number
    page_size?: number
    admin_id?: string
    action?: string
    resource_type?: string
    created_after?: string
    created_before?: string
  } = {}): Promise<ApiResponse<AdminAuditLogsResponse>> {
    const response = await apiClient.get<ApiResponse<AdminAuditLogsResponse>>('/admin/audit-logs', { params });
    return response.data;
  },
};
