/**
 * Account Service
 * 
 * API calls for account management.
 */

import apiClient from './api';
import type { ApiResponse, User, ApiKey, Subscription } from '../types';

export const accountService = {
  /**
   * Get account info
   */
  async getAccount(): Promise<ApiResponse<{ user: User }>> {
    const response = await apiClient.get<ApiResponse<{ user: User }>>('/account');
    return response.data;
  },

  /**
   * Update account
   */
  async updateAccount(data: { full_name?: string }): Promise<ApiResponse<{ user: User }>> {
    const response = await apiClient.put<ApiResponse<{ user: User }>>('/account', data);
    return response.data;
  },

  /**
   * Get API keys
   */
  async getApiKeys(): Promise<ApiResponse<{ api_keys: ApiKey[] }>> {
    const response = await apiClient.get<ApiResponse<{ api_keys: ApiKey[] }>>('/account/api-keys');
    return response.data;
  },

  /**
   * Create API key
   */
  async createApiKey(name?: string): Promise<ApiResponse<{ api_key: string }>> {
    const response = await apiClient.post<ApiResponse<{ api_key: string }>>('/account/api-keys', { name });
    return response.data;
  },

  /**
   * Delete API key
   */
  async deleteApiKey(keyId: string): Promise<ApiResponse<void>> {
    const response = await apiClient.delete<ApiResponse<void>>(`/account/api-keys/${keyId}`);
    return response.data;
  },
};

export const subscriptionService = {
  /**
   * Get subscription
   */
  async getSubscription(): Promise<ApiResponse<{ subscription: Subscription }>> {
    const response = await apiClient.get<ApiResponse<{ subscription: Subscription }>>('/subscription');
    return response.data;
  },

  /**
   * Upgrade subscription
   */
  async upgradeSubscription(tier: string): Promise<ApiResponse<{ checkout_url: string }>> {
    const response = await apiClient.post<ApiResponse<{ checkout_url: string }>>('/subscription/upgrade', { tier });
    return response.data;
  },

  /**
   * Cancel subscription
   */
  async cancelSubscription(): Promise<ApiResponse<void>> {
    const response = await apiClient.post<ApiResponse<void>>('/subscription/cancel');
    return response.data;
  },
};
