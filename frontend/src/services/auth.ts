/**
 * Authentication Service
 * 
 * API calls for authentication.
 */

import apiClient from './api';
import type { ApiResponse, User } from '../types';

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  full_name?: string;
}

export const authService = {
  /**
   * Register a new user
   */
  async register(data: RegisterRequest): Promise<ApiResponse<{ user: User }>> {
    const response = await apiClient.post<ApiResponse<{ user: User }>>('/auth/register', data);
    return response.data;
  },

  /**
   * Login user
   */
  async login(data: LoginRequest): Promise<ApiResponse<{ user: User }>> {
    const response = await apiClient.post<ApiResponse<{ user: User }>>('/auth/login', data);
    return response.data;
  },

  /**
   * Logout user
   */
  async logout(): Promise<ApiResponse<void>> {
    const response = await apiClient.post<ApiResponse<void>>('/auth/logout');
    return response.data;
  },

  /**
   * Get current user info
   */
  async getCurrentUser(): Promise<ApiResponse<{ user: User }>> {
    const response = await apiClient.get<ApiResponse<{ user: User }>>('/auth/me');
    return response.data;
  },
};
