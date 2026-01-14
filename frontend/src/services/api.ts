/**
 * API Client Service
 * 
 * Axios instance with interceptors for API communication.
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';

// Use relative URL to leverage Vite proxy, or absolute URL if specified
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true, // Include cookies for session auth
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Add API key if available (for programmatic access)
    const apiKey = localStorage.getItem('api_key');
    if (apiKey && config.headers) {
      config.headers['X-API-Key'] = apiKey;
    }
    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error: AxiosError) => {
    // Log network errors for debugging
    if (!error.response) {
      console.error('Network Error:', {
        message: error.message,
        code: error.code,
        config: {
          url: error.config?.url,
          baseURL: error.config?.baseURL,
          method: error.config?.method,
        }
      });
    }
    
    // Handle common errors
    if (error.response?.status === 401) {
      // Unauthorized - clear auth state
      localStorage.removeItem('api_key');
      // Don't redirect here - let the ProtectedRoute handle it
      // Only redirect for non-auth API calls from authenticated views
      const isAuthEndpoint = error.config?.url?.includes('/auth/');
      const isPublicPage = ['/login', '/register', '/', '/features', '/pricing', '/about'].includes(window.location.pathname);
      if (!isAuthEndpoint && !isPublicPage) {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export default apiClient;

// Unified API object for backward compatibility
import { signalsService } from './signals';
import { authService } from './auth';
import { accountService, subscriptionService } from './account';

export const api = {
  // Signals
  async getSignals(params: { limit?: number; symbol?: string; min_confidence?: number; signal_type?: string; days?: number }) {
    const response = await signalsService.getSignals(params);
    return response.data;
  },

  async getSignal(id: string) {
    const response = await signalsService.getSignal(id);
    return response.data?.signal;
  },

  // Auth
  async login(email: string, password: string) {
    const response = await authService.login({ email, password });
    return response.data;
  },

  async register(email: string, password: string, fullName?: string) {
    const response = await authService.register({ email, password, full_name: fullName });
    return response.data;
  },

  async logout() {
    await authService.logout();
  },

  async getCurrentUser() {
    const response = await authService.getCurrentUser();
    return response.data;
  },

  // Account
  account: accountService,
  subscription: subscriptionService,
};
