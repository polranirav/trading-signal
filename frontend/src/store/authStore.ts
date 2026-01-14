/**
 * Authentication Store (Zustand)
 * 
 * Global state management for authentication.
 */

import { create } from 'zustand';
import type { User } from '../types';
import { authService } from '../services/auth';

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => Promise<void>;
  fetchUser: () => Promise<void>;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>()((set) => ({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await authService.login({ email, password });
          if (response.success && response.data?.user) {
            set({
              user: response.data.user,
              isAuthenticated: true,
              isLoading: false,
              error: null,
            });
          } else {
            throw new Error(response.message || 'Login failed');
          }
        } catch (error: any) {
          console.error('Login error:', error);
          let errorMessage = 'Login failed';
          
          if (error.response) {
            // Server responded with error
            errorMessage = error.response.data?.message || error.response.statusText || 'Login failed';
          } else if (error.request) {
            // Request made but no response (network error)
            errorMessage = 'Network error: Could not connect to server. Please check if the backend is running.';
          } else {
            // Something else happened
            errorMessage = error.message || 'Login failed';
          }
          
          set({
            user: null,
            isAuthenticated: false,
            isLoading: false,
            error: errorMessage,
          });
          throw error;
        }
      },

      register: async (email: string, password: string, fullName?: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await authService.register({ email, password, full_name: fullName });
          if (response.success && response.data?.user) {
            set({
              user: response.data.user,
              isAuthenticated: true,
              isLoading: false,
              error: null,
            });
          } else {
            throw new Error(response.message || 'Registration failed');
          }
        } catch (error: any) {
          console.error('Registration error:', error);
          let errorMessage = 'Registration failed';
          
          if (error.response) {
            // Server responded with error
            errorMessage = error.response.data?.message || error.response.statusText || 'Registration failed';
          } else if (error.request) {
            // Request made but no response (network error)
            errorMessage = 'Network error: Could not connect to server. Please check if the backend is running.';
          } else {
            // Something else happened
            errorMessage = error.message || 'Registration failed';
          }
          
          set({
            user: null,
            isAuthenticated: false,
            isLoading: false,
            error: errorMessage,
          });
          throw error;
        }
      },

      logout: async () => {
        try {
          await authService.logout();
        } catch (error) {
          // Continue with logout even if API call fails
        } finally {
          set({
            user: null,
            isAuthenticated: false,
            error: null,
          });
        }
      },

      fetchUser: async () => {
        set({ isLoading: true });
        try {
          const response = await authService.getCurrentUser();
          if (response.success && response.data?.user) {
            set({
              user: response.data.user,
              isAuthenticated: true,
              isLoading: false,
              error: null,
            });
          } else {
            throw new Error('Failed to fetch user');
          }
        } catch (error) {
          set({
            user: null,
            isAuthenticated: false,
            isLoading: false,
          });
        }
      },

      clearError: () => {
        set({ error: null });
      },
    })
);
