/**
 * Signals Service
 * 
 * API calls for trading signals.
 */

import apiClient from './api';
import type { ApiResponse, Signal } from '../types';

export interface SignalListResponse {
  signals: Signal[];
  count: number;
}

export interface SignalListParams {
  limit?: number;
  symbol?: string;
  min_confidence?: number;
  signal_type?: string;
  days?: number;
}

export const signalsService = {
  /**
   * Get list of signals
   */
  async getSignals(params: SignalListParams = {}): Promise<ApiResponse<SignalListResponse>> {
    const response = await apiClient.get<ApiResponse<SignalListResponse>>('/signals', { params });
    return response.data;
  },

  /**
   * Get signal by ID
   */
  async getSignal(id: string): Promise<ApiResponse<{ signal: Signal }>> {
    const response = await apiClient.get<ApiResponse<{ signal: Signal }>>(`/signals/${id}`);
    return response.data;
  },
};
