/**
 * Portfolio Context
 * 
 * Global state management for user's portfolio.
 * Makes portfolio data available to all dashboard pages.
 */

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import apiClient from '../services/api';

// Types
export interface PortfolioHolding {
    id: string;
    symbol: string;
    shares: number;
    avg_cost: number;
    current_price: number;
    pnl: number;
    pnl_percent: number;
    signal?: string;
}

export interface PortfolioSummary {
    total_value: number;
    total_cost: number;
    total_pnl: number;
    total_pnl_percent: number;
    holdings_count: number;
}

interface PortfolioContextType {
    // Data
    holdings: PortfolioHolding[];
    summary: PortfolioSummary | null;
    portfolioSymbols: string[];

    // State
    isLoading: boolean;
    isLoaded: boolean;
    error: string | null;

    // Actions
    refreshPortfolio: () => Promise<void>;

    // Helpers
    hasPortfolio: boolean;
    topHolding: PortfolioHolding | null;
    getHolding: (symbol: string) => PortfolioHolding | undefined;
    isInPortfolio: (symbol: string) => boolean;
}

const PortfolioContext = createContext<PortfolioContextType | undefined>(undefined);

// Default empty summary
const defaultSummary: PortfolioSummary = {
    total_value: 0,
    total_cost: 0,
    total_pnl: 0,
    total_pnl_percent: 0,
    holdings_count: 0,
};

export function PortfolioProvider({ children }: { children: ReactNode }) {
    const [holdings, setHoldings] = useState<PortfolioHolding[]>([]);
    const [summary, setSummary] = useState<PortfolioSummary | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isLoaded, setIsLoaded] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Derived values
    const portfolioSymbols = holdings.map(h => h.symbol);
    const hasPortfolio = holdings.length > 0;
    const topHolding = holdings.length > 0
        ? holdings.reduce((max, h) => (h.current_price * h.shares > max.current_price * max.shares) ? h : max)
        : null;

    // Helper functions
    const getHolding = useCallback((symbol: string) => {
        return holdings.find(h => h.symbol.toUpperCase() === symbol.toUpperCase());
    }, [holdings]);

    const isInPortfolio = useCallback((symbol: string) => {
        return portfolioSymbols.some(s => s.toUpperCase() === symbol.toUpperCase());
    }, [portfolioSymbols]);

    // Fetch portfolio from API
    const refreshPortfolio = useCallback(async () => {
        setIsLoading(true);
        setError(null);

        try {
            // Fetch holdings and summary from /portfolio/summary endpoint
            // This returns both holdings array and summary object
            const response = await apiClient.get('/portfolio/summary');

            // Extract holdings and summary from response
            const holdingsData = response.data?.holdings || [];
            const summaryData = response.data?.summary || defaultSummary;

            // Transform holdings to match our interface
            const transformedHoldings: PortfolioHolding[] = holdingsData.map((h: any) => ({
                id: h.id,
                symbol: h.symbol,
                shares: h.shares,
                avg_cost: h.avg_cost,
                current_price: h.current_price || h.avg_cost,
                pnl: h.pnl || 0,
                pnl_percent: h.pnl_pct || 0,
                signal: h.signal,
            }));

            setHoldings(transformedHoldings);
            setSummary({
                total_value: summaryData.total_current_value || 0,
                total_cost: summaryData.total_cost_basis || 0,
                total_pnl: summaryData.total_pnl || 0,
                total_pnl_percent: summaryData.total_pnl_pct || 0,
                holdings_count: summaryData.total_holdings || transformedHoldings.length,
            });
            setIsLoaded(true);
        } catch (err: any) {
            console.error('Error fetching portfolio:', err);
            setError(err.message || 'Failed to load portfolio');
            // Don't reset holdings on error - keep stale data
            setIsLoaded(true);
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Load portfolio on mount
    useEffect(() => {
        refreshPortfolio();
    }, [refreshPortfolio]);

    // Context value
    const value: PortfolioContextType = {
        holdings,
        summary,
        portfolioSymbols,
        isLoading,
        isLoaded,
        error,
        refreshPortfolio,
        hasPortfolio,
        topHolding,
        getHolding,
        isInPortfolio,
    };

    return (
        <PortfolioContext.Provider value={value}>
            {children}
        </PortfolioContext.Provider>
    );
}

// Custom hook to use portfolio context
export function usePortfolio() {
    const context = useContext(PortfolioContext);
    if (context === undefined) {
        throw new Error('usePortfolio must be used within a PortfolioProvider');
    }
    return context;
}

// Export context for advanced usage
export { PortfolioContext };
