/**
 * Risk Management Component
 * 
 * Displays critical risk metrics for portfolio:
 * - Stop-loss levels for each position
 * - Position sizing as % of portfolio
 * - Maximum drawdown tracking
 * - Concentration risk indicators
 * - Value at Risk (VaR) calculations
 */

import { Box, Typography, Grid, LinearProgress, Tooltip, Chip } from '@mui/material'
import WarningIcon from '@mui/icons-material/Warning'
import ShieldIcon from '@mui/icons-material/Shield'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import PieChartIcon from '@mui/icons-material/PieChart'

interface Holding {
    symbol: string
    shares: number
    avg_cost: number
    current_price: number
    pnl: number
    pnl_pct: number
}

interface RiskMetrics {
    total_value: number
    holdings: Holding[]
}

export function RiskManagementPanel({ metrics }: { metrics: RiskMetrics }) {
    const { holdings, total_value } = metrics

    // Calculate position weights
    const positionsWithRisk = holdings.map(h => {
        const currentValue = h.shares * h.current_price
        const weight = (currentValue / total_value) * 100

        // Calculate stop-loss levels (dynamic based on volatility proxy)
        const volatility = Math.abs(h.pnl_pct) / 10 + 0.05 // Simplified volatility estimate
        const stopLossLevel = h.current_price * (1 - Math.max(0.05, Math.min(0.15, volatility)))
        const takeProfitLevel = h.current_price * (1 + volatility * 2)

        // Risk rating based on concentration
        let riskLevel: 'low' | 'medium' | 'high' = 'low'
        if (weight > 25) riskLevel = 'high'
        else if (weight > 15) riskLevel = 'medium'

        return {
            ...h,
            currentValue,
            weight,
            stopLossLevel,
            takeProfitLevel,
            riskLevel,
        }
    }).sort((a, b) => b.weight - a.weight)

    // Portfolio-level risk metrics
    const maxPosition = Math.max(...positionsWithRisk.map(p => p.weight))
    const top3Concentration = positionsWithRisk.slice(0, 3).reduce((sum, p) => sum + p.weight, 0)
    const portfolioVaR = total_value * 0.025 // Simplified 2.5% daily VaR
    const maxDrawdown = Math.min(...positionsWithRisk.map(p => p.pnl_pct))

    const getRiskColor = (level: string) => {
        switch (level) {
            case 'high': return '#ef4444'
            case 'medium': return '#f59e0b'
            default: return '#10b981'
        }
    }

    return (
        <Box sx={{
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid rgba(255,255,255,0.05)',
            borderRadius: 3,
            p: 3,
            mt: 3
        }}>
            {/* Header */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                <ShieldIcon sx={{ color: '#8b5cf6' }} />
                <Typography variant="h6" fontWeight={700}>
                    Risk Management
                </Typography>
                <Chip
                    label="Live"
                    size="small"
                    sx={{
                        bgcolor: 'rgba(16, 185, 129, 0.15)',
                        color: '#10b981',
                        fontSize: '0.7rem',
                        height: 20,
                    }}
                />
            </Box>

            {/* Portfolio Risk Metrics */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={6} md={3}>
                    <Box sx={{ p: 2, bgcolor: 'rgba(239, 68, 68, 0.1)', borderRadius: 2, border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                        <Typography variant="caption" color="text.secondary">Daily VaR (95%)</Typography>
                        <Typography variant="h6" sx={{ color: '#f87171', fontWeight: 700 }}>
                            -${portfolioVaR.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            Max expected loss
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                    <Box sx={{ p: 2, bgcolor: 'rgba(245, 158, 11, 0.1)', borderRadius: 2, border: '1px solid rgba(245, 158, 11, 0.2)' }}>
                        <Typography variant="caption" color="text.secondary">Max Drawdown</Typography>
                        <Typography variant="h6" sx={{ color: maxDrawdown < -5 ? '#ef4444' : '#fbbf24', fontWeight: 700 }}>
                            {maxDrawdown.toFixed(1)}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            Worst position
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                    <Box sx={{ p: 2, bgcolor: 'rgba(139, 92, 246, 0.1)', borderRadius: 2, border: '1px solid rgba(139, 92, 246, 0.2)' }}>
                        <Typography variant="caption" color="text.secondary">Top Position</Typography>
                        <Typography variant="h6" sx={{ color: maxPosition > 25 ? '#ef4444' : '#a78bfa', fontWeight: 700 }}>
                            {maxPosition.toFixed(1)}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            Concentration
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                    <Box sx={{ p: 2, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 2, border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                        <Typography variant="caption" color="text.secondary">Top 3 Holdings</Typography>
                        <Typography variant="h6" sx={{ color: top3Concentration > 60 ? '#f59e0b' : '#60a5fa', fontWeight: 700 }}>
                            {top3Concentration.toFixed(1)}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            Combined weight
                        </Typography>
                    </Box>
                </Grid>
            </Grid>

            {/* Position Risk Details */}
            <Typography variant="subtitle2" sx={{ color: '#94a3b8', mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                <PieChartIcon sx={{ fontSize: 16 }} />
                Position Risk Analysis
            </Typography>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {positionsWithRisk.slice(0, 8).map((position) => (
                    <Box
                        key={position.symbol}
                        sx={{
                            p: 2,
                            bgcolor: 'rgba(255,255,255,0.02)',
                            borderRadius: 2,
                            border: `1px solid ${getRiskColor(position.riskLevel)}20`,
                            display: 'flex',
                            flexDirection: 'column',
                            gap: 1,
                        }}
                    >
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                <Typography fontWeight={700} color="#fff">{position.symbol}</Typography>
                                {position.riskLevel !== 'low' && (
                                    <Tooltip title={`${position.riskLevel === 'high' ? 'High' : 'Medium'} concentration risk`}>
                                        <WarningIcon sx={{ color: getRiskColor(position.riskLevel), fontSize: 16 }} />
                                    </Tooltip>
                                )}
                            </Box>
                            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                                <Tooltip title="Stop-Loss Level">
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <TrendingDownIcon sx={{ color: '#ef4444', fontSize: 14 }} />
                                        <Typography variant="caption" sx={{ color: '#ef4444' }}>
                                            ${position.stopLossLevel.toFixed(2)}
                                        </Typography>
                                    </Box>
                                </Tooltip>
                                <Typography variant="caption" color="text.secondary">
                                    → ${position.current_price.toFixed(2)} →
                                </Typography>
                                <Tooltip title="Take-Profit Target">
                                    <Typography variant="caption" sx={{ color: '#10b981' }}>
                                        ${position.takeProfitLevel.toFixed(2)}
                                    </Typography>
                                </Tooltip>
                            </Box>
                        </Box>

                        {/* Position weight bar */}
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <Box sx={{ flex: 1 }}>
                                <LinearProgress
                                    variant="determinate"
                                    value={Math.min(position.weight, 100)}
                                    sx={{
                                        height: 6,
                                        borderRadius: 3,
                                        bgcolor: 'rgba(255,255,255,0.1)',
                                        '& .MuiLinearProgress-bar': {
                                            bgcolor: getRiskColor(position.riskLevel),
                                            borderRadius: 3,
                                        }
                                    }}
                                />
                            </Box>
                            <Typography variant="caption" sx={{ color: '#94a3b8', minWidth: 45 }}>
                                {position.weight.toFixed(1)}%
                            </Typography>
                            <Typography variant="caption" sx={{ color: position.pnl >= 0 ? '#10b981' : '#ef4444' }}>
                                {position.pnl >= 0 ? '+' : ''}{position.pnl_pct.toFixed(1)}%
                            </Typography>
                        </Box>
                    </Box>
                ))}
            </Box>

            {positionsWithRisk.length > 8 && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block', textAlign: 'center' }}>
                    + {positionsWithRisk.length - 8} more positions
                </Typography>
            )}
        </Box>
    )
}

export default RiskManagementPanel
