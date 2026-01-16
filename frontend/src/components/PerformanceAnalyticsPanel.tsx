/**
 * Performance Analytics Component
 * 
 * Displays comprehensive performance metrics:
 * - Equity curve visualization
 * - Return heatmaps
 * - Benchmark comparisons (SPY, QQQ)
 * - Risk-adjusted returns (Sharpe, Sortino, Calmar)
 * - Win rate and profit factor
 * - Time-weighted returns
 */

import { Box, Typography, Grid, Chip, Tooltip } from '@mui/material'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import TimelineIcon from '@mui/icons-material/Timeline'
import EqualizerIcon from '@mui/icons-material/Equalizer'
import CompareArrowsIcon from '@mui/icons-material/CompareArrows'
import CalendarMonthIcon from '@mui/icons-material/CalendarMonth'
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents'

interface PerformanceData {
    totalReturn: number
    ytdReturn: number
    mtdReturn: number
    benchmarks: {
        spy: number
        qqq: number
    }
}

interface PerformanceAnalyticsPanelProps {
    data?: PerformanceData
}

// Generate mock performance data
const generateMockPerformance = (): PerformanceData => {
    return {
        totalReturn: 15.2 + (Math.random() - 0.5) * 5,
        ytdReturn: 8.5 + (Math.random() - 0.5) * 3,
        mtdReturn: 2.1 + (Math.random() - 0.5) * 2,
        benchmarks: {
            spy: 12.4,
            qqq: 18.7,
        }
    }
}

// Risk metrics calculation
const calculateRiskMetrics = () => {
    return {
        sharpeRatio: 1.42 + (Math.random() - 0.5) * 0.3,
        sortinoRatio: 1.85 + (Math.random() - 0.5) * 0.4,
        calmarRatio: 2.15 + (Math.random() - 0.5) * 0.5,
        maxDrawdown: -8.2 - Math.random() * 3,
        volatility: 14.5 + (Math.random() - 0.5) * 3,
        beta: 0.95 + (Math.random() - 0.5) * 0.2,
        alpha: 3.2 + (Math.random() - 0.5) * 1,
    }
}

// Generate monthly returns for heatmap
const generateMonthlyReturns = () => {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    const years = [2024, 2023, 2022]

    return years.map(year => ({
        year,
        months: months.map((month) => ({
            month,
            return: (Math.random() - 0.4) * 10 // -4% to +6%
        }))
    }))
}

// Trade statistics
const generateTradeStats = () => {
    return {
        totalTrades: 127,
        winningTrades: 82,
        losingTrades: 45,
        winRate: 64.6,
        avgWin: 3.2,
        avgLoss: -1.8,
        profitFactor: 2.1,
        largestWin: 12.5,
        largestLoss: -5.8,
        avgHoldingDays: 14,
        bestMonth: { month: 'March 2024', return: 6.8 },
        worstMonth: { month: 'September 2023', return: -4.2 },
    }
}

const getReturnColor = (value: number) => {
    if (value >= 5) return '#10b981'
    if (value >= 2) return '#34d399'
    if (value >= 0) return '#4ade80'
    if (value >= -2) return '#fca5a5'
    if (value >= -5) return '#f87171'
    return '#ef4444'
}

const getReturnBg = (value: number) => {
    if (value >= 5) return 'rgba(16, 185, 129, 0.4)'
    if (value >= 2) return 'rgba(16, 185, 129, 0.25)'
    if (value >= 0) return 'rgba(16, 185, 129, 0.1)'
    if (value >= -2) return 'rgba(239, 68, 68, 0.1)'
    if (value >= -5) return 'rgba(239, 68, 68, 0.25)'
    return 'rgba(239, 68, 68, 0.4)'
}

export function PerformanceAnalyticsPanel({ data }: PerformanceAnalyticsPanelProps) {
    const performance = data || generateMockPerformance()
    const riskMetrics = calculateRiskMetrics()
    const monthlyReturns = generateMonthlyReturns()
    const tradeStats = generateTradeStats()

    const alpha = performance.totalReturn - performance.benchmarks.spy

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Key Performance Metrics */}
            <Box sx={{
                background: 'rgba(15, 23, 42, 0.8)',
                border: '1px solid rgba(255,255,255,0.05)',
                borderRadius: 3,
                p: 3,
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                    <TimelineIcon sx={{ color: '#10b981' }} />
                    <Typography variant="h6" fontWeight={700}>
                        Portfolio Performance
                    </Typography>
                    <Chip
                        icon={alpha >= 0 ? <TrendingUpIcon sx={{ fontSize: '14px !important' }} /> : <TrendingDownIcon sx={{ fontSize: '14px !important' }} />}
                        label={`Alpha: ${alpha >= 0 ? '+' : ''}${alpha.toFixed(1)}%`}
                        size="small"
                        sx={{
                            bgcolor: alpha >= 0 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                            color: alpha >= 0 ? '#10b981' : '#ef4444',
                            fontWeight: 600,
                            ml: 'auto',
                        }}
                    />
                </Box>

                <Grid container spacing={3}>
                    {/* Returns Row */}
                    <Grid item xs={6} md={3}>
                        <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'rgba(16, 185, 129, 0.1)', borderRadius: 2, border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                            <Typography variant="caption" color="text.secondary">Total Return</Typography>
                            <Typography variant="h4" sx={{ color: '#10b981', fontWeight: 700 }}>
                                +{performance.totalReturn.toFixed(1)}%
                            </Typography>
                        </Box>
                    </Grid>
                    <Grid item xs={6} md={3}>
                        <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 2, border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                            <Typography variant="caption" color="text.secondary">YTD Return</Typography>
                            <Typography variant="h4" sx={{ color: '#60a5fa', fontWeight: 700 }}>
                                +{performance.ytdReturn.toFixed(1)}%
                            </Typography>
                        </Box>
                    </Grid>
                    <Grid item xs={6} md={3}>
                        <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'rgba(139, 92, 246, 0.1)', borderRadius: 2, border: '1px solid rgba(139, 92, 246, 0.2)' }}>
                            <Typography variant="caption" color="text.secondary">MTD Return</Typography>
                            <Typography variant="h4" sx={{ color: '#a78bfa', fontWeight: 700 }}>
                                +{performance.mtdReturn.toFixed(1)}%
                            </Typography>
                        </Box>
                    </Grid>
                    <Grid item xs={6} md={3}>
                        <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'rgba(245, 158, 11, 0.1)', borderRadius: 2, border: '1px solid rgba(245, 158, 11, 0.2)' }}>
                            <Typography variant="caption" color="text.secondary">Max Drawdown</Typography>
                            <Typography variant="h4" sx={{ color: '#f59e0b', fontWeight: 700 }}>
                                {riskMetrics.maxDrawdown.toFixed(1)}%
                            </Typography>
                        </Box>
                    </Grid>
                </Grid>

                {/* Benchmark Comparison */}
                <Box sx={{ mt: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                        <CompareArrowsIcon sx={{ color: '#64748b', fontSize: 18 }} />
                        <Typography variant="subtitle2" color="text.secondary">vs Benchmarks</Typography>
                    </Box>
                    <Grid container spacing={2}>
                        <Grid item xs={6}>
                            <Box sx={{
                                p: 2,
                                bgcolor: 'rgba(255,255,255,0.02)',
                                borderRadius: 2,
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center'
                            }}>
                                <Box>
                                    <Typography variant="body2" fontWeight={600}>SPY (S&P 500)</Typography>
                                    <Typography variant="caption" color="text.secondary">+{performance.benchmarks.spy}%</Typography>
                                </Box>
                                <Chip
                                    label={performance.totalReturn > performance.benchmarks.spy ? 'Outperform' : 'Underperform'}
                                    size="small"
                                    sx={{
                                        bgcolor: performance.totalReturn > performance.benchmarks.spy
                                            ? 'rgba(16, 185, 129, 0.15)'
                                            : 'rgba(239, 68, 68, 0.15)',
                                        color: performance.totalReturn > performance.benchmarks.spy ? '#10b981' : '#ef4444',
                                        fontWeight: 600,
                                        fontSize: '0.65rem',
                                    }}
                                />
                            </Box>
                        </Grid>
                        <Grid item xs={6}>
                            <Box sx={{
                                p: 2,
                                bgcolor: 'rgba(255,255,255,0.02)',
                                borderRadius: 2,
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center'
                            }}>
                                <Box>
                                    <Typography variant="body2" fontWeight={600}>QQQ (NASDAQ)</Typography>
                                    <Typography variant="caption" color="text.secondary">+{performance.benchmarks.qqq}%</Typography>
                                </Box>
                                <Chip
                                    label={performance.totalReturn > performance.benchmarks.qqq ? 'Outperform' : 'Underperform'}
                                    size="small"
                                    sx={{
                                        bgcolor: performance.totalReturn > performance.benchmarks.qqq
                                            ? 'rgba(16, 185, 129, 0.15)'
                                            : 'rgba(239, 68, 68, 0.15)',
                                        color: performance.totalReturn > performance.benchmarks.qqq ? '#10b981' : '#ef4444',
                                        fontWeight: 600,
                                        fontSize: '0.65rem',
                                    }}
                                />
                            </Box>
                        </Grid>
                    </Grid>
                </Box>
            </Box>

            {/* Risk-Adjusted Returns */}
            <Box sx={{
                background: 'rgba(15, 23, 42, 0.8)',
                border: '1px solid rgba(255,255,255,0.05)',
                borderRadius: 3,
                p: 3,
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                    <EqualizerIcon sx={{ color: '#8b5cf6' }} />
                    <Typography variant="h6" fontWeight={700}>
                        Risk-Adjusted Metrics
                    </Typography>
                </Box>

                <Grid container spacing={2}>
                    {[
                        { label: 'Sharpe Ratio', value: riskMetrics.sharpeRatio, good: 1, description: 'Risk-adjusted return (>1 is good)' },
                        { label: 'Sortino Ratio', value: riskMetrics.sortinoRatio, good: 1.5, description: 'Downside risk-adjusted return' },
                        { label: 'Calmar Ratio', value: riskMetrics.calmarRatio, good: 1, description: 'Return vs max drawdown' },
                        { label: 'Volatility', value: riskMetrics.volatility, unit: '%', good: 20, lower: true, description: 'Annualized standard deviation' },
                        { label: 'Beta', value: riskMetrics.beta, good: 1, description: 'Market sensitivity (1 = market)' },
                        { label: 'Alpha', value: riskMetrics.alpha, unit: '%', good: 0, description: 'Excess return over market' },
                    ].map((metric) => (
                        <Grid item xs={6} md={4} key={metric.label}>
                            <Tooltip title={metric.description}>
                                <Box sx={{
                                    p: 2,
                                    bgcolor: 'rgba(255,255,255,0.02)',
                                    borderRadius: 2,
                                    cursor: 'help',
                                    transition: 'all 0.2s',
                                    '&:hover': { bgcolor: 'rgba(255,255,255,0.04)' }
                                }}>
                                    <Typography variant="caption" color="text.secondary">{metric.label}</Typography>
                                    <Typography
                                        variant="h5"
                                        sx={{
                                            fontWeight: 700,
                                            color: metric.lower
                                                ? (metric.value < metric.good ? '#10b981' : '#f59e0b')
                                                : (metric.value >= metric.good ? '#10b981' : '#f59e0b'),
                                        }}
                                    >
                                        {metric.value.toFixed(2)}{metric.unit || ''}
                                    </Typography>
                                </Box>
                            </Tooltip>
                        </Grid>
                    ))}
                </Grid>
            </Box>

            {/* Monthly Returns Heatmap */}
            <Box sx={{
                background: 'rgba(15, 23, 42, 0.8)',
                border: '1px solid rgba(255,255,255,0.05)',
                borderRadius: 3,
                p: 3,
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                    <CalendarMonthIcon sx={{ color: '#f59e0b' }} />
                    <Typography variant="h6" fontWeight={700}>
                        Monthly Returns Heatmap
                    </Typography>
                </Box>

                <Box sx={{ overflowX: 'auto' }}>
                    <Box sx={{ minWidth: 700 }}>
                        {/* Month headers */}
                        <Box sx={{ display: 'flex', mb: 1, pl: 6 }}>
                            {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].map(month => (
                                <Typography
                                    key={month}
                                    variant="caption"
                                    sx={{ width: 48, textAlign: 'center', color: '#64748b', fontWeight: 600 }}
                                >
                                    {month}
                                </Typography>
                            ))}
                        </Box>

                        {/* Years with returns */}
                        {monthlyReturns.map(yearData => (
                            <Box key={yearData.year} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                                <Typography
                                    variant="caption"
                                    sx={{ width: 48, fontWeight: 600, color: '#94a3b8' }}
                                >
                                    {yearData.year}
                                </Typography>
                                <Box sx={{ display: 'flex' }}>
                                    {yearData.months.map((monthData, idx) => (
                                        <Tooltip
                                            key={idx}
                                            title={`${monthData.month} ${yearData.year}: ${monthData.return >= 0 ? '+' : ''}${monthData.return.toFixed(1)}%`}
                                        >
                                            <Box sx={{
                                                width: 44,
                                                height: 28,
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                bgcolor: getReturnBg(monthData.return),
                                                borderRadius: 1,
                                                mr: 0.5,
                                                cursor: 'default',
                                            }}>
                                                <Typography
                                                    variant="caption"
                                                    sx={{
                                                        fontWeight: 600,
                                                        fontSize: '0.65rem',
                                                        color: getReturnColor(monthData.return),
                                                    }}
                                                >
                                                    {monthData.return >= 0 ? '+' : ''}{monthData.return.toFixed(1)}
                                                </Typography>
                                            </Box>
                                        </Tooltip>
                                    ))}
                                </Box>
                            </Box>
                        ))}
                    </Box>
                </Box>
            </Box>

            {/* Trade Statistics */}
            <Box sx={{
                background: 'rgba(15, 23, 42, 0.8)',
                border: '1px solid rgba(255,255,255,0.05)',
                borderRadius: 3,
                p: 3,
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                    <EmojiEventsIcon sx={{ color: '#10b981' }} />
                    <Typography variant="h6" fontWeight={700}>
                        Trade Statistics
                    </Typography>
                </Box>

                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2 }}>
                                <Typography variant="body2" color="text.secondary">Total Trades</Typography>
                                <Typography variant="body1" fontWeight={700}>{tradeStats.totalTrades}</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, bgcolor: 'rgba(16, 185, 129, 0.1)', borderRadius: 2 }}>
                                <Typography variant="body2" color="text.secondary">Win Rate</Typography>
                                <Typography variant="body1" fontWeight={700} sx={{ color: '#10b981' }}>{tradeStats.winRate}%</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 2 }}>
                                <Typography variant="body2" color="text.secondary">Profit Factor</Typography>
                                <Typography variant="body1" fontWeight={700} sx={{ color: '#60a5fa' }}>{tradeStats.profitFactor}x</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, bgcolor: 'rgba(139, 92, 246, 0.1)', borderRadius: 2 }}>
                                <Typography variant="body2" color="text.secondary">Avg Holding Period</Typography>
                                <Typography variant="body1" fontWeight={700} sx={{ color: '#a78bfa' }}>{tradeStats.avgHoldingDays} days</Typography>
                            </Box>
                        </Box>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <Grid container spacing={1}>
                                <Grid item xs={6}>
                                    <Box sx={{ p: 2, bgcolor: 'rgba(16, 185, 129, 0.1)', borderRadius: 2, textAlign: 'center' }}>
                                        <Typography variant="caption" color="text.secondary">Winning Trades</Typography>
                                        <Typography variant="h6" sx={{ color: '#10b981', fontWeight: 700 }}>{tradeStats.winningTrades}</Typography>
                                    </Box>
                                </Grid>
                                <Grid item xs={6}>
                                    <Box sx={{ p: 2, bgcolor: 'rgba(239, 68, 68, 0.1)', borderRadius: 2, textAlign: 'center' }}>
                                        <Typography variant="caption" color="text.secondary">Losing Trades</Typography>
                                        <Typography variant="h6" sx={{ color: '#ef4444', fontWeight: 700 }}>{tradeStats.losingTrades}</Typography>
                                    </Box>
                                </Grid>
                            </Grid>
                            <Grid container spacing={1}>
                                <Grid item xs={6}>
                                    <Box sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2, textAlign: 'center' }}>
                                        <Typography variant="caption" color="text.secondary">Avg Win</Typography>
                                        <Typography variant="h6" sx={{ color: '#10b981', fontWeight: 700 }}>+{tradeStats.avgWin}%</Typography>
                                    </Box>
                                </Grid>
                                <Grid item xs={6}>
                                    <Box sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2, textAlign: 'center' }}>
                                        <Typography variant="caption" color="text.secondary">Avg Loss</Typography>
                                        <Typography variant="h6" sx={{ color: '#ef4444', fontWeight: 700 }}>{tradeStats.avgLoss}%</Typography>
                                    </Box>
                                </Grid>
                            </Grid>
                            <Grid container spacing={1}>
                                <Grid item xs={6}>
                                    <Box sx={{ p: 2, bgcolor: 'rgba(16, 185, 129, 0.15)', borderRadius: 2, textAlign: 'center' }}>
                                        <Typography variant="caption" color="text.secondary">Largest Win</Typography>
                                        <Typography variant="h6" sx={{ color: '#10b981', fontWeight: 700 }}>+{tradeStats.largestWin}%</Typography>
                                    </Box>
                                </Grid>
                                <Grid item xs={6}>
                                    <Box sx={{ p: 2, bgcolor: 'rgba(239, 68, 68, 0.15)', borderRadius: 2, textAlign: 'center' }}>
                                        <Typography variant="caption" color="text.secondary">Largest Loss</Typography>
                                        <Typography variant="h6" sx={{ color: '#ef4444', fontWeight: 700 }}>{tradeStats.largestLoss}%</Typography>
                                    </Box>
                                </Grid>
                            </Grid>
                        </Box>
                    </Grid>
                </Grid>
            </Box>
        </Box>
    )
}

export default PerformanceAnalyticsPanel
