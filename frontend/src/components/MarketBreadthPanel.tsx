/**
 * Market Breadth Component
 * 
 * Displays market health indicators:
 * - Advance/Decline ratio
 * - New Highs vs New Lows
 * - Fear & Greed Index
 * - Sector performance
 * - Market sentiment gauges
 */

import { useMemo } from 'react'
import { Box, Typography, Grid, LinearProgress, Chip, Tooltip } from '@mui/material'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import DonutLargeIcon from '@mui/icons-material/DonutLarge'
import SentimentVeryDissatisfiedIcon from '@mui/icons-material/SentimentVeryDissatisfied'
import SentimentVerySatisfiedIcon from '@mui/icons-material/SentimentVerySatisfied'
import PublicIcon from '@mui/icons-material/Public'

// Mock market breadth data
const generateMarketBreadthData = () => {
    return {
        advanceDecline: {
            advances: Math.floor(280 + Math.random() * 40),
            declines: Math.floor(200 + Math.random() * 40),
            unchanged: Math.floor(10 + Math.random() * 10),
        },
        newHighsLows: {
            newHighs: Math.floor(40 + Math.random() * 30),
            newLows: Math.floor(10 + Math.random() * 20),
        },
        fearGreed: Math.floor(35 + Math.random() * 40), // 0-100
        vix: 13.2 + (Math.random() - 0.5) * 4,
        putCallRatio: 0.75 + (Math.random() - 0.5) * 0.3,
        sectors: [
            { name: 'Technology', change: 1.2 + (Math.random() - 0.5) * 2 },
            { name: 'Healthcare', change: 0.8 + (Math.random() - 0.5) * 2 },
            { name: 'Financial', change: 0.5 + (Math.random() - 0.5) * 2 },
            { name: 'Energy', change: -0.3 + (Math.random() - 0.5) * 2 },
            { name: 'Consumer', change: 0.7 + (Math.random() - 0.5) * 2 },
            { name: 'Industrial', change: 0.4 + (Math.random() - 0.5) * 2 },
            { name: 'Real Estate', change: -0.5 + (Math.random() - 0.5) * 2 },
            { name: 'Materials', change: 0.3 + (Math.random() - 0.5) * 2 },
            { name: 'Utilities', change: -0.2 + (Math.random() - 0.5) * 1 },
            { name: 'Communication', change: 0.9 + (Math.random() - 0.5) * 2 },
        ].sort((a, b) => b.change - a.change),
        indices: {
            sp500: { value: 4783.45, change: 0.45 },
            nasdaq: { value: 15062.83, change: 0.72 },
            dow: { value: 37695.25, change: 0.32 },
            russell: { value: 2015.32, change: -0.18 },
        },
    }
}

const getFearGreedColor = (value: number) => {
    if (value >= 75) return '#10b981' // Extreme Greed
    if (value >= 55) return '#34d399' // Greed
    if (value >= 45) return '#fbbf24' // Neutral
    if (value >= 25) return '#f59e0b' // Fear
    return '#ef4444' // Extreme Fear
}

const getFearGreedLabel = (value: number) => {
    if (value >= 75) return 'Extreme Greed'
    if (value >= 55) return 'Greed'
    if (value >= 45) return 'Neutral'
    if (value >= 25) return 'Fear'
    return 'Extreme Fear'
}

export function MarketBreadthPanel() {
    const data = useMemo(() => generateMarketBreadthData(), [])

    const adRatio = data.advanceDecline.advances / data.advanceDecline.declines
    const adPercent = (data.advanceDecline.advances / (data.advanceDecline.advances + data.advanceDecline.declines)) * 100

    return (
        <Box sx={{
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid rgba(255,255,255,0.05)',
            borderRadius: 3,
            p: 3,
        }}>
            {/* Header */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                <PublicIcon sx={{ color: '#3b82f6' }} />
                <Typography variant="h6" fontWeight={700}>
                    Market Breadth
                </Typography>
                <Chip
                    label="Live"
                    size="small"
                    sx={{
                        bgcolor: 'rgba(16, 185, 129, 0.15)',
                        color: '#10b981',
                        fontSize: '0.65rem',
                        height: 20,
                        ml: 'auto',
                    }}
                />
            </Box>

            <Grid container spacing={3}>
                {/* Fear & Greed Index */}
                <Grid item xs={12} md={4}>
                    <Box sx={{
                        p: 2,
                        bgcolor: 'rgba(255,255,255,0.02)',
                        borderRadius: 2,
                        border: '1px solid rgba(255,255,255,0.05)',
                        textAlign: 'center',
                    }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                            Fear & Greed Index
                        </Typography>
                        <Box sx={{ position: 'relative', display: 'inline-flex', my: 2 }}>
                            <Box sx={{
                                width: 100,
                                height: 100,
                                borderRadius: '50%',
                                background: `conic-gradient(${getFearGreedColor(data.fearGreed)} ${data.fearGreed}%, rgba(255,255,255,0.1) 0)`,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                            }}>
                                <Box sx={{
                                    width: 80,
                                    height: 80,
                                    borderRadius: '50%',
                                    bgcolor: '#0a0b14',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    flexDirection: 'column',
                                }}>
                                    <Typography variant="h4" sx={{ color: getFearGreedColor(data.fearGreed), fontWeight: 700 }}>
                                        {data.fearGreed}
                                    </Typography>
                                </Box>
                            </Box>
                        </Box>
                        <Chip
                            icon={data.fearGreed >= 50 ? (
                                <SentimentVerySatisfiedIcon sx={{ fontSize: '14px !important' }} />
                            ) : (
                                <SentimentVeryDissatisfiedIcon sx={{ fontSize: '14px !important' }} />
                            )}
                            label={getFearGreedLabel(data.fearGreed)}
                            size="small"
                            sx={{
                                bgcolor: `${getFearGreedColor(data.fearGreed)}20`,
                                color: getFearGreedColor(data.fearGreed),
                                fontWeight: 600,
                            }}
                        />
                    </Box>
                </Grid>

                {/* Advance/Decline */}
                <Grid item xs={12} md={4}>
                    <Box sx={{
                        p: 2,
                        bgcolor: 'rgba(255,255,255,0.02)',
                        borderRadius: 2,
                        border: '1px solid rgba(255,255,255,0.05)',
                    }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                            Advance/Decline (S&P 500)
                        </Typography>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <TrendingUpIcon sx={{ color: '#10b981', fontSize: 16 }} />
                                <Typography variant="body2" color="#10b981">{data.advanceDecline.advances}</Typography>
                            </Box>
                            <Typography variant="body2" color="text.secondary">{data.advanceDecline.unchanged}</Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <TrendingDownIcon sx={{ color: '#ef4444', fontSize: 16 }} />
                                <Typography variant="body2" color="#ef4444">{data.advanceDecline.declines}</Typography>
                            </Box>
                        </Box>
                        <LinearProgress
                            variant="determinate"
                            value={adPercent}
                            sx={{
                                height: 8,
                                borderRadius: 4,
                                bgcolor: 'rgba(239, 68, 68, 0.3)',
                                '& .MuiLinearProgress-bar': {
                                    bgcolor: '#10b981',
                                    borderRadius: 4,
                                }
                            }}
                        />
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                            <Typography variant="caption" color="text.secondary">
                                A/D Ratio: {adRatio.toFixed(2)}
                            </Typography>
                            <Chip
                                label={adRatio > 1 ? 'Bullish' : 'Bearish'}
                                size="small"
                                sx={{
                                    height: 18,
                                    bgcolor: adRatio > 1 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                                    color: adRatio > 1 ? '#10b981' : '#ef4444',
                                    fontSize: '0.6rem',
                                }}
                            />
                        </Box>

                        <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                            <Typography variant="caption" color="text.secondary">New Highs vs New Lows</Typography>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                                <Typography variant="body2" sx={{ color: '#10b981' }}>
                                    {data.newHighsLows.newHighs} Highs
                                </Typography>
                                <Typography variant="body2" sx={{ color: '#ef4444' }}>
                                    {data.newHighsLows.newLows} Lows
                                </Typography>
                            </Box>
                        </Box>
                    </Box>
                </Grid>

                {/* Market Indicators */}
                <Grid item xs={12} md={4}>
                    <Box sx={{
                        p: 2,
                        bgcolor: 'rgba(255,255,255,0.02)',
                        borderRadius: 2,
                        border: '1px solid rgba(255,255,255,0.05)',
                    }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                            Key Indicators
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="body2" color="text.secondary">VIX (Fear Index)</Typography>
                                <Typography
                                    variant="body2"
                                    sx={{
                                        color: data.vix > 20 ? '#ef4444' : data.vix > 15 ? '#f59e0b' : '#10b981',
                                        fontWeight: 600,
                                    }}
                                >
                                    {data.vix.toFixed(2)}
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="body2" color="text.secondary">Put/Call Ratio</Typography>
                                <Typography
                                    variant="body2"
                                    sx={{
                                        color: data.putCallRatio > 1 ? '#ef4444' : data.putCallRatio > 0.7 ? '#f59e0b' : '#10b981',
                                        fontWeight: 600,
                                    }}
                                >
                                    {data.putCallRatio.toFixed(2)}
                                </Typography>
                            </Box>
                        </Box>

                        <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                            <Typography variant="caption" color="text.secondary">Major Indices</Typography>
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 1 }}>
                                {[
                                    { name: 'S&P 500', ...data.indices.sp500 },
                                    { name: 'NASDAQ', ...data.indices.nasdaq },
                                    { name: 'DOW', ...data.indices.dow },
                                ].map(index => (
                                    <Box key={index.name} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography variant="caption">{index.name}</Typography>
                                        <Typography
                                            variant="caption"
                                            sx={{ color: index.change >= 0 ? '#10b981' : '#ef4444', fontWeight: 600 }}
                                        >
                                            {index.change >= 0 ? '+' : ''}{index.change.toFixed(2)}%
                                        </Typography>
                                    </Box>
                                ))}
                            </Box>
                        </Box>
                    </Box>
                </Grid>
            </Grid>

            {/* Sector Performance */}
            <Box sx={{ mt: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <DonutLargeIcon sx={{ color: '#8b5cf6', fontSize: 18 }} />
                    <Typography variant="subtitle2" fontWeight={600}>Sector Performance (Today)</Typography>
                </Box>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {data.sectors.map(sector => (
                        <Tooltip key={sector.name} title={`${sector.name}: ${sector.change >= 0 ? '+' : ''}${sector.change.toFixed(2)}%`}>
                            <Chip
                                label={
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <span>{sector.name}</span>
                                        {sector.change >= 0 ? (
                                            <TrendingUpIcon sx={{ fontSize: 12 }} />
                                        ) : (
                                            <TrendingDownIcon sx={{ fontSize: 12 }} />
                                        )}
                                        <span>{sector.change >= 0 ? '+' : ''}{sector.change.toFixed(1)}%</span>
                                    </Box>
                                }
                                size="small"
                                sx={{
                                    bgcolor: sector.change >= 0 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                                    color: sector.change >= 0 ? '#10b981' : '#ef4444',
                                    border: `1px solid ${sector.change >= 0 ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                                    fontSize: '0.7rem',
                                }}
                            />
                        </Tooltip>
                    ))}
                </Box>
            </Box>
        </Box>
    )
}

export default MarketBreadthPanel
