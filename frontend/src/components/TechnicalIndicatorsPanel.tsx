/**
 * Technical Indicators Panel
 * 
 * Displays key technical analysis indicators:
 * - RSI (Relative Strength Index)
 * - MACD (Moving Average Convergence Divergence)
 * - Bollinger Bands
 * - Moving Averages (20/50/200 SMA/EMA)
 * - Support/Resistance Levels
 * - VWAP
 */

import { Box, Typography, Grid, LinearProgress, Tooltip, Chip, Divider } from '@mui/material'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import TimelineIcon from '@mui/icons-material/Timeline'
import StackedLineChartIcon from '@mui/icons-material/StackedLineChart'

interface TechnicalIndicatorsProps {
    symbol: string
    currentPrice: number
    timeframe: string
}

// Generate mock indicator data based on symbol and price
const generateIndicators = (symbol: string, currentPrice: number) => {
    // Create deterministic variation based on symbol
    const seed = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
    const variation = (seed % 100) / 100

    // RSI (0-100)
    const rsi = 30 + (variation * 50) + (Math.random() - 0.5) * 10

    // MACD
    const macdLine = (variation - 0.5) * 5
    const signalLine = macdLine + (Math.random() - 0.5) * 2
    const histogram = macdLine - signalLine

    // Bollinger Bands (based on current price)
    const volatility = 0.02 + variation * 0.03
    const middleBand = currentPrice
    const upperBand = currentPrice * (1 + volatility * 2)
    const lowerBand = currentPrice * (1 - volatility * 2)

    // Moving Averages
    const sma20 = currentPrice * (1 + (Math.random() - 0.5) * 0.03)
    const sma50 = currentPrice * (1 + (Math.random() - 0.5) * 0.05)
    const sma200 = currentPrice * (1 + (Math.random() - 0.5) * 0.08)
    const ema20 = currentPrice * (1 + (Math.random() - 0.5) * 0.02)

    // Support/Resistance
    const support1 = currentPrice * (1 - 0.02 - variation * 0.02)
    const support2 = currentPrice * (1 - 0.05 - variation * 0.03)
    const resistance1 = currentPrice * (1 + 0.02 + variation * 0.02)
    const resistance2 = currentPrice * (1 + 0.05 + variation * 0.03)

    // VWAP
    const vwap = currentPrice * (1 + (Math.random() - 0.5) * 0.01)

    // ATR (Average True Range)
    const atr = currentPrice * 0.015 * (1 + variation * 0.5)

    return {
        rsi,
        macd: { line: macdLine, signal: signalLine, histogram },
        bollingerBands: { upper: upperBand, middle: middleBand, lower: lowerBand },
        movingAverages: { sma20, sma50, sma200, ema20 },
        supportResistance: { support1, support2, resistance1, resistance2 },
        vwap,
        atr,
    }
}

const getRSIColor = (value: number) => {
    if (value >= 70) return '#ef4444' // Overbought
    if (value <= 30) return '#10b981' // Oversold
    return '#94a3b8'
}

const getRSILabel = (value: number) => {
    if (value >= 70) return 'Overbought'
    if (value <= 30) return 'Oversold'
    return 'Neutral'
}

export function TechnicalIndicatorsPanel({ symbol, currentPrice, timeframe }: TechnicalIndicatorsProps) {
    const indicators = generateIndicators(symbol, currentPrice)

    return (
        <Box sx={{
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid rgba(255,255,255,0.05)',
            borderRadius: 3,
            p: 3,
        }}>
            {/* Header */}
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <TimelineIcon sx={{ color: '#8b5cf6' }} />
                    <Typography variant="h6" fontWeight={700}>
                        Technical Indicators
                    </Typography>
                </Box>
                <Chip
                    label={timeframe}
                    size="small"
                    sx={{ bgcolor: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', fontWeight: 600 }}
                />
            </Box>

            <Grid container spacing={3}>
                {/* RSI */}
                <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2 }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                            RSI (14)
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
                            <Typography variant="h4" sx={{ color: getRSIColor(indicators.rsi), fontWeight: 700 }}>
                                {indicators.rsi.toFixed(1)}
                            </Typography>
                            <Chip
                                label={getRSILabel(indicators.rsi)}
                                size="small"
                                sx={{
                                    bgcolor: `${getRSIColor(indicators.rsi)}20`,
                                    color: getRSIColor(indicators.rsi),
                                    fontWeight: 600,
                                    fontSize: '0.7rem',
                                }}
                            />
                        </Box>
                        <Box sx={{ position: 'relative', height: 8, bgcolor: 'rgba(255,255,255,0.1)', borderRadius: 4 }}>
                            <Box
                                sx={{
                                    position: 'absolute',
                                    left: '30%',
                                    right: '30%',
                                    height: '100%',
                                    bgcolor: 'rgba(16, 185, 129, 0.3)',
                                    borderRadius: 4,
                                }}
                            />
                            <Box
                                sx={{
                                    position: 'absolute',
                                    left: `${indicators.rsi}%`,
                                    top: '-4px',
                                    width: 16,
                                    height: 16,
                                    bgcolor: getRSIColor(indicators.rsi),
                                    borderRadius: '50%',
                                    transform: 'translateX(-50%)',
                                    boxShadow: `0 0 10px ${getRSIColor(indicators.rsi)}80`,
                                }}
                            />
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                            <Typography variant="caption" color="text.secondary">0</Typography>
                            <Typography variant="caption" sx={{ color: '#10b981' }}>30</Typography>
                            <Typography variant="caption" sx={{ color: '#ef4444' }}>70</Typography>
                            <Typography variant="caption" color="text.secondary">100</Typography>
                        </Box>
                    </Box>
                </Grid>

                {/* MACD */}
                <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2 }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                            MACD (12,26,9)
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            {indicators.macd.histogram > 0 ? (
                                <TrendingUpIcon sx={{ color: '#10b981', fontSize: 20 }} />
                            ) : (
                                <TrendingDownIcon sx={{ color: '#ef4444', fontSize: 20 }} />
                            )}
                            <Typography variant="h5" sx={{
                                color: indicators.macd.histogram > 0 ? '#10b981' : '#ef4444',
                                fontWeight: 700
                            }}>
                                {indicators.macd.histogram > 0 ? '+' : ''}{indicators.macd.histogram.toFixed(3)}
                            </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption" color="text.secondary">MACD Line</Typography>
                                <Typography variant="caption" sx={{ color: '#60a5fa' }}>
                                    {indicators.macd.line.toFixed(3)}
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption" color="text.secondary">Signal Line</Typography>
                                <Typography variant="caption" sx={{ color: '#f59e0b' }}>
                                    {indicators.macd.signal.toFixed(3)}
                                </Typography>
                            </Box>
                        </Box>
                        <Chip
                            label={indicators.macd.histogram > 0 ? 'Bullish Crossover' : 'Bearish Crossover'}
                            size="small"
                            sx={{
                                mt: 1,
                                bgcolor: indicators.macd.histogram > 0 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                                color: indicators.macd.histogram > 0 ? '#10b981' : '#ef4444',
                                fontSize: '0.65rem',
                            }}
                        />
                    </Box>
                </Grid>

                {/* Bollinger Bands */}
                <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2 }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                            Bollinger Bands (20,2)
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Typography variant="caption" color="text.secondary">Upper</Typography>
                                <Typography variant="body2" sx={{ color: '#ef4444', fontWeight: 600 }}>
                                    ${indicators.bollingerBands.upper.toFixed(2)}
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Typography variant="caption" color="text.secondary">Price</Typography>
                                <Typography variant="body2" sx={{ color: '#fff', fontWeight: 700 }}>
                                    ${currentPrice.toFixed(2)}
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Typography variant="caption" color="text.secondary">Middle (SMA20)</Typography>
                                <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                                    ${indicators.bollingerBands.middle.toFixed(2)}
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Typography variant="caption" color="text.secondary">Lower</Typography>
                                <Typography variant="body2" sx={{ color: '#10b981', fontWeight: 600 }}>
                                    ${indicators.bollingerBands.lower.toFixed(2)}
                                </Typography>
                            </Box>
                        </Box>
                        {/* Band Width indicator */}
                        <Box sx={{ mt: 1.5 }}>
                            <LinearProgress
                                variant="determinate"
                                value={((currentPrice - indicators.bollingerBands.lower) /
                                    (indicators.bollingerBands.upper - indicators.bollingerBands.lower)) * 100}
                                sx={{
                                    height: 8,
                                    borderRadius: 4,
                                    bgcolor: 'rgba(255,255,255,0.1)',
                                    '& .MuiLinearProgress-bar': {
                                        bgcolor: currentPrice > indicators.bollingerBands.middle ? '#ef4444' : '#10b981',
                                        borderRadius: 4,
                                    }
                                }}
                            />
                        </Box>
                    </Box>
                </Grid>
            </Grid>

            <Divider sx={{ my: 3, borderColor: 'rgba(255,255,255,0.05)' }} />

            {/* Moving Averages & Support/Resistance */}
            <Grid container spacing={3}>
                {/* Moving Averages */}
                <Grid item xs={12} md={6}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                        <StackedLineChartIcon sx={{ color: '#60a5fa', fontSize: 18 }} />
                        <Typography variant="subtitle2" fontWeight={600}>Moving Averages</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {[
                            { label: 'SMA 20', value: indicators.movingAverages.sma20, color: '#f59e0b' },
                            { label: 'SMA 50', value: indicators.movingAverages.sma50, color: '#8b5cf6' },
                            { label: 'SMA 200', value: indicators.movingAverages.sma200, color: '#ef4444' },
                            { label: 'EMA 20', value: indicators.movingAverages.ema20, color: '#10b981' },
                            { label: 'VWAP', value: indicators.vwap, color: '#3b82f6' },
                        ].map((ma) => (
                            <Tooltip key={ma.label} title={`${ma.label}: $${ma.value.toFixed(2)}`}>
                                <Chip
                                    label={
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: ma.color }} />
                                            <span>{ma.label}</span>
                                            {currentPrice > ma.value ? (
                                                <TrendingUpIcon sx={{ fontSize: 12, color: '#10b981' }} />
                                            ) : (
                                                <TrendingDownIcon sx={{ fontSize: 12, color: '#ef4444' }} />
                                            )}
                                        </Box>
                                    }
                                    size="small"
                                    sx={{
                                        bgcolor: `${ma.color}15`,
                                        color: ma.color,
                                        border: `1px solid ${ma.color}30`,
                                    }}
                                />
                            </Tooltip>
                        ))}
                    </Box>
                </Grid>

                {/* Support/Resistance */}
                <Grid item xs={12} md={6}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                        <ShowChartIcon sx={{ color: '#f59e0b', fontSize: 18 }} />
                        <Typography variant="subtitle2" fontWeight={600}>Support & Resistance</Typography>
                    </Box>
                    <Grid container spacing={1}>
                        <Grid item xs={6}>
                            <Typography variant="caption" color="#ef4444" fontWeight={600}>Resistance</Typography>
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 0.5 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption" color="text.secondary">R2</Typography>
                                    <Typography variant="caption" sx={{ color: '#ef4444' }}>
                                        ${indicators.supportResistance.resistance2.toFixed(2)}
                                    </Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption" color="text.secondary">R1</Typography>
                                    <Typography variant="caption" sx={{ color: '#f87171' }}>
                                        ${indicators.supportResistance.resistance1.toFixed(2)}
                                    </Typography>
                                </Box>
                            </Box>
                        </Grid>
                        <Grid item xs={6}>
                            <Typography variant="caption" color="#10b981" fontWeight={600}>Support</Typography>
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 0.5 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption" color="text.secondary">S1</Typography>
                                    <Typography variant="caption" sx={{ color: '#34d399' }}>
                                        ${indicators.supportResistance.support1.toFixed(2)}
                                    </Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption" color="text.secondary">S2</Typography>
                                    <Typography variant="caption" sx={{ color: '#10b981' }}>
                                        ${indicators.supportResistance.support2.toFixed(2)}
                                    </Typography>
                                </Box>
                            </Box>
                        </Grid>
                    </Grid>
                </Grid>
            </Grid>

            {/* ATR */}
            <Box sx={{ mt: 3, p: 2, bgcolor: 'rgba(245, 158, 11, 0.1)', borderRadius: 2, border: '1px solid rgba(245, 158, 11, 0.2)' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box>
                        <Typography variant="subtitle2" fontWeight={600} sx={{ color: '#fbbf24' }}>
                            ATR (14) - Volatility
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            Average True Range indicates expected price movement
                        </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'right' }}>
                        <Typography variant="h6" sx={{ color: '#fbbf24', fontWeight: 700 }}>
                            ${indicators.atr.toFixed(2)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            ±{((indicators.atr / currentPrice) * 100).toFixed(2)}%
                        </Typography>
                    </Box>
                </Box>
            </Box>
        </Box>
    )
}

export default TechnicalIndicatorsPanel
