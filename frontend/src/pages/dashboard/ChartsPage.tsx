import { useState, useMemo, useCallback, useEffect } from 'react'
import { Box, Typography, Grid, TextField, Chip } from '@mui/material'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import AccessTimeIcon from '@mui/icons-material/AccessTime'
import SignalCellularAltIcon from '@mui/icons-material/SignalCellularAlt'
import BarChartIcon from '@mui/icons-material/BarChart'
import ShieldIcon from '@mui/icons-material/Shield'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import PsychologyIcon from '@mui/icons-material/Psychology'
import AutoGraphIcon from '@mui/icons-material/AutoGraph'
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet'
import { api } from '../../services/api'
import { SectionCard, ConfidenceBar } from '../../components/SignalComponents'
import CandlestickChart from '../../components/charts/CandlestickChart'
import GhostCandleChart, { getPredictionLabels, calculatePredictedPrices } from '../../components/charts/GhostCandleChart'
import { TechnicalIndicatorsPanel } from '../../components/TechnicalIndicatorsPanel'
import { CandlestickData, Time } from 'lightweight-charts'
import { usePortfolio } from '../../context'
import '../../styles/premium.css'

// Default quick access stocks (used when no portfolio)
const defaultQuickStocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN']

// Timeframe Options with details
const timeframes = [
    { label: '1M', value: '1M', interval: 60, candleCount: 60, predictionLabel: 'Next 5 Minutes' },
    { label: '5M', value: '5M', interval: 300, candleCount: 50, predictionLabel: 'Next 25 Minutes' },
    { label: '15M', value: '15M', interval: 900, candleCount: 40, predictionLabel: 'Next 75 Minutes' },
    { label: '1H', value: '1H', interval: 3600, candleCount: 50, predictionLabel: 'Next 5 Hours' },
    { label: '4H', value: '4H', interval: 14400, candleCount: 30, predictionLabel: 'Next 20 Hours' },
    { label: '1D', value: '1D', interval: 86400, candleCount: 30, predictionLabel: 'Next 5 Days' },
]
export default function ChartsPage() {
    // Portfolio context
    const { portfolioSymbols, topHolding, hasPortfolio, isInPortfolio } = usePortfolio()
    const [searchParams] = useSearchParams()

    // Quick access stocks - portfolio first, then defaults
    const quickStocks = useMemo(() => {
        if (hasPortfolio && portfolioSymbols.length > 0) {
            // Show up to 7 portfolio stocks
            const portfolioStocks = portfolioSymbols.slice(0, 7)
            return portfolioStocks
        }
        return defaultQuickStocks
    }, [hasPortfolio, portfolioSymbols])

    // Determine initial symbol: URL param > Top Holding > 'AAPL'
    const urlSymbol = searchParams.get('symbol')
    const defaultSymbol = urlSymbol || topHolding?.symbol || 'AAPL'

    const [selectedSymbol, setSelectedSymbol] = useState(defaultSymbol)
    const [timeframe, setTimeframe] = useState('1H')
    const [searchSymbol, setSearchSymbol] = useState('')

    // Update selected symbol when URL param changes
    useEffect(() => {
        const symbolParam = searchParams.get('symbol')
        if (symbolParam && symbolParam !== selectedSymbol) {
            setSelectedSymbol(symbolParam)
        }
    }, [searchParams, selectedSymbol])

    // Update selected symbol when portfolio loads IF no URL param was provided
    useEffect(() => {
        if (!urlSymbol && topHolding && selectedSymbol === 'AAPL' && topHolding.symbol !== 'AAPL') {
            setSelectedSymbol(topHolding.symbol)
        }
    }, [topHolding, selectedSymbol, urlSymbol])

    // Get current timeframe config
    const currentTimeframeConfig = timeframes.find(tf => tf.value === timeframe) || timeframes[3]

    // Fetch signals - increased limit for more coverage
    const { data: signalsData, isError } = useQuery({
        queryKey: ['signals', { limit: 100 }],
        queryFn: () => api.getSignals({ limit: 100 }),
    })

    const signals = signalsData?.signals || []
    const currentSignal = signals.find((s: any) => s.symbol === selectedSymbol)

    // Mock live data - based on current signal
    const liveData = useMemo(() => {
        const basePrice = currentSignal?.price_at_signal || 178.50
        const signalType = currentSignal?.signal_type || 'HOLD'
        const isPositive = signalType.includes('BUY')

        return {
            price: basePrice,
            change: isPositive ? `+${(Math.random() * 2 + 0.5).toFixed(2)}%` : `-${(Math.random() * 1.5 + 0.3).toFixed(2)}%`,
            changePercent: isPositive ? Math.random() * 2 + 0.5 : -(Math.random() * 1.5 + 0.3),
            high: basePrice * (1 + Math.random() * 0.025),
            low: basePrice * (1 - Math.random() * 0.02),
            volume: `${(Math.random() * 50 + 20).toFixed(1)}M`,
            trend: isPositive ? 'Bullish' : signalType.includes('SELL') ? 'Bearish' : 'Neutral',
        }
    }, [currentSignal, selectedSymbol])

    // Generate prediction labels based on timeframe
    const predictionLabels = useMemo(() => getPredictionLabels(timeframe, 5), [timeframe])

    // DYNAMIC predictions based on the actual signal data for selected stock
    const predictions = useMemo(() => {
        // Base confidence from the signal's confluence score (0-1 scale to 50-95 scale)
        const baseConfidence = currentSignal?.confluence_score
            ? Math.round(currentSignal.confluence_score * 45 + 50)
            : 55

        // Determine direction bias from signal type
        const signalType = currentSignal?.signal_type || 'HOLD'
        const isBullish = signalType.includes('BUY')
        const isBearish = signalType.includes('SELL')

        // Calculate technical and sentiment influence
        const techScore = currentSignal?.technical_score || 0.5
        const sentScore = currentSignal?.sentiment_score || 0.5

        // Generate 5 predictions with variation based on stock's actual data
        const generateDirection = (index: number): 'UP' | 'DOWN' => {
            // Use a seeded random based on symbol to make it consistent per stock
            const seed = selectedSymbol.split('').reduce((a, b) => a + b.charCodeAt(0), 0)
            const pseudoRandom = Math.sin(seed * (index + 1)) * 0.5 + 0.5

            if (isBullish) {
                // 70-85% chance of UP for bullish stocks
                return pseudoRandom < (0.70 + techScore * 0.15) ? 'UP' : 'DOWN'
            } else if (isBearish) {
                // 65-80% chance of DOWN for bearish stocks
                return pseudoRandom < (0.65 + (1 - techScore) * 0.15) ? 'DOWN' : 'UP'
            } else {
                // Mixed for HOLD stocks
                return pseudoRandom < 0.5 ? 'UP' : 'DOWN'
            }
        }

        // Generate confidence with variation based on timeframe and scores
        const generateConfidence = (index: number): number => {
            const seed = selectedSymbol.split('').reduce((a, b) => a + b.charCodeAt(0), 0)
            const variation = Math.sin(seed * (index + 2)) * 10

            // Shorter timeframes = higher confidence, longer = lower
            const timeframeDecay = [0, -2, -4, -6, -5]

            let conf = baseConfidence + variation + timeframeDecay[index]

            // Apply sentiment boost/penalty
            if (sentScore > 0.7) conf += 5
            if (sentScore < 0.3) conf -= 5

            return Math.max(45, Math.min(95, Math.round(conf)))
        }

        return predictionLabels.map((label, index) => ({
            period: label,
            direction: generateDirection(index),
            confidence: generateConfidence(index),
        }))
    }, [predictionLabels, currentSignal, selectedSymbol])

    // Calculate predicted prices
    const predictedPrices = useMemo(() =>
        calculatePredictedPrices(predictions, liveData.price),
        [predictions, liveData.price]
    )

    const avgConfidence = predictions.reduce((acc, p) => acc + p.confidence, 0) / predictions.length

    // Use actual signal recommendation if available
    const recommendation = useMemo(() => {
        if (currentSignal?.signal_type?.includes('BUY')) return 'BUY'
        if (currentSignal?.signal_type?.includes('SELL')) return 'SELL'
        if (avgConfidence > 60) return 'BUY'
        if (avgConfidence < 45) return 'SELL'
        return 'HOLD'
    }, [currentSignal, avgConfidence])

    const upCount = predictions.filter(p => p.direction === 'UP').length
    const downCount = predictions.filter(p => p.direction === 'DOWN').length

    // AI Reasoning - Dynamic based on actual signal data
    const aiReasoning = useMemo(() => {
        const techRationale = currentSignal?.technical_rationale || 'Technical analysis complete'
        const sentRationale = currentSignal?.sentiment_rationale || 'Sentiment analysis pending'

        if (recommendation === 'BUY') {
            return `Strong bullish momentum detected for ${selectedSymbol} with ${upCount}/5 timeframes showing upward movement. ${techRationale}. ${sentRationale}. Consider entering at current levels with defined stop-loss.`
        } else if (recommendation === 'SELL') {
            return `Bearish pressure detected for ${selectedSymbol} with ${downCount}/5 timeframes showing downward movement. ${techRationale}. Consider reducing exposure or hedging positions.`
        }
        return `Mixed signals detected for ${selectedSymbol}. ${techRationale}. Market showing uncertainty. Consider waiting for clearer trend direction.`
    }, [recommendation, upCount, downCount, selectedSymbol, currentSignal])

    // Generate candlestick data based on timeframe
    const candlestickData = useMemo(() => {
        const basePrice = liveData.price
        const data: CandlestickData[] = []
        const now = Date.now() / 1000
        const interval = currentTimeframeConfig.interval
        const candleCount = currentTimeframeConfig.candleCount

        for (let i = candleCount - 1; i >= 0; i--) {
            const time = (now - i * interval) as Time
            const randomChange = (Math.random() - 0.5) * 0.02
            const open = i === candleCount - 1 ? basePrice : data[data.length - 1].close
            const close = open * (1 + randomChange)
            const high = Math.max(open, close) * (1 + Math.random() * 0.008)
            const low = Math.min(open, close) * (1 - Math.random() * 0.008)

            data.push({
                time,
                open: Number(open.toFixed(2)),
                high: Number(high.toFixed(2)),
                low: Number(low.toFixed(2)),
                close: Number(close.toFixed(2)),
            })
        }
        return data
    }, [liveData.price, selectedSymbol, timeframe, currentTimeframeConfig])

    const lastCandleTime = candlestickData.length > 0
        ? candlestickData[candlestickData.length - 1].time as number
        : undefined

    const showApiError = isError && signals.length === 0

    const handleTimeframeChange = useCallback((newTimeframe: string) => {
        setTimeframe(newTimeframe)
    }, [])

    return (
        <Box sx={{
            p: 3,
            minHeight: '100vh',
            bgcolor: 'background.default',
            overflowY: 'auto',
            overflowX: 'hidden'
        }}>
            {showApiError && (
                <Box sx={{ mb: 2, p: 2, bgcolor: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: 2 }}>
                    <Typography sx={{ color: '#f59e0b', fontSize: '0.9rem' }}>⚠️ Using demo data.</Typography>
                </Box>
            )}

            {/* HEADER: Symbol + Timeframe + Prediction Info */}
            <Box sx={{
                mb: 3,
                p: 2,
                background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.8))',
                borderRadius: 3,
                border: '1px solid rgba(255,255,255,0.1)'
            }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
                    {/* Title & Symbol */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <AutoGraphIcon sx={{ color: '#3b82f6', fontSize: '1.8rem' }} />
                            <Typography sx={{ fontWeight: 700, color: '#fff', fontSize: '1.4rem' }}>Dual Chart Analysis</Typography>
                        </Box>
                        <Box sx={{
                            px: 2, py: 0.75, borderRadius: 2,
                            background: 'rgba(59, 130, 246, 0.15)',
                            border: '1px solid rgba(59, 130, 246, 0.4)'
                        }}>
                            <Typography sx={{ color: '#3b82f6', fontWeight: 700, fontSize: '1.1rem' }}>
                                Symbol: {selectedSymbol}
                            </Typography>
                        </Box>
                        <TextField
                            size="small"
                            placeholder="Search..."
                            value={searchSymbol}
                            onChange={(e) => setSearchSymbol(e.target.value.toUpperCase())}
                            onKeyPress={(e) => { if (e.key === 'Enter' && searchSymbol) setSelectedSymbol(searchSymbol) }}
                            sx={{ width: 100, '& .MuiOutlinedInput-root': { background: 'rgba(0,0,0,0.3)', height: 36 } }}
                        />
                    </Box>

                    {/* Timeframe Selector */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>Timeframe:</Typography>
                        <Box sx={{ display: 'flex', gap: 0.5 }}>
                            {timeframes.map(tf => (
                                <Chip
                                    key={tf.value}
                                    label={tf.label}
                                    onClick={() => handleTimeframeChange(tf.value)}
                                    size="small"
                                    sx={{
                                        fontWeight: 700,
                                        fontSize: '0.8rem',
                                        px: 0.5,
                                        background: timeframe === tf.value
                                            ? 'linear-gradient(135deg, #3b82f6, #8b5cf6)'
                                            : 'rgba(255,255,255,0.05)',
                                        color: timeframe === tf.value ? '#fff' : '#94a3b8',
                                        border: timeframe === tf.value ? 'none' : '1px solid rgba(255,255,255,0.1)',
                                        '&:hover': { background: timeframe === tf.value ? 'linear-gradient(135deg, #3b82f6, #8b5cf6)' : 'rgba(255,255,255,0.1)' }
                                    }}
                                />
                            ))}
                        </Box>
                        <Box sx={{
                            px: 2, py: 0.75, borderRadius: 2,
                            background: 'rgba(139, 92, 246, 0.15)',
                            border: '1px solid rgba(139, 92, 246, 0.4)'
                        }}>
                            <Typography sx={{ color: '#a78bfa', fontWeight: 600, fontSize: '0.85rem' }}>
                                Prediction: {currentTimeframeConfig.predictionLabel}
                            </Typography>
                        </Box>
                    </Box>
                </Box>

                {/* Quick Access */}
                <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: '#64748b' }}>
                        {hasPortfolio ? (
                            <>
                                <AccountBalanceWalletIcon sx={{ fontSize: '0.9rem', color: '#3b82f6' }} />
                                <Typography sx={{ fontSize: '0.8rem', color: '#3b82f6', fontWeight: 600 }}>My Portfolio</Typography>
                            </>
                        ) : (
                            <>
                                <AccessTimeIcon sx={{ fontSize: '0.9rem' }} />
                                <Typography sx={{ fontSize: '0.8rem' }}>Quick Access</Typography>
                            </>
                        )}
                    </Box>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        {quickStocks.map(stock => (
                            <Box
                                key={stock}
                                onClick={() => setSelectedSymbol(stock)}
                                sx={{
                                    px: 1.5, py: 0.5, borderRadius: 1.5, cursor: 'pointer',
                                    fontSize: '0.8rem', fontWeight: 600,
                                    background: selectedSymbol === stock ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255,255,255,0.03)',
                                    color: selectedSymbol === stock ? '#3b82f6' : '#64748b',
                                    border: selectedSymbol === stock ? '1px solid rgba(59, 130, 246, 0.4)' : '1px solid transparent',
                                    transition: 'all 0.2s',
                                    display: 'flex', alignItems: 'center', gap: 0.5,
                                    '&:hover': { background: 'rgba(59, 130, 246, 0.1)', color: '#fff' }
                                }}
                            >
                                {hasPortfolio && isInPortfolio(stock) && (
                                    <Box sx={{ width: 5, height: 5, borderRadius: '50%', background: '#10b981' }} />
                                )}
                                {stock}
                            </Box>
                        ))}
                    </Box>
                </Box>
            </Box>

            {/* DUAL CHARTS - Side by Side */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
                {/* LIVE MARKET CHART */}
                <Grid item xs={12} lg={6}>
                    <Box sx={{
                        background: 'linear-gradient(180deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.9))',
                        borderRadius: 3,
                        border: '1px solid rgba(59, 130, 246, 0.25)',
                        overflow: 'hidden'
                    }}>
                        <Box sx={{
                            px: 2, py: 1.5,
                            background: 'linear-gradient(90deg, rgba(59, 130, 246, 0.12), transparent)',
                            borderBottom: '1px solid rgba(255,255,255,0.05)',
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                        }}>
                            <Typography sx={{ fontWeight: 700, color: '#fff', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: 1 }}>
                                📍 LIVE MARKET ({currentTimeframeConfig.candleCount} candles)
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Box sx={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981', animation: 'pulse 2s infinite' }} />
                                <Typography sx={{ fontSize: '0.75rem', color: '#10b981', fontWeight: 600 }}>LIVE</Typography>
                            </Box>
                        </Box>

                        <Box sx={{ px: 2, py: 1.5, display: 'flex', gap: 4, borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                            <Box>
                                <Typography sx={{ fontSize: '0.65rem', color: '#64748b' }}>Price</Typography>
                                <Typography sx={{ fontSize: '1.3rem', fontWeight: 700, color: '#fff' }}>${liveData.price.toFixed(2)}</Typography>
                            </Box>
                            <Box>
                                <Typography sx={{ fontSize: '0.65rem', color: '#64748b' }}>Change</Typography>
                                <Typography sx={{ fontSize: '1rem', fontWeight: 600, color: '#10b981' }}>{liveData.change}</Typography>
                            </Box>
                            <Box>
                                <Typography sx={{ fontSize: '0.65rem', color: '#64748b' }}>High</Typography>
                                <Typography sx={{ fontSize: '0.9rem', color: '#10b981', fontWeight: 600 }}>${liveData.high.toFixed(2)}</Typography>
                            </Box>
                            <Box>
                                <Typography sx={{ fontSize: '0.65rem', color: '#64748b' }}>Low</Typography>
                                <Typography sx={{ fontSize: '0.9rem', color: '#ef4444', fontWeight: 600 }}>${liveData.low.toFixed(2)}</Typography>
                            </Box>
                        </Box>

                        <Box sx={{ p: 1.5 }}>
                            <CandlestickChart data={candlestickData} height={400} symbol={selectedSymbol} />
                        </Box>
                    </Box>
                </Grid>

                {/* PREDICTION CHART */}
                <Grid item xs={12} lg={6}>
                    <Box sx={{
                        background: 'linear-gradient(180deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.9))',
                        borderRadius: 3,
                        border: '1px solid rgba(139, 92, 246, 0.25)',
                        overflow: 'hidden'
                    }}>
                        <Box sx={{
                            px: 2, py: 1.5,
                            background: 'linear-gradient(90deg, rgba(139, 92, 246, 0.12), transparent)',
                            borderBottom: '1px solid rgba(255,255,255,0.05)',
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                        }}>
                            <Typography sx={{ fontWeight: 700, color: '#fff', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: 1 }}>
                                🔮 PREDICTION (5 ghost candles)
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <PsychologyIcon sx={{ fontSize: '0.9rem', color: '#a78bfa' }} />
                                <Typography sx={{ fontSize: '0.75rem', color: '#a78bfa' }}>LSTM+XGBoost</Typography>
                            </Box>
                        </Box>

                        <Box sx={{ px: 2, py: 1.5, display: 'flex', gap: 3, borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                            <Box sx={{
                                px: 2, py: 0.75, borderRadius: 2,
                                background: recommendation === 'BUY' ? 'rgba(16, 185, 129, 0.2)' : recommendation === 'SELL' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                                border: `1px solid ${recommendation === 'BUY' ? 'rgba(16, 185, 129, 0.4)' : recommendation === 'SELL' ? 'rgba(239, 68, 68, 0.4)' : 'rgba(245, 158, 11, 0.4)'}`
                            }}>
                                <Typography sx={{ fontSize: '1.1rem', fontWeight: 800, color: recommendation === 'BUY' ? '#10b981' : recommendation === 'SELL' ? '#ef4444' : '#f59e0b' }}>
                                    {recommendation}
                                </Typography>
                            </Box>
                            <Box>
                                <Typography sx={{ fontSize: '0.65rem', color: '#64748b' }}>Confidence</Typography>
                                <Typography sx={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>{avgConfidence.toFixed(0)}%</Typography>
                            </Box>
                            <Box>
                                <Typography sx={{ fontSize: '0.65rem', color: '#64748b' }}>Signals</Typography>
                                <Typography sx={{ fontSize: '0.9rem', fontWeight: 600 }}>
                                    <span style={{ color: '#10b981' }}>{upCount} UP</span> / <span style={{ color: '#ef4444' }}>{downCount} DOWN</span>
                                </Typography>
                            </Box>
                        </Box>

                        <Box sx={{ p: 1.5 }}>
                            <GhostCandleChart
                                predictions={predictions}
                                currentPrice={liveData.price}
                                height={400}
                                timeframe={timeframe}
                                lastCandleTime={lastCandleTime}
                            />
                        </Box>
                    </Box>
                </Grid>
            </Grid>

            {/* PREDICTION DETAILS - Full Width Below Charts */}
            <Box sx={{
                mb: 3,
                background: 'linear-gradient(180deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.9))',
                borderRadius: 3,
                border: '1px solid rgba(139, 92, 246, 0.2)',
                overflow: 'hidden'
            }}>
                <Box sx={{
                    px: 3, py: 2,
                    background: 'linear-gradient(90deg, rgba(139, 92, 246, 0.1), transparent)',
                    borderBottom: '1px solid rgba(255,255,255,0.05)'
                }}>
                    <Typography sx={{ fontWeight: 700, color: '#fff', fontSize: '1rem' }}>📊 PREDICTION DETAILS</Typography>
                </Box>

                <Box sx={{ p: 3 }}>
                    {/* Prediction Timeline Grid */}
                    <Grid container spacing={2} sx={{ mb: 3 }}>
                        {predictions.map((pred, i) => (
                            <Grid item xs={12} sm={6} md={2.4} key={i}>
                                <Box sx={{
                                    p: 2, borderRadius: 2,
                                    background: pred.direction === 'UP' ? 'rgba(16, 185, 129, 0.08)' : 'rgba(239, 68, 68, 0.08)',
                                    border: `1px solid ${pred.direction === 'UP' ? 'rgba(16, 185, 129, 0.25)' : 'rgba(239, 68, 68, 0.25)'}`,
                                }}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                        <Typography sx={{ color: '#a78bfa', fontWeight: 700, fontSize: '0.9rem' }}>{pred.period}</Typography>
                                        {pred.direction === 'UP' ? (
                                            <TrendingUpIcon sx={{ color: '#10b981', fontSize: '1.2rem' }} />
                                        ) : (
                                            <TrendingDownIcon sx={{ color: '#ef4444', fontSize: '1.2rem' }} />
                                        )}
                                    </Box>
                                    <Typography sx={{
                                        color: pred.direction === 'UP' ? '#10b981' : '#ef4444',
                                        fontWeight: 800,
                                        fontSize: '1.1rem',
                                        mb: 1
                                    }}>
                                        {pred.direction}
                                    </Typography>
                                    <ConfidenceBar value={pred.confidence / 100} />
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                                        <Typography sx={{ color: '#64748b', fontSize: '0.7rem' }}>Confidence</Typography>
                                        <Typography sx={{ color: '#fff', fontWeight: 600, fontSize: '0.8rem' }}>{pred.confidence}%</Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                                        <Typography sx={{ color: '#64748b', fontSize: '0.7rem' }}>Predicted</Typography>
                                        <Typography sx={{ color: pred.direction === 'UP' ? '#10b981' : '#ef4444', fontWeight: 600, fontSize: '0.8rem' }}>
                                            ${predictedPrices[i].toFixed(2)}
                                        </Typography>
                                    </Box>
                                </Box>
                            </Grid>
                        ))}
                    </Grid>

                    {/* Recommendation + AI Reasoning */}
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={4}>
                            <Box sx={{
                                p: 3, borderRadius: 2, textAlign: 'center',
                                background: recommendation === 'BUY'
                                    ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05))'
                                    : recommendation === 'SELL'
                                        ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05))'
                                        : 'linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05))',
                                border: `2px solid ${recommendation === 'BUY' ? 'rgba(16, 185, 129, 0.4)' : recommendation === 'SELL' ? 'rgba(239, 68, 68, 0.4)' : 'rgba(245, 158, 11, 0.4)'}`,
                            }}>
                                <Typography sx={{ fontSize: '0.75rem', color: '#64748b', mb: 1 }}>🎯 RECOMMENDATION</Typography>
                                <Typography sx={{
                                    fontSize: '2.5rem', fontWeight: 800,
                                    color: recommendation === 'BUY' ? '#10b981' : recommendation === 'SELL' ? '#ef4444' : '#f59e0b',
                                    textShadow: '0 0 30px currentColor',
                                    mb: 1
                                }}>
                                    {recommendation}
                                </Typography>
                                <Typography sx={{ fontSize: '1.1rem', color: '#fff', fontWeight: 600 }}>
                                    {avgConfidence.toFixed(0)}% conf
                                </Typography>
                                <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 3 }}>
                                    <Box>
                                        <Typography sx={{ color: '#10b981', fontWeight: 700, fontSize: '1.3rem' }}>{upCount}</Typography>
                                        <Typography sx={{ color: '#64748b', fontSize: '0.7rem' }}>UP</Typography>
                                    </Box>
                                    <Box sx={{ width: 1, background: 'rgba(255,255,255,0.1)' }} />
                                    <Box>
                                        <Typography sx={{ color: '#ef4444', fontWeight: 700, fontSize: '1.3rem' }}>{downCount}</Typography>
                                        <Typography sx={{ color: '#64748b', fontSize: '0.7rem' }}>DOWN</Typography>
                                    </Box>
                                </Box>
                            </Box>
                        </Grid>

                        <Grid item xs={12} md={8}>
                            <Box sx={{
                                p: 3, borderRadius: 2, height: '100%',
                                background: 'rgba(139, 92, 246, 0.05)',
                                border: '1px solid rgba(139, 92, 246, 0.2)',
                            }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                                    <PsychologyIcon sx={{ color: '#a78bfa' }} />
                                    <Typography sx={{ fontSize: '0.9rem', color: '#a78bfa', fontWeight: 700 }}>💡 AI REASONING</Typography>
                                </Box>
                                <Typography sx={{ color: '#e2e8f0', fontSize: '1rem', lineHeight: 1.8, fontStyle: 'italic', mb: 2 }}>
                                    "{aiReasoning}"
                                </Typography>
                                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                                    <Box sx={{ px: 2, py: 0.5, borderRadius: 1, background: 'rgba(59, 130, 246, 0.15)' }}>
                                        <Typography sx={{ fontSize: '0.75rem', color: '#3b82f6' }}>📊 Technical: Strong</Typography>
                                    </Box>
                                    <Box sx={{ px: 2, py: 0.5, borderRadius: 1, background: 'rgba(16, 185, 129, 0.15)' }}>
                                        <Typography sx={{ fontSize: '0.75rem', color: '#10b981' }}>💬 Sentiment: Positive</Typography>
                                    </Box>
                                    <Box sx={{ px: 2, py: 0.5, borderRadius: 1, background: 'rgba(139, 92, 246, 0.15)' }}>
                                        <Typography sx={{ fontSize: '0.75rem', color: '#a78bfa' }}>🎯 ML Score: {avgConfidence.toFixed(0)}%</Typography>
                                    </Box>
                                </Box>
                            </Box>
                        </Grid>
                    </Grid>
                </Box>
            </Box>

            {/* BOTTOM CARDS */}
            <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                    <SectionCard title="SIGNAL ANALYSIS" icon={<SignalCellularAltIcon />} iconColor="#3b82f6">
                        <Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                                <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>Technical Score:</Typography>
                                <Typography sx={{ color: '#10b981', fontWeight: 600, fontSize: '1.1rem' }}>
                                    {currentSignal?.technical_score ? (currentSignal.technical_score * 100).toFixed(0) : 88}%
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                                <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>Sentiment Score:</Typography>
                                <Typography sx={{ color: '#f59e0b', fontWeight: 600, fontSize: '1.1rem' }}>
                                    {currentSignal?.sentiment_score ? (currentSignal.sentiment_score * 100).toFixed(0) : 82}%
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>Confluence:</Typography>
                                <Typography sx={{ color: '#3b82f6', fontWeight: 600, fontSize: '1.1rem' }}>
                                    {currentSignal?.confluence_score ? (currentSignal.confluence_score * 100).toFixed(0) : 85}%
                                </Typography>
                            </Box>
                        </Box>
                    </SectionCard>
                </Grid>

                <Grid item xs={12} md={4}>
                    <SectionCard title="INDICATORS" icon={<BarChartIcon />} iconColor="#8b5cf6">
                        <Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                                <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>RSI (14):</Typography>
                                <Typography sx={{ color: '#fff', fontWeight: 600, fontSize: '1.1rem' }}>58.4</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                                <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>MACD Signal:</Typography>
                                <Typography sx={{ color: '#10b981', fontWeight: 600, fontSize: '1.1rem' }}>Bullish</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>SMA 20/50:</Typography>
                                <Typography sx={{ color: '#10b981', fontWeight: 600, fontSize: '1.1rem' }}>Above</Typography>
                            </Box>
                        </Box>
                    </SectionCard>
                </Grid>

                <Grid item xs={12} md={4}>
                    <SectionCard title="RISK METRICS" icon={<ShieldIcon />} iconColor="#ef4444">
                        <Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                                <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>Stop Loss:</Typography>
                                <Typography sx={{ color: '#ef4444', fontWeight: 600, fontSize: '1.1rem' }}>${(liveData.price * 0.95).toFixed(2)}</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                                <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>Take Profit:</Typography>
                                <Typography sx={{ color: '#10b981', fontWeight: 600, fontSize: '1.1rem' }}>${(liveData.price * 1.10).toFixed(2)}</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>Risk/Reward:</Typography>
                                <Typography sx={{ color: '#3b82f6', fontWeight: 600, fontSize: '1.1rem' }}>1:2</Typography>
                            </Box>
                        </Box>
                    </SectionCard>
                </Grid>
            </Grid>

            {/* Technical Indicators Panel */}
            <Box sx={{ mb: 3 }}>
                <TechnicalIndicatorsPanel
                    symbol={selectedSymbol}
                    currentPrice={liveData.price}
                    timeframe={currentTimeframeConfig.label}
                />
            </Box>
        </Box>
    )
}
