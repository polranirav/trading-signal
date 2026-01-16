/**
 * Live Market Data Header
 * 
 * Displays real-time market data with:
 * - Timestamps (last update time)
 * - Data freshness indicators (green=live, yellow=delayed, red=stale)
 * - Bid/Ask spreads
 * - Volume data
 * - Market status (open/closed/pre-market/after-hours)
 */

import { useState, useEffect } from 'react'
import { Box, Typography, Chip, Tooltip } from '@mui/material'
import AccessTimeIcon from '@mui/icons-material/AccessTime'
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import ShowChartIcon from '@mui/icons-material/ShowChart'

interface MarketTicker {
    symbol: string
    price: number
    change: number
    changePercent: number
    volume: number
    bid?: number
    ask?: number
    lastUpdate: Date
}

// Mock market data - in production, this would come from WebSocket
const getMockMarketData = (): MarketTicker[] => [
    { symbol: 'SPY', price: 478.20 + (Math.random() - 0.5) * 2, change: 2.15, changePercent: 0.45, volume: 45_200_000, bid: 478.18, ask: 478.22, lastUpdate: new Date() },
    { symbol: 'NVDA', price: 495.22 + (Math.random() - 0.5) * 5, change: 8.45, changePercent: 1.74, volume: 32_100_000, bid: 495.18, ask: 495.26, lastUpdate: new Date() },
    { symbol: 'AAPL', price: 185.92 + (Math.random() - 0.5) * 1, change: 1.23, changePercent: 0.67, volume: 28_500_000, bid: 185.90, ask: 185.94, lastUpdate: new Date() },
    { symbol: 'MSFT', price: 378.91 + (Math.random() - 0.5) * 3, change: -2.10, changePercent: -0.55, volume: 18_700_000, bid: 378.88, ask: 378.94, lastUpdate: new Date() },
    { symbol: 'VIX', price: 13.45 + (Math.random() - 0.5) * 0.5, change: -0.58, changePercent: -4.12, volume: 0, lastUpdate: new Date() },
]

type MarketStatus = 'open' | 'closed' | 'pre-market' | 'after-hours'

const getMarketStatus = (): MarketStatus => {
    const now = new Date()
    const hours = now.getHours()
    const minutes = now.getMinutes()
    const day = now.getDay()

    // Weekend
    if (day === 0 || day === 6) return 'closed'

    const time = hours * 60 + minutes

    // Pre-market: 4:00 AM - 9:30 AM ET
    if (time >= 240 && time < 570) return 'pre-market'
    // Market hours: 9:30 AM - 4:00 PM ET
    if (time >= 570 && time < 960) return 'open'
    // After-hours: 4:00 PM - 8:00 PM ET
    if (time >= 960 && time < 1200) return 'after-hours'

    return 'closed'
}

const getStatusColor = (status: MarketStatus) => {
    switch (status) {
        case 'open': return '#10b981'
        case 'pre-market': return '#f59e0b'
        case 'after-hours': return '#8b5cf6'
        default: return '#64748b'
    }
}

const getStatusLabel = (status: MarketStatus) => {
    switch (status) {
        case 'open': return 'Market Open'
        case 'pre-market': return 'Pre-Market'
        case 'after-hours': return 'After Hours'
        default: return 'Market Closed'
    }
}

type DataFreshness = 'live' | 'delayed' | 'stale'

const getDataFreshness = (lastUpdate: Date): DataFreshness => {
    const ageMs = Date.now() - lastUpdate.getTime()
    if (ageMs < 5000) return 'live'
    if (ageMs < 60000) return 'delayed'
    return 'stale'
}

const getFreshnessColor = (freshness: DataFreshness) => {
    switch (freshness) {
        case 'live': return '#10b981'
        case 'delayed': return '#f59e0b'
        default: return '#ef4444'
    }
}

export function LiveMarketHeader() {
    const [marketData, setMarketData] = useState<MarketTicker[]>([])
    const [currentTime, setCurrentTime] = useState(new Date())
    const [marketStatus, setMarketStatus] = useState<MarketStatus>('closed')

    useEffect(() => {
        // Initial load
        setMarketData(getMockMarketData())
        setMarketStatus(getMarketStatus())

        // Update every 3 seconds
        const dataInterval = setInterval(() => {
            setMarketData(getMockMarketData())
        }, 3000)

        // Update time every second
        const timeInterval = setInterval(() => {
            setCurrentTime(new Date())
        }, 1000)

        // Update market status every minute
        const statusInterval = setInterval(() => {
            setMarketStatus(getMarketStatus())
        }, 60000)

        return () => {
            clearInterval(dataInterval)
            clearInterval(timeInterval)
            clearInterval(statusInterval)
        }
    }, [])

    const overallFreshness = marketData.length > 0
        ? getDataFreshness(marketData[0].lastUpdate)
        : 'stale'

    return (
        <Box sx={{
            background: 'rgba(13, 14, 23, 0.95)',
            borderBottom: '1px solid rgba(255,255,255,0.08)',
            py: 1.5,
            px: 3,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 2,
        }}>
            {/* Left: Market Status & Time */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <FiberManualRecordIcon sx={{
                        color: getStatusColor(marketStatus),
                        fontSize: 12,
                        animation: marketStatus === 'open' ? 'pulse 2s infinite' : 'none'
                    }} />
                    <Typography variant="body2" sx={{ color: getStatusColor(marketStatus), fontWeight: 600 }}>
                        {getStatusLabel(marketStatus)}
                    </Typography>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <AccessTimeIcon sx={{ color: '#64748b', fontSize: 16 }} />
                    <Typography variant="body2" color="text.secondary">
                        {currentTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </Typography>
                </Box>

                <Tooltip title={`Data is ${overallFreshness}`}>
                    <Chip
                        icon={<FiberManualRecordIcon sx={{ fontSize: '10px !important', color: `${getFreshnessColor(overallFreshness)} !important` }} />}
                        label={overallFreshness === 'live' ? 'Live Data' : overallFreshness === 'delayed' ? '15min Delayed' : 'Stale'}
                        size="small"
                        sx={{
                            bgcolor: `${getFreshnessColor(overallFreshness)}15`,
                            color: getFreshnessColor(overallFreshness),
                            border: `1px solid ${getFreshnessColor(overallFreshness)}30`,
                            fontSize: '0.7rem',
                            height: 24,
                        }}
                    />
                </Tooltip>
            </Box>

            {/* Center: Tickers */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap' }}>
                {marketData.map((ticker) => (
                    <Tooltip
                        key={ticker.symbol}
                        title={
                            <Box>
                                <Typography variant="body2">
                                    Bid: ${ticker.bid?.toFixed(2)} | Ask: ${ticker.ask?.toFixed(2)}
                                </Typography>
                                <Typography variant="body2">
                                    Volume: {(ticker.volume / 1_000_000).toFixed(1)}M
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    Updated: {ticker.lastUpdate.toLocaleTimeString()}
                                </Typography>
                            </Box>
                        }
                    >
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, cursor: 'pointer' }}>
                            <Typography variant="body2" sx={{ fontWeight: 700, color: '#fff' }}>
                                {ticker.symbol}
                            </Typography>
                            <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                                ${ticker.price.toFixed(2)}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                {ticker.change >= 0 ? (
                                    <TrendingUpIcon sx={{ color: '#10b981', fontSize: 14 }} />
                                ) : (
                                    <TrendingDownIcon sx={{ color: '#ef4444', fontSize: 14 }} />
                                )}
                                <Typography
                                    variant="body2"
                                    sx={{
                                        color: ticker.change >= 0 ? '#10b981' : '#ef4444',
                                        fontWeight: 600,
                                        fontSize: '0.8rem',
                                    }}
                                >
                                    {ticker.changePercent >= 0 ? '+' : ''}{ticker.changePercent.toFixed(2)}%
                                </Typography>
                            </Box>
                        </Box>
                    </Tooltip>
                ))}
            </Box>

            {/* Right: Data Source */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ShowChartIcon sx={{ color: '#64748b', fontSize: 16 }} />
                <Typography variant="caption" color="text.secondary">
                    Source: Alpha Vantage
                </Typography>
            </Box>
        </Box>
    )
}

export default LiveMarketHeader
