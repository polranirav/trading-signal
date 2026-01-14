import React, { useState } from 'react'
import { Box, Typography, Grid, TextField, Select, MenuItem, FormControl, InputLabel, Button, Slider, ToggleButtonGroup, ToggleButton } from '@mui/material'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import SettingsIcon from '@mui/icons-material/Settings'
import BoltIcon from '@mui/icons-material/Bolt'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import GppGoodIcon from '@mui/icons-material/GppGood'
import WarningIcon from '@mui/icons-material/Warning'
import DownloadIcon from '@mui/icons-material/Download'
import { SectionCard } from '../../components/SignalComponents'
import MetricCard from '../../components/MetricCard'
import '../../styles/premium.css'

export default function BacktestPage() {
    const [mode, setMode] = useState<'simple' | 'advanced'>('simple')
    const [symbol, setSymbol] = useState('AAPL')
    const [dateRange, setDateRange] = useState('1y')
    const [strategy, setStrategy] = useState('rsi')
    const [isRunning, setIsRunning] = useState(false)
    const [hasResults, setHasResults] = useState(false)

    // Advanced params
    const [oversold, setOversold] = useState(30)
    const [overbought, setOverbought] = useState(70)
    const [stopLoss, setStopLoss] = useState(5)
    const [takeProfit, setTakeProfit] = useState(10)

    const runBacktest = () => {
        setIsRunning(true)
        setTimeout(() => {
            setIsRunning(false)
            setHasResults(true)
        }, 2000)
    }

    // Mock results
    const results = {
        totalReturn: '+42.5%',
        annualReturn: '+18.2%',
        sharpe: '1.85',
        maxDrawdown: '-12.3%',
        winRate: '68%',
        profitFactor: '2.1',
        totalTrades: 45,
        overfitting: 'Low',
    }

    return (
        <Box className="fade-in">
            {/* Header */}
            <Box sx={{ mb: 3 }}>
                <Typography variant="h4" sx={{ fontWeight: 700, color: '#fff', fontSize: '1.75rem', mb: 0.5 }}>
                    Strategy Backtesting
                </Typography>
                <Typography sx={{ color: '#64748b', fontSize: '0.9rem' }}>
                    Test trading strategies with walk-forward validation
                </Typography>
            </Box>

            {/* Mode Toggle */}
            <Box sx={{ mb: 3 }}>
                <ToggleButtonGroup
                    value={mode}
                    exclusive
                    onChange={(_, v) => v && setMode(v)}
                    sx={{ background: 'rgba(0,0,0,0.3)', borderRadius: 2 }}
                >
                    <ToggleButton value="simple" sx={{ px: 3, color: mode === 'simple' ? '#3b82f6' : '#64748b' }}>
                        <BoltIcon sx={{ mr: 1 }} /> Simple
                    </ToggleButton>
                    <ToggleButton value="advanced" sx={{ px: 3, color: mode === 'advanced' ? '#8b5cf6' : '#64748b' }}>
                        <SettingsIcon sx={{ mr: 1 }} /> Advanced
                    </ToggleButton>
                </ToggleButtonGroup>
            </Box>

            <Grid container spacing={3}>
                {/* Left Column - Configuration */}
                <Grid item xs={12} lg={4}>
                    <SectionCard
                        title={mode === 'simple' ? 'Quick Backtest' : 'Strategy Builder'}
                        icon={mode === 'simple' ? <BoltIcon /> : <SettingsIcon />}
                        iconColor={mode === 'simple' ? '#3b82f6' : '#8b5cf6'}
                    >
                        {/* Symbol */}
                        <Box sx={{ mb: 2 }}>
                            <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 0.5 }}>Symbol</Typography>
                            <TextField
                                fullWidth
                                size="small"
                                value={symbol}
                                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                                sx={{ '& .MuiOutlinedInput-root': { background: 'rgba(0,0,0,0.2)' } }}
                            />
                        </Box>

                        {/* Date Range */}
                        <Box sx={{ mb: 2 }}>
                            <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 0.5 }}>Date Range</Typography>
                            <Select
                                fullWidth
                                size="small"
                                value={dateRange}
                                onChange={(e) => setDateRange(e.target.value)}
                                sx={{ background: 'rgba(0,0,0,0.2)' }}
                            >
                                <MenuItem value="1y">Last 1 Year</MenuItem>
                                <MenuItem value="2y">Last 2 Years</MenuItem>
                                <MenuItem value="5y">Last 5 Years</MenuItem>
                            </Select>
                        </Box>

                        {/* Strategy */}
                        <Box sx={{ mb: 2 }}>
                            <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 0.5 }}>Strategy</Typography>
                            <Select
                                fullWidth
                                size="small"
                                value={strategy}
                                onChange={(e) => setStrategy(e.target.value)}
                                sx={{ background: 'rgba(0,0,0,0.2)' }}
                            >
                                <MenuItem value="my_signals">📊 Test My Signals</MenuItem>
                                <MenuItem value="rsi">📈 RSI Mean Reversion</MenuItem>
                                <MenuItem value="ma_crossover">📉 Moving Average Crossover</MenuItem>
                                <MenuItem value="momentum">🚀 Momentum Strategy</MenuItem>
                            </Select>
                        </Box>

                        {/* Advanced Options */}
                        {mode === 'advanced' && (
                            <>
                                <Typography sx={{ color: '#94a3b8', fontSize: '0.9rem', fontWeight: 600, mt: 3, mb: 2 }}>Entry Rules</Typography>

                                <Box sx={{ mb: 2 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 1 }}>RSI Oversold: {oversold}</Typography>
                                    <Slider value={oversold} min={20} max={40} onChange={(_, v) => setOversold(v as number)} />
                                </Box>

                                <Box sx={{ mb: 2 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 1 }}>RSI Overbought: {overbought}</Typography>
                                    <Slider value={overbought} min={60} max={80} onChange={(_, v) => setOverbought(v as number)} />
                                </Box>

                                <Typography sx={{ color: '#94a3b8', fontSize: '0.9rem', fontWeight: 600, mt: 3, mb: 2 }}>Exit Rules</Typography>

                                <Box sx={{ mb: 2 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 1 }}>Stop Loss: {stopLoss}%</Typography>
                                    <Slider value={stopLoss} min={1} max={10} onChange={(_, v) => setStopLoss(v as number)} />
                                </Box>

                                <Box sx={{ mb: 2 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 1 }}>Take Profit: {takeProfit}%</Typography>
                                    <Slider value={takeProfit} min={5} max={20} onChange={(_, v) => setTakeProfit(v as number)} />
                                </Box>
                            </>
                        )}

                        {/* Run Button */}
                        <Button
                            fullWidth
                            variant="contained"
                            size="large"
                            onClick={runBacktest}
                            disabled={isRunning}
                            sx={{ mt: 2 }}
                        >
                            <PlayArrowIcon sx={{ mr: 1 }} />
                            {isRunning ? 'Running...' : 'Run Backtest'}
                        </Button>
                    </SectionCard>
                </Grid>

                {/* Right Column - Results */}
                <Grid item xs={12} lg={8}>
                    {hasResults ? (
                        <>
                            {/* Metrics Row 1 */}
                            <Grid container spacing={2} sx={{ mb: 2 }}>
                                <Grid item xs={6} md={3}>
                                    <MetricCard icon="chart" iconColor="blue" label="Total Return" value={results.totalReturn} subText="Cumulative" sentiment="bullish" />
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <MetricCard icon="trending-up" iconColor="green" label="Annual Return" value={results.annualReturn} subText="Annualized" sentiment="bullish" />
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <MetricCard icon="bar-chart" iconColor="purple" label="Sharpe Ratio" value={results.sharpe} subText="Risk-adjusted" />
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <MetricCard icon="trending-down" iconColor="red" label="Max Drawdown" value={results.maxDrawdown} subText="Peak to trough" sentiment="bearish" />
                                </Grid>
                            </Grid>

                            {/* Metrics Row 2 */}
                            <Grid container spacing={2} sx={{ mb: 3 }}>
                                <Grid item xs={6} md={3}>
                                    <MetricCard icon="chart" iconColor="green" label="Win Rate" value={results.winRate} subText="Profitable" sentiment="bullish" />
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <MetricCard icon="bar-chart" iconColor="purple" label="Profit Factor" value={results.profitFactor} subText="Gross P/L" />
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <MetricCard icon="chart" iconColor="blue" label="Total Trades" value={results.totalTrades} subText="Executed" />
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <MetricCard icon="server" iconColor="green" label="Overfitting" value={results.overfitting} subText="Risk level" />
                                </Grid>
                            </Grid>

                            {/* Equity Curve */}
                            <SectionCard title="Equity Curve" icon={<ShowChartIcon />} iconColor="#10b981">
                                <Box sx={{ height: 250, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: 2 }}>
                                    <Box sx={{ textAlign: 'center' }}>
                                        <TrendingUpIcon sx={{ fontSize: 48, color: '#10b981', mb: 1 }} />
                                        <Typography sx={{ color: '#64748b' }}>Cumulative returns over time</Typography>
                                    </Box>
                                </Box>
                            </SectionCard>

                            {/* Actions */}
                            <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                                <Button variant="outlined" startIcon={<DownloadIcon />}>Export Results</Button>
                                <Button variant="outlined" color="primary" onClick={() => setHasResults(false)}>Run Again</Button>
                            </Box>
                        </>
                    ) : (
                        <Box sx={{
                            height: 400,
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: 'rgba(0,0,0,0.2)',
                            borderRadius: 3
                        }}>
                            <ShowChartIcon sx={{ fontSize: 64, color: '#64748b', mb: 2 }} />
                            <Typography sx={{ color: '#94a3b8', fontSize: '1.1rem', mb: 1 }}>Ready to Backtest</Typography>
                            <Typography sx={{ color: '#64748b' }}>Configure your strategy and click 'Run Backtest' to see results</Typography>
                        </Box>
                    )}
                </Grid>
            </Grid>
        </Box>
    )
}
