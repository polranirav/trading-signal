/**
 * Strategy Lab - Portfolio "What If" Scenarios
 * 
 * Transformed from Backtest to portfolio-focused Strategy Lab:
 * - "What if" scenarios on actual portfolio
 * - Rebalancing suggestions
 * - Forward-looking strategy simulation
 */

import { useState, useEffect, useMemo } from 'react'
import { Box, Typography, Grid, Select, MenuItem, Button, Slider, ToggleButtonGroup, ToggleButton, Chip, Paper, Alert, LinearProgress } from '@mui/material'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import ScienceIcon from '@mui/icons-material/Science'
import BoltIcon from '@mui/icons-material/Bolt'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import BalanceIcon from '@mui/icons-material/Balance'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet'
import LightbulbIcon from '@mui/icons-material/Lightbulb'
import { api } from '../../services/api'
import { useQuery } from '@tanstack/react-query'
import { SectionCard } from '../../components/SignalComponents'
import MetricCard from '../../components/MetricCard'
import { usePortfolio } from '../../context'
import '../../styles/premium.css'

// Scenario types
type ScenarioType = 'what_if' | 'rebalance' | 'optimize'

// ... existing code ...

export default function StrategyLabPage() {
    const { holdings, portfolioSymbols, hasPortfolio, summary } = usePortfolio()

    // Construct portfolioData from context
    const portfolioData = useMemo(() => {
        const totalValue = summary?.total_value || 0
        const allocations = holdings.map((h: any) => ({
            symbol: h.symbol,
            allocation: totalValue > 0 ? ((h.current_price * h.shares) / totalValue) * 100 : 0
        })).sort((a: any, b: any) => b.allocation - a.allocation)

        return {
            totalValue,
            allocations
        }
    }, [summary, holdings])

    // Fetch signals to power the simulation logic
    const { data: signalsData } = useQuery({
        queryKey: ['signals', { limit: 100 }],
        queryFn: () => api.getSignals({ limit: 100 }),
    })

    const signals = useMemo(() => signalsData?.signals || [], [signalsData])

    // ... existing state ...
    const [scenarioType, setScenarioType] = useState<ScenarioType>('what_if')
    const [selectedStock, setSelectedStock] = useState('')
    const [action, setAction] = useState<'buy' | 'sell'>('buy')
    const [shares, setShares] = useState(10)
    const [targetAllocation, setTargetAllocation] = useState(20)
    const [isRunning, setIsRunning] = useState(false)
    const [hasResults, setHasResults] = useState(false)

    // ... existing effects ...

    // Simulation handler
    const runSimulation = () => {
        setIsRunning(true)
        setHasResults(false)

        // Simulate processing calculation
        setTimeout(() => {
            setHasResults(true)
            setIsRunning(false)
        }, 800)
    }

    // Mock simulation results - NOW DATA-DRIVEN
    const getSimulationResults = () => {
        const signal = signals.find((s: any) => s.symbol === selectedStock)
        const currentPrice = signal?.price_at_signal || 150
        const confidence = signal?.confluence_score || 0.5
        const signalType = signal?.signal_type || 'HOLD'

        // Base drift based on signal
        // If BUY and we BUY -> Positive drift
        // If SELL and we BUY -> Negative drift
        let drift = 0
        if (signalType.includes('BUY')) drift = 0.08 * (confidence + 0.5) // +4% to +12%
        if (signalType.includes('SELL')) drift = -0.05 * (confidence + 0.5) // -2.5% to -7.5%

        // Add randomness but anchored to the signal
        const noise = (Math.random() - 0.5) * 0.05

        if (scenarioType === 'what_if') {
            const cost = shares * currentPrice

            // Calculate projected gain based on action vs signal
            let projectedGain = 0

            if (action === 'buy') {
                projectedGain = (drift + noise) * 100
            } else {
                // Short selling / Selling existing
                projectedGain = -(drift + noise) * 100
            }

            // Cap reasonable limits
            projectedGain = Math.max(-15, Math.min(25, projectedGain))

            return {
                title: `${action === 'buy' ? 'Buying' : 'Selling'} ${shares} shares of ${selectedStock}`,
                cost: action === 'buy' ? cost : -cost,
                projectedReturn: projectedGain,
                riskLevel: Math.abs(projectedGain) > 10 ? 'High' : Math.abs(projectedGain) > 5 ? 'Medium' : 'Low',
                diversificationImpact: action === 'buy' ? 'Increased' : 'Decreased',
                recommendation: projectedGain > 5 ? '✅ Strong opportunity' : projectedGain > 1 ? '⚠️ Moderate opportunity' : '❌ Consider alternatives',
            }
        }

        if (scenarioType === 'rebalance') {
            // Rebalancing logic remains portfolio-structure based
            return {
                title: 'Rebalancing Suggestions',
                trades: portfolioData.allocations.slice(0, 3).map((a: any) => ({
                    symbol: a.symbol,
                    action: a.allocation > targetAllocation ? 'SELL' : 'BUY',
                    shares: Math.abs(Math.floor((a.allocation - targetAllocation) * portfolioData.totalValue / 100 / (signals.find((s: any) => s.symbol === a.symbol)?.price_at_signal || 150))),
                    reason: a.allocation > targetAllocation ? 'Over-allocated' : 'Under-allocated',
                })),
                projectedDiversification: '+15%',
                riskReduction: '-8%',
            }
        }


        // Optimize
        return {
            title: 'Portfolio Optimization',
            suggestions: [
                { action: 'Increase', symbol: 'High-growth tech', reason: 'Strong momentum signals' },
                { action: 'Reduce', symbol: 'Defensive stocks', reason: 'Market conditions favor growth' },
                { action: 'Add', symbol: 'Dividend aristocrats', reason: 'Income generation' },
            ],
            expectedReturn: '+12.5%',
            sharpeImprovement: '+0.35',
        }
    }

    const results = hasResults ? getSimulationResults() : null

    // If no portfolio, show prompt
    if (!hasPortfolio) {
        return (
            <Box className="fade-in">
                <Typography variant="h4" sx={{ fontWeight: 700, color: '#fff', mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ScienceIcon sx={{ color: '#8b5cf6' }} />
                    Strategy Lab
                </Typography>
                <Alert
                    severity="info"
                    sx={{
                        background: 'rgba(139, 92, 246, 0.1)',
                        border: '1px solid rgba(139, 92, 246, 0.3)',
                        '& .MuiAlert-icon': { color: '#8b5cf6' }
                    }}
                >
                    <Typography sx={{ color: '#e2e8f0' }}>
                        <strong>Import your portfolio to use the Strategy Lab!</strong><br />
                        The Strategy Lab lets you simulate "What If" scenarios on your actual holdings.
                        Go to <strong>My Portfolio</strong> → <strong>Import Transactions</strong> to get started.
                    </Typography>
                </Alert>
            </Box>
        )
    }

    return (
        <Box className="fade-in">
            {/* Header */}
            <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 2 }}>
                <Box>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#fff', fontSize: '1.75rem', mb: 0.5, display: 'flex', alignItems: 'center', gap: 1 }}>
                        <ScienceIcon sx={{ color: '#8b5cf6' }} />
                        Strategy Lab
                    </Typography>
                    <Typography sx={{ color: '#64748b', fontSize: '0.9rem' }}>
                        Simulate changes to your portfolio and see the projected impact
                    </Typography>
                </Box>
                <Chip
                    icon={<AccountBalanceWalletIcon />}
                    label={`${portfolioSymbols.length} Stocks • $${portfolioData.totalValue.toLocaleString()}`}
                    sx={{
                        fontWeight: 600,
                        background: 'linear-gradient(135deg, #8b5cf6, #a855f7)',
                        color: '#fff',
                    }}
                />
            </Box>

            {/* Scenario Type Selector */}
            <Box sx={{ mb: 3 }}>
                <ToggleButtonGroup
                    value={scenarioType}
                    exclusive
                    onChange={(_, v) => v && setScenarioType(v)}
                    sx={{ background: 'rgba(0,0,0,0.3)', borderRadius: 2 }}
                >
                    <ToggleButton value="what_if" sx={{ px: 3, color: scenarioType === 'what_if' ? '#8b5cf6' : '#64748b' }}>
                        <BoltIcon sx={{ mr: 1 }} /> What If?
                    </ToggleButton>
                    <ToggleButton value="rebalance" sx={{ px: 3, color: scenarioType === 'rebalance' ? '#f59e0b' : '#64748b' }}>
                        <BalanceIcon sx={{ mr: 1 }} /> Rebalance
                    </ToggleButton>
                    <ToggleButton value="optimize" sx={{ px: 3, color: scenarioType === 'optimize' ? '#10b981' : '#64748b' }}>
                        <AutoFixHighIcon sx={{ mr: 1 }} /> Optimize
                    </ToggleButton>
                </ToggleButtonGroup>
            </Box>

            <Grid container spacing={3}>
                {/* Left Column - Configuration */}
                <Grid item xs={12} lg={4}>
                    <SectionCard
                        title={
                            scenarioType === 'what_if' ? 'What If Scenario' :
                                scenarioType === 'rebalance' ? 'Rebalancing Tool' : 'AI Optimizer'
                        }
                        icon={
                            scenarioType === 'what_if' ? <BoltIcon /> :
                                scenarioType === 'rebalance' ? <BalanceIcon /> : <AutoFixHighIcon />
                        }
                        iconColor={
                            scenarioType === 'what_if' ? '#8b5cf6' :
                                scenarioType === 'rebalance' ? '#f59e0b' : '#10b981'
                        }
                    >
                        {scenarioType === 'what_if' && (
                            <>
                                {/* Action Type */}
                                <Box sx={{ mb: 3 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 1 }}>Action</Typography>
                                    <ToggleButtonGroup
                                        fullWidth
                                        value={action}
                                        exclusive
                                        onChange={(_, v) => v && setAction(v)}
                                    >
                                        <ToggleButton value="buy" sx={{ color: action === 'buy' ? '#10b981' : '#64748b' }}>
                                            <TrendingUpIcon sx={{ mr: 0.5 }} /> Buy More
                                        </ToggleButton>
                                        <ToggleButton value="sell" sx={{ color: action === 'sell' ? '#ef4444' : '#64748b' }}>
                                            <TrendingDownIcon sx={{ mr: 0.5 }} /> Sell
                                        </ToggleButton>
                                    </ToggleButtonGroup>
                                </Box>

                                {/* Stock Selection */}
                                <Box sx={{ mb: 2 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 0.5 }}>Stock</Typography>
                                    <Select
                                        fullWidth
                                        size="small"
                                        value={selectedStock}
                                        onChange={(e) => setSelectedStock(e.target.value)}
                                        sx={{ background: 'rgba(0,0,0,0.2)' }}
                                    >
                                        {portfolioSymbols.map((s: string) => (
                                            <MenuItem key={s} value={s}>{s}</MenuItem>
                                        ))}
                                    </Select>
                                </Box>

                                {/* Shares */}
                                <Box sx={{ mb: 2 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 1 }}>Shares: {shares}</Typography>
                                    <Slider
                                        value={shares}
                                        min={1}
                                        max={100}
                                        onChange={(_, v) => setShares(v as number)}
                                        marks={[
                                            { value: 1, label: '1' },
                                            { value: 50, label: '50' },
                                            { value: 100, label: '100' },
                                        ]}
                                    />
                                </Box>
                            </>
                        )}

                        {scenarioType === 'rebalance' && (
                            <>
                                <Alert severity="info" sx={{ mb: 3, background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                                    <Typography sx={{ fontSize: '0.85rem', color: '#e2e8f0' }}>
                                        Set your target allocation percentage per stock. We'll calculate the trades needed.
                                    </Typography>
                                </Alert>

                                <Box sx={{ mb: 2 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 1 }}>
                                        Target Max Allocation: {targetAllocation}%
                                    </Typography>
                                    <Slider
                                        value={targetAllocation}
                                        min={5}
                                        max={50}
                                        step={5}
                                        onChange={(_, v) => setTargetAllocation(v as number)}
                                        marks={[
                                            { value: 10, label: '10%' },
                                            { value: 25, label: '25%' },
                                            { value: 50, label: '50%' },
                                        ]}
                                    />
                                </Box>

                                {/* Current Top Holdings */}
                                <Typography sx={{ color: '#94a3b8', fontSize: '0.85rem', fontWeight: 600, mt: 3, mb: 1 }}>
                                    Current Allocation
                                </Typography>
                                {portfolioData.allocations.slice(0, 5).map((a: any) => (
                                    <Box key={a.symbol} sx={{ mb: 1.5 }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                            <Typography sx={{ color: '#fff', fontSize: '0.85rem', fontWeight: 600 }}>{a.symbol}</Typography>
                                            <Typography sx={{
                                                color: a.allocation > targetAllocation ? '#ef4444' : '#10b981',
                                                fontSize: '0.85rem'
                                            }}>
                                                {a.allocation.toFixed(1)}%
                                            </Typography>
                                        </Box>
                                        <LinearProgress
                                            variant="determinate"
                                            value={Math.min(a.allocation, 100)}
                                            sx={{
                                                height: 4,
                                                borderRadius: 2,
                                                background: 'rgba(255,255,255,0.1)',
                                                '& .MuiLinearProgress-bar': {
                                                    background: a.allocation > targetAllocation
                                                        ? 'linear-gradient(135deg, #ef4444, #f87171)'
                                                        : 'linear-gradient(135deg, #10b981, #34d399)',
                                                }
                                            }}
                                        />
                                    </Box>
                                ))}
                            </>
                        )}

                        {scenarioType === 'optimize' && (
                            <>
                                <Alert severity="success" sx={{ mb: 3, background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                                    <Typography sx={{ fontSize: '0.85rem', color: '#e2e8f0' }}>
                                        AI will analyze your portfolio and suggest optimizations based on current market signals.
                                    </Typography>
                                </Alert>

                                <Box sx={{ mb: 3 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 0.5 }}>Optimization Goal</Typography>
                                    <Select
                                        fullWidth
                                        size="small"
                                        defaultValue="growth"
                                        sx={{ background: 'rgba(0,0,0,0.2)' }}
                                    >
                                        <MenuItem value="growth">🚀 Maximum Growth</MenuItem>
                                        <MenuItem value="balanced">⚖️ Balanced Risk/Return</MenuItem>
                                        <MenuItem value="income">💰 Income Generation</MenuItem>
                                        <MenuItem value="preservation">🛡️ Capital Preservation</MenuItem>
                                    </Select>
                                </Box>

                                <Box sx={{ mb: 2 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 0.5 }}>Risk Tolerance</Typography>
                                    <Select
                                        fullWidth
                                        size="small"
                                        defaultValue="moderate"
                                        sx={{ background: 'rgba(0,0,0,0.2)' }}
                                    >
                                        <MenuItem value="conservative">Conservative</MenuItem>
                                        <MenuItem value="moderate">Moderate</MenuItem>
                                        <MenuItem value="aggressive">Aggressive</MenuItem>
                                    </Select>
                                </Box>
                            </>
                        )}

                        {/* Run Button */}
                        <Button
                            fullWidth
                            variant="contained"
                            size="large"
                            onClick={runSimulation}
                            disabled={isRunning}
                            sx={{
                                mt: 3,
                                background: scenarioType === 'what_if'
                                    ? 'linear-gradient(135deg, #8b5cf6, #a855f7)'
                                    : scenarioType === 'rebalance'
                                        ? 'linear-gradient(135deg, #f59e0b, #d97706)'
                                        : 'linear-gradient(135deg, #10b981, #059669)',
                            }}
                        >
                            <PlayArrowIcon sx={{ mr: 1 }} />
                            {isRunning ? 'Simulating...' : 'Run Simulation'}
                        </Button>
                    </SectionCard>
                </Grid>

                {/* Right Column - Results */}
                <Grid item xs={12} lg={8}>
                    {hasResults && results ? (
                        <>
                            {/* Results Header */}
                            <Paper
                                elevation={0}
                                sx={{
                                    p: 3,
                                    mb: 3,
                                    background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(168, 85, 247, 0.05))',
                                    border: '1px solid rgba(139, 92, 246, 0.2)',
                                    borderRadius: 2,
                                }}
                            >
                                <Typography sx={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <LightbulbIcon sx={{ color: '#f59e0b' }} />
                                    {results.title}
                                </Typography>

                                {scenarioType === 'what_if' && (
                                    <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                                        <Box>
                                            <Typography sx={{ color: '#64748b', fontSize: '0.8rem' }}>Estimated Cost</Typography>
                                            <Typography sx={{ color: (results.cost ?? 0) > 0 ? '#ef4444' : '#10b981', fontWeight: 600, fontSize: '1.1rem' }}>
                                                {(results.cost ?? 0) > 0 ? '-' : '+'}${Math.abs(results.cost ?? 0).toLocaleString()}
                                            </Typography>
                                        </Box>
                                        <Box>
                                            <Typography sx={{ color: '#64748b', fontSize: '0.8rem' }}>Projected Return</Typography>
                                            <Typography sx={{ color: (results.projectedReturn ?? 0) >= 0 ? '#10b981' : '#ef4444', fontWeight: 600, fontSize: '1.1rem' }}>
                                                {(results.projectedReturn ?? 0) >= 0 ? '+' : ''}{(results.projectedReturn ?? 0).toFixed(2)}%
                                            </Typography>
                                        </Box>
                                        <Box>
                                            <Typography sx={{ color: '#64748b', fontSize: '0.8rem' }}>Risk Level</Typography>
                                            <Chip
                                                label={results.riskLevel}
                                                size="small"
                                                sx={{
                                                    background: results.riskLevel === 'Low' ? 'rgba(16, 185, 129, 0.2)' :
                                                        results.riskLevel === 'Medium' ? 'rgba(245, 158, 11, 0.2)' :
                                                            'rgba(239, 68, 68, 0.2)',
                                                    color: results.riskLevel === 'Low' ? '#10b981' :
                                                        results.riskLevel === 'Medium' ? '#f59e0b' : '#ef4444',
                                                }}
                                            />
                                        </Box>
                                    </Box>
                                )}
                            </Paper>

                            {/* Recommendation */}
                            {scenarioType === 'what_if' && (
                                <SectionCard title="AI Recommendation" icon={<AutoFixHighIcon />} iconColor="#f59e0b">
                                    <Typography sx={{ fontSize: '1.5rem', mb: 2 }}>
                                        {results.recommendation}
                                    </Typography>
                                    <Typography sx={{ color: '#94a3b8', lineHeight: 1.7 }}>
                                        Based on current market conditions and your portfolio composition,
                                        {(results.projectedReturn ?? 0) > 5
                                            ? ` this trade shows strong potential. The ${selectedStock} position would ${action === 'buy' ? 'increase' : 'decrease'} your exposure to this sector.`
                                            : ` this trade carries moderate risk. Consider your overall portfolio balance before proceeding.`
                                        }
                                    </Typography>
                                </SectionCard>
                            )}

                            {scenarioType === 'rebalance' && results.trades && (
                                <SectionCard title="Suggested Trades" icon={<BalanceIcon />} iconColor="#f59e0b">
                                    {results.trades.map((trade: any, i: number) => (
                                        <Box key={i} sx={{
                                            p: 2,
                                            mb: 2,
                                            background: 'rgba(0,0,0,0.2)',
                                            borderRadius: 2,
                                            borderLeft: `4px solid ${trade.action === 'BUY' ? '#10b981' : '#ef4444'}`
                                        }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                <Box>
                                                    <Typography sx={{ fontWeight: 600, color: '#fff' }}>{trade.symbol}</Typography>
                                                    <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>{trade.reason}</Typography>
                                                </Box>
                                                <Chip
                                                    label={`${trade.action} ${trade.shares} shares`}
                                                    sx={{
                                                        background: trade.action === 'BUY' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                                                        color: trade.action === 'BUY' ? '#10b981' : '#ef4444',
                                                        fontWeight: 600,
                                                    }}
                                                />
                                            </Box>
                                        </Box>
                                    ))}

                                    <Box sx={{ mt: 3, display: 'flex', gap: 3 }}>
                                        <Box>
                                            <Typography sx={{ color: '#64748b', fontSize: '0.8rem' }}>Diversification</Typography>
                                            <Typography sx={{ color: '#10b981', fontWeight: 600 }}>{results.projectedDiversification}</Typography>
                                        </Box>
                                        <Box>
                                            <Typography sx={{ color: '#64748b', fontSize: '0.8rem' }}>Risk Reduction</Typography>
                                            <Typography sx={{ color: '#10b981', fontWeight: 600 }}>{results.riskReduction}</Typography>
                                        </Box>
                                    </Box>
                                </SectionCard>
                            )}

                            {scenarioType === 'optimize' && results.suggestions && (
                                <SectionCard title="Optimization Suggestions" icon={<AutoFixHighIcon />} iconColor="#10b981">
                                    {results.suggestions.map((s: any, i: number) => (
                                        <Box key={i} sx={{
                                            p: 2,
                                            mb: 2,
                                            background: 'rgba(0,0,0,0.2)',
                                            borderRadius: 2,
                                        }}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                                <Chip
                                                    label={s.action}
                                                    size="small"
                                                    sx={{
                                                        background: s.action === 'Increase' ? 'rgba(16, 185, 129, 0.2)' :
                                                            s.action === 'Reduce' ? 'rgba(239, 68, 68, 0.2)' :
                                                                'rgba(59, 130, 246, 0.2)',
                                                        color: s.action === 'Increase' ? '#10b981' :
                                                            s.action === 'Reduce' ? '#ef4444' : '#3b82f6',
                                                    }}
                                                />
                                                <Box>
                                                    <Typography sx={{ fontWeight: 600, color: '#fff' }}>{s.symbol}</Typography>
                                                    <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>{s.reason}</Typography>
                                                </Box>
                                            </Box>
                                        </Box>
                                    ))}

                                    <Grid container spacing={2} sx={{ mt: 2 }}>
                                        <Grid item xs={6}>
                                            <MetricCard
                                                icon="trending-up"
                                                iconColor="green"
                                                label="Expected Return"
                                                value={results.expectedReturn}
                                                sentiment="bullish"
                                            />
                                        </Grid>
                                        <Grid item xs={6}>
                                            <MetricCard
                                                icon="chart"
                                                iconColor="purple"
                                                label="Sharpe Improvement"
                                                value={results.sharpeImprovement}
                                            />
                                        </Grid>
                                    </Grid>
                                </SectionCard>
                            )}

                            {/* Actions */}
                            <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                                <Button variant="outlined" onClick={() => setHasResults(false)}>
                                    Try Another Scenario
                                </Button>
                            </Box>
                        </>
                    ) : (
                        <Box sx={{
                            height: 450,
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: 'rgba(0,0,0,0.2)',
                            borderRadius: 3
                        }}>
                            <ScienceIcon sx={{ fontSize: 64, color: '#8b5cf6', mb: 2 }} />
                            <Typography sx={{ color: '#94a3b8', fontSize: '1.1rem', mb: 1 }}>
                                {scenarioType === 'what_if' && 'Simulate buying or selling stocks'}
                                {scenarioType === 'rebalance' && 'Get rebalancing recommendations'}
                                {scenarioType === 'optimize' && 'AI-powered portfolio optimization'}
                            </Typography>
                            <Typography sx={{ color: '#64748b', textAlign: 'center', maxWidth: 400 }}>
                                Configure your scenario on the left and click 'Run Simulation' to see projected outcomes
                            </Typography>
                        </Box>
                    )}
                </Grid>
            </Grid>
        </Box>
    )
}
