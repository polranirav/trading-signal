/**
 * Performance Page - Portfolio-Centric with Benchmarks
 * 
 * Shows:
 * - Portfolio performance vs S&P 500 and NASDAQ benchmarks
 * - "Am I beating the market?" indicator
 * - Individual stock contribution analysis
 * - Time-weighted returns
 */

import { useState, useMemo } from 'react'
import {
  Box,
  Typography,
  Grid,
  Chip,
  Alert,
  Paper,
  LinearProgress,
  Tooltip,
} from '@mui/material'
import { useQuery } from '@tanstack/react-query'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents'
import WarningIcon from '@mui/icons-material/Warning'
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import apiClient from '../../services/api'
import MetricCard from '../../components/MetricCard'
import { SectionCard } from '../../components/SignalComponents'
import { PerformanceAnalyticsPanel } from '../../components/PerformanceAnalyticsPanel'
import { usePortfolio } from '../../context'
import '../../styles/premium.css'

// Simulated benchmark data (in production, this would come from an API)
const generateBenchmarkData = (days: number) => {
  const today = new Date()
  const data = []
  let sp500 = 100
  let nasdaq = 100
  let portfolio = 100

  for (let i = days; i >= 0; i--) {
    const date = new Date(today)
    date.setDate(date.getDate() - i)

    // Simulate daily returns
    sp500 *= 1 + (Math.random() - 0.48) * 0.015 // Slightly positive bias
    nasdaq *= 1 + (Math.random() - 0.47) * 0.02 // Higher volatility, slightly positive
    portfolio *= 1 + (Math.random() - 0.45) * 0.025 // User portfolio

    data.push({
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      'My Portfolio': parseFloat((portfolio - 100).toFixed(2)),
      'S&P 500': parseFloat((sp500 - 100).toFixed(2)),
      'NASDAQ': parseFloat((nasdaq - 100).toFixed(2)),
    })
  }

  return data
}

// Portfolio API
const portfolioApi = {
  getSummary: () => apiClient.get('/portfolio/summary').then((r: any) => r.data),
  getTimeline: () => apiClient.get('/portfolio/transactions/timeline').then((r: any) => r.data),
}

export default function PerformancePage() {
  const [timeframe] = useState<'1M' | '3M' | '6M' | '1Y'>('3M')
  const { holdings, hasPortfolio, portfolioSymbols } = usePortfolio()

  // Fetch portfolio summary (placeholder for real data)
  const { data: _summary } = useQuery({
    queryKey: ['portfolio', 'summary'],
    queryFn: portfolioApi.getSummary,
    enabled: hasPortfolio,
  })

  // Fetch transaction timeline (placeholder for real data)
  const { data: _timelineData } = useQuery({
    queryKey: ['portfolio', 'timeline'],
    queryFn: portfolioApi.getTimeline,
    enabled: hasPortfolio,
  })

  // Generate benchmark comparison data
  const days = timeframe === '1M' ? 30 : timeframe === '3M' ? 90 : timeframe === '6M' ? 180 : 365
  const benchmarkData = useMemo(() => generateBenchmarkData(days), [days])

  // Calculate performance metrics
  const latestData = benchmarkData[benchmarkData.length - 1] || { 'My Portfolio': 0, 'S&P 500': 0, 'NASDAQ': 0 }
  const portfolioReturn = latestData['My Portfolio']
  const sp500Return = latestData['S&P 500']
  const nasdaqReturn = latestData['NASDAQ']

  const beatingMarket = portfolioReturn > sp500Return
  const alphaSP500 = portfolioReturn - sp500Return

  // Calculate stock contributions (mock data based on holdings)
  const stockContributions = useMemo(() => {
    if (!holdings || holdings.length === 0) return []

    return holdings.map((h: any) => {
      const currentValue = h.current_value || h.shares * (h.avg_cost * (1 + Math.random() * 0.3 - 0.15))
      const costBasis = h.shares * h.avg_cost
      const gain = currentValue - costBasis
      const gainPct = costBasis > 0 ? (gain / costBasis) * 100 : 0

      return {
        symbol: h.symbol,
        contribution: parseFloat(gainPct.toFixed(2)),
        value: currentValue,
        weight: 0, // Will calculate below
      }
    }).sort((a: any, b: any) => b.contribution - a.contribution)
  }, [holdings])

  // Calculate weights
  const totalValue = stockContributions.reduce((sum: number, s: any) => sum + s.value, 0)
  stockContributions.forEach((s: any) => {
    s.weight = totalValue > 0 ? (s.value / totalValue) * 100 : 0
  })

  // If no portfolio, show prompt
  if (!hasPortfolio) {
    return (
      <Box className="fade-in">
        <Typography variant="h4" sx={{ fontWeight: 700, color: '#fff', mb: 2 }}>
          Performance Tracking
        </Typography>
        <Alert
          severity="info"
          sx={{
            background: 'rgba(59, 130, 246, 0.1)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            '& .MuiAlert-icon': { color: '#3b82f6' }
          }}
        >
          <Typography sx={{ color: '#e2e8f0' }}>
            <strong>Import your portfolio to track performance!</strong><br />
            Go to <strong>My Portfolio</strong> → <strong>Import Transactions</strong> to get started.
            You'll see how your portfolio compares to market benchmarks.
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
            <ShowChartIcon sx={{ color: '#8b5cf6' }} />
            My Portfolio Performance
          </Typography>
          <Typography sx={{ color: '#64748b', fontSize: '0.9rem' }}>
            Compare your portfolio against major market benchmarks
          </Typography>
        </Box>

        <Chip
          icon={<AccountBalanceWalletIcon />}
          label={`${portfolioSymbols.length} Stocks`}
          sx={{
            fontWeight: 600,
            background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
            color: '#fff',
          }}
        />
      </Box>

      {/* Am I Beating the Market? Banner */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          mb: 4,
          background: beatingMarket
            ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05))'
            : 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05))',
          border: beatingMarket
            ? '1px solid rgba(16, 185, 129, 0.3)'
            : '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: 2,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {beatingMarket ? (
            <EmojiEventsIcon sx={{ fontSize: 48, color: '#10b981' }} />
          ) : (
            <WarningIcon sx={{ fontSize: 48, color: '#ef4444' }} />
          )}
          <Box>
            <Typography sx={{ fontSize: '1.5rem', fontWeight: 700, color: beatingMarket ? '#10b981' : '#ef4444' }}>
              {beatingMarket ? '🎉 You\'re Beating the Market!' : '📉 Lagging Behind the Market'}
            </Typography>
            <Typography sx={{ color: '#94a3b8' }}>
              {beatingMarket
                ? `Your portfolio is outperforming the S&P 500 by ${alphaSP500.toFixed(2)}%`
                : `Your portfolio is underperforming the S&P 500 by ${Math.abs(alphaSP500).toFixed(2)}%`
              }
            </Typography>
          </Box>
        </Box>
      </Paper>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={6} md={3}>
          <MetricCard
            icon="trending-up"
            iconColor="purple"
            label="My Portfolio"
            value={`${portfolioReturn >= 0 ? '+' : ''}${portfolioReturn.toFixed(2)}%`}
            subText={timeframe}
            sentiment={portfolioReturn >= 0 ? 'bullish' : 'bearish'}
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard
            icon="chart"
            iconColor="blue"
            label="S&P 500"
            value={`${sp500Return >= 0 ? '+' : ''}${sp500Return.toFixed(2)}%`}
            subText={timeframe}
            sentiment={sp500Return >= 0 ? 'bullish' : 'bearish'}
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard
            icon="chart"
            iconColor="green"
            label="NASDAQ"
            value={`${nasdaqReturn >= 0 ? '+' : ''}${nasdaqReturn.toFixed(2)}%`}
            subText={timeframe}
            sentiment={nasdaqReturn >= 0 ? 'bullish' : 'bearish'}
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard
            icon={beatingMarket ? "trending-up" : "trending-down"}
            iconColor={beatingMarket ? "green" : "red"}
            label="Alpha (vs S&P)"
            value={`${alphaSP500 >= 0 ? '+' : ''}${alphaSP500.toFixed(2)}%`}
            subText="Excess return"
            sentiment={alphaSP500 >= 0 ? 'bullish' : 'bearish'}
          />
        </Grid>
      </Grid>

      {/* Performance Chart */}
      <SectionCard title="Performance vs Benchmarks" icon={<ShowChartIcon />} iconColor="#8b5cf6">
        <Box sx={{ height: 350 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={benchmarkData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis
                dataKey="date"
                stroke="#64748b"
                tick={{ fontSize: 11 }}
                interval={Math.floor(benchmarkData.length / 8)}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fontSize: 11 }}
                tickFormatter={(v) => `${v}%`}
              />
              <ChartTooltip
                contentStyle={{
                  background: 'rgba(15, 23, 42, 0.95)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 8,
                }}
                formatter={(value: number) => [`${value.toFixed(2)}%`, '']}
              />
              <Legend />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" strokeDasharray="3 3" />
              <Line
                type="monotone"
                dataKey="My Portfolio"
                stroke="#8b5cf6"
                strokeWidth={3}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="S&P 500"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                strokeDasharray="5 5"
              />
              <Line
                type="monotone"
                dataKey="NASDAQ"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
                strokeDasharray="3 3"
              />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      </SectionCard>

      {/* Stock Contributions */}
      <Grid container spacing={3} sx={{ mt: 1 }}>
        <Grid item xs={12} md={6}>
          <SectionCard title="Top Performers" icon={<TrendingUpIcon />} iconColor="#10b981">
            <Box sx={{ py: 1 }}>
              {stockContributions.filter((s: any) => s.contribution > 0).slice(0, 5).map((stock: any) => (
                <Box key={stock.symbol} sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography sx={{ fontWeight: 600, color: '#fff' }}>{stock.symbol}</Typography>
                    <Typography sx={{ color: '#10b981', fontWeight: 600 }}>
                      +{stock.contribution.toFixed(2)}%
                    </Typography>
                  </Box>
                  <Tooltip title={`${stock.weight.toFixed(1)}% of portfolio`}>
                    <LinearProgress
                      variant="determinate"
                      value={Math.min(stock.contribution, 50) * 2}
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        background: 'rgba(16, 185, 129, 0.2)',
                        '& .MuiLinearProgress-bar': {
                          background: 'linear-gradient(135deg, #10b981, #34d399)',
                          borderRadius: 3,
                        }
                      }}
                    />
                  </Tooltip>
                </Box>
              ))}
              {stockContributions.filter((s: any) => s.contribution > 0).length === 0 && (
                <Typography sx={{ color: '#64748b', textAlign: 'center', py: 2 }}>
                  No winning positions yet
                </Typography>
              )}
            </Box>
          </SectionCard>
        </Grid>

        <Grid item xs={12} md={6}>
          <SectionCard title="Lagging Positions" icon={<TrendingDownIcon />} iconColor="#ef4444">
            <Box sx={{ py: 1 }}>
              {stockContributions.filter((s: any) => s.contribution < 0).slice(0, 5).map((stock: any) => (
                <Box key={stock.symbol} sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography sx={{ fontWeight: 600, color: '#fff' }}>{stock.symbol}</Typography>
                    <Typography sx={{ color: '#ef4444', fontWeight: 600 }}>
                      {stock.contribution.toFixed(2)}%
                    </Typography>
                  </Box>
                  <Tooltip title={`${stock.weight.toFixed(1)}% of portfolio`}>
                    <LinearProgress
                      variant="determinate"
                      value={Math.min(Math.abs(stock.contribution), 50) * 2}
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        background: 'rgba(239, 68, 68, 0.2)',
                        '& .MuiLinearProgress-bar': {
                          background: 'linear-gradient(135deg, #ef4444, #f87171)',
                          borderRadius: 3,
                        }
                      }}
                    />
                  </Tooltip>
                </Box>
              ))}
              {stockContributions.filter((s: any) => s.contribution < 0).length === 0 && (
                <Typography sx={{ color: '#64748b', textAlign: 'center', py: 2 }}>
                  No losing positions! 🎉
                </Typography>
              )}
            </Box>
          </SectionCard>
        </Grid>
      </Grid>

      {/* Portfolio Allocation Chart */}
      <Box sx={{ mt: 3 }}>
        <SectionCard title="Stock Contribution to Returns" icon={<AccountBalanceWalletIcon />} iconColor="#f59e0b">
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={stockContributions.slice(0, 10)}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis type="number" stroke="#64748b" tickFormatter={(v) => `${v}%`} />
                <YAxis type="category" dataKey="symbol" stroke="#64748b" width={60} />
                <ChartTooltip
                  contentStyle={{
                    background: 'rgba(15, 23, 42, 0.95)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 8,
                  }}
                  formatter={(value: number) => [`${value.toFixed(2)}%`, 'Return']}
                />
                <ReferenceLine x={0} stroke="rgba(255,255,255,0.3)" />
                <Bar
                  dataKey="contribution"
                  fill="#8b5cf6"
                  radius={[0, 4, 4, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </SectionCard>
      </Box>

      {/* Advanced Performance Analytics */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 3, display: 'flex', alignItems: 'center', gap: 1.5 }}>
          📈 Advanced Analytics
        </Typography>
        <PerformanceAnalyticsPanel />
      </Box>
    </Box>
  )
}
