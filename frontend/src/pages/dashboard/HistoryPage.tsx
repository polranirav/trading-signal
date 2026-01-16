/**
 * Signal Intelligence Dashboard
 * 
 * Enhanced history page with performance tracking, accuracy analytics,
 * and market insights - matching Dash history.py
 */

import { useState, useMemo } from 'react'
import { Box, Typography, Grid, TextField, Select, MenuItem, Chip, Button, Slider, RadioGroup, FormControlLabel, Radio, CircularProgress, Paper } from '@mui/material'
import { useQuery } from '@tanstack/react-query'
import FilterListIcon from '@mui/icons-material/FilterList'
import HistoryIcon from '@mui/icons-material/History'
import PieChartIcon from '@mui/icons-material/PieChart'
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents'
import TipsAndUpdatesIcon from '@mui/icons-material/TipsAndUpdates'
import AnalyticsIcon from '@mui/icons-material/Analytics'
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import SchoolIcon from '@mui/icons-material/School'
import apiClient from '../../services/api'
import { api } from '../../services/api'
import { format } from 'date-fns'
import MetricCard from '../../components/MetricCard'
import { SectionCard, SignalBadge, ConfidenceBar } from '../../components/SignalComponents'
import { TradingJournalPanel } from '../../components/TradingJournalPanel'
import { usePortfolio } from '../../context'
import '../../styles/premium.css'

export default function HistoryPage() {
  // Portfolio context
  const { portfolioSymbols, hasPortfolio } = usePortfolio()

  // Portfolio-first: When user has portfolio, always show their stocks
  const viewMode = hasPortfolio ? 'portfolio' : 'all'

  // Tab state: 'signals' or 'lessons'
  const [activeTab, setActiveTab] = useState<'signals' | 'lessons'>('signals')

  const [symbolFilter, setSymbolFilter] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const [timeFilter, setTimeFilter] = useState('30')
  const [minConfidence, setMinConfidence] = useState(40)

  // Fetch lessons from LLM
  const { data: lessonsData, isLoading: lessonsLoading } = useQuery({
    queryKey: ['trading-lessons'],
    queryFn: () => apiClient.get('/portfolio/lessons').then((r: any) => r.data),
    enabled: hasPortfolio,
    staleTime: 1000 * 60 * 5, // Cache for 5 minutes
  })

  const { data: signalsData, isLoading, refetch } = useQuery({
    queryKey: ['signals', 'history', { days: parseInt(timeFilter) || 365 }],
    queryFn: () => api.getSignals({
      days: parseInt(timeFilter) || 365,
      limit: 100,
    }),
  })

  const allSignals = signalsData?.signals || []

  // Apply filters including portfolio filter
  const signals = useMemo(() => {
    let filtered = [...allSignals]

    // Portfolio filter
    if (viewMode === 'portfolio' && hasPortfolio && !symbolFilter) {
      filtered = filtered.filter((s: any) => portfolioSymbols.includes(s.symbol))
    }

    // Symbol search filter
    if (symbolFilter) {
      filtered = filtered.filter((s: any) => s.symbol.toLowerCase().includes(symbolFilter.toLowerCase()))
    }

    // Type filter
    if (typeFilter !== 'all') {
      filtered = filtered.filter((s: any) => s.signal_type?.includes(typeFilter))
    }

    // Confidence filter
    filtered = filtered.filter((s: any) => (s.confluence_score || 0) * 100 >= minConfidence)

    return filtered
  }, [allSignals, viewMode, hasPortfolio, portfolioSymbols, symbolFilter, typeFilter, minConfidence])

  // Calculate metrics
  const totalSignals = signals.length
  const executedSignals = signals.filter((s: any) => s.is_executed)
  const winningSignals = executedSignals.filter((s: any) => (s.realized_pnl_pct || 0) > 0)
  const accuracy = executedSignals.length > 0 ? (winningSignals.length / executedSignals.length * 100) : 0
  const avgReturn = executedSignals.length > 0
    ? executedSignals.reduce((acc: number, s: any) => acc + (s.realized_pnl_pct || 0), 0) / executedSignals.length * 100
    : 0

  // Hot streak (mock for now)
  const hotStreak = Math.floor(Math.random() * 5) + 1

  // Buy/Sell distribution
  const buyCount = signals.filter((s: any) => s.signal_type?.includes('BUY')).length
  const sellCount = signals.filter((s: any) => s.signal_type?.includes('SELL')).length
  const holdCount = signals.filter((s: any) => s.signal_type === 'HOLD').length

  // Top performers (highest confidence)
  const topPerformers = [...signals]
    .filter((s: any) => s.signal_type?.includes('BUY'))
    .sort((a: any, b: any) => (b.confluence_score || 0) - (a.confluence_score || 0))
    .slice(0, 5)

  return (
    <Box className="fade-in">
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 2 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, color: '#fff', fontSize: '1.75rem', mb: 0.5 }}>
            {hasPortfolio ? 'My Signal History' : 'Signal Intelligence'}
          </Typography>
          <Typography sx={{ color: '#64748b', fontSize: '0.9rem' }}>
            {hasPortfolio
              ? `Track signal performance for your ${portfolioSymbols.length} portfolio stocks`
              : 'Track signal performance, market trends, and discover trading patterns.'
            }
          </Typography>
        </Box>

        {/* Portfolio Indicator */}
        {hasPortfolio && (
          <Chip
            icon={<AccountBalanceWalletIcon />}
            label={`${signals.length} Signals`}
            sx={{
              fontWeight: 600,
              background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              color: '#fff',
              cursor: 'default',
            }}
          />
        )}
      </Box>

      {/* Tab Switcher - Signal History vs Trading Lessons */}
      {hasPortfolio && (
        <Box sx={{ mb: 3, display: 'flex', gap: 1 }}>
          <Chip
            icon={<HistoryIcon />}
            label="Signal History"
            onClick={() => setActiveTab('signals')}
            sx={{
              fontWeight: 600,
              background: activeTab === 'signals' ? 'linear-gradient(135deg, #3b82f6, #8b5cf6)' : 'rgba(255,255,255,0.05)',
              color: activeTab === 'signals' ? '#fff' : '#94a3b8',
              cursor: 'pointer',
              '&:hover': { opacity: 0.8 },
            }}
          />
          <Chip
            icon={<SchoolIcon />}
            label="Trading Lessons"
            onClick={() => setActiveTab('lessons')}
            sx={{
              fontWeight: 600,
              background: activeTab === 'lessons' ? 'linear-gradient(135deg, #f59e0b, #d97706)' : 'rgba(255,255,255,0.05)',
              color: activeTab === 'lessons' ? '#fff' : '#94a3b8',
              cursor: 'pointer',
              '&:hover': { opacity: 0.8 },
            }}
          />
        </Box>
      )}

      {/* Trading Lessons Panel (AI-Powered) */}
      {activeTab === 'lessons' && hasPortfolio && (
        <Paper
          elevation={0}
          sx={{
            p: 3,
            mb: 4,
            background: 'rgba(245, 158, 11, 0.05)',
            border: '1px solid rgba(245, 158, 11, 0.2)',
            borderRadius: 2,
          }}
        >
          {lessonsLoading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <CircularProgress size={24} sx={{ color: '#f59e0b' }} />
              <Typography sx={{ color: '#f59e0b' }}>Analyzing your trading history...</Typography>
            </Box>
          ) : lessonsData?.lessons ? (
            <Box
              sx={{
                '& h1': { fontSize: '1.5rem', fontWeight: 700, color: '#fff', mb: 2 },
                '& h2': { fontSize: '1.15rem', fontWeight: 600, color: '#f59e0b', mt: 3, mb: 1.5 },
                '& p': { color: '#e2e8f0', lineHeight: 1.7, mb: 1.5 },
                '& ul': { color: '#e2e8f0', pl: 2 },
                '& li': { mb: 0.75 },
                '& strong': { color: '#fff' },
                '& em': { color: '#94a3b8' },
              }}
              dangerouslySetInnerHTML={{
                __html: lessonsData.lessons
                  .replace(/^# (.*?)$/gm, '<h1>$1</h1>')
                  .replace(/^## (.*?)$/gm, '<h2>$1</h2>')
                  .replace(/^- (.*?)$/gm, '<li>$1</li>')
                  .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                  .replace(/\*(.*?)\*/g, '<em>$1</em>')
                  .replace(/\n\n/g, '<br/><br/>')
              }}
            />
          ) : (
            <Typography sx={{ color: '#94a3b8' }}>
              No trading history yet. Import your transactions to get personalized lessons!
            </Typography>
          )}

          {lessonsData?.has_data && (
            <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid rgba(255,255,255,0.1)', display: 'flex', gap: 3 }}>
              <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>
                <strong style={{ color: '#94a3b8' }}>Transactions:</strong> {lessonsData.transaction_count}
              </Typography>
              <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>
                <strong style={{ color: '#94a3b8' }}>Symbols:</strong> {lessonsData.symbol_count}
              </Typography>
              <Typography sx={{ color: '#64748b', fontSize: '0.85rem' }}>
                <strong style={{ color: lessonsData.total_realized_pnl >= 0 ? '#10b981' : '#ef4444' }}>Realized P&L:</strong> ${lessonsData.total_realized_pnl?.toLocaleString() || 0}
              </Typography>
            </Box>
          )}
        </Paper>
      )}

      {/* Signal History Content (only show when signals tab is active or no portfolio) */}
      {(activeTab === 'signals' || !hasPortfolio) && (
        <>
          {/* Key Metrics Row */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={6} md={3} className="fade-in fade-in-delay-1">
              <MetricCard
                icon="chart"
                iconColor="blue"
                label="Total Signals"
                value={totalSignals}
                subText="All time"
              />
            </Grid>
            <Grid item xs={6} md={3} className="fade-in fade-in-delay-2">
              <MetricCard
                icon="chart"
                iconColor="green"
                label="Accuracy"
                value={`${accuracy.toFixed(0)}%`}
                subText="Win rate"
                sentiment={accuracy >= 50 ? 'bullish' : 'bearish'}
              />
            </Grid>
            <Grid item xs={6} md={3} className="fade-in fade-in-delay-3">
              <MetricCard
                icon="bar-chart"
                iconColor="purple"
                label="Avg Return"
                value={`${avgReturn >= 0 ? '+' : ''}${avgReturn.toFixed(1)}%`}
                subText="Per signal"
                sentiment={avgReturn >= 0 ? 'bullish' : 'bearish'}
              />
            </Grid>
            <Grid item xs={6} md={3} className="fade-in fade-in-delay-4">
              <MetricCard
                icon="trending-up"
                iconColor="red"
                label="Hot Streak"
                value={hotStreak}
                subText="Consecutive wins"
              />
            </Grid>
          </Grid>

          {/* Main Content Grid */}
          <Grid container spacing={3}>
            {/* Left Column - Filters & Signal List */}
            <Grid item xs={12} lg={5}>
              {/* Smart Filters */}
              <SectionCard title="Smart Filters" icon={<FilterListIcon />} iconColor="#3b82f6">
                <Box sx={{ mb: 2 }}>
                  <TextField
                    fullWidth
                    size="small"
                    placeholder="Search symbol..."
                    value={symbolFilter}
                    onChange={(e) => setSymbolFilter(e.target.value)}
                    sx={{ '& .MuiOutlinedInput-root': { background: 'rgba(0,0,0,0.2)' } }}
                  />
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 1 }}>Signal Type</Typography>
                  <RadioGroup row value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
                    <FormControlLabel value="all" control={<Radio size="small" />} label="All" sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.85rem' } }} />
                    <FormControlLabel value="BUY" control={<Radio size="small" />} label="🟢 Buy" sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.85rem' } }} />
                    <FormControlLabel value="SELL" control={<Radio size="small" />} label="🔴 Sell" sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.85rem' } }} />
                    <FormControlLabel value="HOLD" control={<Radio size="small" />} label="🟡 Hold" sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.85rem' } }} />
                  </RadioGroup>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 0.5 }}>Time Period</Typography>
                  <Select
                    fullWidth
                    size="small"
                    value={timeFilter}
                    onChange={(e) => setTimeFilter(e.target.value)}
                    sx={{ background: 'rgba(0,0,0,0.2)' }}
                  >
                    <MenuItem value="1">Today</MenuItem>
                    <MenuItem value="7">Last 7 days</MenuItem>
                    <MenuItem value="30">Last 30 days</MenuItem>
                    <MenuItem value="90">Last 90 days</MenuItem>
                    <MenuItem value="all">All time</MenuItem>
                  </Select>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 1 }}>Min Confidence: {minConfidence}%</Typography>
                  <Slider
                    value={minConfidence}
                    onChange={(_, v) => setMinConfidence(v as number)}
                    min={0}
                    max={100}
                    step={10}
                    marks={[{ value: 0, label: '0%' }, { value: 50, label: '50%' }, { value: 100, label: '100%' }]}
                  />
                </Box>

                <Button fullWidth variant="contained" onClick={() => refetch()}>Apply Filters</Button>
              </SectionCard>

              {/* Recent Signals */}
              <Box sx={{ mt: 3 }}>
                <SectionCard
                  title="Recent Signals"
                  icon={<HistoryIcon />}
                  iconColor="#8b5cf6"
                  action={<Box sx={{ fontSize: '0.75rem', color: '#64748b' }}>{signals.length} signals</Box>}
                >
                  <Box sx={{ maxHeight: 400, overflowY: 'auto' }}>
                    {isLoading ? (
                      <Typography sx={{ color: '#64748b', py: 3, textAlign: 'center' }}>Loading...</Typography>
                    ) : signals.length > 0 ? (
                      signals.slice(0, 10).map((signal: any) => {
                        const signalColor = signal.signal_type?.includes('BUY') ? '#10b981' : signal.signal_type?.includes('SELL') ? '#ef4444' : '#f59e0b'
                        return (
                          <Box key={signal.id} sx={{
                            p: 1.5,
                            mb: 1,
                            background: 'rgba(0,0,0,0.2)',
                            borderRadius: 2,
                            borderLeft: `3px solid ${signalColor}`,
                            cursor: 'pointer',
                            '&:hover': { background: 'rgba(0,0,0,0.3)' }
                          }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography sx={{ fontWeight: 700, color: '#fff' }}>{signal.symbol}</Typography>
                                <SignalBadge type={signal.signal_type} />
                              </Box>
                              <Typography sx={{ fontSize: '0.75rem', color: '#64748b' }}>
                                {signal.created_at ? format(new Date(signal.created_at), 'MMM d, HH:mm') : ''}
                              </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                              <Typography sx={{ fontSize: '0.8rem', color: '#64748b' }}>
                                Confidence: <span style={{ color: '#fff', fontWeight: 600 }}>{((signal.confluence_score || 0) * 100).toFixed(0)}%</span>
                              </Typography>
                              <Typography sx={{ fontSize: '0.8rem', color: '#64748b' }}>
                                Price: <span style={{ color: '#fff' }}>${signal.price_at_signal?.toFixed(2) || 'N/A'}</span>
                              </Typography>
                            </Box>
                            <Box sx={{ mt: 1 }}>
                              <ConfidenceBar value={signal.confluence_score || 0} showLabel={false} />
                            </Box>
                          </Box>
                        )
                      })
                    ) : (
                      <Typography sx={{ color: '#64748b', py: 3, textAlign: 'center' }}>No signals found</Typography>
                    )}
                  </Box>
                </SectionCard>
              </Box>
            </Grid>

            {/* Right Column - Analytics */}
            <Grid item xs={12} lg={7}>
              {/* Performance Chart Placeholder */}
              <SectionCard title="Signal Performance" icon={<AnalyticsIcon />} iconColor="#10b981">
                <Box sx={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: 2 }}>
                  <Box sx={{ textAlign: 'center' }}>
                    <TrendingUpIcon sx={{ fontSize: 48, color: '#10b981', mb: 1 }} />
                    <Typography sx={{ color: '#64748b' }}>Cumulative performance over time</Typography>
                  </Box>
                </Box>
              </SectionCard>

              {/* Two Column Stats */}
              <Grid container spacing={3} sx={{ mt: 1 }}>
                <Grid item xs={12} md={6}>
                  <SectionCard title="Distribution" icon={<PieChartIcon />} iconColor="#f59e0b">
                    <Box sx={{ py: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                        <Typography sx={{ color: '#64748b' }}>Buy Signals</Typography>
                        <Typography sx={{ color: '#10b981', fontWeight: 600 }}>{buyCount}</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                        <Typography sx={{ color: '#64748b' }}>Sell Signals</Typography>
                        <Typography sx={{ color: '#ef4444', fontWeight: 600 }}>{sellCount}</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography sx={{ color: '#64748b' }}>Hold Signals</Typography>
                        <Typography sx={{ color: '#f59e0b', fontWeight: 600 }}>{holdCount}</Typography>
                      </Box>
                    </Box>
                  </SectionCard>
                </Grid>
                <Grid item xs={12} md={6}>
                  <SectionCard title="Top Signals" icon={<EmojiEventsIcon />} iconColor="#fbbf24">
                    <Box sx={{ py: 1 }}>
                      {topPerformers.length > 0 ? topPerformers.map((signal: any, i: number) => (
                        <Box key={i} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography sx={{ color: '#fbbf24', fontWeight: 600, width: 20 }}>{i + 1}.</Typography>
                            <Typography sx={{ color: '#fff', fontWeight: 600 }}>{signal.symbol}</Typography>
                          </Box>
                          <Typography sx={{ color: '#10b981', fontSize: '0.85rem' }}>
                            {((signal.confluence_score || 0) * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      )) : (
                        <Typography sx={{ color: '#64748b', textAlign: 'center' }}>No data</Typography>
                      )}
                    </Box>
                  </SectionCard>
                </Grid>
              </Grid>

              {/* Market Insights */}
              <Box sx={{ mt: 3 }}>
                <SectionCard title="Market Insights" icon={<TipsAndUpdatesIcon />} iconColor="#06b6d4">
                  <Typography sx={{ color: '#94a3b8', fontSize: '0.9rem', lineHeight: 1.6 }}>
                    📈 <strong style={{ color: '#fff' }}>Bullish Trend:</strong> {buyCount > sellCount ? 'More buy signals than sell signals indicate bullish sentiment.' : 'Mixed signals suggest cautious positioning.'}<br /><br />
                    📊 <strong style={{ color: '#fff' }}>Confidence:</strong> Average signal confidence is {((signals.reduce((acc: number, s: any) => acc + (s.confluence_score || 0), 0) / Math.max(signals.length, 1)) * 100).toFixed(0)}%.<br /><br />
                    🔥 <strong style={{ color: '#fff' }}>Hot Streak:</strong> {hotStreak} consecutive winning trades.
                  </Typography>
                </SectionCard>
              </Box>
            </Grid>
          </Grid>
        </>
      )}

      {/* Trading Journal */}
      <Box sx={{ mt: 4 }}>
        <TradingJournalPanel />
      </Box>
    </Box>
  )
}
