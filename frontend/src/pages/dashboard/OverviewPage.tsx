import { useState, useMemo, useEffect } from 'react'
import {
  Box, Typography, Grid, Chip, Alert, TextField,
  Button, Select, MenuItem, FormControl, InputLabel,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  CircularProgress
} from '@mui/material'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, Link, useSearchParams } from 'react-router-dom'
import { format } from 'date-fns'
import { api } from '../../services/api'
import BoltIcon from '@mui/icons-material/Bolt'
import AnalyticsIcon from '@mui/icons-material/Analytics'
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet'
import FilterListIcon from '@mui/icons-material/FilterList'
import SensorsIcon from '@mui/icons-material/Sensors'

import MetricCard from '../../components/MetricCard'
import { SignalBadge, ConfidenceBar, OpportunityCard, SectionCard } from '../../components/SignalComponents'
import { MarketBreadthPanel } from '../../components/MarketBreadthPanel'
import { usePortfolio } from '../../context'
import '../../styles/premium.css'

export default function OverviewPage() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  // Portfolio context
  const { portfolioSymbols, hasPortfolio, isInPortfolio, isLoaded: portfolioLoaded } = usePortfolio()

  // View Mode: 'portfolio' (default if user has one) or 'all'
  const viewMode = hasPortfolio ? 'portfolio' : 'all'

  // --- FILTERS STATE ---
  const urlSymbol = searchParams.get('symbol') || ''
  const [symbol, setSymbol] = useState(() => urlSymbol.toUpperCase())
  const [signalType, setSignalType] = useState('')
  const [minConfidence, setMinConfidence] = useState(0)
  const [days, setDays] = useState(7)

  // Sync symbol state with URL
  useEffect(() => {
    const currentUrlSymbol = searchParams.get('symbol') || ''
    if (currentUrlSymbol && currentUrlSymbol.toUpperCase() !== symbol.toUpperCase()) {
      setSymbol(currentUrlSymbol.toUpperCase())
    }
  }, [searchParams])

  // --- QUERIES ---

  // 1. Overview Query (Top Stats & Opportunities) - Unfiltered (or filtered by portfolio for stats)
  const { data: overviewData, isLoading: isLoadingOverview } = useQuery({
    queryKey: ['overview_signals', { limit: 50 }],
    queryFn: () => api.getSignals({ limit: 50 }),
  })

  // 2. Table Query (Detailed Analysis) - Filtered
  const { data: tableData, isLoading: isLoadingTable, refetch: refetchTable } = useQuery({
    queryKey: ['analysis_signals', { symbol, signal_type: signalType, min_confidence: minConfidence, days }],
    queryFn: () => api.getSignals({
      symbol: symbol || undefined,
      signal_type: signalType || undefined,
      min_confidence: minConfidence || undefined,
      days,
      limit: 100,
    }),
  })

  // --- DERIVED DATA ---

  const allOverviewSignals = overviewData?.signals || []

  // Stats Signals: Filter manually by portfolio if in portfolio mode
  const statsSignals = viewMode === 'portfolio'
    ? allOverviewSignals.filter((s: any) => portfolioSymbols.includes(s.symbol))
    : allOverviewSignals

  const buySignals = statsSignals.filter((s: any) => s.signal_type?.includes('BUY'))
  const sellSignals = statsSignals.filter((s: any) => s.signal_type?.includes('SELL'))

  // Top Opportunities (always derive from the broad stats query to ensure we show best of class)
  const topOpportunities = [...statsSignals]
    .sort((a: any, b: any) => (b.confluence_score || 0) - (a.confluence_score || 0))
    .slice(0, 4) // Show top 4

  // Table Signals: The main list for analysis
  const tableSignalsRaw = tableData?.signals || []

  // Filter table signals by portfolio if in portfolio mode, unless user has searched for a specific symbol
  const finalTableSignals = useMemo(() => {
    // If user specifically typed a symbol, show it regardless of portfolio
    if (symbol) return tableSignalsRaw

    // Otherwise, if in portfolio mode, restrict to portfolio
    if (viewMode === 'portfolio') {
      return tableSignalsRaw.filter((s: any) => portfolioSymbols.includes(s.symbol))
    }

    return tableSignalsRaw
  }, [tableSignalsRaw, viewMode, portfolioSymbols, symbol])

  // --- HANDLERS ---

  const handleFilter = () => {
    refetchTable()
  }

  const handleClear = () => {
    setSymbol('')
    setSignalType('')
    setMinConfidence(0)
    setDays(7)
    setSearchParams({})
  }

  const handleSymbolChange = (value: string) => {
    const upperValue = value.toUpperCase().trim()
    setSymbol(upperValue)

    const newParams = new URLSearchParams(searchParams)
    if (upperValue) {
      newParams.set('symbol', upperValue)
    } else {
      newParams.delete('symbol')
    }
    setSearchParams(newParams, { replace: true })
  }

  return (
    <Box className="fade-in">
      {/* Header Section */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 2 }}>
        <Box>
          <Typography
            variant="h4"
            component="h1"
            sx={{ fontWeight: 700, color: '#fff', fontSize: '1.75rem', mb: 0.5 }}
          >
            {viewMode === 'portfolio' ? 'My Portfolio Dashboard' : 'Market Dashboard'}
          </Typography>
          <Typography sx={{ color: '#64748b', fontSize: '0.9rem' }}>
            {viewMode === 'portfolio'
              ? `Monitoring ${portfolioSymbols.length} active assets in your portfolio`
              : 'Real-time market intelligence and trading signals'
            }
          </Typography>
        </Box>

        {hasPortfolio && (
          <Chip
            icon={<AccountBalanceWalletIcon />}
            label={`${portfolioSymbols.length} Holdings`}
            sx={{
              fontWeight: 600,
              background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              color: '#fff'
            }}
          />
        )}
      </Box>

      {/* Import Portfolio Prompt */}
      {!hasPortfolio && portfolioLoaded && (
        <Alert
          severity="info"
          sx={{
            mb: 3,
            background: 'rgba(59, 130, 246, 0.1)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            '& .MuiAlert-message': { color: '#94a3b8' }
          }}
        >
          <Typography sx={{ color: '#e2e8f0' }}>
            💡 <strong>Tip:</strong> Import your portfolio to see personalized signals.{' '}
            <Link to="/dashboard" style={{ color: '#3b82f6', fontWeight: 600 }}>
              Go to Portfolio →
            </Link>
          </Typography>
        </Alert>
      )}

      {/* Metric Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={6} md={3} className="fade-in fade-in-delay-1">
          <MetricCard
            icon="chart"
            iconColor="blue"
            label={viewMode === 'portfolio' ? 'My Signals' : 'Total Signals'}
            value={statsSignals.length}
            subText={viewMode === 'portfolio' ? 'Actionable items' : 'Market-wide'}
          />
        </Grid>
        <Grid item xs={6} md={3} className="fade-in fade-in-delay-2">
          <MetricCard
            icon="trending-up"
            iconColor="green"
            label="Buy Opportunities"
            value={buySignals.length}
            subText="Bullish setups"
            sentiment="bullish"
          />
        </Grid>
        <Grid item xs={6} md={3} className="fade-in fade-in-delay-3">
          <MetricCard
            icon="trending-down"
            iconColor="red"
            label="Sell Warnings"
            value={sellSignals.length}
            subText="Bearish setups"
            sentiment="bearish"
          />
        </Grid>
        <Grid item xs={6} md={3} className="fade-in fade-in-delay-4">
          <MetricCard
            icon="server"
            iconColor="purple"
            label="Avg Confidence"
            value={
              statsSignals.length > 0
                ? ((statsSignals.reduce((acc: number, s: any) => acc + (s.confluence_score || 0), 0) / statsSignals.length) * 100).toFixed(0) + '%'
                : '0%'
            }
            subText="Signal strength"
          />
        </Grid>
      </Grid>

      {/* Top Opportunities Row */}
      <Box sx={{ mb: 4 }}>
        <SectionCard
          title="Top Opportunities"
          icon={<BoltIcon />}
          iconColor="#f59e0b"
        >
          <Grid container spacing={3}>
            {isLoadingOverview ? (
              <Grid item xs={12}><Typography sx={{ color: '#64748b', textAlign: 'center', py: 4 }}>Loading top picks...</Typography></Grid>
            ) : topOpportunities.length > 0 ? (
              topOpportunities.map((signal: any) => (
                <Grid item xs={12} sm={6} md={3} key={signal.id}>
                  <Box sx={{ cursor: 'pointer', height: '100%' }} onClick={() => navigate(`/dashboard/charts?symbol=${signal.symbol}`)}>
                    <OpportunityCard
                      symbol={signal.symbol}
                      signalType={signal.signal_type}
                      confidence={signal.confluence_score}
                      price={signal.price_at_signal}
                    />
                  </Box>
                </Grid>
              ))
            ) : (
              <Grid item xs={12}>
                <Typography sx={{ color: '#64748b', textAlign: 'center', py: 4 }}>
                  No high-confidence opportunities found right now.
                </Typography>
              </Grid>
            )}
          </Grid>
        </SectionCard>
      </Box>

      {/* Detailed Analysis Section (Merged from AnalysisPage) */}
      <Box sx={{ mb: 4 }}>
        <SectionCard
          title="Market Analysis & Filters"
          icon={<FilterListIcon />}
          iconColor="#3b82f6"
        >
          {/* Filters Bar */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                label="Symbol"
                value={symbol}
                onChange={(e) => handleSymbolChange(e.target.value)}
                placeholder="e.g. CMG, AAPL"
                size="small"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Signal Type</InputLabel>
                <Select
                  value={signalType}
                  onChange={(e) => setSignalType(e.target.value)}
                  label="Signal Type"
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="BUY">Buy</MenuItem>
                  <MenuItem value="SELL">Sell</MenuItem>
                  <MenuItem value="HOLD">Hold</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={2}>
              <TextField
                fullWidth
                label="Min Conf (%)"
                type="number"
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
                size="small"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={2}>
              <TextField
                fullWidth
                label="Days"
                type="number"
                value={days}
                onChange={(e) => setDays(Number(e.target.value))}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={2}>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button variant="contained" onClick={handleFilter} fullWidth size="medium">
                  Filter
                </Button>
                <Button variant="outlined" onClick={handleClear} fullWidth size="medium">
                  Clear
                </Button>
              </Box>
            </Grid>
          </Grid>

          {/* Table */}
          {isLoadingTable ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : finalTableSignals.length > 0 ? (
            <TableContainer>
              <Table className="premium-table">
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>R/R Ratio</TableCell>
                    <TableCell>Price</TableCell>
                    <TableCell>Date</TableCell>
                    <TableCell align="right">Action</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {finalTableSignals.map((signal: any) => (
                    <TableRow key={signal.id} hover>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {hasPortfolio && isInPortfolio(signal.symbol) && (
                            <Box sx={{
                              width: 6, height: 6, borderRadius: '50%',
                              background: '#3b82f6',
                              flexShrink: 0
                            }} />
                          )}
                          <Typography variant="body1" fontWeight={700} color="#fff">
                            {signal.symbol}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <SignalBadge type={signal.signal_type} />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ width: 100 }}>
                          <ConfidenceBar value={signal.confluence_score} />
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography color="#94a3b8">{signal.risk_reward_ratio ? signal.risk_reward_ratio.toFixed(2) : '-'}</Typography>
                      </TableCell>
                      <TableCell>
                        <Typography color="#fff">${signal.price_at_signal ? signal.price_at_signal.toFixed(2) : '-'}</Typography>
                      </TableCell>
                      <TableCell>
                        <Typography color="#64748b" fontSize="0.85rem">
                          {signal.created_at ? format(new Date(signal.created_at), 'MMM d, HH:mm') : '-'}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                          <Button
                            component={Link}
                            to={`/dashboard/signals?symbol=${signal.symbol}`}
                            variant="outlined"
                            size="small"
                            startIcon={<SensorsIcon fontSize="small" />}
                            sx={{
                              textTransform: 'none',
                              borderColor: 'rgba(139, 92, 246, 0.3)',
                              color: '#a78bfa',
                              '&:hover': { borderColor: '#a78bfa', bgcolor: 'rgba(139, 92, 246, 0.1)' }
                            }}
                          >
                            Signals
                          </Button>
                          <Button
                            component={Link}
                            to={`/dashboard/charts?symbol=${signal.symbol}`}
                            variant="outlined"
                            size="small"
                            startIcon={<AnalyticsIcon fontSize="small" />}
                            sx={{ textTransform: 'none', borderColor: 'rgba(255,255,255,0.2)', color: '#94a3b8' }}
                          >
                            Chart
                          </Button>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Box sx={{ textAlign: 'center', py: 6, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
              <Typography color="#64748b">
                {viewMode === 'portfolio' && !symbol
                  ? 'No active signals found for your portfolio. Switch Signal Type or search directly.'
                  : 'No signals match your criteria.'}
              </Typography>
            </Box>
          )}
        </SectionCard>
      </Box>

      {/* Market Breadth Panel */}
      <Box sx={{ mb: 4 }}>
        <MarketBreadthPanel />
      </Box>
    </Box>
  )
}
