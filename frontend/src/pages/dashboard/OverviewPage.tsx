import { useState } from 'react'
import { Box, Typography, Grid, Chip, Alert } from '@mui/material'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../../services/api'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import StorageIcon from '@mui/icons-material/Storage'
import BoltIcon from '@mui/icons-material/Bolt'
import AnalyticsIcon from '@mui/icons-material/Analytics'
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet'
import AllInclusiveIcon from '@mui/icons-material/AllInclusive'

import MetricCard from '../../components/MetricCard'
import { SignalBadge, ConfidenceBar, OpportunityCard, SectionCard } from '../../components/SignalComponents'
import { usePortfolio } from '../../context'
import '../../styles/premium.css'

export default function OverviewPage() {
  // Portfolio context - get user's holdings
  const { portfolioSymbols, holdings, hasPortfolio, isLoaded: portfolioLoaded } = usePortfolio()

  // View mode: "portfolio" or "all"
  const [viewMode, setViewMode] = useState<'portfolio' | 'all'>(hasPortfolio ? 'portfolio' : 'all')

  const { data: signalsData, isLoading } = useQuery({
    queryKey: ['signals', { limit: 50 }],
    queryFn: () => api.getSignals({ limit: 50 }),
  })

  const allSignals = signalsData?.signals || []

  // Filter signals based on view mode
  const signals = viewMode === 'portfolio' && hasPortfolio
    ? allSignals.filter((s: any) => portfolioSymbols.includes(s.symbol))
    : allSignals

  const buySignals = signals.filter((s: any) => s.signal_type?.includes('BUY'))
  const sellSignals = signals.filter((s: any) => s.signal_type?.includes('SELL'))
  const holdSignals = signals.filter((s: any) => s.signal_type === 'HOLD')

  // Get top opportunities (highest confidence signals)
  const topOpportunities = [...signals]
    .sort((a: any, b: any) => (b.confluence_score || 0) - (a.confluence_score || 0))
    .slice(0, 5)

  // Portfolio signals count
  const portfolioSignalsCount = allSignals.filter((s: any) => portfolioSymbols.includes(s.symbol)).length

  return (
    <Box className="fade-in">
      {/* Header Section */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 2 }}>
        <Box>
          <Typography
            variant="h4"
            component="h1"
            sx={{
              fontWeight: 700,
              color: '#fff',
              fontSize: '1.75rem',
              mb: 0.5
            }}
          >
            {viewMode === 'portfolio' ? 'My Portfolio Signals' : 'Market Overview'}
          </Typography>
          <Typography sx={{ color: '#64748b', fontSize: '0.9rem' }}>
            {viewMode === 'portfolio'
              ? `Signals for your ${portfolioSymbols.length} portfolio stocks`
              : 'Real-time signal monitoring and market sentiment'
            }
          </Typography>
        </Box>

        {/* View Mode Toggle */}
        {hasPortfolio && (
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              icon={<AccountBalanceWalletIcon />}
              label="My Portfolio"
              onClick={() => setViewMode('portfolio')}
              sx={{
                fontWeight: 600,
                background: viewMode === 'portfolio'
                  ? 'linear-gradient(135deg, #3b82f6, #8b5cf6)'
                  : 'rgba(255,255,255,0.05)',
                color: viewMode === 'portfolio' ? '#fff' : '#94a3b8',
                border: viewMode === 'portfolio' ? 'none' : '1px solid rgba(255,255,255,0.1)',
                '&:hover': { opacity: 0.9 }
              }}
            />
            <Chip
              icon={<AllInclusiveIcon />}
              label="All Market"
              onClick={() => setViewMode('all')}
              sx={{
                fontWeight: 600,
                background: viewMode === 'all'
                  ? 'linear-gradient(135deg, #3b82f6, #8b5cf6)'
                  : 'rgba(255,255,255,0.05)',
                color: viewMode === 'all' ? '#fff' : '#94a3b8',
                border: viewMode === 'all' ? 'none' : '1px solid rgba(255,255,255,0.1)',
                '&:hover': { opacity: 0.9 }
              }}
            />
          </Box>
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
            💡 <strong>Tip:</strong> Import your portfolio to see personalized signals for YOUR stocks.{' '}
            <Link to="/dashboard" style={{ color: '#3b82f6', fontWeight: 600 }}>
              Go to Portfolio →
            </Link>
          </Typography>
        </Alert>
      )}

      {/* Premium Metric Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={6} md={3} className="fade-in fade-in-delay-1">
          <MetricCard
            icon="chart"
            iconColor="blue"
            label={viewMode === 'portfolio' ? 'Portfolio Signals' : 'Total Signals'}
            value={viewMode === 'portfolio' ? portfolioSignalsCount : signalsData?.count || 0}
            subText={viewMode === 'portfolio' ? `From ${portfolioSymbols.length} stocks` : `${signals.length} in view`}
          />
        </Grid>
        <Grid item xs={6} md={3} className="fade-in fade-in-delay-2">
          <MetricCard
            icon="trending-up"
            iconColor="green"
            label="Buy Signals"
            value={buySignals.length}
            subText="Active opportunities"
            sentiment="bullish"
          />
        </Grid>
        <Grid item xs={6} md={3} className="fade-in fade-in-delay-3">
          <MetricCard
            icon="trending-down"
            iconColor="red"
            label="Sell Signals"
            value={sellSignals.length}
            subText="Risk warnings"
            sentiment="bearish"
          />
        </Grid>
        <Grid item xs={6} md={3} className="fade-in fade-in-delay-4">
          <MetricCard
            icon="server"
            iconColor="purple"
            label="System Status"
            value={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box className="status-dot online" />
                <span style={{ fontSize: '1.25rem', fontWeight: 600 }}>Online</span>
              </Box>
            }
            subText={`Updated: ${new Date().toLocaleTimeString()}`}
          />
        </Grid>
      </Grid>

      {/* Main Content Grid */}
      <Grid container spacing={3}>
        {/* Left Column - Top Opportunities */}
        <Grid item xs={12} lg={7}>
          <SectionCard
            title={viewMode === 'portfolio' ? 'My Top Opportunities' : 'Top Opportunities'}
            icon={<BoltIcon />}
            iconColor="#f59e0b"
          >
            {isLoading ? (
              <Typography sx={{ color: '#64748b', py: 4, textAlign: 'center' }}>
                Loading signals...
              </Typography>
            ) : topOpportunities.length > 0 ? (
              topOpportunities.map((signal: any) => (
                <OpportunityCard
                  key={signal.id}
                  symbol={signal.symbol}
                  signalType={signal.signal_type}
                  confidence={signal.confluence_score}
                  price={signal.price_at_signal}
                />
              ))
            ) : (
              <Typography sx={{ color: '#64748b', py: 4, textAlign: 'center' }}>
                {viewMode === 'portfolio'
                  ? 'No signals for your portfolio stocks. Try adding more stocks or switch to All Market view.'
                  : 'No opportunities found'
                }
              </Typography>
            )}
          </SectionCard>
        </Grid>

        {/* Right Column - Signal Feed */}
        <Grid item xs={12} lg={5}>
          <SectionCard
            title={viewMode === 'portfolio' ? 'My Signal Feed' : 'Signal Feed'}
            icon={<AnalyticsIcon />}
            iconColor="#10b981"
          >
            <Box sx={{ maxHeight: 400, overflowY: 'auto', pr: 1 }}>
              {isLoading ? (
                <Typography sx={{ color: '#64748b', py: 4, textAlign: 'center' }}>
                  Loading...
                </Typography>
              ) : signals.length > 0 ? (
                <table className="premium-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Signal</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {signals.slice(0, 10).map((signal: any) => (
                      <tr key={signal.id}>
                        <td>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {hasPortfolio && portfolioSymbols.includes(signal.symbol) && (
                              <Box sx={{
                                width: 6, height: 6, borderRadius: '50%',
                                background: '#3b82f6',
                                flexShrink: 0
                              }} />
                            )}
                            <Box>
                              <Typography sx={{ fontWeight: 700, color: '#fff' }}>
                                {signal.symbol}
                              </Typography>
                              <Typography sx={{ fontSize: '0.75rem', color: '#64748b' }}>
                                ${signal.price_at_signal?.toFixed(2) || 'N/A'}
                              </Typography>
                            </Box>
                          </Box>
                        </td>
                        <td>
                          <SignalBadge type={signal.signal_type} />
                        </td>
                        <td>
                          <Box sx={{ width: 80 }}>
                            <ConfidenceBar value={signal.confluence_score} />
                          </Box>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <Typography sx={{ color: '#64748b', py: 4, textAlign: 'center' }}>
                  {viewMode === 'portfolio'
                    ? 'No signals for portfolio stocks'
                    : 'No signals available'
                  }
                </Typography>
              )}
            </Box>
          </SectionCard>
        </Grid>
      </Grid>

      {/* Quick Stats Row */}
      <Grid container spacing={3} sx={{ mt: 3 }}>
        <Grid item xs={12} md={4}>
          <Box className="glass-card" sx={{ p: 2 }}>
            <Typography sx={{ fontSize: '0.8rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, mb: 1 }}>
              Avg Confidence
            </Typography>
            <Typography sx={{ fontSize: '1.5rem', fontWeight: 700, color: '#fff' }}>
              {signals.length > 0
                ? (signals.reduce((acc: number, s: any) => acc + (s.confluence_score || 0), 0) / signals.length * 100).toFixed(1)
                : 0}%
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={12} md={4}>
          <Box className="glass-card" sx={{ p: 2 }}>
            <Typography sx={{ fontSize: '0.8rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, mb: 1 }}>
              Hold Signals
            </Typography>
            <Typography sx={{ fontSize: '1.5rem', fontWeight: 700, color: '#f59e0b' }}>
              {holdSignals.length}
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={12} md={4}>
          <Box className="glass-card" sx={{ p: 2 }}>
            <Typography sx={{ fontSize: '0.8rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, mb: 1 }}>
              Strong Signals
            </Typography>
            <Typography sx={{ fontSize: '1.5rem', fontWeight: 700, color: '#3b82f6' }}>
              {signals.filter((s: any) => s.signal_type?.includes('STRONG')).length}
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Box>
  )
}
