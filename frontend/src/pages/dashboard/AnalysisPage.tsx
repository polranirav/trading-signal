/**
 * Analysis Page
 * 
 * Detailed signal analysis with portfolio integration.
 */

import { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { useSearchParams, Link } from 'react-router-dom';
import { api } from '../../services/api';
import { format } from 'date-fns';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import AllInclusiveIcon from '@mui/icons-material/AllInclusive';
import { usePortfolio } from '../../context';

export default function AnalysisPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlSymbol = searchParams.get('symbol') || '';

  // Portfolio context
  const { portfolioSymbols, hasPortfolio, isInPortfolio, isLoaded: portfolioLoaded } = usePortfolio();

  // View mode: "portfolio" or "all"
  const [viewMode, setViewMode] = useState<'portfolio' | 'all'>('portfolio');

  // Initialize symbol from URL on mount
  const [symbol, setSymbol] = useState(() => urlSymbol.toUpperCase());
  const [signalType, setSignalType] = useState('');
  const [minConfidence, setMinConfidence] = useState(0);
  const [days, setDays] = useState(7);

  // Sync symbol state with URL query params when URL changes
  useEffect(() => {
    const currentUrlSymbol = searchParams.get('symbol') || '';
    if (currentUrlSymbol && currentUrlSymbol.toUpperCase() !== symbol.toUpperCase()) {
      setSymbol(currentUrlSymbol.toUpperCase());
    }
  }, [searchParams]);

  const { data: signalsData, isLoading, refetch } = useQuery({
    queryKey: ['signals', { symbol, signal_type: signalType, min_confidence: minConfidence, days }],
    queryFn: () => api.getSignals({
      symbol: symbol || undefined,
      signal_type: signalType || undefined,
      min_confidence: minConfidence || undefined,
      days,
      limit: 100,
    }),
  });

  // Get all signals
  const allSignals = signalsData?.signals || [];

  // Filter by portfolio if in portfolio mode
  const signals = useMemo(() => {
    if (viewMode === 'portfolio' && hasPortfolio && !symbol) {
      return allSignals.filter((s: any) => portfolioSymbols.includes(s.symbol));
    }
    return allSignals;
  }, [allSignals, viewMode, hasPortfolio, portfolioSymbols, symbol]);

  // Portfolio signal counts
  const portfolioSignals = allSignals.filter((s: any) => portfolioSymbols.includes(s.symbol));

  const handleFilter = () => {
    refetch();
  };

  const handleClear = () => {
    setSymbol('');
    setSignalType('');
    setMinConfidence(0);
    setDays(7);
    setSearchParams({});
  };

  const handleSymbolChange = (value: string) => {
    const upperValue = value.toUpperCase().trim();
    setSymbol(upperValue);

    const newParams = new URLSearchParams(searchParams);
    if (upperValue) {
      newParams.set('symbol', upperValue);
    } else {
      newParams.delete('symbol');
    }
    setSearchParams(newParams, { replace: true });
  };

  return (
    <Box>
      {/* Header with View Mode Toggle */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 4, flexWrap: 'wrap', gap: 2 }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom fontWeight={700}>
            {viewMode === 'portfolio' && !symbol ? 'Portfolio Signal Analysis' : 'Signal Analysis'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {viewMode === 'portfolio' && !symbol
              ? `Analyzing signals for your ${portfolioSymbols.length} portfolio stocks`
              : 'Analyze trading signals with advanced filters and detailed metrics.'
            }
          </Typography>
        </Box>

        {/* View Mode Toggle */}
        {hasPortfolio && (
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              icon={<AccountBalanceWalletIcon />}
              label={`My Portfolio (${portfolioSignals.length})`}
              onClick={() => setViewMode('portfolio')}
              sx={{
                fontWeight: 600,
                background: viewMode === 'portfolio'
                  ? 'linear-gradient(135deg, #3b82f6, #8b5cf6)'
                  : 'rgba(255,255,255,0.05)',
                color: viewMode === 'portfolio' ? '#fff' : '#94a3b8',
                border: viewMode === 'portfolio' ? 'none' : '1px solid rgba(255,255,255,0.1)',
              }}
            />
            <Chip
              icon={<AllInclusiveIcon />}
              label="All Signals"
              onClick={() => setViewMode('all')}
              sx={{
                fontWeight: 600,
                background: viewMode === 'all'
                  ? 'linear-gradient(135deg, #3b82f6, #8b5cf6)'
                  : 'rgba(255,255,255,0.05)',
                color: viewMode === 'all' ? '#fff' : '#94a3b8',
                border: viewMode === 'all' ? 'none' : '1px solid rgba(255,255,255,0.1)',
              }}
            />
          </Box>
        )}
      </Box>

      {/* Import Prompt */}
      {!hasPortfolio && portfolioLoaded && (
        <Alert
          severity="info"
          sx={{
            mb: 3,
            background: 'rgba(59, 130, 246, 0.1)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
          }}
        >
          <Typography>
            💡 <strong>Tip:</strong> Import your portfolio to filter signals for YOUR stocks.{' '}
            <Link to="/dashboard" style={{ color: '#3b82f6', fontWeight: 600 }}>
              Go to Portfolio →
            </Link>
          </Typography>
        </Alert>
      )}

      {/* Filters */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Filters
          </Typography>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                label="Symbol"
                value={symbol}
                onChange={(e) => handleSymbolChange(e.target.value)}
                placeholder="e.g., AAPL, TSLA"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth>
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
                label="Min Confidence"
                type="number"
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
                inputProps={{ min: 0, max: 100, step: 1 }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={2}>
              <TextField
                fullWidth
                label="Days"
                type="number"
                value={days}
                onChange={(e) => setDays(Number(e.target.value))}
                inputProps={{ min: 1, max: 365 }}
              />
            </Grid>
            <Grid item xs={12} md={2}>
              <Box sx={{ display: 'flex', gap: 1, height: '100%', alignItems: 'center' }}>
                <Button variant="contained" onClick={handleFilter} fullWidth>
                  Filter
                </Button>
                <Button variant="outlined" onClick={handleClear} fullWidth>
                  Clear
                </Button>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Results Summary */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Total Signals
              </Typography>
              <Typography variant="h4" fontWeight={700}>
                {signals.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Avg Confidence
              </Typography>
              <Typography variant="h4" fontWeight={700} color="primary.main">
                {signals.length > 0
                  ? ((signals.reduce((sum: number, s: any) => sum + (s.confluence_score || 0), 0) / signals.length) * 100).toFixed(1)
                  : '0'}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Buy Signals
              </Typography>
              <Typography variant="h4" fontWeight={700} color="success.main">
                {signals.filter((s: any) => s.signal_type?.includes('BUY')).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Sell Signals
              </Typography>
              <Typography variant="h4" fontWeight={700} color="error.main">
                {signals.filter((s: any) => s.signal_type?.includes('SELL')).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Signals Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {viewMode === 'portfolio' && !symbol ? 'Portfolio Signals' : 'Signals'}
          </Typography>
          {isLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : signals.length > 0 ? (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Risk-Reward</TableCell>
                    <TableCell>Price</TableCell>
                    <TableCell>Date</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {signals.map((signal: any) => (
                    <TableRow key={signal.id}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {hasPortfolio && isInPortfolio(signal.symbol) && (
                            <Box sx={{
                              width: 6, height: 6, borderRadius: '50%',
                              background: '#3b82f6',
                              flexShrink: 0
                            }} />
                          )}
                          <Typography variant="body1" fontWeight={600}>
                            {signal.symbol}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={signal.signal_type}
                          color={signal.signal_type?.includes('BUY') ? 'success' : signal.signal_type?.includes('SELL') ? 'error' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        {signal.confluence_score ? ((signal.confluence_score * 100).toFixed(1) + '%') : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {signal.risk_reward_ratio ? signal.risk_reward_ratio.toFixed(2) : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {signal.price_at_signal ? `$${signal.price_at_signal.toFixed(2)}` : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {signal.created_at ? format(new Date(signal.created_at), 'MMM d, yyyy HH:mm') : 'N/A'}
                      </TableCell>
                      <TableCell>
                        <Button
                          component={Link}
                          to={`/dashboard/charts?symbol=${signal.symbol}`}
                          variant="outlined"
                          size="small"
                          startIcon={<AnalyticsIcon />}
                        >
                          Chart
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Typography color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
              {viewMode === 'portfolio' && !symbol
                ? 'No signals for your portfolio stocks. Try switching to All Signals view.'
                : 'No signals found'
              }
            </Typography>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}
