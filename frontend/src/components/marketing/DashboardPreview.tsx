/**
 * Dashboard Preview Component
 * 
 * Shows a preview of the dashboard interface.
 */

import { Box, Typography, Card, CardContent, Grid, Chip } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

export default function DashboardPreview() {
  // Mock signal data for preview
  const previewSignals = [
    { symbol: 'AAPL', type: 'BUY', confidence: 87, price: 185.32, change: '+2.3%' },
    { symbol: 'MSFT', type: 'BUY', confidence: 82, price: 378.45, change: '+1.8%' },
    { symbol: 'GOOGL', type: 'HOLD', confidence: 65, price: 142.56, change: '-0.5%' },
    { symbol: 'TSLA', type: 'SELL', confidence: 78, price: 248.12, change: '-3.2%' },
  ];

  const getSignalColor = (type: string) => {
    if (type === 'BUY') return 'success';
    if (type === 'SELL') return 'warning'; // Using warning (amber) instead of error (red)
    return 'default';
  };

  return (
    <Box
      sx={{
        bgcolor: 'background.paper',
        borderRadius: 3,
        overflow: 'hidden',
        boxShadow: 4,
        border: 1,
        borderColor: 'divider',
      }}
    >
      {/* Dashboard Header */}
      <Box
        sx={{
          bgcolor: 'primary.main',
          color: 'primary.contrastText',
          p: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Typography variant="h6" fontWeight={600}>
          Trading Signals Dashboard
        </Typography>
        <Typography variant="caption" sx={{ opacity: 0.9 }}>
          Live Preview
        </Typography>
      </Box>

      {/* Stats Cards */}
      <Box sx={{ p: 2, bgcolor: 'background.default' }}>
        <Grid container spacing={2}>
          <Grid item xs={6} sm={3}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Typography variant="caption" color="text.secondary">
                  Today's Signals
                </Typography>
                <Typography variant="h5" fontWeight={700}>
                  24
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Typography variant="caption" color="text.secondary">
                  Win Rate
                </Typography>
                <Typography variant="h5" fontWeight={700} color="success.main">
                  73%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Typography variant="caption" color="text.secondary">
                  Watchlist
                </Typography>
                <Typography variant="h5" fontWeight={700}>
                  12
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Typography variant="caption" color="text.secondary">
                  Portfolio
                </Typography>
                <Typography variant="h5" fontWeight={700} color="success.main">
                  +18.5%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>

      {/* Signals List */}
      <Box sx={{ p: 2 }}>
        <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
          Latest Signals
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          {previewSignals.map((signal, index) => (
            <Box
              key={index}
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                p: 2,
                bgcolor: 'background.default',
                borderRadius: 2,
                border: 1,
                borderColor: 'divider',
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flex: 1 }}>
                <Box>
                  <Typography variant="h6" fontWeight={600}>
                    {signal.symbol}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    ${signal.price}
                  </Typography>
                </Box>
                <Chip
                  label={signal.type}
                  color={getSignalColor(signal.type) as any}
                  size="small"
                  sx={{ fontWeight: 600 }}
                />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  {signal.change.startsWith('+') ? (
                    <TrendingUpIcon fontSize="small" color="success" />
                  ) : (
                    <TrendingDownIcon fontSize="small" color="warning" />
                  )}
                  <Typography
                    variant="body2"
                    color={signal.change.startsWith('+') ? 'success.main' : 'warning.main'}
                    fontWeight={600}
                  >
                    {signal.change}
                  </Typography>
                </Box>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Typography variant="body2" color="text.secondary">
                  Confidence
                </Typography>
                <Typography variant="h6" fontWeight={700} color="primary.main">
                  {signal.confidence}%
                </Typography>
              </Box>
            </Box>
          ))}
        </Box>
      </Box>

      {/* Footer Note */}
      <Box
        sx={{
          bgcolor: 'background.default',
          borderTop: 1,
          borderColor: 'divider',
          p: 1.5,
          textAlign: 'center',
        }}
      >
        <Typography variant="caption" color="text.secondary">
          This is a preview. Actual dashboard may vary.
        </Typography>
      </Box>
    </Box>
  );
}
