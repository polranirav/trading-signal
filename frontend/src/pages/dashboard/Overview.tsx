/**
 * Dashboard Overview Page
 */

import { useState, useEffect } from 'react';
import { Container, Typography, Grid, Card, CardContent, Box } from '@mui/material';
import { signalsService } from '../../services/signals';
import type { Signal } from '../../types';

export default function Overview() {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSignals();
  }, []);

  const fetchSignals = async () => {
    try {
      setLoading(true);
      const response = await signalsService.getSignals({ limit: 20 });
      if (response.success && response.data?.signals) {
        setSignals(response.data.signals);
      }
    } catch (error) {
      console.error('Failed to fetch signals:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Dashboard Overview
      </Typography>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Total Signals
              </Typography>
              <Typography variant="h4">{signals.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Buy Signals
              </Typography>
              <Typography variant="h4">
                {signals.filter(s => s.signal_type?.includes('BUY')).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Sell Signals
              </Typography>
              <Typography variant="h4">
                {signals.filter(s => s.signal_type?.includes('SELL')).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mt: 4 }}>
        <Typography variant="h5" gutterBottom>
          Recent Signals
        </Typography>
        {loading ? (
          <Typography>Loading...</Typography>
        ) : (
          <Box>
            {signals.length === 0 ? (
              <Typography>No signals found</Typography>
            ) : (
              signals.map((signal) => (
                <Card key={signal.id} sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="h6">{signal.symbol}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {signal.signal_type} - Confidence: {signal.confluence_score?.toFixed(2)}
                    </Typography>
                  </CardContent>
                </Card>
              ))
            )}
          </Box>
        )}
      </Box>
    </Container>
  );
}
