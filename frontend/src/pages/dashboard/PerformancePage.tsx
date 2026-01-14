/**
 * Performance Page
 * 
 * Analyze signal performance with metrics and charts.
 */

import { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  TextField,
  Button,
  CircularProgress,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { api } from '../../services/api';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

export default function PerformancePage() {
  const [days, setDays] = useState(30);

  const { data: signalsData, isLoading } = useQuery({
    queryKey: ['signals', 'performance', { days }],
    queryFn: () => api.getSignals({ days, limit: 1000 }),
  });

  const signals = signalsData?.signals || [];

  // Calculate metrics
  const executedSignals = signals.filter(s => s.is_executed && s.realized_pnl_pct !== null && s.realized_pnl_pct !== undefined);
  const winningSignals = executedSignals.filter(s => (s.realized_pnl_pct || 0) > 0);
  const winRate = executedSignals.length > 0 ? (winningSignals.length / executedSignals.length) * 100 : 0;
  const avgReturn = executedSignals.length > 0 
    ? executedSignals.reduce((sum, s) => sum + (s.realized_pnl_pct || 0), 0) / executedSignals.length 
    : 0;
  const totalReturn = executedSignals.reduce((sum, s) => sum + (s.realized_pnl_pct || 0), 0);

  // Cumulative returns data
  const cumulativeReturns = executedSignals.reduce((acc, signal, index) => {
    const prevReturn = index === 0 ? 0 : acc[index - 1].return;
    const newReturn = prevReturn + (signal.realized_pnl_pct || 0);
    acc.push({
      date: signal.created_at ? new Date(signal.created_at).toLocaleDateString() : '',
      return: newReturn,
    });
    return acc;
  }, [] as { date: string; return: number }[]);

  // Signal type distribution
  const signalDistribution = [
    { name: 'Buy', value: signals.filter(s => s.signal_type?.includes('BUY')).length },
    { name: 'Sell', value: signals.filter(s => s.signal_type?.includes('SELL')).length },
    { name: 'Hold', value: signals.filter(s => s.signal_type?.includes('HOLD')).length },
  ].filter(item => item.value > 0);

  // Returns distribution (bins)
  const returnsBins = [-10, -5, 0, 5, 10, 15, 20];
  const returnsDistribution = returnsBins.slice(0, -1).map((bin, index) => {
    const min = bin;
    const max = returnsBins[index + 1];
    const count = executedSignals.filter(s => {
      const pnl = s.realized_pnl_pct || 0;
      return pnl >= min && pnl < max;
    }).length;
    return { range: `${min}% - ${max}%`, count };
  });

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom fontWeight={700}>
        Performance Tracking
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
        Analyze signal performance, win rates, and returns over time.
      </Typography>

      {/* Time Period Filter */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <TextField
              label="Days"
              type="number"
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              inputProps={{ min: 1, max: 365 }}
              sx={{ width: 150 }}
            />
            <Button variant="contained" onClick={() => window.location.reload()}>
              Update
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Metrics Summary */}
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
                Executed Signals
              </Typography>
              <Typography variant="h4" fontWeight={700}>
                {executedSignals.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Win Rate
              </Typography>
              <Typography variant="h4" fontWeight={700} color={winRate >= 50 ? 'success.main' : 'error.main'}>
                {winRate.toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Avg Return
              </Typography>
              <Typography variant="h4" fontWeight={700} color={avgReturn >= 0 ? 'success.main' : 'error.main'}>
                {avgReturn.toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Total Return
              </Typography>
              <Typography variant="h4" fontWeight={700} color={totalReturn >= 0 ? 'success.main' : 'error.main'}>
                {totalReturn.toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Winning Signals
              </Typography>
              <Typography variant="h4" fontWeight={700} color="success.main">
                {winningSignals.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Grid container spacing={3}>
          {/* Cumulative Returns */}
          {cumulativeReturns.length > 0 && (
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Cumulative Returns
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={cumulativeReturns}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="return" stroke="#667eea" strokeWidth={2} name="Cumulative Return (%)" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Signal Distribution */}
          {signalDistribution.length > 0 && (
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Signal Distribution
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={signalDistribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {signalDistribution.map((_entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Returns Distribution */}
          {returnsDistribution.some(r => r.count > 0) && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Returns Distribution
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={returnsDistribution}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="range" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" fill="#667eea" name="Number of Signals" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          )}

          {signals.length === 0 && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
                    No signals found for the selected period
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}
    </Box>
  );
}
