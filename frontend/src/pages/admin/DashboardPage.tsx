/**
 * Admin Dashboard Page
 * 
 * Overview statistics and platform metrics.
 */

import { Box, Typography, Grid, Card, CardContent, CircularProgress } from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { adminService } from '../../services/admin';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import PeopleIcon from '@mui/icons-material/People';
import SignalCellularAltIcon from '@mui/icons-material/SignalCellularAlt';
import CardMembershipIcon from '@mui/icons-material/CardMembership';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

export default function DashboardPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['admin-dashboard-stats'],
    queryFn: () => adminService.getDashboardStats(),
  });

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error || !data?.data) {
    return (
      <Box>
        <Typography variant="h5" color="error">
          Failed to load dashboard statistics
        </Typography>
      </Box>
    );
  }

  const stats = data.data;

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom fontWeight={700}>
        Admin Dashboard
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
        Platform overview and key metrics
      </Typography>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <PeopleIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6" color="text.secondary">
                  Total Users
                </Typography>
              </Box>
              <Typography variant="h3" fontWeight={700}>
                {stats.users.total}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {stats.users.active} active
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SignalCellularAltIcon sx={{ mr: 1, color: 'success.main' }} />
                <Typography variant="h6" color="text.secondary">
                  Total Signals
                </Typography>
              </Box>
              <Typography variant="h3" fontWeight={700}>
                {stats.signals.total.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {stats.signals.today} today
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <CardMembershipIcon sx={{ mr: 1, color: 'warning.main' }} />
                <Typography variant="h6" color="text.secondary">
                  Subscriptions
                </Typography>
              </Box>
              <Typography variant="h3" fontWeight={700}>
                {stats.subscriptions.active}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {stats.subscriptions.total} total
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUpIcon sx={{ mr: 1, color: 'info.main' }} />
                <Typography variant="h6" color="text.secondary">
                  Signups (30d)
                </Typography>
              </Box>
              <Typography variant="h3" fontWeight={700}>
                {stats.users.signups_last_30_days}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {stats.users.admins} admins
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        {/* Signups Trend */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Signups (Last 7 Days)
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={stats.users.signups_by_day}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} />
                  <YAxis />
                  <Tooltip labelFormatter={(date) => new Date(date).toLocaleDateString()} />
                  <Line type="monotone" dataKey="count" stroke="#667eea" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Subscription Tier Breakdown */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Subscription Tiers
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={Object.entries(stats.subscriptions.tier_breakdown).map(([tier, count]) => ({ tier, count }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="tier" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#667eea" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
