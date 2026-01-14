import {
  Container,
  Box,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  AppBar,
  Toolbar,
} from '@mui/material'
import { Link } from 'react-router-dom'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import SpeedIcon from '@mui/icons-material/Speed'
import SecurityIcon from '@mui/icons-material/Security'

export default function LandingPage() {
  return (
    <Box>
      {/* Navigation */}
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Trading Signals Pro
          </Typography>
          <Button color="inherit" component={Link} to="/features">
            Features
          </Button>
          <Button color="inherit" component={Link} to="/pricing">
            Pricing
          </Button>
          <Button color="inherit" component={Link} to="/about">
            About
          </Button>
          <Button color="inherit" component={Link} to="/login" sx={{ ml: 1 }}>
            Login
          </Button>
          <Button variant="contained" component={Link} to="/register" sx={{ ml: 1 }}>
            Sign Up
          </Button>
        </Toolbar>
      </AppBar>

      {/* Hero Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          py: 10,
          color: 'white',
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={7}>
              <Typography variant="h2" component="h1" gutterBottom fontWeight={700}>
                AI-Powered Trading Signals
              </Typography>
              <Typography variant="h5" sx={{ mb: 4, opacity: 0.9 }}>
                Professional-grade trading signals powered by machine learning, technical
                analysis, and sentiment analysis. Make data-driven trading decisions.
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  size="large"
                  component={Link}
                  to="/register"
                  sx={{ bgcolor: 'white', color: 'primary.main', '&:hover': { bgcolor: 'grey.100' } }}
                >
                  Start Free Trial
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  component={Link}
                  to="/pricing"
                  sx={{ borderColor: 'white', color: 'white', '&:hover': { borderColor: 'white', bgcolor: 'rgba(255,255,255,0.1)' } }}
                >
                  View Pricing
                </Button>
              </Box>
              <Box sx={{ mt: 3, display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                <Typography variant="body2">✓ No credit card required</Typography>
                <Typography variant="body2">✓ 7-day free trial</Typography>
                <Typography variant="body2">✓ Cancel anytime</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={5}>
              <Box sx={{ textAlign: 'center' }}>
                {/* Placeholder for dashboard preview image */}
                <Box
                  sx={{
                    bgcolor: 'rgba(255,255,255,0.1)',
                    borderRadius: 2,
                    p: 4,
                    minHeight: 400,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography>Dashboard Preview</Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ py: 10 }}>
        <Typography variant="h3" component="h2" align="center" gutterBottom fontWeight={700}>
          Why Choose Trading Signals Pro?
        </Typography>
        <Grid container spacing={4} sx={{ mt: 2 }}>
          {[
            {
              icon: <TrendingUpIcon sx={{ fontSize: 48 }} />,
              title: 'AI-Powered Analysis',
              description:
                'Advanced machine learning models analyze market patterns, sentiment, and technical indicators to generate high-confidence signals.',
            },
            {
              icon: <SpeedIcon sx={{ fontSize: 48 }} />,
              title: 'Real-Time Signals',
              description:
                'Get instant notifications when high-probability trading opportunities are detected. Never miss a signal with email and dashboard alerts.',
            },
            {
              icon: <SecurityIcon sx={{ fontSize: 48 }} />,
              title: 'Risk Management',
              description:
                'Every signal includes comprehensive risk metrics: VaR, CVaR, stop-loss, take-profit, and position sizing recommendations.',
            },
          ].map((feature, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center', p: 4 }}>
                  <Box sx={{ color: 'primary.main', mb: 2 }}>{feature.icon}</Box>
                  <Typography variant="h5" gutterBottom fontWeight={600}>
                    {feature.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* CTA Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          py: 8,
          color: 'white',
        }}
      >
        <Container maxWidth="md">
          <Typography variant="h3" component="h2" align="center" gutterBottom fontWeight={700}>
            Ready to Get Started?
          </Typography>
          <Typography variant="h6" align="center" sx={{ mb: 4, opacity: 0.9 }}>
            Join thousands of traders using AI-powered signals to make better trading decisions.
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
            <Button
              variant="contained"
              size="large"
              component={Link}
              to="/register"
              sx={{ bgcolor: 'white', color: 'primary.main', '&:hover': { bgcolor: 'grey.100' } }}
            >
              Start Your Free Trial
            </Button>
            <Button
              variant="outlined"
              size="large"
              component={Link}
              to="/pricing"
              sx={{ borderColor: 'white', color: 'white', '&:hover': { borderColor: 'white', bgcolor: 'rgba(255,255,255,0.1)' } }}
            >
              View Pricing
            </Button>
          </Box>
        </Container>
      </Box>
    </Box>
  )
}
