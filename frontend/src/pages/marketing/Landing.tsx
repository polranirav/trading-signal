/**
 * Enhanced Landing Page (Marketing)
 * 
 * Improved landing page with trust signals, testimonials, and better UX.
 */

import { Link } from 'react-router-dom';
import {
  Container,
  Typography,
  Button,
  Box,
  Grid,
  Card,
  CardContent,
  AppBar,
  Toolbar,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import SpeedIcon from '@mui/icons-material/Speed';
import SecurityIcon from '@mui/icons-material/Security';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

// Components
import TrustSignals from '../../components/marketing/TrustSignals';
import Testimonials from '../../components/marketing/Testimonials';
import DashboardPreview from '../../components/marketing/DashboardPreview';
import FAQ from '../../components/marketing/FAQ';

export default function Landing() {
  return (
    <Box>
      {/* Navigation */}
      <AppBar 
        position="sticky" 
        elevation={0} 
        sx={{ 
          bgcolor: 'rgba(15, 23, 42, 0.8)',
          backdropFilter: 'blur(20px)',
          borderBottom: 1, 
          borderColor: 'divider',
        }}
      >
        <Toolbar sx={{ py: 1 }}>
          <Typography 
            variant="h6" 
            sx={{ 
              flexGrow: 1, 
              fontWeight: 800, 
              color: 'text.primary',
              fontSize: '1.25rem',
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            Trading Signals Pro
          </Typography>
          <Button 
            color="inherit" 
            component={Link} 
            to="/features" 
            sx={{ 
              color: 'text.primary', 
              mr: 1,
              fontWeight: 500,
              '&:hover': { color: 'primary.main' },
            }}
          >
            Features
          </Button>
          <Button 
            color="inherit" 
            component={Link} 
            to="/pricing" 
            sx={{ 
              color: 'text.primary', 
              mr: 1,
              fontWeight: 500,
              '&:hover': { color: 'primary.main' },
            }}
          >
            Pricing
          </Button>
          <Button 
            color="inherit" 
            component={Link} 
            to="/about" 
            sx={{ 
              color: 'text.primary', 
              mr: 2,
              fontWeight: 500,
              '&:hover': { color: 'primary.main' },
            }}
          >
            About
          </Button>
          <Button 
            component={Link} 
            to="/login" 
            variant="outlined" 
            sx={{ 
              mr: 1.5,
              borderColor: 'rgba(148, 163, 184, 0.3)',
              color: 'text.primary',
              '&:hover': {
                borderColor: 'primary.main',
                bgcolor: 'rgba(99, 102, 241, 0.1)',
              },
            }}
          >
            Login
          </Button>
          <Button 
            component={Link} 
            to="/register" 
            variant="contained"
            sx={{
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
              },
            }}
          >
            Sign Up Free
          </Button>
        </Toolbar>
      </AppBar>

      {/* Trust Signals Bar */}
      <TrustSignals />

      {/* Hero Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%)',
          color: 'white',
          py: { xs: 10, md: 16 },
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(255,255,255,0.1) 0%, transparent 50%)',
            pointerEvents: 'none',
          },
        }}
      >
        {/* Animated Background Pattern */}
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            opacity: 0.15,
            backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)',
            backgroundSize: '50px 50px',
            animation: 'float 20s ease-in-out infinite',
            '@keyframes float': {
              '0%, 100%': { transform: 'translateY(0px)' },
              '50%': { transform: 'translateY(-20px)' },
            },
          }}
        />
        
        {/* Gradient Orbs */}
        <Box
          sx={{
            position: 'absolute',
            width: '600px',
            height: '600px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%)',
            top: '-300px',
            right: '-300px',
            animation: 'pulse 8s ease-in-out infinite',
            '@keyframes pulse': {
              '0%, 100%': { transform: 'scale(1)', opacity: 0.5 },
              '50%': { transform: 'scale(1.1)', opacity: 0.8 },
            },
          }}
        />
        <Box
          sx={{
            position: 'absolute',
            width: '400px',
            height: '400px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%)',
            bottom: '-200px',
            left: '-200px',
            animation: 'pulse 6s ease-in-out infinite',
            animationDelay: '1s',
          }}
        />
        
        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography 
                variant="h1" 
                component="h1" 
                gutterBottom 
                sx={{ 
                  fontSize: { xs: '2.75rem', md: '4rem', lg: '4.5rem' },
                  lineHeight: 1.1,
                  mb: 3,
                  background: 'linear-gradient(135deg, #ffffff 0%, rgba(255,255,255,0.9) 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                Turn Market Data Into{' '}
                <Box component="span" sx={{ 
                  background: 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}>
                  Profitable Trades
                </Box>
              </Typography>
              <Typography 
                variant="h5" 
                sx={{ 
                  mb: 4, 
                  opacity: 0.95, 
                  fontWeight: 400,
                  fontSize: { xs: '1.1rem', md: '1.25rem' },
                  lineHeight: 1.6,
                }}
              >
                AI-powered trading signals with <strong>73% win rate</strong>. Join <strong>1,247+ traders</strong> using machine learning to make better trading decisions.
              </Typography>
              
              {/* Key Benefits */}
              <Box sx={{ mb: 4 }}>
                {[
                  'AI-powered analysis with 73% accuracy',
                  'Real-time signals delivered instantly',
                  'Comprehensive risk metrics included',
                  'No credit card required for trial',
                ].map((benefit, index) => (
                  <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 1.5 }}>
                    <CheckCircleIcon sx={{ mr: 1.5, fontSize: 24, opacity: 0.9 }} />
                    <Typography variant="body1" sx={{ opacity: 0.95 }}>
                      {benefit}
                    </Typography>
                  </Box>
                ))}
              </Box>

              {/* CTAs */}
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 4 }}>
                <Button
                  component={Link}
                  to="/register"
                  variant="contained"
                  size="large"
                  sx={{
                    bgcolor: 'white',
                    color: '#6366f1',
                    px: 5,
                    py: 1.75,
                    fontWeight: 700,
                    fontSize: '1.1rem',
                    borderRadius: 3,
                    boxShadow: '0 10px 30px rgba(0,0,0,0.3)',
                    '&:hover': { 
                      bgcolor: 'rgba(255,255,255,0.95)', 
                      transform: 'translateY(-3px)',
                      boxShadow: '0 15px 40px rgba(0,0,0,0.4)',
                    },
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  }}
                >
                  Start Free Trial
                </Button>
                <Button
                  component={Link}
                  to="/pricing"
                  variant="outlined"
                  size="large"
                  sx={{
                    borderColor: 'rgba(255,255,255,0.5)',
                    borderWidth: 2,
                    color: 'white',
                    px: 5,
                    py: 1.75,
                    fontWeight: 700,
                    fontSize: '1.1rem',
                    borderRadius: 3,
                    backdropFilter: 'blur(10px)',
                    bgcolor: 'rgba(255,255,255,0.1)',
                    '&:hover': {
                      borderColor: 'white',
                      bgcolor: 'rgba(255,255,255,0.2)',
                      transform: 'translateY(-3px)',
                      boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
                    },
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  }}
                >
                  View Pricing
                </Button>
              </Box>

              {/* Trust Indicators */}
              <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap', alignItems: 'center' }}>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  ✓ 7-day free trial
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  ✓ Cancel anytime
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  ✓ No credit card required
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Box sx={{ position: 'relative' }}>
                <DashboardPreview />
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ py: 10 }}>
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Typography variant="h3" component="h2" gutterBottom fontWeight={700}>
            Why Choose Trading Signals Pro?
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
            Everything you need to make data-driven trading decisions with confidence
          </Typography>
        </Box>

        <Grid container spacing={4}>
          {[
            {
              icon: <AnalyticsIcon sx={{ fontSize: 48 }} />,
              title: 'AI-Powered Analysis',
              description:
                'Advanced machine learning models analyze market patterns, sentiment, and technical indicators to generate high-confidence signals with comprehensive risk metrics.',
            },
            {
              icon: <SpeedIcon sx={{ fontSize: 48 }} />,
              title: 'Real-Time Signals',
              description:
                'Get instant notifications when high-probability trading opportunities are detected. Never miss a signal with email alerts and dashboard updates.',
            },
            {
              icon: <SecurityIcon sx={{ fontSize: 48 }} />,
              title: 'Risk Management',
              description:
                'Every signal includes comprehensive risk metrics: VaR, CVaR, stop-loss, take-profit, and position sizing recommendations to help you manage risk effectively.',
            },
            {
              icon: <TrendingUpIcon sx={{ fontSize: 48 }} />,
              title: 'Proven Results',
              description:
                'Our signals have a 73% win rate with an average 23% better returns compared to market benchmarks. Join thousands of successful traders.',
            },
          ].map((feature, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card
                sx={{
                  height: '100%',
                  textAlign: 'center',
                  p: 4,
                  position: 'relative',
                  overflow: 'hidden',
                  background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%)',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: '0 20px 40px rgba(99, 102, 241, 0.2)',
                    '& .feature-icon': {
                      transform: 'scale(1.1) rotate(5deg)',
                    },
                  },
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '4px',
                    background: 'linear-gradient(90deg, #6366f1, #8b5cf6)',
                    opacity: 0,
                    transition: 'opacity 0.3s',
                  },
                  '&:hover::before': {
                    opacity: 1,
                  },
                }}
              >
                <Box 
                  className="feature-icon"
                  sx={{ 
                    color: 'primary.main', 
                    mb: 3, 
                    display: 'flex', 
                    justifyContent: 'center',
                    transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  }}
                >
                  {feature.icon}
                </Box>
                <Typography variant="h5" gutterBottom fontWeight={700} sx={{ mb: 2 }}>
                  {feature.title}
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8, fontSize: '0.95rem' }}>
                  {feature.description}
                </Typography>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* Testimonials Section */}
      <Testimonials />

      {/* How It Works Section */}
      <Box sx={{ py: { xs: 8, md: 12 }, bgcolor: 'background.default' }}>
        <Container maxWidth="lg">
          <Box sx={{ textAlign: 'center', mb: 8 }}>
            <Typography 
              variant="h2" 
              component="h2" 
              gutterBottom 
              sx={{ 
                fontSize: { xs: '2.25rem', md: '3rem' },
                mb: 2,
              }}
            >
              How It Works
            </Typography>
            <Typography 
              variant="h6" 
              color="text.secondary" 
              sx={{ 
                maxWidth: 700, 
                mx: 'auto',
                fontSize: { xs: '1rem', md: '1.125rem' },
                lineHeight: 1.7,
              }}
            >
              Get started in minutes and start receiving high-quality trading signals
            </Typography>
          </Box>

          <Grid container spacing={4}>
            {[
              {
                step: '1',
                title: 'Sign Up Free',
                description: 'Create your account in seconds. No credit card required. Start with our free tier or 7-day premium trial.',
              },
              {
                step: '2',
                title: 'Set Your Preferences',
                description: 'Choose your risk tolerance, preferred sectors, and trading style. Our system personalizes signals for you.',
              },
              {
                step: '3',
                title: 'Add Stocks to Watchlist',
                description: 'Select the stocks you want to monitor. Our AI analyzes them in real-time for trading opportunities.',
              },
              {
                step: '4',
                title: 'Receive Signals',
                description: 'Get instant notifications when high-confidence signals are detected. Each signal includes detailed analysis and risk metrics.',
              },
            ].map((item, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Box 
                  sx={{ 
                    textAlign: 'center',
                    p: 3,
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                    },
                  }}
                >
                  <Box
                    sx={{
                      width: 80,
                      height: 80,
                      borderRadius: '50%',
                      background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                      color: 'white',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '2rem',
                      fontWeight: 800,
                      mx: 'auto',
                      mb: 3,
                      boxShadow: '0 8px 24px rgba(99, 102, 241, 0.3)',
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      '&:hover': {
                        transform: 'scale(1.1)',
                        boxShadow: '0 12px 32px rgba(99, 102, 241, 0.4)',
                      },
                    }}
                  >
                    {item.step}
                  </Box>
                  <Typography 
                    variant="h5" 
                    gutterBottom 
                    fontWeight={700}
                    sx={{ mb: 2 }}
                  >
                    {item.title}
                  </Typography>
                  <Typography 
                    variant="body1" 
                    color="text.secondary"
                    sx={{ 
                      lineHeight: 1.7,
                      fontSize: '0.95rem',
                    }}
                  >
                    {item.description}
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* FAQ Section */}
      <FAQ />

      {/* Final CTA Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%)',
          color: 'white',
          py: { xs: 10, md: 14 },
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%)',
            pointerEvents: 'none',
          },
        }}
      >
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            opacity: 0.15,
            backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)',
            backgroundSize: '50px 50px',
          }}
        />
        
        <Container maxWidth="md" sx={{ position: 'relative', zIndex: 1 }}>
          <Typography 
            variant="h2" 
            component="h2" 
            align="center" 
            gutterBottom 
            sx={{ 
              fontSize: { xs: '2.25rem', md: '3rem' },
              mb: 2,
            }}
          >
            Ready to Get Started?
          </Typography>
          <Typography 
            variant="h6" 
            align="center" 
            sx={{ 
              mb: 5, 
              opacity: 0.95, 
              fontWeight: 400,
              fontSize: { xs: '1rem', md: '1.125rem' },
              lineHeight: 1.7,
            }}
          >
            Join <strong>1,247+ traders</strong> using AI-powered signals to make better trading decisions. Start your free trial today.
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap', mb: 4 }}>
            <Button
              component={Link}
              to="/register"
              variant="contained"
              size="large"
              sx={{
                bgcolor: 'white',
                color: '#6366f1',
                px: 6,
                py: 1.75,
                fontWeight: 700,
                fontSize: '1.1rem',
                borderRadius: 3,
                boxShadow: '0 10px 30px rgba(0,0,0,0.3)',
                '&:hover': { 
                  bgcolor: 'rgba(255,255,255,0.95)', 
                  transform: 'translateY(-3px)',
                  boxShadow: '0 15px 40px rgba(0,0,0,0.4)',
                },
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              }}
            >
              Start Your Free Trial
            </Button>
            <Button
              component={Link}
              to="/pricing"
              variant="outlined"
              size="large"
              sx={{
                borderColor: 'rgba(255,255,255,0.5)',
                borderWidth: 2,
                color: 'white',
                px: 6,
                py: 1.75,
                fontWeight: 700,
                fontSize: '1.1rem',
                borderRadius: 3,
                backdropFilter: 'blur(10px)',
                bgcolor: 'rgba(255,255,255,0.1)',
                '&:hover': {
                  borderColor: 'white',
                  bgcolor: 'rgba(255,255,255,0.2)',
                  transform: 'translateY(-3px)',
                  boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
                },
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              }}
            >
              View Pricing Plans
            </Button>
          </Box>
          <Typography variant="body1" align="center" sx={{ opacity: 0.9, fontSize: '0.95rem' }}>
            No credit card required • 7-day free trial • Cancel anytime
          </Typography>
        </Container>
      </Box>
    </Box>
  );
}
