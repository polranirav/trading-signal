/**
 * Features Page
 * 
 * Detailed feature list and capabilities.
 */

import { Box, Container, Typography, Grid, Card, CardContent } from '@mui/material';
import { Link } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Button from '@mui/material/Button';

export default function FeaturesPage() {
  const features = [
    {
      category: 'Technical Analysis',
      items: [
        { title: 'RSI (Relative Strength Index)', description: 'Momentum oscillator identifying overbought/oversold conditions' },
        { title: 'MACD (Moving Average Convergence Divergence)', description: 'Trend-following momentum indicator' },
        { title: 'Bollinger Bands', description: 'Volatility bands for price action analysis' },
        { title: 'Multiple SMAs', description: 'Simple Moving Averages across different timeframes' },
        { title: 'ATR (Average True Range)', description: 'Volatility measurement for risk assessment' },
        { title: 'OBV & MFI', description: 'Volume-based indicators for market sentiment' },
      ],
    },
    {
      category: 'Sentiment Analysis',
      items: [
        { title: 'FinBERT Model', description: 'State-of-the-art financial sentiment analysis using transformer models' },
        { title: 'Time-Weighted Sentiment', description: 'Recent news weighted more heavily in analysis' },
        { title: 'LLM Integration', description: 'GPT-4 powered analysis for nuanced market interpretation' },
        { title: 'RAG with Pinecone', description: 'Retrieval-Augmented Generation for context-aware insights' },
      ],
    },
    {
      category: 'Machine Learning',
      items: [
        { title: 'Attention-Based LSTM', description: 'Deep learning models capturing temporal patterns' },
        { title: 'Ensemble Stacking', description: 'Combining multiple models for improved predictions' },
        { title: 'Temporal Fusion Transformer', description: 'Advanced time series forecasting with attention mechanisms' },
        { title: 'Confluence Engine', description: 'Intelligent signal fusion with adaptive weighting' },
      ],
    },
    {
      category: 'Risk Management',
      items: [
        { title: 'Value at Risk (VaR)', description: '95% confidence interval risk measurement' },
        { title: 'Conditional VaR (CVaR)', description: 'Expected shortfall beyond VaR threshold' },
        { title: 'Risk-Reward Ratio', description: 'Expected return vs. potential loss calculation' },
        { title: 'Position Sizing', description: 'Kelly Criterion-inspired optimal position sizing' },
        { title: 'Stop-Loss Recommendations', description: 'Dynamic stop-loss levels based on volatility' },
      ],
    },
    {
      category: 'Platform Features',
      items: [
        { title: 'Real-Time Signals', description: 'Live trading signal generation and alerts' },
        { title: 'Email Notifications', description: 'Instant email alerts for new signals' },
        { title: 'Signal History', description: 'Complete historical record of all signals' },
        { title: 'Performance Tracking', description: 'Win rate, returns, and performance analytics' },
        { title: 'REST API', description: 'Programmatic access with API keys' },
        { title: 'Multiple Subscription Tiers', description: 'Free, Essential, and Advanced plans' },
      ],
    },
  ];

  return (
    <Box>
      <AppBar 
        position="static" 
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
              fontSize: '1.25rem',
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }} 
            component={Link} 
            to="/" 
            style={{ textDecoration: 'none' }}
          >
            Trading Signals Pro
          </Typography>
          <Button 
            color="inherit" 
            component={Link} 
            to="/pricing"
            sx={{ 
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
            to="/login"
            sx={{ 
              mr: 1.5,
              fontWeight: 500,
              '&:hover': { color: 'primary.main' },
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
            Sign Up
          </Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: { xs: 6, md: 10 } }}>
        <Box sx={{ textAlign: 'center', mb: 8 }}>
          <Typography 
            variant="h2" 
            component="h1" 
            gutterBottom 
            sx={{ 
              fontSize: { xs: '2.25rem', md: '3rem' },
              mb: 2,
            }}
          >
            Features
          </Typography>
          <Typography 
            variant="h6" 
            color="text.secondary" 
            sx={{ 
              mb: 6,
              fontSize: { xs: '1rem', md: '1.125rem' },
              maxWidth: 700,
              mx: 'auto',
              lineHeight: 1.7,
            }} 
            align="center"
          >
            Comprehensive trading signal platform with advanced analytics and risk management.
          </Typography>
        </Box>

        {features.map((category, categoryIndex) => (
          <Box key={categoryIndex} sx={{ mb: 8 }}>
            <Typography 
              variant="h3" 
              component="h2" 
              gutterBottom 
              sx={{ 
                mb: 4,
                fontSize: { xs: '1.75rem', md: '2.25rem' },
                position: 'relative',
                display: 'inline-block',
                '&::after': {
                  content: '""',
                  position: 'absolute',
                  bottom: -8,
                  left: 0,
                  width: '60px',
                  height: '4px',
                  background: 'linear-gradient(90deg, #6366f1, #8b5cf6)',
                  borderRadius: 2,
                },
              }}
            >
              {category.category}
            </Typography>
            <Grid container spacing={3}>
              {category.items.map((item, itemIndex) => (
                <Grid item xs={12} md={6} key={itemIndex}>
                  <Card
                    sx={{
                      height: '100%',
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%)',
                      '&:hover': {
                        transform: 'translateY(-4px)',
                        boxShadow: '0 12px 24px rgba(99, 102, 241, 0.15)',
                        '&::before': {
                          opacity: 1,
                        },
                      },
                      position: 'relative',
                      overflow: 'hidden',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        height: '3px',
                        background: 'linear-gradient(90deg, #6366f1, #8b5cf6)',
                        opacity: 0,
                        transition: 'opacity 0.3s',
                      },
                    }}
                  >
                    <CardContent sx={{ p: 3 }}>
                      <Typography 
                        variant="h6" 
                        gutterBottom
                        sx={{ 
                          fontWeight: 700,
                          mb: 1.5,
                        }}
                      >
                        {item.title}
                      </Typography>
                      <Typography 
                        variant="body1" 
                        color="text.secondary"
                        sx={{ 
                          lineHeight: 1.7,
                        }}
                      >
                        {item.description}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        ))}

        <Box 
          sx={{ 
            textAlign: 'center', 
            mt: 10,
            p: 6,
            borderRadius: 4,
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
            border: '1px solid rgba(99, 102, 241, 0.2)',
          }}
        >
          <Typography 
            variant="h4" 
            gutterBottom
            sx={{ 
              fontWeight: 700,
              mb: 2,
            }}
          >
            Ready to Get Started?
          </Typography>
          <Typography 
            variant="body1" 
            color="text.secondary"
            sx={{ 
              mb: 4,
              maxWidth: 500,
              mx: 'auto',
            }}
          >
            Experience the power of AI-powered trading signals. Start your free trial today.
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Button 
              variant="contained" 
              size="large" 
              component={Link} 
              to="/register"
              sx={{
                px: 5,
                py: 1.5,
                fontWeight: 700,
                background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
                },
              }}
            >
              Start Free Trial
            </Button>
            <Button 
              variant="outlined" 
              size="large" 
              component={Link} 
              to="/pricing"
              sx={{
                px: 5,
                py: 1.5,
                fontWeight: 700,
                borderColor: 'primary.main',
                color: 'primary.main',
                '&:hover': {
                  borderColor: 'primary.dark',
                  bgcolor: 'rgba(99, 102, 241, 0.1)',
                },
              }}
            >
              View Pricing
            </Button>
          </Box>
        </Box>
      </Container>
    </Box>
  );
}
