/**
 * Pricing Page
 * 
 * Subscription tiers and pricing information.
 */

import { Box, Container, Typography, Grid, Card, CardContent, Button, Chip, List, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import { Link } from 'react-router-dom';
import CheckIcon from '@mui/icons-material/Check';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';

export default function PricingPage() {
  const tiers = [
    {
      name: 'Free',
      price: '$0',
      period: 'month',
      description: 'Perfect for getting started',
      features: [
        '10 signals per day',
        'Basic technical indicators',
        'Email notifications',
        '7-day signal history',
        'Community support',
      ],
      limits: {
        signals: 10,
        apiCalls: 100,
      },
      buttonText: 'Start Free',
      buttonVariant: 'outlined' as const,
    },
    {
      name: 'Essential',
      price: '$29',
      period: 'month',
      description: 'For active traders',
      popular: true,
      features: [
        '100 signals per day',
        'All technical indicators',
        'Sentiment analysis',
        'ML predictions',
        '30-day signal history',
        'Risk metrics (VaR, CVaR)',
        '1,000 API calls/day',
        'Email & priority support',
      ],
      limits: {
        signals: 100,
        apiCalls: 1000,
      },
      buttonText: 'Start Free Trial',
      buttonVariant: 'contained' as const,
    },
    {
      name: 'Advanced',
      price: '$99',
      period: 'month',
      description: 'For professional traders',
      features: [
        'Unlimited signals',
        'All features included',
        'Advanced ML models (TFT)',
        'Unlimited history',
        'Advanced risk metrics',
        '10,000 API calls/day',
        'Custom integrations',
        'Priority & phone support',
      ],
      limits: {
        signals: -1,
        apiCalls: 10000,
      },
      buttonText: 'Start Free Trial',
      buttonVariant: 'contained' as const,
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
            to="/features"
            sx={{ 
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
            Pricing
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
            Choose the plan that's right for you. All plans include a free trial.
          </Typography>
        </Box>

        <Grid container spacing={4} sx={{ mb: 8 }}>
          {tiers.map((tier, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  position: 'relative',
                  ...(tier.popular && {
                    border: 2,
                    borderColor: 'primary.main',
                  }),
                }}
              >
                {tier.popular && (
                  <Chip
                    label="Most Popular"
                    color="primary"
                    sx={{ position: 'absolute', top: 16, right: 16 }}
                  />
                )}
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h5" component="h2" gutterBottom fontWeight={700}>
                    {tier.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    {tier.description}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'baseline', mb: 3 }}>
                    <Typography variant="h3" component="span" fontWeight={700}>
                      {tier.price}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                      /{tier.period}
                    </Typography>
                  </Box>
                  <List>
                    {tier.features.map((feature, featureIndex) => (
                      <ListItem key={featureIndex} disablePadding sx={{ py: 0.5 }}>
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          <CheckIcon color="success" fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={feature} />
                      </ListItem>
                    ))}
                  </List>
                  <Button
                    variant={tier.buttonVariant}
                    fullWidth
                    size="large"
                    component={Link}
                    to="/register"
                    sx={{ mt: 3 }}
                  >
                    {tier.buttonText}
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="h5" gutterBottom>
            Questions about pricing?
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Contact us for enterprise pricing or custom solutions.
          </Typography>
          <Button variant="outlined" component={Link} to="/about">
            Learn More
          </Button>
        </Box>
      </Container>
    </Box>
  );
}
