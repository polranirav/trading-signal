/**
 * Pricing Page - Modern Tech Design
 */

import { Box, Container, Typography, Grid, Card, CardContent, Button, Chip, List, ListItem, ListItemIcon, ListItemText, useTheme, alpha, Stack, Paper } from '@mui/material';
import { Link } from 'react-router-dom';
import CheckIcon from '@mui/icons-material/Check';
import BoltIcon from '@mui/icons-material/Bolt';
import SendIcon from '@mui/icons-material/Send';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';

// Premium Button Component (Copied from Landing.tsx for consistency)
const PremiumButton = ({ children, variant = "contained", ...props }: any) => {
  const theme = useTheme();
  return (
    <Button
      variant={variant}
      {...props}
      sx={{
        py: 1.5,
        px: 4,
        fontSize: '1rem',
        fontWeight: 700,
        borderRadius: '8px',
        textTransform: 'none',
        position: 'relative',
        overflow: 'hidden',
        transition: 'all 0.3s ease',
        ...(variant === 'contained' && {
          background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
          color: 'white',
          boxShadow: `0 4px 20px ${alpha(theme.palette.primary.main, 0.4)}`,
          '&:hover': {
            boxShadow: `0 8px 30px ${alpha(theme.palette.primary.main, 0.6)}`,
            transform: 'translateY(-2px)',
          },
          '&::after': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            background: 'linear-gradient(rgba(255,255,255,0.2), transparent)',
            opacity: 0,
            transition: 'opacity 0.2s',
          },
          '&:hover::after': {
            opacity: 1,
          }
        }),
        ...(variant === 'outlined' && {
          border: `1px solid ${alpha(theme.palette.common.white, 0.2)}`,
          color: 'white',
          background: alpha(theme.palette.common.white, 0.02),
          backdropFilter: 'blur(10px)',
          '&:hover': {
            border: `1px solid ${theme.palette.primary.light}`,
            background: alpha(theme.palette.primary.main, 0.1),
          }
        }),
        ...props.sx
      }}
    >
      {children}
    </Button>
  );
};

// Modern Tech Navbar (Copied from Landing.tsx for consistency)
const Navbar = () => {
  const theme = useTheme();
  return (
    <Box
      sx={{
        py: 2,
        px: 3,
        position: 'sticky',
        top: 0,
        zIndex: 1000,
        backdropFilter: 'blur(20px)',
        backgroundColor: alpha(theme.palette.background.default, 0.8),
        borderBottom: `1px solid ${alpha(theme.palette.common.white, 0.05)}`,
      }}
    >
      <Container maxWidth="xl">
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Link to="/" style={{ textDecoration: 'none' }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Box sx={{
                p: 0.5,
                borderRadius: 1,
                bgcolor: alpha(theme.palette.primary.main, 0.1),
                color: 'primary.main',
                display: 'flex'
              }}>
                <BoltIcon />
              </Box>
              <Typography variant="h6" fontWeight="bold" sx={{ color: 'common.white', letterSpacing: -0.5 }}>
                TRADING<Typography component="span" variant="h6" sx={{ color: 'primary.main', fontWeight: 'bold' }}>PRO</Typography>
              </Typography>
            </Stack>
          </Link>

          <Stack direction="row" spacing={4} sx={{ display: { xs: 'none', md: 'flex' } }}>
            {['Features', 'Pricing'].map((item) => (
              <Link key={item} to={`/${item.toLowerCase()}`} style={{ textDecoration: 'none' }}>
                <Typography variant="button" sx={{
                  color: 'text.secondary',
                  fontWeight: 500,
                  transition: 'color 0.2s',
                  '&:hover': { color: 'common.white' }
                }}>
                  {item}
                </Typography>
              </Link>
            ))}
          </Stack>

          <Stack direction="row" spacing={2} alignItems="center">
            <Button component={Link} to="/login" variant="text" sx={{ color: 'text.secondary', '&:hover': { color: 'white' } }}>
              Log In
            </Button>
            <PremiumButton component={Link} to="/register" endIcon={<ArrowForwardIcon fontSize="small" />}>
              Get Started
            </PremiumButton>
          </Stack>
        </Stack>
      </Container>
    </Box>
  );
};

export default function PricingPage() {
  const theme = useTheme();

  const gradientText = (theme: any) => ({
    background: `linear-gradient(135deg, ${theme.palette.common.white} 0%, ${theme.palette.primary.light} 100%)`,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  });

  const tiers = [
    {
      name: 'Starter',
      price: 'Free',
      period: 'forever',
      description: 'Essential tools for casual analysis',
      features: [
        '10 AI Signals / Day',
        'Basic Charts',
        'Daily Market Summary',
        'Community Access',
      ],
      buttonText: 'Start Free',
      buttonVariant: 'outlined' as const,
    },
    {
      name: 'Pro Trader',
      price: '$49',
      period: 'month',
      description: 'Advanced intelligence for active traders',
      popular: true,
      features: [
        'Unlimited AI Signals',
        'Real-time Latency (<50ms)',
        'Sentiment Analysis Engine',
        'Risk Metrics (VaR)',
        'Copy Trading API',
        'Priority Support',
      ],
      buttonText: 'Start 14-Day Trial',
      buttonVariant: 'contained' as const,
    },
    {
      name: 'Enterprise',
      price: 'Custom',
      period: 'quote',
      description: 'Dedicated infrastructure for funds',
      features: [
        'Everything in Pro',
        'Dedicated GPU Nodes',
        'Custom Strategy Development',
        'White Label Reports',
        'SLA Guarantee',
        '24/7 Account Manager',
      ],
      buttonText: 'Contact Sales',
      buttonVariant: 'outlined' as const,
    },
  ];

  return (
    <Box sx={{ bgcolor: 'background.default', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar />

      <Box sx={{
        py: 10,
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Ambient Back Glow */}
        <Box
          sx={{
            position: 'absolute',
            top: '20%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '1200px',
            height: '800px',
            background: `radial-gradient(circle, ${alpha(theme.palette.primary.dark, 0.15)} 0%, transparent 70%)`,
            filter: 'blur(80px)',
            zIndex: 0,
          }}
        />

        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <Box sx={{ textAlign: 'center', mb: 10 }}>
            <Chip
              label="SUBSCRIPTION PLANS"
              sx={{
                mb: 3,
                bgcolor: alpha(theme.palette.primary.main, 0.1),
                color: 'primary.main',
                fontWeight: 'bold',
                border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`
              }}
            />
            <Typography
              variant="h2"
              component="h1"
              gutterBottom
              sx={{ fontWeight: 800, mb: 2 }}
            >
              Scalable <span style={gradientText(theme)}>Intelligence</span>
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
              Choose the power you need. Upgrade or downgrade at any time.
              Transparent pricing, no hidden fees.
            </Typography>
          </Box>

          <Grid container spacing={4} alignItems="center">
            {tiers.map((tier, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Paper
                  elevation={tier.popular ? 24 : 1}
                  sx={{
                    position: 'relative',
                    height: tier.popular ? '110%' : '100%',
                    border: tier.popular ? `1px solid ${alpha(theme.palette.primary.main, 0.5)}` : `1px solid ${alpha(theme.palette.common.white, 0.1)}`,
                    background: tier.popular ? alpha(theme.palette.background.paper, 0.6) : alpha(theme.palette.background.paper, 0.3),
                    backdropFilter: 'blur(20px)',
                    borderRadius: 4,
                    transition: 'all 0.3s ease',
                    boxShadow: tier.popular ? `0 0 40px -10px ${alpha(theme.palette.primary.main, 0.3)}` : 'none',
                    '&:hover': {
                      transform: 'translateY(-10px)',
                      borderColor: theme.palette.primary.main,
                      boxShadow: `0 0 50px -10px ${alpha(theme.palette.primary.main, 0.4)}`
                    },
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'visible'
                  }}
                >
                  {tier.popular && (
                    <Box
                      sx={{
                        position: 'absolute',
                        top: -16,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                        color: 'common.white',
                        fontWeight: 'bold',
                        borderRadius: 20,
                        px: 2,
                        py: 0.5,
                        fontSize: '0.75rem',
                        boxShadow: `0 4px 20px ${alpha(theme.palette.primary.main, 0.4)}`,
                        zIndex: 2
                      }}
                    >
                      MOST POPULAR
                    </Box>
                  )}

                  <CardContent sx={{ p: 4, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="h5" component="h2" gutterBottom fontWeight={700} color="common.white">
                      {tier.name}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'baseline', mb: 1 }}>
                      <Typography variant="h3" component="span" fontWeight={800} color="common.white">
                        {tier.price}
                      </Typography>
                      {tier.price !== 'Custom' && (
                        <Typography variant="subtitle1" color="text.secondary">
                          /{tier.period}
                        </Typography>
                      )}
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
                      {tier.description}
                    </Typography>

                    <List sx={{ mb: 4, flexGrow: 1 }}>
                      {tier.features.map((feature, idx) => (
                        <ListItem key={idx} disablePadding sx={{ py: 1 }}>
                          <ListItemIcon sx={{ minWidth: 32 }}>
                            <CheckIcon sx={{ color: tier.popular ? 'primary.main' : 'text.secondary', fontSize: 20 }} />
                          </ListItemIcon>
                          <ListItemText
                            primary={feature}
                            primaryTypographyProps={{
                              variant: 'body2',
                              color: 'text.primary',
                              fontWeight: 500
                            }}
                          />
                        </ListItem>
                      ))}
                    </List>

                    <PremiumButton
                      fullWidth
                      variant={tier.buttonVariant}
                      component={Link}
                      to="/register"
                    >
                      {tier.buttonText}
                    </PremiumButton>

                  </CardContent>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>
    </Box>
  );
}
