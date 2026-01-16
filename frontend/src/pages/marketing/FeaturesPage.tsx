/**
 * Features Page - Modern Tech Design
 */

import { Box, Container, Typography, Grid, Card, CardContent, Button, useTheme, alpha, Stack, Chip, Paper } from '@mui/material';
import { Link } from 'react-router-dom';
import BoltIcon from '@mui/icons-material/Bolt';
import QueryStatsIcon from '@mui/icons-material/QueryStats';
import PsychologyIcon from '@mui/icons-material/Psychology';
import ShieldIcon from '@mui/icons-material/Shield';
import SendIcon from '@mui/icons-material/Send';
import DataObjectIcon from '@mui/icons-material/DataObject';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import CandlestickChartIcon from '@mui/icons-material/CandlestickChart';
import HubIcon from '@mui/icons-material/Hub';

// Premium Button Component (Consistent)
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

// Navbar (Consistent)
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

export default function FeaturesPage() {
  const theme = useTheme();

  const features = [
    {
      category: 'Technical Indicators',
      icon: <QueryStatsIcon fontSize="large" />,
      color: theme.palette.primary.main,
      items: [
        { title: 'RSI & Stochastic', description: 'Momentum oscillators identifying overbought/oversold conditions using multi-frame analysis.', icon: <AutoGraphIcon fontSize="small" /> },
        { title: 'MACD Divergence', description: 'Trend-following momentum indicator designed to detect early potential price reversals.', icon: <CandlestickChartIcon fontSize="small" /> },
        { title: 'Bollinger Bands', description: 'Volatility bands for dynamic price action analysis and squeeze detection.', icon: <QueryStatsIcon fontSize="small" /> },
        { title: 'Ichimoku Cloud AI', description: 'Automated cloud breakout detection with lagging span verification.', icon: <BoltIcon fontSize="small" /> },
        { title: 'Fibonacci Auto-Levels', description: 'Real-time support/resistance mapping using golden ratio algorithms.', icon: <DataObjectIcon fontSize="small" /> },
      ],
    },
    {
      category: 'AI & Sentiment',
      icon: <PsychologyIcon fontSize="large" />,
      color: theme.palette.secondary.main,
      items: [
        { title: 'FinBERT Sentiment', description: 'NLP analysis of 50,000+ financial news sources and social media feeds daily.', icon: <HubIcon fontSize="small" /> },
        { title: 'GPT-4 Integration', description: 'LLM-powered reasoning for complex market events and macro-economic correlation.', icon: <PsychologyIcon fontSize="small" /> },
        { title: 'Transformer Models', description: 'Deep learning for non-linear time-series forecasting of crypto assets.', icon: <AutoGraphIcon fontSize="small" /> },
        { title: 'Whale Wallet Tracking', description: 'On-chain analysis alerting you to large institutional movements before they impact price.', icon: <BoltIcon fontSize="small" /> },
      ],
    },
    {
      category: 'Risk Protocols',
      icon: <ShieldIcon fontSize="large" />,
      color: theme.palette.success.main,
      items: [
        { title: 'Smart Stop-Loss', description: 'Dynamic levels based on current market volatility (ATR) to prevent premature exits.', icon: <ShieldIcon fontSize="small" /> },
        { title: 'Value at Risk (VaR)', description: 'Institutional-grade risk assessment probabilities for portfolio management.', icon: <QueryStatsIcon fontSize="small" /> },
        { title: 'Position Sizing', description: 'Kelly Criterion calculator for optimal capital allocation per trade.', icon: <DataObjectIcon fontSize="small" /> },
        { title: 'Correlation Matrix', description: 'Real-time asset correlation warnings to prevent over-exposure.', icon: <HubIcon fontSize="small" /> },
      ],
    },
  ];

  const gradientText = (theme: any) => ({
    background: `linear-gradient(135deg, ${theme.palette.common.white} 0%, ${theme.palette.primary.light} 100%)`,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  });

  return (
    <Box sx={{ bgcolor: 'background.default', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar />

      <Box sx={{ py: 10, position: 'relative', overflow: 'visible' }}>
        {/* Background Glow */}
        <Box
          sx={{
            position: 'absolute',
            top: '0',
            left: '20%',
            width: '100%',
            height: '100%',
            background: `radial-gradient(circle at 50% 30%, ${alpha(theme.palette.secondary.dark, 0.15)} 0%, transparent 60%)`,
            filter: 'blur(100px)',
            zIndex: 0,
            pointerEvents: 'none',
          }}
        />

        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <Box sx={{ textAlign: 'center', mb: 10 }}>
            <Chip
              label="SYSTEM CAPABILITIES"
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
              Institutional Grade <br />
              <span style={gradientText(theme)}>Architecture</span>
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 700, mx: 'auto' }}>
              Our platform combines traditional technical analysis with cutting-edge AI to provide a comprehensive market view.
            </Typography>
          </Box>

          {features.map((category, idx) => (
            <Box key={idx} sx={{ mb: 12 }}>
              <Stack direction="row" spacing={3} alignItems="center" sx={{ mb: 5 }}>
                <Box sx={{
                  p: 2,
                  borderRadius: 3,
                  bgcolor: alpha(category.color, 0.1),
                  color: category.color,
                  display: 'flex',
                  boxShadow: `0 0 20px ${alpha(category.color, 0.3)}`
                }}>
                  {category.icon}
                </Box>
                <Typography variant="h3" fontWeight={800} sx={{ color: 'white' }}>
                  {category.category}
                </Typography>
              </Stack>

              <Grid container spacing={4}>
                {category.items.map((item, itemIndex) => (
                  <Grid item xs={12} md={4} key={itemIndex}>
                    <Paper
                      sx={{
                        height: '100%',
                        p: 4,
                        background: alpha(theme.palette.background.paper, 0.3),
                        backdropFilter: 'blur(10px)',
                        border: `1px solid ${alpha(theme.palette.common.white, 0.05)}`,
                        borderRadius: 4,
                        transition: 'all 0.3s ease',
                        display: 'flex',
                        flexDirection: 'column',
                        '&:hover': {
                          borderColor: category.color,
                          transform: 'translateY(-5px)',
                          boxShadow: `0 10px 40px -10px ${alpha(category.color, 0.2)}`,
                          background: alpha(theme.palette.background.paper, 0.5),
                        }
                      }}
                    >
                      <Box sx={{ mb: 2, color: category.color, opacity: 0.8 }}>
                        {item.icon}
                      </Box>
                      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
                        <Typography variant="h6" fontWeight="bold" color="common.white">
                          {item.title}
                        </Typography>
                      </Stack>
                      <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                        {item.description}
                      </Typography>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </Box>
          ))}

          <Box sx={{ textAlign: 'center', mt: 10, p: 6, bgcolor: alpha(theme.palette.primary.main, 0.05), border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`, borderRadius: 4 }}>
            <Typography variant="h4" fontWeight="bold" gutterBottom color="white">
              Ready to upgrade your trading?
            </Typography>
            <Typography color="text.secondary" sx={{ mb: 4 }}>
              Join thousands of traders using our institutional-grade tools.
            </Typography>
            <PremiumButton component={Link} to="/register" size="large">
              Start Free Trial
            </PremiumButton>
          </Box>
        </Container>
      </Box>
    </Box>
  );
}
