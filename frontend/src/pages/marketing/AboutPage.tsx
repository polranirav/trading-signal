/**
 * About Page - Modern Tech Design
 */

import { Box, Container, Typography, Grid, Card, CardContent, Button, useTheme, alpha, Stack, Avatar } from '@mui/material';
import { Link } from 'react-router-dom';
import BoltIcon from '@mui/icons-material/Bolt';
import SecurityIcon from '@mui/icons-material/Security';
import SpeedIcon from '@mui/icons-material/Speed';
import PsychologyIcon from '@mui/icons-material/Psychology';
import GroupsIcon from '@mui/icons-material/Groups';
import SendIcon from '@mui/icons-material/Send';
import CodeIcon from '@mui/icons-material/Code';

// Reusing Navbar component for consistency
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
        backgroundColor: alpha(theme.palette.background.default, 0.7),
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

          <Stack direction="row" spacing={2} alignItems="center">
            <Button component={Link} to="/features" color="inherit" sx={{ color: 'text.secondary' }}>Features</Button>
            <Button component={Link} to="/pricing" color="inherit" sx={{ color: 'text.secondary' }}>Pricing</Button>
            <Button component={Link} to="/login" variant="text" sx={{ color: 'text.secondary' }}>
              Log In
            </Button>
            <Button
              component={Link}
              to="/register"
              variant="contained"
              endIcon={<SendIcon fontSize="small" />}
              sx={{
                background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                color: 'white',
                fontWeight: 600,
                borderRadius: 2,
              }}
            >
              Get Started
            </Button>
          </Stack>
        </Stack>
      </Container>
    </Box>
  );
};

export default function AboutPage() {
  const theme = useTheme();

  return (
    <Box sx={{ bgcolor: 'background.default', minHeight: '100vh', paddingBottom: 10 }}>
      <Navbar />

      {/* Hero Section */}
      <Box sx={{
        position: 'relative',
        overflow: 'hidden',
        py: 15,
        background: `radial-gradient(circle at 50% 0%, ${alpha(theme.palette.primary.dark, 0.2)} 0%, ${theme.palette.background.default} 70%)`,
      }}>
        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <Typography variant="overline" display="block" align="center" sx={{ color: 'primary.main', letterSpacing: 3, mb: 1, fontWeight: 'bold' }}>
            OUR VISION
          </Typography>
          <Typography variant="h2" component="h1" gutterBottom fontWeight={800} align="center" sx={{ fontSize: { xs: '2.5rem', md: '4rem' } }}>
            Decoding the Market <br />
            <Box component="span" sx={{
              background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}>
              Through Intelligence
            </Box>
          </Typography>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 6, maxWidth: '800px', mx: 'auto', lineHeight: 1.6 }} align="center">
            We are a collective of data scientists, quant traders, and engineers building the next generation of financial intelligence infrastructure.
          </Typography>
        </Container>
      </Box>

      {/* Core Values */}
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%', bgcolor: alpha(theme.palette.background.paper, 0.4), border: `1px solid ${alpha(theme.palette.common.white, 0.05)}` }}>
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ bgcolor: alpha(theme.palette.primary.main, 0.1), color: 'primary.main', width: 48, height: 48, borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 3 }}>
                  <CodeIcon />
                </Box>
                <Typography variant="h5" gutterBottom fontWeight="bold">Data First</Typography>
                <Typography variant="body2" color="text.secondary">
                  We believe in hard data over intuition. Our models process petabytes of market history to find statistically significant edges.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%', bgcolor: alpha(theme.palette.background.paper, 0.4), border: `1px solid ${alpha(theme.palette.common.white, 0.05)}` }}>
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ bgcolor: alpha(theme.palette.secondary.main, 0.1), color: 'secondary.main', width: 48, height: 48, borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 3 }}>
                  <SpeedIcon />
                </Box>
                <Typography variant="h5" gutterBottom fontWeight="bold">Low Latency</Typography>
                <Typography variant="body2" color="text.secondary">
                  In high-frequency markets, speed is alpha. Our infrastructure is optimized for sub-millisecond signal delivery.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%', bgcolor: alpha(theme.palette.background.paper, 0.4), border: `1px solid ${alpha(theme.palette.common.white, 0.05)}` }}>
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ bgcolor: alpha(theme.palette.info.main, 0.1), color: 'info.main', width: 48, height: 48, borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 3 }}>
                  <SecurityIcon />
                </Box>
                <Typography variant="h5" gutterBottom fontWeight="bold">Transparency</Typography>
                <Typography variant="body2" color="text.secondary">
                  No black boxes. We provide full performance metrics, historical drawdowns, and risk parameters for every signal.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Team/Tech Section */}
        <Box sx={{ mt: 15, display: 'flex', flexDirection: { xs: 'column', md: 'row' }, alignItems: 'center', gap: 8 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h3" gutterBottom fontWeight="bold">The Engine</Typography>
            <Typography variant="body1" paragraph color="text.secondary" >
              Our proprietary "DeepWave" engine combines LSTM networks with Transformer architectures to analyze multi-modal data streams: price action, order book depth, and social sentiment.
            </Typography>
            <Typography variant="body1" paragraph color="text.secondary">
              Running on a distributed cluster of GPU nodes, we re-train our models weekly to adapt to shifting market regimes.
            </Typography>
            <Button variant="outlined" size="large" sx={{ mt: 2, borderColor: theme.palette.primary.main, color: theme.palette.primary.main }}>
              View Technical Whitepaper
            </Button>
          </Box>
          <Box sx={{ flex: 1, position: 'relative' }}>
            <Box sx={{
              border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
              borderRadius: 4,
              p: 4,
              position: 'relative',
              background: alpha(theme.palette.background.paper, 0.3),
              backdropFilter: 'blur(10px)'
            }}>
              <Typography variant="h4" gutterBottom fontWeight="bold">
                System Status
              </Typography>
              <Stack spacing={2}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', borderBottom: `1px solid ${alpha(theme.palette.common.white, 0.1)}`, pb: 1 }}>
                  <Typography color="text.secondary">Model Accuracy (24h)</Typography>
                  <Typography color="success.main" fontWeight="bold">94.2%</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', borderBottom: `1px solid ${alpha(theme.palette.common.white, 0.1)}`, pb: 1 }}>
                  <Typography color="text.secondary">Active Nodes</Typography>
                  <Typography color="primary.main" fontWeight="bold">128/128</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography color="text.secondary">Sentiment Analysis</Typography>
                  <Typography color="info.main" fontWeight="bold">ONLINE</Typography>
                </Box>
              </Stack>
            </Box>
          </Box>
        </Box>
      </Container>
    </Box>
  );
}
