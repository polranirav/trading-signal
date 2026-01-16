import {
  Box,
  Button,
  Container,
  Grid,
  Typography,
  Stack,
  useTheme,
  alpha,
  Chip,
  Avatar,
  Paper,
  IconButton,
  Divider,
} from '@mui/material';
import { Link } from 'react-router-dom';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import SecurityIcon from '@mui/icons-material/Security';
import SpeedIcon from '@mui/icons-material/Speed';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import InsightsIcon from '@mui/icons-material/Insights';
import BoltIcon from '@mui/icons-material/Bolt';
import CodeIcon from '@mui/icons-material/Code';
import SendIcon from '@mui/icons-material/Send';
import GitHubIcon from '@mui/icons-material/GitHub';
import TwitterIcon from '@mui/icons-material/Twitter';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import { keyframes } from '@mui/system';

// Animations
const float = keyframes`
  0% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
  100% { transform: translateY(0px); }
`;

const pulse = keyframes`
  0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
  70% { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
  100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
`;

const gradientText = (theme: any) => ({
  background: `linear-gradient(135deg, ${theme.palette.common.white} 0%, ${theme.palette.primary.light} 100%)`,
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  backgroundClip: 'text',
});

// Premium Button Component
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


// Modern Tech Navbar
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

// Epic Futuristic Dashboard Mockup
const DashboardPreview = () => {
  const theme = useTheme();

  // Scanline animation
  const scan = keyframes`
        0% { transform: translateY(-100%); }
        100% { transform: translateY(400px); }
    `;

  // Holographic pulse
  const holoPulse = keyframes`
        0% { opacity: 0.3; }
        50% { opacity: 0.7; }
        100% { opacity: 0.3; }
    `;

  return (
    <Box sx={{ position: 'relative', perspective: '1000px', transformStyle: 'preserve-3d' }}>
      {/* Massive Behind Glow */}
      <Box
        sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '140%',
          height: '140%',
          background: `radial-gradient(circle, ${alpha(theme.palette.primary.main, 0.4)} 0%, transparent 60%)`,
          filter: 'blur(80px)',
          zIndex: 0,
          animation: `${pulse} 4s infinite ease-in-out`
        }}
      />

      {/* Main Window Container */}
      <Paper
        elevation={24}
        sx={{
          position: 'relative',
          zIndex: 1,
          background: alpha('#0a0b14', 0.8),
          backdropFilter: 'blur(20px)',
          border: `1px solid ${alpha(theme.palette.primary.light, 0.3)}`,
          borderRadius: 4,
          overflow: 'hidden',
          display: 'flex',
          height: 420,
          boxShadow: `0 0 50px -10px ${alpha(theme.palette.primary.main, 0.5)}`, // Neon glow shadow
          transform: 'rotateX(5deg) rotateY(-5deg)', // Slight 3D Tilt
          transition: 'transform 0.5s ease',
          '&:hover': {
            transform: 'rotateX(0deg) rotateY(0deg) scale(1.02)',
            boxShadow: `0 0 80px -10px ${alpha(theme.palette.secondary.main, 0.6)}`,
          }
        }}
      >
        {/* Overlay: Scanline Effect */}
        <Box
          sx={{
            position: 'absolute',
            top: 0, left: 0, right: 0, height: '20%',
            background: `linear-gradient(to bottom, transparent, ${alpha(theme.palette.primary.main, 0.2)}, transparent)`,
            zIndex: 10,
            pointerEvents: 'none',
            animation: `${scan} 3s linear infinite`,
            opacity: 0.5
          }}
        />

        {/* Overlay: Grid Pattern */}
        <Box
          sx={{
            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
            backgroundImage: `linear-gradient(${alpha(theme.palette.common.white, 0.03)} 1px, transparent 1px), linear-gradient(90deg, ${alpha(theme.palette.common.white, 0.03)} 1px, transparent 1px)`,
            backgroundSize: '20px 20px',
            zIndex: 0,
            pointerEvents: 'none'
          }}
        />

        {/* Sidebar */}
        <Box sx={{ width: 70, borderRight: `1px solid ${alpha(theme.palette.common.white, 0.05)}`, display: 'flex', flexDirection: 'column', alignItems: 'center', py: 3, gap: 3, bgcolor: alpha(theme.palette.background.paper, 0.3), zIndex: 2, backdropFilter: 'blur(5px)' }}>
          <Box sx={{
            width: 40, height: 40, borderRadius: 2,
            background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
            boxShadow: `0 0 20px ${theme.palette.primary.main}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <BoltIcon sx={{ color: 'white', fontSize: 24 }} />
          </Box>
          <Divider sx={{ width: '60%', borderColor: alpha(theme.palette.common.white, 0.1) }} />
          {[1, 2, 3, 4, 5].map(i => (
            <Box key={i} sx={{
              width: 36, height: 36, borderRadius: 2,
              bgcolor: i === 1 ? alpha(theme.palette.primary.main, 0.2) : 'transparent',
              border: i === 1 ? `1px solid ${alpha(theme.palette.primary.main, 0.5)}` : 'none',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 0.2s',
              '&:hover': { bgcolor: alpha(theme.palette.common.white, 0.1) }
            }}>
              <Box sx={{ width: 18, height: 2, borderRadius: 1, bgcolor: i === 1 ? theme.palette.primary.light : alpha(theme.palette.common.white, 0.3), boxShadow: i === 1 ? `0 0 10px ${theme.palette.primary.main}` : 'none' }} />
            </Box>
          ))}
        </Box>

        {/* Content */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', zIndex: 2 }}>

          {/* Ticker */}
          <Box sx={{ height: 40, borderBottom: `1px solid ${alpha(theme.palette.common.white, 0.05)}`, display: 'flex', alignItems: 'center', px: 3, gap: 4, bgcolor: alpha(theme.palette.background.default, 0.6) }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 'bold' }}>BTC/USD</Typography>
              <Typography variant="caption" sx={{ color: '#10b981', fontWeight: 'bold', textShadow: '0 0 10px rgba(16, 185, 129, 0.5)' }}>+2.45%</Typography>
            </Stack>
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 'bold' }}>ETH/USD</Typography>
              <Typography variant="caption" sx={{ color: '#ef4444', fontWeight: 'bold', textShadow: '0 0 10px rgba(239, 68, 68, 0.5)' }}>-0.85%</Typography>
            </Stack>
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 'bold' }}>AI CONFIDENCE</Typography>
              <Box sx={{ width: 60, height: 4, bgcolor: alpha(theme.palette.common.white, 0.1), borderRadius: 1 }}>
                <Box sx={{ width: '85%', height: '100%', bgcolor: theme.palette.secondary.main, borderRadius: 1, boxShadow: `0 0 10px ${theme.palette.secondary.main}` }} />
              </Box>
            </Stack>
          </Box>

          {/* Dashboard Area */}
          <Box sx={{ p: 3, display: 'flex', gap: 3, flex: 1 }}>

            {/* Main Chart */}
            <Box sx={{ flex: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Paper sx={{
                flex: 1,
                background: `linear-gradient(180deg, ${alpha(theme.palette.background.paper, 0.4)} 0%, ${alpha(theme.palette.background.paper, 0.1)} 100%)`,
                border: `1px solid ${alpha(theme.palette.common.white, 0.05)}`,
                borderRadius: 2,
                p: 2,
                position: 'relative',
                overflow: 'hidden'
              }}>
                {/* Header */}
                <Stack direction="row" justifyContent="space-between" sx={{ mb: 2 }}>
                  <Stack direction="row" spacing={2} alignItems="center">
                    <Typography variant="subtitle2" fontWeight="bold" color="white">BTC/USD Perp</Typography>
                    <Box sx={{ px: 1, py: 0.5, borderRadius: 1, bgcolor: alpha(theme.palette.primary.main, 0.2), border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}` }}>
                      <Typography variant="caption" color="primary.light" fontWeight="bold">15m</Typography>
                    </Box>
                  </Stack>
                  <Typography variant="h6" color="#10b981" fontWeight="bold" sx={{ textShadow: '0 0 15px rgba(16, 185, 129, 0.4)' }}>$48,294.10</Typography>
                </Stack>

                {/* Candles */}
                <Box sx={{ display: 'flex', alignItems: 'flex-end', height: '80%', gap: 1, pb: 1, px: 1 }}>
                  {[45, 50, 48, 55, 65, 60, 75, 80, 78, 85, 95, 90, 100, 110].map((h, i) => {
                    const isUp = i === 0 || h > [45, 50, 48, 55, 65, 60, 75, 80, 78, 85, 95, 90, 100, 110][i - 1];
                    const color = isUp ? '#10b981' : '#ef4444';
                    return (
                      <Box key={i} sx={{ flex: 1, position: 'relative', height: '100%' }}>
                        {/* Wick */}
                        <Box sx={{
                          position: 'absolute', left: '50%', top: `${100 - h - 15}%`, height: `${h + 20}%`, width: 1,
                          bgcolor: color, opacity: 0.6,
                          boxShadow: `0 0 5px ${color}`
                        }} />
                        {/* Body */}
                        <Box sx={{
                          position: 'absolute',
                          left: 0, right: 0,
                          bottom: `${i * 3 + 10}%`,
                          height: `${Math.random() * 25 + 5}%`,
                          bgcolor: color,
                          borderRadius: 1,
                          boxShadow: `0 0 15px ${alpha(color, 0.4)}`, // Neon Glow
                          transition: 'all 0.3s ease',
                          '&:hover': { transform: 'scaleY(1.1)', boxShadow: `0 0 25px ${color}` }
                        }} />
                      </Box>
                    )
                  })}
                </Box>
              </Paper>
            </Box>

            {/* Signals Panel */}
            <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#ef4444', animation: `${pulse} 1s infinite` }} />
                <Typography variant="caption" fontWeight="bold" color="text.secondary" sx={{ letterSpacing: 1 }}>LIVE FEED</Typography>
              </Box>

              {[
                { sym: 'BTC', type: 'LONG', lev: '20x', pnl: '+124%', speed: '12ms' },
                { sym: 'ETH', type: 'SHORT', lev: '10x', pnl: '+45%', speed: '15ms' },
                { sym: 'SOL', type: 'LONG', lev: '50x', pnl: '+89%', speed: '11ms' },
              ].map((sig, i) => (
                <Paper key={i} sx={{
                  p: 2,
                  bgcolor: alpha(theme.palette.background.paper, 0.2),
                  border: `1px solid ${alpha(theme.palette.common.white, 0.05)}`,
                  borderRadius: 2,
                  position: 'relative',
                  overflow: 'hidden',
                  transition: 'all 0.3s',
                  '&:hover': {
                    borderColor: theme.palette.primary.main,
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    transform: 'translateX(5px)'
                  }
                }}>
                  {/* Left Accent */}
                  <Box sx={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, bgcolor: sig.type === 'LONG' ? '#10b981' : '#ef4444' }} />

                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                    <Typography variant="subtitle2" fontWeight="bold" color="white">{sig.sym}</Typography>
                    <Chip label={sig.type} size="small" sx={{
                      height: 20, fontSize: 10, fontWeight: 'bold', borderRadius: 1,
                      bgcolor: sig.type === 'LONG' ? alpha('#10b981', 0.2) : alpha('#ef4444', 0.2),
                      color: sig.type === 'LONG' ? '#10b981' : '#ef4444',
                      border: `1px solid ${sig.type === 'LONG' ? alpha('#10b981', 0.3) : alpha('#ef4444', 0.3)}`
                    }} />
                  </Stack>
                  <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Typography variant="caption" color="text.secondary">Lev: <span style={{ color: 'white' }}>{sig.lev}</span></Typography>
                    <Typography variant="caption" color="#10b981" fontWeight="bold">{sig.pnl}</Typography>
                  </Stack>
                </Paper>
              ))}

              <Box sx={{ mt: 'auto', p: 2, borderRadius: 2, bgcolor: alpha(theme.palette.primary.main, 0.1), border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}` }}>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                  <AutoGraphIcon fontSize="small" color="primary" />
                  <Typography variant="caption" fontWeight="bold" color="primary.light">SYSTEM STATUS</Typography>
                </Stack>
                <Typography variant="body2" color="white" fontWeight="bold">Neural Engine v2.4 Active</Typography>
                <Typography variant="caption" color="text.secondary">Processing 1.2M events/sec</Typography>
              </Box>
            </Box>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

// High-Tech Hero
const Hero = () => {
  const theme = useTheme();
  return (
    <Box
      sx={{
        minHeight: '90vh',
        pt: 15,
        pb: 10,
        display: 'flex',
        alignItems: 'center',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Dynamic Background */}
      <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: -1 }}>
        {/* Mesh Gradient 1 */}
        <Box sx={{
          position: 'absolute',
          top: '-20%',
          left: '-10%',
          width: '800px',
          height: '800px',
          background: `radial-gradient(circle, ${alpha(theme.palette.primary.dark, 0.25)} 0%, transparent 70%)`,
          filter: 'blur(80px)',
          animation: `${float} 10s ease-in-out infinite`,
        }} />
        {/* Mesh Gradient 2 */}
        <Box sx={{
          position: 'absolute',
          bottom: '-20%',
          right: '-10%',
          width: '800px',
          height: '800px',
          background: `radial-gradient(circle, ${alpha(theme.palette.secondary.dark, 0.25)} 0%, transparent 70%)`,
          filter: 'blur(80px)',
          animation: `${float} 12s ease-in-out infinite reverse`,
        }} />
        <Box sx={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          backgroundImage: 'linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
          maskImage: 'radial-gradient(ellipse at center, black 40%, transparent 80%)',
        }} />
      </Box>

      <Container maxWidth="xl" sx={{ position: 'relative', zIndex: 1 }}>
        <Grid container spacing={8} alignItems="center">
          <Grid item xs={12} md={6}>
            <Chip
              icon={<BoltIcon sx={{ color: `${theme.palette.warning.main} !important` }} />}
              label="v2.0 Live: Enhanced AI Latency"
              sx={{
                bgcolor: alpha(theme.palette.warning.main, 0.1),
                color: theme.palette.warning.light,
                border: `1px solid ${alpha(theme.palette.warning.main, 0.2)}`,
                mb: 4,
                fontWeight: 600
              }}
            />
            <Typography variant="h1" gutterBottom sx={{ fontSize: { xs: '3rem', md: '5rem' }, fontWeight: 800, lineHeight: 1 }}>
              Algorithmic <br />
              <Box component="span" sx={gradientText(theme)}>
                Intelligence
              </Box>
            </Typography>
            <Typography variant="h5" sx={{ color: 'text.secondary', mb: 6, maxWidth: '600px', lineHeight: 1.6, fontWeight: 400 }}>
              Deploy institutional-grade trading strategies powered by our proprietary Neural Engine.
              Real-time processing. Sub-millisecond execution signals.
            </Typography>

            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={3}>
              <PremiumButton
                component={Link}
                to="/register"
                endIcon={<ArrowForwardIcon />}
              >
                Start Free Trial
              </PremiumButton>
              <PremiumButton
                variant="outlined"
                component={Link}
                to="/features"
                startIcon={<CodeIcon />}
              >
                View Documentation
              </PremiumButton>
            </Stack>

            <Stack direction="row" spacing={6} sx={{ mt: 8 }}>
              {[
                { label: 'Active Users', value: '12k+' },
                { label: 'Daily Signals', value: '8.5M' },
                { label: 'Uptime', value: '99.9%' },
              ].map((stat) => (
                <Box key={stat.label}>
                  <Typography variant="h4" fontWeight="bold" sx={{ color: 'white' }}>{stat.value}</Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>{stat.label}</Typography>
                </Box>
              ))}
            </Stack>
          </Grid>

          <Grid item xs={12} md={6}>
            <DashboardPreview />
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

// Features Section (Enhanced)
const Features = () => {
  const theme = useTheme();
  const techs = [
    { icon: <AutoGraphIcon fontSize="large" />, title: "Predictive Modeling", desc: "LSTM & Transformer architecture for price forecasting." },
    { icon: <InsightsIcon fontSize="large" />, title: "Sentiment Engine", desc: "NLP processing of 50k+ news sources daily." },
    { icon: <SecurityIcon fontSize="large" />, title: "Risk Protocol", desc: "Automated hedging and stop-losses via API." },
  ];

  return (
    <Box sx={{ py: 15, bgcolor: 'background.default', position: 'relative', borderTop: `1px solid ${alpha(theme.palette.common.white, 0.05)}` }}>
      <Container maxWidth="xl">
        <Typography variant="h2" align="center" gutterBottom fontWeight={800} sx={{ mb: 2 }}>
          Powered by <span style={gradientText(theme)}>Next-Gen</span> Tech
        </Typography>
        <Typography variant="h6" align="center" color="text.secondary" sx={{ mb: 10, maxWidth: '700px', mx: 'auto' }}>
          We've packaged hedge-fund grade technology into an accessible platform for every trader.
        </Typography>

        <Grid container spacing={4}>
          {techs.map((tech, i) => (
            <Grid item xs={12} md={4} key={i}>
              <Paper
                elevation={0}
                sx={{
                  p: 5,
                  height: '100%',
                  bgcolor: alpha(theme.palette.background.paper, 0.3),
                  border: `1px solid ${alpha(theme.palette.common.white, 0.05)}`,
                  borderRadius: 4,
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    bgcolor: alpha(theme.palette.background.paper, 0.6),
                    borderColor: theme.palette.primary.main,
                    transform: 'translateY(-5px)',
                    boxShadow: `0 10px 40px -10px ${alpha(theme.palette.primary.main, 0.2)}`
                  }
                }}
              >
                <Box sx={{
                  width: 70, height: 70,
                  mb: 3,
                  borderRadius: 3,
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  color: 'primary.main',
                  display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}>
                  {tech.icon}
                </Box>
                <Typography variant="h5" gutterBottom fontWeight="bold" color="common.white">{tech.title}</Typography>
                <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.7 }}>{tech.desc}</Typography>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

// New "How It Works" Section to add content bulk
const HowItWorks = () => {
  const theme = useTheme();
  const steps = [
    { num: '01', title: 'Connect Account', desc: 'Securely link your exchange API keys via encrypted tunnel.' },
    { num: '02', title: 'Select Strategy', desc: 'Choose from conservative to aggressive high-frequency models.' },
    { num: '03', title: 'Automate Trading', desc: 'Let our Neural Engine execute trades with sub-ms latency.' }
  ];

  return (
    <Box sx={{ py: 15, bgcolor: alpha(theme.palette.background.paper, 0.2) }}>
      <Container maxWidth="xl">
        <Grid container spacing={8} alignItems="center">
          <Grid item xs={12} md={5}>
            <Typography variant="overline" sx={{ color: 'secondary.main', fontWeight: 'bold', letterSpacing: 2 }}>WORKFLOW</Typography>
            <Typography variant="h2" fontWeight={800} gutterBottom sx={{ mt: 1, mb: 3 }}>
              Setup in <br />
              <span style={{ color: theme.palette.primary.main }}>Minutes</span>
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph sx={{ fontSize: '1.1rem', mb: 4 }}>
              No complex coding required. Our platform is designed for plug-and-play efficiency, allowing you to focus on strategy while we handle the execution.
            </Typography>
            <PremiumButton endIcon={<ArrowForwardIcon />}>
              View Integration Guide
            </PremiumButton>
          </Grid>
          <Grid item xs={12} md={7}>
            <Stack spacing={3}>
              {steps.map((step, i) => (
                <Paper key={i} sx={{
                  p: 3,
                  bgcolor: alpha(theme.palette.background.default, 0.5),
                  border: `1px solid ${alpha(theme.palette.common.white, 0.05)}`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 3,
                  transition: 'transform 0.2s',
                  '&:hover': { transform: 'translateX(10px)', borderColor: theme.palette.secondary.main }
                }}>
                  <Typography variant="h3" fontWeight={900} sx={{ color: alpha(theme.palette.common.white, 0.1) }}>
                    {step.num}
                  </Typography>
                  <Box>
                    <Typography variant="h6" fontWeight="bold" color="common.white">{step.title}</Typography>
                    <Typography variant="body2" color="text.secondary">{step.desc}</Typography>
                  </Box>
                </Paper>
              ))}
            </Stack>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

// Footer Component (Added as requested)
const Footer = () => {
  const theme = useTheme();
  return (
    <Box sx={{ bgcolor: theme.palette.background.paper, pt: 10, pb: 4, borderTop: `1px solid ${alpha(theme.palette.common.white, 0.05)}` }}>
      <Container maxWidth="xl">
        <Grid container spacing={8}>
          <Grid item xs={12} md={4}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 3 }}>
              <Box sx={{ p: 0.5, borderRadius: 1, bgcolor: alpha(theme.palette.primary.main, 0.1), color: 'primary.main', display: 'flex' }}>
                <BoltIcon />
              </Box>
              <Typography variant="h6" fontWeight="bold" sx={{ color: 'common.white', letterSpacing: -0.5 }}>
                TRADING<Typography component="span" variant="h6" sx={{ color: 'primary.main', fontWeight: 'bold' }}>PRO</Typography>
              </Typography>
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 300, mb: 3 }}>
              Instituional-grade AI trading infrastructure for the modern trader.
              Powered by DeepWave™ Neural Engine.
            </Typography>
            <Stack direction="row" spacing={2}>
              {[<GitHubIcon />, <TwitterIcon />, <LinkedInIcon />].map((icon, i) => (
                <IconButton key={i} size="small" sx={{ color: 'text.secondary', border: `1px solid ${alpha(theme.palette.common.white, 0.1)}` }}>
                  {icon}
                </IconButton>
              ))}
            </Stack>
          </Grid>

          <Grid item xs={6} md={2}>
            <Typography variant="subtitle2" fontWeight="bold" color="common.white" sx={{ mb: 3 }}>PRODUCT</Typography>
            <Stack spacing={2}>
              {['Features', 'Pricing'].map(item => (
                <Link key={item} to="#" style={{ textDecoration: 'none' }}>
                  <Typography variant="body2" color="text.secondary" sx={{ '&:hover': { color: 'primary.main' } }}>{item}</Typography>
                </Link>
              ))}
            </Stack>
          </Grid>

          <Grid item xs={6} md={2}>
            <Typography variant="subtitle2" fontWeight="bold" color="common.white" sx={{ mb: 3 }}>RESOURCES</Typography>
            <Stack spacing={2}>
              {['Documentation', 'Whitepaper', 'Community', 'Status'].map(item => (
                <Link key={item} to="#" style={{ textDecoration: 'none' }}>
                  <Typography variant="body2" color="text.secondary" sx={{ '&:hover': { color: 'primary.main' } }}>{item}</Typography>
                </Link>
              ))}
            </Stack>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, bgcolor: alpha(theme.palette.background.default, 0.5), border: `1px solid ${alpha(theme.palette.common.white, 0.1)}` }}>
              <Typography variant="subtitle2" fontWeight="bold" color="common.white" sx={{ mb: 2 }}>Subscribe to our newsletter</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>Get the latest AI market insights.</Typography>
              <PremiumButton size="small" fullWidth>Subscribe</PremiumButton>
            </Paper>
          </Grid>
        </Grid>

        <Divider sx={{ my: 4, borderColor: alpha(theme.palette.common.white, 0.05) }} />

        <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" alignItems="center" spacing={2}>
          <Typography variant="caption" color="text.secondary">© 2024 Trading Signals Pro. All rights reserved.</Typography>
          <Stack direction="row" spacing={4}>
            <Typography variant="caption" color="text.secondary">Privacy Policy</Typography>
            <Typography variant="caption" color="text.secondary">Terms of Service</Typography>
          </Stack>
        </Stack>
      </Container>
    </Box>
  );
};

export default function Landing() {
  return (
    <Box sx={{ bgcolor: 'background.default', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar />
      <Hero />
      <Features />
      <HowItWorks />
      <Footer />
    </Box>
  )
}
