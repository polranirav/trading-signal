/**
 * Login Page - Modern Tech Design
 */

import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Alert,
  Box,
  Checkbox,
  FormControlLabel,
  useTheme,
  alpha,
  Stack,
} from '@mui/material';
import BoltIcon from '@mui/icons-material/Bolt';
import { useAuthStore } from '../../store/authStore';
import { keyframes } from '@mui/system';

const float = keyframes`
  0% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
  100% { transform: translateY(0px); }
`;

export default function Login() {
  const navigate = useNavigate();
  const theme = useTheme();
  const { login, isAuthenticated, error, isLoading, clearError } = useAuthStore();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  useEffect(() => {
    clearError();
    setLocalError(null);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);
    clearError();

    try {
      await login(email, password);
      navigate('/dashboard');
    } catch (err: any) {
      setLocalError(err.response?.data?.message || err.message || 'Login failed');
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        position: 'relative',
        overflow: 'hidden',
        bgcolor: 'background.default',
      }}
    >
      {/* Dynamic Background */}
      <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 0 }}>
        <Box sx={{
          position: 'absolute',
          top: '-20%',
          right: '-10%',
          width: '600px',
          height: '600px',
          background: `radial-gradient(circle, ${alpha(theme.palette.primary.dark, 0.2)} 0%, transparent 70%)`,
          filter: 'blur(80px)',
          animation: `${float} 10s ease-in-out infinite`,
        }} />
        <Box sx={{
          position: 'absolute',
          bottom: '-20%',
          left: '-10%',
          width: '600px',
          height: '600px',
          background: `radial-gradient(circle, ${alpha(theme.palette.secondary.dark, 0.2)} 0%, transparent 70%)`,
          filter: 'blur(80px)',
          animation: `${float} 12s ease-in-out infinite reverse`,
        }} />
        <Box sx={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          backgroundImage: 'linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px)',
          backgroundSize: '50px 50px',
          maskImage: 'linear-gradient(to bottom, black 40%, transparent 100%)',
        }} />
      </Box>

      <Container maxWidth="sm" sx={{ position: 'relative', zIndex: 1 }}>
        <Paper
          elevation={24}
          sx={{
            p: { xs: 4, md: 6 },
            borderRadius: '24px',
            background: alpha(theme.palette.background.paper, 0.6),
            backdropFilter: 'blur(20px)',
            border: `1px solid ${alpha(theme.palette.common.white, 0.05)}`,
            boxShadow: `0 20px 60px ${alpha(theme.palette.common.black, 0.5)}`,
          }}
        >
          <Box sx={{ textAlign: 'center', mb: 4 }}>
            <Stack direction="row" spacing={1} justifyContent="center" alignItems="center" sx={{ mb: 3 }}>
              <Box sx={{ p: 1, borderRadius: 2, bgcolor: alpha(theme.palette.primary.main, 0.1), color: 'primary.main' }}>
                <BoltIcon fontSize="large" />
              </Box>
            </Stack>

            <Typography
              variant="h4"
              gutterBottom
              sx={{ fontWeight: 700, color: 'common.white' }}
            >
              Welcome Back
            </Typography>
            <Typography
              variant="body1"
              sx={{ color: 'text.secondary' }}
            >
              Access your intelligent trading dashboard
            </Typography>
          </Box>

          {(error || localError) && (
            <Alert severity="error" sx={{ mb: 3 }} onClose={() => { clearError(); setLocalError(null); }}>
              {error || localError}
            </Alert>
          )}

          <Box component="form" onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              margin="normal"
              autoComplete="email"
              autoFocus
              variant="outlined"
            />
            <TextField
              fullWidth
              label="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              margin="normal"
              autoComplete="current-password"
            />
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mt: 1 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    sx={{ color: 'text.secondary', '&.Mui-checked': { color: 'primary.main' } }}
                  />
                }
                label="Remember me"
                sx={{ color: 'text.secondary' }}
              />
              <Link to="/forgot-password" style={{ color: theme.palette.primary.main, textDecoration: 'none' }}>
                Forgot Password?
              </Link>
            </Stack>

            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              sx={{
                mt: 4,
                mb: 3,
                py: 1.5,
                fontSize: '1rem',
                background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                color: 'common.white',
                fontWeight: 600,
                boxShadow: `0 0 20px ${alpha(theme.palette.primary.main, 0.3)}`,
                '&:hover': {
                  boxShadow: `0 0 30px ${alpha(theme.palette.primary.main, 0.5)}`,
                }
              }}
              disabled={isLoading}
            >
              {isLoading ? 'Authenticating...' : 'Sign In'}
            </Button>

            <Typography variant="body1" align="center" sx={{ color: 'text.secondary' }}>
              Don't have an account?{' '}
              <Link to="/register" style={{ color: theme.palette.primary.main, textDecoration: 'none', fontWeight: 600 }}>
                Get Started
              </Link>
            </Typography>
          </Box>
        </Paper>
      </Container>
    </Box>
  );
}
