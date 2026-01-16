/**
 * Register Page - Modern Tech Design
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
  useTheme,
  alpha,
  Stack,
  LinearProgress,
} from '@mui/material';
import BoltIcon from '@mui/icons-material/Bolt';
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser';
import { useAuthStore } from '../../store/authStore';
import { keyframes } from '@mui/system';

const float = keyframes`
  0% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
  100% { transform: translateY(0px); }
`;

export default function Register() {
  const navigate = useNavigate();
  const theme = useTheme();
  const { register, isAuthenticated, error, isLoading, clearError } = useAuthStore();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');
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

    // Validation
    if (password !== confirmPassword) {
      setLocalError('Passwords do not match');
      return;
    }

    if (password.length < 8) {
      setLocalError('Password must be at least 8 characters');
      return;
    }

    try {
      await register(email, password, fullName || undefined);
      navigate('/dashboard');
    } catch (err: any) {
      setLocalError(err.response?.data?.message || err.message || 'Registration failed');
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
          bottom: '-20%',
          right: '-10%',
          width: '600px',
          height: '600px',
          background: `radial-gradient(circle, ${alpha(theme.palette.primary.dark, 0.2)} 0%, transparent 70%)`,
          filter: 'blur(80px)',
          animation: `${float} 10s ease-in-out infinite`,
        }} />
        <Box sx={{
          position: 'absolute',
          top: '-20%',
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
              <Box sx={{ p: 1, borderRadius: 2, bgcolor: alpha(theme.palette.secondary.main, 0.1), color: 'secondary.main' }}>
                <BoltIcon fontSize="large" />
              </Box>
            </Stack>

            <Typography variant="overline" sx={{ color: 'primary.main', fontWeight: 'bold', letterSpacing: 2 }}>
              Join the Network
            </Typography>
            <Typography
              variant="h4"
              gutterBottom
              sx={{ fontWeight: 700, color: 'common.white', mt: 1 }}
            >
              Create Account
            </Typography>
            <Typography
              variant="body1"
              sx={{ color: 'text.secondary' }}
            >
              Start your 14-day Pro trial instantly
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
              label="Full Name (Optional)"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              margin="normal"
              autoComplete="name"
              variant="outlined"
            />
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
              autoComplete="new-password"
              helperText="Must be at least 8 characters"
            />
            <TextField
              fullWidth
              label="Confirm Password"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              margin="normal"
              autoComplete="new-password"
            />

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
              {isLoading ? 'Creating Account...' : 'Initialize Account'}
            </Button>

            <Stack spacing={1} alignItems="center">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'success.main' }}>
                <VerifiedUserIcon fontSize="small" />
                <Typography variant="caption">Bank-grade encryption</Typography>
              </Box>
              <Typography variant="body1" align="center" sx={{ color: 'text.secondary' }}>
                Already registered?{' '}
                <Link to="/login" style={{ color: theme.palette.primary.main, textDecoration: 'none', fontWeight: 600 }}>
                  Log In
                </Link>
              </Typography>
            </Stack>
          </Box>
        </Paper>
      </Container>
    </Box>
  );
}
