/**
 * Protected Admin Route Component
 * 
 * Route wrapper that requires admin authentication.
 */

import { Navigate } from 'react-router-dom';
import { Box, CircularProgress } from '@mui/material';
import { useAuthStore } from '../store/authStore';

interface ProtectedAdminRouteProps {
  children: React.ReactNode;
}

export default function ProtectedAdminRoute({ children }: ProtectedAdminRouteProps) {
  const { user, isAuthenticated } = useAuthStore();

  // Show loading state if user is being fetched
  if (user === undefined) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '100vh',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated || !user) {
    return <Navigate to="/login" replace />;
  }

  // Redirect to dashboard if not admin
  if (!user.is_admin) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
}
