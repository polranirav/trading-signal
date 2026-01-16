/**
 * Main App Component
 * 
 * Sets up routing and layout.
 */

import { Routes, Route, Navigate } from 'react-router-dom';
import { useEffect } from 'react';

// Pages
import Login from './pages/auth/Login';
import Register from './pages/auth/Register';
import Landing from './pages/marketing/Landing';
import FeaturesPage from './pages/marketing/FeaturesPage';
import PricingPage from './pages/marketing/PricingPage';
import AboutPage from './pages/marketing/AboutPage';
import OverviewPage from './pages/dashboard/OverviewPage';
import HistoryPage from './pages/dashboard/HistoryPage';
import PerformancePage from './pages/dashboard/PerformancePage';
import AccountPage from './pages/dashboard/AccountPage';
import StocksPage from './pages/dashboard/StocksPage';
import ChartsPage from './pages/dashboard/ChartsPage';
import BacktestPage from './pages/dashboard/BacktestPage';
import PortfolioPage from './pages/dashboard/PortfolioPage';
import SignalIntelligencePage from './pages/dashboard/SignalIntelligencePage';
import AlertsPage from './pages/dashboard/AlertsPage';

// Admin Pages
import AdminDashboard from './pages/admin/DashboardPage';
import AdminUsers from './pages/admin/UsersPage';
import AdminSignals from './pages/admin/SignalsPage';
import AdminSubscriptions from './pages/admin/SubscriptionsPage';
import AdminSettings from './pages/admin/SettingsPage';
import AdminAuditLogs from './pages/admin/AuditLogsPage';

// Components
import ProtectedRoute from './components/ProtectedRoute';
import ProtectedAdminRoute from './components/ProtectedAdminRoute';
import AdminLayout from './components/layout/AdminLayout';
import DashboardLayout from './components/layout/DashboardLayout';

// Store
import { useAuthStore } from './store/authStore';

// Context
import { PortfolioProvider } from './context';

function App() {
  const { fetchUser } = useAuthStore();

  useEffect(() => {
    // Fetch user on app load
    fetchUser();
  }, [fetchUser]);

  return (
    <Routes>
      {/* Public Routes */}
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/features" element={<FeaturesPage />} />
      <Route path="/pricing" element={<PricingPage />} />
      <Route path="/about" element={<AboutPage />} />

      {/* Protected Routes - Dashboard */}
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <PortfolioProvider>
              <DashboardLayout />
            </PortfolioProvider>
          </ProtectedRoute>
        }
      >
        <Route index element={<PortfolioPage />} />
        <Route path="overview" element={<OverviewPage />} />
        <Route path="stocks" element={<StocksPage />} />
        <Route path="charts" element={<ChartsPage />} />
        <Route path="signals" element={<SignalIntelligencePage />} />
        <Route path="alerts" element={<AlertsPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="performance" element={<PerformancePage />} />
        <Route path="backtest" element={<BacktestPage />} />
        <Route path="account" element={<AccountPage />} />
      </Route>

      {/* Admin Routes */}
      <Route
        path="/admin"
        element={
          <ProtectedAdminRoute>
            <AdminLayout />
          </ProtectedAdminRoute>
        }
      >
        <Route index element={<AdminDashboard />} />
        <Route path="users" element={<AdminUsers />} />
        <Route path="signals" element={<AdminSignals />} />
        <Route path="subscriptions" element={<AdminSubscriptions />} />
        <Route path="settings" element={<AdminSettings />} />
        <Route path="audit-logs" element={<AdminAuditLogs />} />
      </Route>

      {/* Default redirect */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
