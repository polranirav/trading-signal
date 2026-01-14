import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom'
import {
  Box,
  Drawer,
  List,
  Typography,
  Divider,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
} from '@mui/material'
import MenuIcon from '@mui/icons-material/Menu'
import DashboardIcon from '@mui/icons-material/Dashboard'
import AnalyticsIcon from '@mui/icons-material/Analytics'
import HistoryIcon from '@mui/icons-material/History'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import PersonIcon from '@mui/icons-material/Person'
import CodeIcon from '@mui/icons-material/Code'
import LogoutIcon from '@mui/icons-material/Logout'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import BarChartIcon from '@mui/icons-material/BarChart'
import SpeedIcon from '@mui/icons-material/Speed'
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet'
import { useState } from 'react'
import { useAuthStore } from '../../store/authStore'
import TickerBar from '../TickerBar'
import '../../styles/premium.css'

const drawerWidth = 260

const menuItems = [
  { text: 'My Portfolio', icon: <AccountBalanceWalletIcon />, path: '/dashboard' },
  { text: 'Market Overview', icon: <DashboardIcon />, path: '/dashboard/overview' },
  { text: 'Analysis', icon: <AnalyticsIcon />, path: '/dashboard/analysis' },
  { text: 'Stocks', icon: <TrendingUpIcon />, path: '/dashboard/stocks' },
  { text: 'Charts', icon: <ShowChartIcon />, path: '/dashboard/charts' },
  { text: 'History', icon: <HistoryIcon />, path: '/dashboard/history' },
  { text: 'Performance', icon: <BarChartIcon />, path: '/dashboard/performance' },
  { text: 'Backtest', icon: <SpeedIcon />, path: '/dashboard/backtest' },
  { text: 'Account', icon: <PersonIcon />, path: '/dashboard/account' },
]

export default function DashboardLayout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const { logout, user } = useAuthStore()

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen)
  }

  const handleLogout = async () => {
    await logout()
    navigate('/login')
  }

  const drawer = (
    <Box sx={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      background: 'rgba(13, 14, 23, 0.95)',
      borderRight: '1px solid rgba(255, 255, 255, 0.08)'
    }}>
      {/* Logo Header */}
      <Box sx={{
        p: 2,
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)'
      }}>
        <Box sx={{
          width: 36,
          height: 36,
          background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
          borderRadius: '10px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 'bold',
          fontSize: '1.1rem'
        }}>
          TS
        </Box>
        <Typography sx={{ fontWeight: 700, fontSize: '1.1rem', color: '#fff' }}>
          TradeSignals
        </Typography>
      </Box>

      {/* Navigation */}
      <List sx={{ flex: 1, py: 2 }}>
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path
          return (
            <ListItem key={item.text} disablePadding sx={{ px: 1, mb: 0.5 }}>
              <ListItemButton
                component={Link}
                to={item.path}
                sx={{
                  borderRadius: '8px',
                  py: 1.2,
                  color: isActive ? '#3b82f6' : '#94a3b8',
                  background: isActive ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                  borderLeft: isActive ? '3px solid #3b82f6' : '3px solid transparent',
                  '&:hover': {
                    background: 'rgba(255, 255, 255, 0.05)',
                    color: '#fff',
                  }
                }}
              >
                <ListItemIcon sx={{
                  color: 'inherit',
                  minWidth: 40,
                  '& svg': { fontSize: '1.2rem' }
                }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.text}
                  primaryTypographyProps={{
                    fontWeight: isActive ? 600 : 500,
                    fontSize: '0.9rem'
                  }}
                />
              </ListItemButton>
            </ListItem>
          )
        })}
      </List>

      <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.08)' }} />

      {/* Logout & Status */}
      <Box sx={{ p: 2 }}>
        <ListItemButton
          onClick={handleLogout}
          sx={{
            borderRadius: '8px',
            py: 1.2,
            color: '#94a3b8',
            '&:hover': {
              background: 'rgba(239, 68, 68, 0.1)',
              color: '#f87171',
            }
          }}
        >
          <ListItemIcon sx={{ color: 'inherit', minWidth: 40 }}>
            <LogoutIcon />
          </ListItemIcon>
          <ListItemText primary="Logout" primaryTypographyProps={{ fontWeight: 500, fontSize: '0.9rem' }} />
        </ListItemButton>

        {/* System Status */}
        <Box sx={{ mt: 2, px: 1 }}>
          <Typography sx={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, mb: 1 }}>
            System Status
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box className="status-dot online" />
            <Typography sx={{ fontSize: '0.85rem', color: '#10b981', fontWeight: 500 }}>
              Online
            </Typography>
          </Box>
          <Typography sx={{ fontSize: '0.7rem', color: '#64748b', mt: 1 }}>
            v1.0.0 Enterprise
          </Typography>
        </Box>
      </Box>
    </Box>
  )

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', background: '#0a0b14' }}>
      {/* Sidebar */}
      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
      >
        {/* Mobile Drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              background: 'transparent',
              border: 'none'
            },
          }}
        >
          {drawer}
        </Drawer>

        {/* Desktop Drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              background: 'transparent',
              border: 'none'
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        {/* Top Bar with Mobile Menu & User Info */}
        <Box sx={{
          height: 56,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 2,
          background: 'rgba(13, 14, 23, 0.8)',
          backdropFilter: 'blur(8px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          position: 'sticky',
          top: 0,
          zIndex: 10
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ display: { md: 'none' }, color: '#94a3b8' }}
            >
              <MenuIcon />
            </IconButton>
            <Typography sx={{ fontWeight: 600, color: '#fff', fontSize: '1rem' }}>
              Trading Signals Pro
            </Typography>
          </Box>
          {user && (
            <Typography sx={{ color: '#94a3b8', fontSize: '0.85rem' }}>
              {user.email}
            </Typography>
          )}
        </Box>

        {/* Ticker Bar */}
        <TickerBar />

        {/* Page Content */}
        <Box sx={{ flex: 1, p: 3 }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  )
}
