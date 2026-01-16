/**
 * Alert System Component
 * 
 * Comprehensive alert management:
 * - Price threshold alerts
 * - Technical indicator crossovers
 * - Volatility spike alerts
 * - Custom alert creation
 * - Alert history and notifications
 */

import { useState } from 'react'
import {
    Box,
    Typography,
    Grid,
    Button,
    TextField,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Chip,
    IconButton,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Switch,
    Tooltip,
    Badge,
} from '@mui/material'
import NotificationsIcon from '@mui/icons-material/Notifications'
import NotificationsActiveIcon from '@mui/icons-material/NotificationsActive'
import AddAlertIcon from '@mui/icons-material/AddAlert'
import DeleteIcon from '@mui/icons-material/Delete'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import WarningIcon from '@mui/icons-material/Warning'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import AccessTimeIcon from '@mui/icons-material/AccessTime'

// Alert types
type AlertType = 'price_above' | 'price_below' | 'rsi_overbought' | 'rsi_oversold' | 'macd_bullish' | 'macd_bearish' | 'volatility_spike' | 'golden_cross' | 'death_cross'

interface Alert {
    id: string
    symbol: string
    type: AlertType
    value?: number
    enabled: boolean
    triggered: boolean
    triggeredAt?: Date
    createdAt: Date
}

interface TriggeredAlert {
    id: string
    symbol: string
    message: string
    severity: 'info' | 'warning' | 'success' | 'error'
    timestamp: Date
    read: boolean
}

const alertTypeLabels: Record<AlertType, { label: string; description: string; icon: JSX.Element }> = {
    price_above: { label: 'Price Above', description: 'Alert when price rises above target', icon: <TrendingUpIcon /> },
    price_below: { label: 'Price Below', description: 'Alert when price falls below target', icon: <TrendingDownIcon /> },
    rsi_overbought: { label: 'RSI Overbought', description: 'Alert when RSI > 70', icon: <ShowChartIcon /> },
    rsi_oversold: { label: 'RSI Oversold', description: 'Alert when RSI < 30', icon: <ShowChartIcon /> },
    macd_bullish: { label: 'MACD Bullish Crossover', description: 'Signal line crosses above MACD', icon: <TrendingUpIcon /> },
    macd_bearish: { label: 'MACD Bearish Crossover', description: 'Signal line crosses below MACD', icon: <TrendingDownIcon /> },
    volatility_spike: { label: 'Volatility Spike', description: 'VIX jumps above threshold', icon: <WarningIcon /> },
    golden_cross: { label: 'Golden Cross', description: '50 SMA crosses above 200 SMA', icon: <TrendingUpIcon /> },
    death_cross: { label: 'Death Cross', description: '50 SMA crosses below 200 SMA', icon: <TrendingDownIcon /> },
}

// Mock initial alerts
const initialAlerts: Alert[] = [
    { id: '1', symbol: 'AAPL', type: 'price_above', value: 200, enabled: true, triggered: false, createdAt: new Date() },
    { id: '2', symbol: 'NVDA', type: 'price_below', value: 450, enabled: true, triggered: false, createdAt: new Date() },
    { id: '3', symbol: 'TSLA', type: 'rsi_oversold', enabled: true, triggered: true, triggeredAt: new Date(Date.now() - 3600000), createdAt: new Date() },
    { id: '4', symbol: 'SPY', type: 'volatility_spike', value: 25, enabled: true, triggered: false, createdAt: new Date() },
    { id: '5', symbol: 'MSFT', type: 'golden_cross', enabled: false, triggered: false, createdAt: new Date() },
]

// Mock triggered alerts/notifications
const initialNotifications: TriggeredAlert[] = [
    { id: '1', symbol: 'TSLA', message: 'RSI dropped below 30 (Oversold)', severity: 'warning', timestamp: new Date(Date.now() - 3600000), read: false },
    { id: '2', symbol: 'NVDA', message: 'Price reached $495.22 (+1.7%)', severity: 'success', timestamp: new Date(Date.now() - 7200000), read: true },
    { id: '3', symbol: 'SPY', message: 'MACD bullish crossover detected', severity: 'info', timestamp: new Date(Date.now() - 14400000), read: true },
    { id: '4', symbol: 'VIX', message: 'Volatility spike: VIX at 18.5 (+15%)', severity: 'error', timestamp: new Date(Date.now() - 86400000), read: true },
]

export function AlertSystemPanel() {
    const [alerts, setAlerts] = useState<Alert[]>(initialAlerts)
    const [notifications, setNotifications] = useState<TriggeredAlert[]>(initialNotifications)
    const [createDialogOpen, setCreateDialogOpen] = useState(false)
    const [newAlert, setNewAlert] = useState({ symbol: '', type: 'price_above' as AlertType, value: '' })

    const unreadCount = notifications.filter(n => !n.read).length

    const handleCreateAlert = () => {
        if (!newAlert.symbol) return

        const alert: Alert = {
            id: Date.now().toString(),
            symbol: newAlert.symbol.toUpperCase(),
            type: newAlert.type,
            value: newAlert.value ? parseFloat(newAlert.value) : undefined,
            enabled: true,
            triggered: false,
            createdAt: new Date(),
        }

        setAlerts([...alerts, alert])
        setCreateDialogOpen(false)
        setNewAlert({ symbol: '', type: 'price_above', value: '' })
    }

    const handleToggleAlert = (id: string) => {
        setAlerts(alerts.map(a => a.id === id ? { ...a, enabled: !a.enabled } : a))
    }

    const handleDeleteAlert = (id: string) => {
        setAlerts(alerts.filter(a => a.id !== id))
    }

    const handleMarkAllRead = () => {
        setNotifications(notifications.map(n => ({ ...n, read: true })))
    }

    const getSeverityColor = (severity: TriggeredAlert['severity']) => {
        switch (severity) {
            case 'success': return '#10b981'
            case 'warning': return '#f59e0b'
            case 'error': return '#ef4444'
            default: return '#3b82f6'
        }
    }

    const needsValue = (type: AlertType) => ['price_above', 'price_below', 'volatility_spike'].includes(type)

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Badge badgeContent={unreadCount} color="error">
                        <NotificationsActiveIcon sx={{ color: '#f59e0b', fontSize: 28 }} />
                    </Badge>
                    <Typography variant="h5" fontWeight={700}>
                        Alert Center
                    </Typography>
                </Box>
                <Button
                    variant="contained"
                    startIcon={<AddAlertIcon />}
                    onClick={() => setCreateDialogOpen(true)}
                    sx={{
                        background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                        '&:hover': { background: 'linear-gradient(135deg, #2563eb, #7c3aed)' }
                    }}
                >
                    Create Alert
                </Button>
            </Box>

            <Grid container spacing={3}>
                {/* Active Alerts */}
                <Grid item xs={12} md={7}>
                    <Box sx={{
                        background: 'rgba(15, 23, 42, 0.8)',
                        border: '1px solid rgba(255,255,255,0.05)',
                        borderRadius: 3,
                        p: 3,
                    }}>
                        <Typography variant="h6" fontWeight={700} sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                            <NotificationsIcon sx={{ color: '#3b82f6' }} />
                            Active Alerts ({alerts.filter(a => a.enabled).length})
                        </Typography>

                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            {alerts.length === 0 ? (
                                <Typography color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                                    No alerts configured. Create your first alert to get started.
                                </Typography>
                            ) : (
                                alerts.map(alert => (
                                    <Box
                                        key={alert.id}
                                        sx={{
                                            p: 2,
                                            borderRadius: 2,
                                            bgcolor: alert.enabled ? 'rgba(255,255,255,0.03)' : 'rgba(255,255,255,0.01)',
                                            border: `1px solid ${alert.triggered ? 'rgba(16, 185, 129, 0.3)' : 'rgba(255,255,255,0.05)'}`,
                                            opacity: alert.enabled ? 1 : 0.5,
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                        }}
                                    >
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                            <Box sx={{
                                                width: 40,
                                                height: 40,
                                                borderRadius: 2,
                                                bgcolor: alert.triggered ? 'rgba(16, 185, 129, 0.15)' : 'rgba(59, 130, 246, 0.15)',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                color: alert.triggered ? '#10b981' : '#60a5fa',
                                            }}>
                                                {alertTypeLabels[alert.type].icon}
                                            </Box>
                                            <Box>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <Typography fontWeight={700}>{alert.symbol}</Typography>
                                                    <Chip
                                                        label={alertTypeLabels[alert.type].label}
                                                        size="small"
                                                        sx={{
                                                            bgcolor: 'rgba(139, 92, 246, 0.15)',
                                                            color: '#a78bfa',
                                                            fontSize: '0.65rem',
                                                        }}
                                                    />
                                                    {alert.triggered && (
                                                        <Chip
                                                            icon={<CheckCircleIcon sx={{ fontSize: '14px !important' }} />}
                                                            label="Triggered"
                                                            size="small"
                                                            sx={{
                                                                bgcolor: 'rgba(16, 185, 129, 0.15)',
                                                                color: '#10b981',
                                                                fontSize: '0.65rem',
                                                            }}
                                                        />
                                                    )}
                                                </Box>
                                                <Typography variant="caption" color="text.secondary">
                                                    {alertTypeLabels[alert.type].description}
                                                    {alert.value !== undefined && ` • Target: $${alert.value}`}
                                                </Typography>
                                            </Box>
                                        </Box>

                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Switch
                                                checked={alert.enabled}
                                                onChange={() => handleToggleAlert(alert.id)}
                                                size="small"
                                            />
                                            <IconButton
                                                size="small"
                                                onClick={() => handleDeleteAlert(alert.id)}
                                                sx={{ color: '#ef4444' }}
                                            >
                                                <DeleteIcon fontSize="small" />
                                            </IconButton>
                                        </Box>
                                    </Box>
                                ))
                            )}
                        </Box>
                    </Box>
                </Grid>

                {/* Notifications */}
                <Grid item xs={12} md={5}>
                    <Box sx={{
                        background: 'rgba(15, 23, 42, 0.8)',
                        border: '1px solid rgba(255,255,255,0.05)',
                        borderRadius: 3,
                        p: 3,
                        height: '100%',
                    }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                            <Typography variant="h6" fontWeight={700} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <AccessTimeIcon sx={{ color: '#f59e0b' }} />
                                Recent Notifications
                            </Typography>
                            {unreadCount > 0 && (
                                <Button size="small" onClick={handleMarkAllRead} sx={{ textTransform: 'none' }}>
                                    Mark all read
                                </Button>
                            )}
                        </Box>

                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, maxHeight: 400, overflowY: 'auto' }}>
                            {notifications.map(notification => (
                                <Box
                                    key={notification.id}
                                    sx={{
                                        p: 2,
                                        borderRadius: 2,
                                        bgcolor: notification.read ? 'rgba(255,255,255,0.02)' : 'rgba(59, 130, 246, 0.1)',
                                        borderLeft: `3px solid ${getSeverityColor(notification.severity)}`,
                                    }}
                                >
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                        <Box>
                                            <Typography variant="body2" fontWeight={notification.read ? 400 : 700}>
                                                {notification.symbol}
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                {notification.message}
                                            </Typography>
                                        </Box>
                                        {!notification.read && (
                                            <Box sx={{
                                                width: 8,
                                                height: 8,
                                                borderRadius: '50%',
                                                bgcolor: '#3b82f6',
                                            }} />
                                        )}
                                    </Box>
                                    <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                                        {notification.timestamp.toLocaleString()}
                                    </Typography>
                                </Box>
                            ))}
                        </Box>
                    </Box>
                </Grid>
            </Grid>

            {/* Quick Alert Templates */}
            <Box sx={{
                background: 'rgba(15, 23, 42, 0.8)',
                border: '1px solid rgba(255,255,255,0.05)',
                borderRadius: 3,
                p: 3,
            }}>
                <Typography variant="h6" fontWeight={700} sx={{ mb: 2 }}>
                    Quick Alert Templates
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.5 }}>
                    {[
                        { label: 'RSI Overbought Alert', type: 'rsi_overbought', color: '#ef4444' },
                        { label: 'RSI Oversold Alert', type: 'rsi_oversold', color: '#10b981' },
                        { label: 'Golden Cross Alert', type: 'golden_cross', color: '#f59e0b' },
                        { label: 'Death Cross Alert', type: 'death_cross', color: '#ef4444' },
                        { label: 'MACD Bullish Signal', type: 'macd_bullish', color: '#10b981' },
                        { label: 'VIX Spike (>20)', type: 'volatility_spike', color: '#8b5cf6' },
                    ].map(template => (
                        <Tooltip key={template.type} title={`Create ${template.label}`}>
                            <Chip
                                label={template.label}
                                onClick={() => {
                                    setNewAlert({ symbol: '', type: template.type as AlertType, value: template.type === 'volatility_spike' ? '20' : '' })
                                    setCreateDialogOpen(true)
                                }}
                                sx={{
                                    bgcolor: `${template.color}15`,
                                    color: template.color,
                                    border: `1px solid ${template.color}30`,
                                    cursor: 'pointer',
                                    '&:hover': { bgcolor: `${template.color}25` }
                                }}
                            />
                        </Tooltip>
                    ))}
                </Box>
            </Box>

            {/* Create Alert Dialog */}
            <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle sx={{
                    background: 'linear-gradient(135deg, #0a0b14, #1a1b2e)',
                    borderBottom: '1px solid rgba(255,255,255,0.1)'
                }}>
                    Create New Alert
                </DialogTitle>
                <DialogContent sx={{ background: '#0a0b14', pt: 3 }}>
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                        <Grid item xs={12}>
                            <TextField
                                fullWidth
                                label="Symbol"
                                value={newAlert.symbol}
                                onChange={(e) => setNewAlert({ ...newAlert, symbol: e.target.value.toUpperCase() })}
                                placeholder="e.g., AAPL"
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <FormControl fullWidth>
                                <InputLabel>Alert Type</InputLabel>
                                <Select
                                    value={newAlert.type}
                                    onChange={(e) => setNewAlert({ ...newAlert, type: e.target.value as AlertType })}
                                    label="Alert Type"
                                >
                                    {Object.entries(alertTypeLabels).map(([type, config]) => (
                                        <MenuItem key={type} value={type}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                {config.icon}
                                                {config.label}
                                            </Box>
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        {needsValue(newAlert.type) && (
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label={newAlert.type === 'volatility_spike' ? 'VIX Threshold' : 'Target Price'}
                                    type="number"
                                    value={newAlert.value}
                                    onChange={(e) => setNewAlert({ ...newAlert, value: e.target.value })}
                                    placeholder={newAlert.type === 'volatility_spike' ? 'e.g., 25' : 'e.g., 150.00'}
                                    InputProps={{
                                        startAdornment: newAlert.type !== 'volatility_spike' ? (
                                            <Typography sx={{ mr: 0.5, color: '#94a3b8' }}>$</Typography>
                                        ) : undefined
                                    }}
                                />
                            </Grid>
                        )}
                    </Grid>

                    <Box sx={{ mt: 3, p: 2, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 2 }}>
                        <Typography variant="caption" color="text.secondary">
                            💡 <strong>Tip:</strong> {alertTypeLabels[newAlert.type].description}
                        </Typography>
                    </Box>
                </DialogContent>
                <DialogActions sx={{ background: '#0a0b14', p: 2 }}>
                    <Button onClick={() => setCreateDialogOpen(false)} color="inherit">Cancel</Button>
                    <Button
                        onClick={handleCreateAlert}
                        variant="contained"
                        disabled={!newAlert.symbol || (needsValue(newAlert.type) && !newAlert.value)}
                    >
                        Create Alert
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    )
}

export default AlertSystemPanel
