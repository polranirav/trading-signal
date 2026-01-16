/**
 * Alerts Page
 * 
 * Dedicated page for managing trading alerts:
 * - View and manage active alerts
 * - Create new alerts (price, technical, volatility)
 * - View notification history
 * - Quick alert templates
 */

import { Box, Typography } from '@mui/material'
import { AlertSystemPanel } from '../../components/AlertSystemPanel'
import '../../styles/premium.css'

export default function AlertsPage() {
    return (
        <Box className="fade-in" sx={{ pb: 4 }}>
            {/* Page Header */}
            <Box sx={{ mb: 4 }}>
                <Typography
                    variant="h4"
                    sx={{ fontWeight: 700, color: '#fff', fontSize: '1.75rem', mb: 0.5 }}
                >
                    🔔 Alerts & Notifications
                </Typography>
                <Typography sx={{ color: '#64748b', fontSize: '0.9rem' }}>
                    Stay ahead of the market with real-time price, technical, and volatility alerts
                </Typography>
            </Box>

            {/* Alert System Panel */}
            <AlertSystemPanel />
        </Box>
    )
}
