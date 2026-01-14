import React from 'react'
import { Box, Typography } from '@mui/material'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import StorageIcon from '@mui/icons-material/Storage'
import BarChartIcon from '@mui/icons-material/BarChart'
import AttachMoneyIcon from '@mui/icons-material/AttachMoney'
import '../styles/premium.css'

// Icon mapping
const iconMap: Record<string, React.ReactNode> = {
    'trending-up': <TrendingUpIcon />,
    'trending-down': <TrendingDownIcon />,
    'chart': <ShowChartIcon />,
    'server': <StorageIcon />,
    'bar-chart': <BarChartIcon />,
    'money': <AttachMoneyIcon />,
}

// Color variants
type ColorVariant = 'blue' | 'green' | 'red' | 'purple' | 'amber'
type SentimentVariant = 'bullish' | 'bearish' | 'neutral' | undefined

interface MetricCardProps {
    icon?: string
    iconColor?: ColorVariant
    label: string
    value: React.ReactNode
    subText?: string
    sentiment?: SentimentVariant
    onClick?: () => void
    className?: string
}

export default function MetricCard({
    icon = 'chart',
    iconColor = 'blue',
    label,
    value,
    subText,
    sentiment,
    onClick,
    className = ''
}: MetricCardProps) {
    // Determine value color based on sentiment
    const getValueColor = () => {
        switch (sentiment) {
            case 'bullish': return '#10b981'
            case 'bearish': return '#ef4444'
            case 'neutral': return '#f59e0b'
            default: return '#ffffff'
        }
    }

    return (
        <Box
            className={`metric-card ${sentiment || ''} ${className} fade-in`}
            onClick={onClick}
            sx={{ cursor: onClick ? 'pointer' : 'default' }}
        >
            {/* Icon */}
            <Box className={`metric-icon ${iconColor}`}>
                {iconMap[icon] || <ShowChartIcon />}
            </Box>

            {/* Label */}
            <Typography
                sx={{
                    fontSize: '0.75rem',
                    color: '#64748b',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    fontWeight: 600,
                    mb: 0.5
                }}
            >
                {label}
            </Typography>

            {/* Value */}
            <Typography
                sx={{
                    fontSize: '1.75rem',
                    fontWeight: 700,
                    color: getValueColor(),
                    lineHeight: 1.2
                }}
            >
                {value}
            </Typography>

            {/* Sub Text */}
            {subText && (
                <Typography
                    sx={{
                        fontSize: '0.8rem',
                        color: '#64748b',
                        mt: 0.5
                    }}
                >
                    {subText}
                </Typography>
            )}
        </Box>
    )
}

// Convenience components for specific card types
export function BuySignalsCard({ count, subText }: { count: number, subText?: string }) {
    return (
        <MetricCard
            icon="trending-up"
            iconColor="green"
            label="Buy Signals"
            value={count}
            subText={subText || 'Active opportunities'}
            sentiment="bullish"
        />
    )
}

export function SellSignalsCard({ count, subText }: { count: number, subText?: string }) {
    return (
        <MetricCard
            icon="trending-down"
            iconColor="red"
            label="Sell Signals"
            value={count}
            subText={subText || 'Risk warnings'}
            sentiment="bearish"
        />
    )
}

export function SystemStatusCard({ status = 'Online' }: { status?: string }) {
    return (
        <MetricCard
            icon="server"
            iconColor="purple"
            label="System Status"
            value={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box className="status-dot online" />
                    <span>{status}</span>
                </Box>
            }
            subText={`Last updated: ${new Date().toLocaleTimeString()}`}
        />
    )
}
