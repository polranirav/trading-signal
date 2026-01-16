import React from 'react'
import { Box, Typography } from '@mui/material'
import '../styles/premium.css'

interface SignalBadgeProps {
    type: 'BUY' | 'SELL' | 'HOLD' | 'STRONG_BUY' | 'STRONG_SELL' | 'TAKE_PROFIT' | 'REVIEW' | 'STOP_LOSS' | string
}

export function SignalBadge({ type }: SignalBadgeProps) {
    const getBadgeClass = () => {
        const upperType = type.toUpperCase()
        if (upperType.includes('STRONG_BUY') || upperType === 'BUY') return 'buy'
        if (upperType.includes('STRONG_SELL') || upperType === 'SELL') return 'sell'
        if (upperType === 'TAKE_PROFIT') return 'take-profit'
        if (upperType === 'REVIEW') return 'review'
        if (upperType === 'STOP_LOSS') return 'stop-loss'
        return 'hold'
    }

    return (
        <span className={`signal-badge ${getBadgeClass()}`}>
            {type.replace('_', ' ')}
        </span>
    )
}

interface ConfidenceBarProps {
    value: number // 0-1 or 0-100
    showLabel?: boolean
}

export function ConfidenceBar({ value, showLabel = true }: ConfidenceBarProps) {
    // Normalize to 0-100
    const percentage = value > 1 ? value : value * 100

    const getLevel = () => {
        if (percentage >= 70) return 'high'
        if (percentage >= 40) return 'medium'
        return 'low'
    }

    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
            <Box className="confidence-bar" sx={{ flex: 1 }}>
                <Box
                    className={`confidence-fill ${getLevel()}`}
                    sx={{ width: `${percentage}%` }}
                />
            </Box>
            {showLabel && (
                <Typography sx={{ fontSize: '0.8rem', fontWeight: 600, color: '#94a3b8', minWidth: 40 }}>
                    {percentage.toFixed(0)}%
                </Typography>
            )}
        </Box>
    )
}

interface OpportunityCardProps {
    symbol: string
    signalType: string
    confidence: number
    price?: number
    change?: string
    onClick?: () => void
}

export function OpportunityCard({
    symbol,
    signalType,
    confidence,
    price,
    change,
    onClick
}: OpportunityCardProps) {
    return (
        <Box className="opportunity-card" onClick={onClick}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography sx={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>
                    {symbol}
                </Typography>
                <SignalBadge type={signalType} />
            </Box>

            {price && (
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography sx={{ color: '#94a3b8', fontSize: '0.9rem' }}>
                        ${price.toFixed(2)}
                    </Typography>
                    {change && (
                        <Typography sx={{
                            color: change.startsWith('+') ? '#10b981' : '#ef4444',
                            fontSize: '0.85rem',
                            fontWeight: 600
                        }}>
                            {change}
                        </Typography>
                    )}
                </Box>
            )}

            <ConfidenceBar value={confidence} />
        </Box>
    )
}

interface SectionCardProps {
    title: string
    icon?: React.ReactNode
    iconColor?: string
    action?: React.ReactNode
    children: React.ReactNode
}

export function SectionCard({ title, icon, iconColor = '#3b82f6', action, children }: SectionCardProps) {
    return (
        <Box className="section-card">
            <Box className="section-header">
                <Typography className="section-title">
                    {icon && <Box component="span" sx={{ color: iconColor, display: 'flex', mr: 1 }}>{icon}</Box>}
                    {title}
                </Typography>
                {action}
            </Box>
            <Box className="section-body">
                {children}
            </Box>
        </Box>
    )
}
