/**
 * Correlation Heatmap Component
 * 
 * Displays correlation matrix between portfolio holdings:
 * - Visual heatmap with color coding
 * - Correlation coefficients
 * - Diversification score
 * - Risk concentration warnings
 */

import { useMemo } from 'react'
import { Box, Typography, Tooltip, Chip } from '@mui/material'
import GridViewIcon from '@mui/icons-material/GridView'
import WarningIcon from '@mui/icons-material/Warning'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'

interface CorrelationHeatmapProps {
    symbols: string[]
}

// Generate mock correlation data
const generateCorrelations = (symbols: string[]): number[][] => {
    const n = symbols.length
    const matrix: number[][] = []

    for (let i = 0; i < n; i++) {
        matrix[i] = []
        for (let j = 0; j < n; j++) {
            if (i === j) {
                matrix[i][j] = 1.0
            } else if (j < i) {
                matrix[i][j] = matrix[j][i]
            } else {
                // Generate realistic-looking correlations
                // Tech stocks tend to be more correlated
                const isTechI = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN'].includes(symbols[i])
                const isTechJ = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN'].includes(symbols[j])

                let baseCorr = 0.3 + Math.random() * 0.4
                if (isTechI && isTechJ) {
                    baseCorr = 0.6 + Math.random() * 0.3 // Higher correlation for tech stocks
                } else if (!isTechI && !isTechJ) {
                    baseCorr = 0.2 + Math.random() * 0.3 // Lower correlation for non-tech
                }

                matrix[i][j] = Math.round(baseCorr * 100) / 100
            }
        }
    }

    return matrix
}

const getCorrelationColor = (value: number) => {
    // Color scale from blue (negative) to white (zero) to red (positive)
    if (value >= 0.8) return 'rgba(239, 68, 68, 0.9)' // Strong positive - red
    if (value >= 0.6) return 'rgba(248, 113, 113, 0.7)' // Moderate positive
    if (value >= 0.4) return 'rgba(252, 165, 165, 0.5)' // Weak positive
    if (value >= 0.2) return 'rgba(255, 255, 255, 0.1)' // Near zero
    if (value >= 0) return 'rgba(147, 197, 253, 0.3)' // Very weak positive
    if (value >= -0.2) return 'rgba(96, 165, 250, 0.5)' // Weak negative
    if (value >= -0.4) return 'rgba(59, 130, 246, 0.6)' // Moderate negative
    return 'rgba(37, 99, 235, 0.8)' // Strong negative - blue
}

const getTextColor = (value: number) => {
    if (Math.abs(value) >= 0.6) return '#fff'
    return '#94a3b8'
}

export function CorrelationHeatmap({ symbols }: CorrelationHeatmapProps) {
    const correlationMatrix = useMemo(() => generateCorrelations(symbols), [symbols])

    // Calculate average correlation (excluding diagonal)
    const avgCorrelation = useMemo(() => {
        let sum = 0
        let count = 0
        for (let i = 0; i < correlationMatrix.length; i++) {
            for (let j = i + 1; j < correlationMatrix.length; j++) {
                sum += correlationMatrix[i][j]
                count++
            }
        }
        return count > 0 ? sum / count : 0
    }, [correlationMatrix])

    // Count high correlations
    const highCorrelationPairs = useMemo(() => {
        const pairs: Array<{ a: string; b: string; corr: number }> = []
        for (let i = 0; i < correlationMatrix.length; i++) {
            for (let j = i + 1; j < correlationMatrix.length; j++) {
                if (correlationMatrix[i][j] >= 0.65) {
                    pairs.push({ a: symbols[i], b: symbols[j], corr: correlationMatrix[i][j] })
                }
            }
        }
        return pairs.sort((a, b) => b.corr - a.corr)
    }, [correlationMatrix, symbols])

    // Diversification score (inverse of average correlation)
    const diversificationScore = Math.max(0, Math.min(100, (1 - avgCorrelation) * 100))

    const getDiversificationLabel = () => {
        if (diversificationScore >= 70) return { label: 'Well Diversified', color: '#10b981' }
        if (diversificationScore >= 50) return { label: 'Moderately Diversified', color: '#f59e0b' }
        return { label: 'Concentrated Risk', color: '#ef4444' }
    }

    const divLabel = getDiversificationLabel()

    if (symbols.length < 2) {
        return (
            <Box sx={{
                background: 'rgba(15, 23, 42, 0.8)',
                border: '1px solid rgba(255,255,255,0.05)',
                borderRadius: 3,
                p: 3,
            }}>
                <Typography color="text.secondary" textAlign="center">
                    Add at least 2 holdings to view correlation analysis
                </Typography>
            </Box>
        )
    }

    return (
        <Box sx={{
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid rgba(255,255,255,0.05)',
            borderRadius: 3,
            p: 3,
        }}>
            {/* Header */}
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <GridViewIcon sx={{ color: '#8b5cf6' }} />
                    <Typography variant="h6" fontWeight={700}>
                        Correlation Heatmap
                    </Typography>
                </Box>
                <Chip
                    icon={diversificationScore >= 50 ?
                        <CheckCircleIcon sx={{ fontSize: '14px !important' }} /> :
                        <WarningIcon sx={{ fontSize: '14px !important' }} />
                    }
                    label={`${divLabel.label} (${diversificationScore.toFixed(0)}%)`}
                    size="small"
                    sx={{
                        bgcolor: `${divLabel.color}15`,
                        color: divLabel.color,
                        fontWeight: 600,
                    }}
                />
            </Box>

            {/* Metrics Row */}
            <Box sx={{ display: 'flex', gap: 3, mb: 3, flexWrap: 'wrap' }}>
                <Box sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2, minWidth: 120 }}>
                    <Typography variant="caption" color="text.secondary">Avg Correlation</Typography>
                    <Typography variant="h6" sx={{ color: avgCorrelation > 0.5 ? '#ef4444' : '#10b981', fontWeight: 700 }}>
                        {avgCorrelation.toFixed(2)}
                    </Typography>
                </Box>
                <Box sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2, minWidth: 120 }}>
                    <Typography variant="caption" color="text.secondary">High Correlation Pairs</Typography>
                    <Typography variant="h6" sx={{ color: highCorrelationPairs.length > 2 ? '#f59e0b' : '#10b981', fontWeight: 700 }}>
                        {highCorrelationPairs.length}
                    </Typography>
                </Box>
                <Box sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2, minWidth: 120 }}>
                    <Typography variant="caption" color="text.secondary">Total Positions</Typography>
                    <Typography variant="h6" sx={{ color: '#60a5fa', fontWeight: 700 }}>
                        {symbols.length}
                    </Typography>
                </Box>
            </Box>

            {/* Heatmap */}
            <Box sx={{ overflowX: 'auto' }}>
                <Box sx={{ minWidth: Math.max(400, symbols.length * 50 + 60) }}>
                    {/* Column headers */}
                    <Box sx={{ display: 'flex', pl: 7 }}>
                        {symbols.map(symbol => (
                            <Typography
                                key={symbol}
                                variant="caption"
                                sx={{
                                    width: 48,
                                    textAlign: 'center',
                                    fontWeight: 600,
                                    color: '#94a3b8',
                                    transform: 'rotate(-45deg)',
                                    transformOrigin: 'left bottom',
                                    whiteSpace: 'nowrap',
                                    mb: 1,
                                }}
                            >
                                {symbol}
                            </Typography>
                        ))}
                    </Box>

                    {/* Matrix rows */}
                    {symbols.map((rowSymbol, i) => (
                        <Box key={rowSymbol} sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography
                                variant="caption"
                                sx={{
                                    width: 56,
                                    textAlign: 'right',
                                    pr: 1,
                                    fontWeight: 600,
                                    color: '#94a3b8',
                                }}
                            >
                                {rowSymbol}
                            </Typography>
                            {symbols.map((colSymbol, j) => (
                                <Tooltip
                                    key={`${rowSymbol}-${colSymbol}`}
                                    title={`${rowSymbol} ↔ ${colSymbol}: ${correlationMatrix[i][j].toFixed(2)}`}
                                >
                                    <Box
                                        sx={{
                                            width: 44,
                                            height: 36,
                                            bgcolor: getCorrelationColor(correlationMatrix[i][j]),
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            m: 0.25,
                                            borderRadius: 1,
                                            cursor: 'default',
                                            border: i === j ? '2px solid rgba(255,255,255,0.2)' : 'none',
                                        }}
                                    >
                                        <Typography
                                            variant="caption"
                                            sx={{
                                                fontSize: '0.65rem',
                                                fontWeight: 600,
                                                color: getTextColor(correlationMatrix[i][j]),
                                            }}
                                        >
                                            {correlationMatrix[i][j].toFixed(2)}
                                        </Typography>
                                    </Box>
                                </Tooltip>
                            ))}
                        </Box>
                    ))}
                </Box>
            </Box>

            {/* Legend */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 3 }}>
                <Typography variant="caption" color="text.secondary">Correlation:</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Box sx={{ width: 20, height: 12, bgcolor: 'rgba(37, 99, 235, 0.8)', borderRadius: 0.5 }} />
                    <Typography variant="caption" color="text.secondary">-1</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Box sx={{ width: 20, height: 12, bgcolor: 'rgba(255, 255, 255, 0.1)', borderRadius: 0.5 }} />
                    <Typography variant="caption" color="text.secondary">0</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Box sx={{ width: 20, height: 12, bgcolor: 'rgba(239, 68, 68, 0.9)', borderRadius: 0.5 }} />
                    <Typography variant="caption" color="text.secondary">+1</Typography>
                </Box>
            </Box>

            {/* High Correlation Warnings */}
            {highCorrelationPairs.length > 0 && (
                <Box sx={{ mt: 3, p: 2, bgcolor: 'rgba(245, 158, 11, 0.1)', borderRadius: 2, border: '1px solid rgba(245, 158, 11, 0.2)' }}>
                    <Typography variant="subtitle2" sx={{ color: '#fbbf24', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                        <WarningIcon sx={{ fontSize: 16 }} />
                        High Correlation Pairs (≥0.65)
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {highCorrelationPairs.slice(0, 5).map(pair => (
                            <Chip
                                key={`${pair.a}-${pair.b}`}
                                label={`${pair.a} ↔ ${pair.b}: ${pair.corr.toFixed(2)}`}
                                size="small"
                                sx={{
                                    bgcolor: 'rgba(239, 68, 68, 0.15)',
                                    color: '#f87171',
                                    fontSize: '0.7rem',
                                }}
                            />
                        ))}
                        {highCorrelationPairs.length > 5 && (
                            <Chip
                                label={`+${highCorrelationPairs.length - 5} more`}
                                size="small"
                                sx={{
                                    bgcolor: 'rgba(255,255,255,0.05)',
                                    color: '#94a3b8',
                                    fontSize: '0.7rem',
                                }}
                            />
                        )}
                    </Box>
                </Box>
            )}
        </Box>
    )
}

export default CorrelationHeatmap
