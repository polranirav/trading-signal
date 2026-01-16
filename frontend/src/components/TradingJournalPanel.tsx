/**
 * Trading Journal Component
 * 
 * Comprehensive trade logging and analysis:
 * - Trade entry/exit records
 * - P&L tracking
 * - Win/Loss analysis
 * - Strategy tagging
 * - Notes and lessons learned
 */

import { useState, useMemo } from 'react'
import {
    Box,
    Typography,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    IconButton,
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField,
    Grid,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Tooltip,
} from '@mui/material'
import BookIcon from '@mui/icons-material/Book'
import AddIcon from '@mui/icons-material/Add'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import NoteAltIcon from '@mui/icons-material/NoteAlt'
import InfoIcon from '@mui/icons-material/Info'

interface Trade {
    id: string
    symbol: string
    entryDate: Date
    exitDate?: Date
    entryPrice: number
    exitPrice?: number
    quantity: number
    type: 'LONG' | 'SHORT'
    status: 'OPEN' | 'CLOSED'
    strategy: string
    notes: string
    lessons?: string
    pnl?: number
    pnlPercent?: number
}

// Mock trades data
const initialTrades: Trade[] = [
    {
        id: '1',
        symbol: 'NVDA',
        entryDate: new Date('2024-01-10'),
        exitDate: new Date('2024-01-15'),
        entryPrice: 485.50,
        exitPrice: 520.25,
        quantity: 50,
        type: 'LONG',
        status: 'CLOSED',
        strategy: 'Momentum',
        notes: 'Strong AI sector momentum, earnings catalyst approaching',
        lessons: 'Held through volatility - patience paid off',
        pnl: 1737.50,
        pnlPercent: 7.16,
    },
    {
        id: '2',
        symbol: 'AAPL',
        entryDate: new Date('2024-01-08'),
        exitDate: new Date('2024-01-12'),
        entryPrice: 182.75,
        exitPrice: 178.50,
        quantity: 100,
        type: 'LONG',
        status: 'CLOSED',
        strategy: 'Swing Trade',
        notes: 'Expected iPhone sales boost, market sentiment shifted',
        lessons: 'Should have set tighter stop-loss',
        pnl: -425.00,
        pnlPercent: -2.33,
    },
    {
        id: '3',
        symbol: 'TSLA',
        entryDate: new Date('2024-01-14'),
        entryPrice: 218.50,
        quantity: 75,
        type: 'LONG',
        status: 'OPEN',
        strategy: 'Breakout',
        notes: 'Breakout above resistance, volume confirmation',
    },
    {
        id: '4',
        symbol: 'MSFT',
        entryDate: new Date('2024-01-05'),
        exitDate: new Date('2024-01-11'),
        entryPrice: 372.80,
        exitPrice: 388.45,
        quantity: 40,
        type: 'LONG',
        status: 'CLOSED',
        strategy: 'Trend Following',
        notes: 'Cloud growth thesis, strong technicals',
        lessons: 'Exited too early - could have captured more upside',
        pnl: 626.00,
        pnlPercent: 4.20,
    },
    {
        id: '5',
        symbol: 'AMZN',
        entryDate: new Date('2024-01-02'),
        exitDate: new Date('2024-01-09'),
        entryPrice: 152.40,
        exitPrice: 167.80,
        quantity: 60,
        type: 'LONG',
        status: 'CLOSED',
        strategy: 'Mean Reversion',
        notes: 'Oversold bounce play after December weakness',
        lessons: 'Mean reversion works well in strong uptrends',
        pnl: 924.00,
        pnlPercent: 10.10,
    },
]

const strategies = ['Momentum', 'Swing Trade', 'Breakout', 'Trend Following', 'Mean Reversion', 'Value', 'Scalp', 'Other']

export function TradingJournalPanel() {
    const [trades, setTrades] = useState<Trade[]>(initialTrades)
    const [addDialogOpen, setAddDialogOpen] = useState(false)
    const [selectedTrade, setSelectedTrade] = useState<Trade | null>(null)
    const [newTrade, setNewTrade] = useState({
        symbol: '',
        entryPrice: '',
        quantity: '',
        type: 'LONG' as 'LONG' | 'SHORT',
        strategy: 'Momentum',
        notes: '',
    })

    // Calculate statistics
    const stats = useMemo(() => {
        const closedTrades = trades.filter(t => t.status === 'CLOSED')
        const winningTrades = closedTrades.filter(t => (t.pnl || 0) > 0)
        const losingTrades = closedTrades.filter(t => (t.pnl || 0) < 0)

        const totalPnl = closedTrades.reduce((sum, t) => sum + (t.pnl || 0), 0)
        const winRate = closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0

        const avgWin = winningTrades.length > 0
            ? winningTrades.reduce((sum, t) => sum + (t.pnl || 0), 0) / winningTrades.length
            : 0
        const avgLoss = losingTrades.length > 0
            ? losingTrades.reduce((sum, t) => sum + (t.pnl || 0), 0) / losingTrades.length
            : 0

        const profitFactor = avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : 0

        return {
            totalTrades: trades.length,
            closedTrades: closedTrades.length,
            openTrades: trades.filter(t => t.status === 'OPEN').length,
            winningTrades: winningTrades.length,
            losingTrades: losingTrades.length,
            winRate,
            totalPnl,
            avgWin,
            avgLoss,
            profitFactor,
        }
    }, [trades])

    const handleAddTrade = () => {
        if (!newTrade.symbol || !newTrade.entryPrice || !newTrade.quantity) return

        const trade: Trade = {
            id: Date.now().toString(),
            symbol: newTrade.symbol.toUpperCase(),
            entryDate: new Date(),
            entryPrice: parseFloat(newTrade.entryPrice),
            quantity: parseInt(newTrade.quantity),
            type: newTrade.type,
            status: 'OPEN',
            strategy: newTrade.strategy,
            notes: newTrade.notes,
        }

        setTrades([trade, ...trades])
        setAddDialogOpen(false)
        setNewTrade({ symbol: '', entryPrice: '', quantity: '', type: 'LONG', strategy: 'Momentum', notes: '' })
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
                    <BookIcon sx={{ color: '#8b5cf6' }} />
                    <Typography variant="h6" fontWeight={700}>
                        Trading Journal
                    </Typography>
                    <Chip
                        label={`${stats.totalTrades} trades`}
                        size="small"
                        sx={{ bgcolor: 'rgba(139, 92, 246, 0.15)', color: '#a78bfa', fontSize: '0.65rem' }}
                    />
                </Box>
                <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    size="small"
                    onClick={() => setAddDialogOpen(true)}
                    sx={{
                        background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
                        '&:hover': { background: 'linear-gradient(135deg, #7c3aed, #4f46e5)' }
                    }}
                >
                    Log Trade
                </Button>
            </Box>

            {/* Stats Row */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6} sm={4} md={2}>
                    <Box sx={{ p: 1.5, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2, textAlign: 'center' }}>
                        <Typography variant="caption" color="text.secondary">Win Rate</Typography>
                        <Typography variant="h6" sx={{ color: stats.winRate >= 50 ? '#10b981' : '#ef4444', fontWeight: 700 }}>
                            {stats.winRate.toFixed(1)}%
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={6} sm={4} md={2}>
                    <Box sx={{ p: 1.5, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2, textAlign: 'center' }}>
                        <Typography variant="caption" color="text.secondary">Total P&L</Typography>
                        <Typography variant="h6" sx={{ color: stats.totalPnl >= 0 ? '#10b981' : '#ef4444', fontWeight: 700 }}>
                            ${stats.totalPnl >= 0 ? '+' : ''}{stats.totalPnl.toLocaleString()}
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={6} sm={4} md={2}>
                    <Box sx={{ p: 1.5, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2, textAlign: 'center' }}>
                        <Typography variant="caption" color="text.secondary">Profit Factor</Typography>
                        <Typography variant="h6" sx={{ color: stats.profitFactor >= 1.5 ? '#10b981' : '#f59e0b', fontWeight: 700 }}>
                            {stats.profitFactor.toFixed(2)}
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={6} sm={4} md={2}>
                    <Box sx={{ p: 1.5, bgcolor: 'rgba(16, 185, 129, 0.1)', borderRadius: 2, textAlign: 'center' }}>
                        <Typography variant="caption" color="text.secondary">Winners</Typography>
                        <Typography variant="h6" sx={{ color: '#10b981', fontWeight: 700 }}>
                            {stats.winningTrades}
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={6} sm={4} md={2}>
                    <Box sx={{ p: 1.5, bgcolor: 'rgba(239, 68, 68, 0.1)', borderRadius: 2, textAlign: 'center' }}>
                        <Typography variant="caption" color="text.secondary">Losers</Typography>
                        <Typography variant="h6" sx={{ color: '#ef4444', fontWeight: 700 }}>
                            {stats.losingTrades}
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={6} sm={4} md={2}>
                    <Box sx={{ p: 1.5, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 2, textAlign: 'center' }}>
                        <Typography variant="caption" color="text.secondary">Open</Typography>
                        <Typography variant="h6" sx={{ color: '#60a5fa', fontWeight: 700 }}>
                            {stats.openTrades}
                        </Typography>
                    </Box>
                </Grid>
            </Grid>

            {/* Trades Table */}
            <TableContainer sx={{ maxHeight: 400 }}>
                <Table stickyHeader size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell sx={{ bgcolor: '#0f172a', color: '#94a3b8', fontWeight: 600 }}>Symbol</TableCell>
                            <TableCell sx={{ bgcolor: '#0f172a', color: '#94a3b8', fontWeight: 600 }}>Type</TableCell>
                            <TableCell sx={{ bgcolor: '#0f172a', color: '#94a3b8', fontWeight: 600 }}>Entry</TableCell>
                            <TableCell sx={{ bgcolor: '#0f172a', color: '#94a3b8', fontWeight: 600 }}>Exit</TableCell>
                            <TableCell sx={{ bgcolor: '#0f172a', color: '#94a3b8', fontWeight: 600 }}>Qty</TableCell>
                            <TableCell sx={{ bgcolor: '#0f172a', color: '#94a3b8', fontWeight: 600 }}>P&L</TableCell>
                            <TableCell sx={{ bgcolor: '#0f172a', color: '#94a3b8', fontWeight: 600 }}>Strategy</TableCell>
                            <TableCell sx={{ bgcolor: '#0f172a', color: '#94a3b8', fontWeight: 600 }}>Status</TableCell>
                            <TableCell sx={{ bgcolor: '#0f172a', color: '#94a3b8', fontWeight: 600 }} align="right">Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {trades.map(trade => (
                            <TableRow
                                key={trade.id}
                                sx={{ '&:hover': { bgcolor: 'rgba(255,255,255,0.02)' } }}
                            >
                                <TableCell>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Typography fontWeight={600}>{trade.symbol}</Typography>
                                    </Box>
                                </TableCell>
                                <TableCell>
                                    <Chip
                                        icon={trade.type === 'LONG' ? <TrendingUpIcon sx={{ fontSize: '12px !important' }} /> : <TrendingDownIcon sx={{ fontSize: '12px !important' }} />}
                                        label={trade.type}
                                        size="small"
                                        sx={{
                                            bgcolor: trade.type === 'LONG' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                                            color: trade.type === 'LONG' ? '#10b981' : '#ef4444',
                                            fontSize: '0.6rem',
                                            height: 20,
                                        }}
                                    />
                                </TableCell>
                                <TableCell>
                                    <Typography variant="body2">${trade.entryPrice.toFixed(2)}</Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        {trade.entryDate.toLocaleDateString()}
                                    </Typography>
                                </TableCell>
                                <TableCell>
                                    {trade.exitPrice ? (
                                        <>
                                            <Typography variant="body2">${trade.exitPrice.toFixed(2)}</Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                {trade.exitDate?.toLocaleDateString()}
                                            </Typography>
                                        </>
                                    ) : (
                                        <Typography variant="caption" color="text.secondary">-</Typography>
                                    )}
                                </TableCell>
                                <TableCell>{trade.quantity}</TableCell>
                                <TableCell>
                                    {trade.status === 'CLOSED' ? (
                                        <Box>
                                            <Typography
                                                variant="body2"
                                                sx={{
                                                    color: (trade.pnl || 0) >= 0 ? '#10b981' : '#ef4444',
                                                    fontWeight: 600,
                                                }}
                                            >
                                                {(trade.pnl || 0) >= 0 ? '+' : ''}${trade.pnl?.toFixed(2)}
                                            </Typography>
                                            <Typography variant="caption" sx={{ color: (trade.pnl || 0) >= 0 ? '#10b981' : '#ef4444' }}>
                                                ({(trade.pnlPercent || 0) >= 0 ? '+' : ''}{trade.pnlPercent?.toFixed(2)}%)
                                            </Typography>
                                        </Box>
                                    ) : (
                                        <Typography variant="caption" color="text.secondary">-</Typography>
                                    )}
                                </TableCell>
                                <TableCell>
                                    <Chip
                                        label={trade.strategy}
                                        size="small"
                                        sx={{
                                            bgcolor: 'rgba(139, 92, 246, 0.15)',
                                            color: '#a78bfa',
                                            fontSize: '0.6rem',
                                            height: 20,
                                        }}
                                    />
                                </TableCell>
                                <TableCell>
                                    <Chip
                                        label={trade.status}
                                        size="small"
                                        sx={{
                                            bgcolor: trade.status === 'OPEN' ? 'rgba(59, 130, 246, 0.15)' : 'rgba(100, 116, 139, 0.15)',
                                            color: trade.status === 'OPEN' ? '#60a5fa' : '#94a3b8',
                                            fontSize: '0.6rem',
                                            height: 20,
                                        }}
                                    />
                                </TableCell>
                                <TableCell align="right">
                                    <Tooltip title="View Details">
                                        <IconButton size="small" onClick={() => setSelectedTrade(trade)}>
                                            <InfoIcon sx={{ fontSize: 16, color: '#64748b' }} />
                                        </IconButton>
                                    </Tooltip>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>

            {/* Add Trade Dialog */}
            <Dialog open={addDialogOpen} onClose={() => setAddDialogOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle sx={{ background: 'linear-gradient(135deg, #0a0b14, #1a1b2e)', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <NoteAltIcon sx={{ color: '#8b5cf6' }} />
                        Log New Trade
                    </Box>
                </DialogTitle>
                <DialogContent sx={{ background: '#0a0b14', pt: 3 }}>
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                label="Symbol"
                                value={newTrade.symbol}
                                onChange={(e) => setNewTrade({ ...newTrade, symbol: e.target.value.toUpperCase() })}
                                placeholder="e.g., AAPL"
                            />
                        </Grid>
                        <Grid item xs={6}>
                            <FormControl fullWidth>
                                <InputLabel>Type</InputLabel>
                                <Select
                                    value={newTrade.type}
                                    onChange={(e) => setNewTrade({ ...newTrade, type: e.target.value as 'LONG' | 'SHORT' })}
                                    label="Type"
                                >
                                    <MenuItem value="LONG">Long</MenuItem>
                                    <MenuItem value="SHORT">Short</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                label="Entry Price"
                                type="number"
                                value={newTrade.entryPrice}
                                onChange={(e) => setNewTrade({ ...newTrade, entryPrice: e.target.value })}
                            />
                        </Grid>
                        <Grid item xs={6}>
                            <TextField
                                fullWidth
                                label="Quantity"
                                type="number"
                                value={newTrade.quantity}
                                onChange={(e) => setNewTrade({ ...newTrade, quantity: e.target.value })}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <FormControl fullWidth>
                                <InputLabel>Strategy</InputLabel>
                                <Select
                                    value={newTrade.strategy}
                                    onChange={(e) => setNewTrade({ ...newTrade, strategy: e.target.value })}
                                    label="Strategy"
                                >
                                    {strategies.map(s => (
                                        <MenuItem key={s} value={s}>{s}</MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12}>
                            <TextField
                                fullWidth
                                label="Notes"
                                multiline
                                rows={3}
                                value={newTrade.notes}
                                onChange={(e) => setNewTrade({ ...newTrade, notes: e.target.value })}
                                placeholder="Entry rationale, market conditions, etc."
                            />
                        </Grid>
                    </Grid>
                </DialogContent>
                <DialogActions sx={{ background: '#0a0b14', p: 2 }}>
                    <Button onClick={() => setAddDialogOpen(false)} color="inherit">Cancel</Button>
                    <Button
                        onClick={handleAddTrade}
                        variant="contained"
                        disabled={!newTrade.symbol || !newTrade.entryPrice || !newTrade.quantity}
                    >
                        Log Trade
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Trade Details Dialog */}
            <Dialog open={!!selectedTrade} onClose={() => setSelectedTrade(null)} maxWidth="sm" fullWidth>
                <DialogTitle sx={{ background: 'linear-gradient(135deg, #0a0b14, #1a1b2e)', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    Trade Details: {selectedTrade?.symbol}
                </DialogTitle>
                <DialogContent sx={{ background: '#0a0b14', pt: 3 }}>
                    {selectedTrade && (
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
                            <Box sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2 }}>
                                <Typography variant="subtitle2" color="text.secondary" gutterBottom>Notes</Typography>
                                <Typography variant="body2">{selectedTrade.notes}</Typography>
                            </Box>
                            {selectedTrade.lessons && (
                                <Box sx={{ p: 2, bgcolor: 'rgba(245, 158, 11, 0.1)', borderRadius: 2, border: '1px solid rgba(245, 158, 11, 0.2)' }}>
                                    <Typography variant="subtitle2" sx={{ color: '#fbbf24' }} gutterBottom>📚 Lessons Learned</Typography>
                                    <Typography variant="body2">{selectedTrade.lessons}</Typography>
                                </Box>
                            )}
                        </Box>
                    )}
                </DialogContent>
                <DialogActions sx={{ background: '#0a0b14', p: 2 }}>
                    <Button onClick={() => setSelectedTrade(null)}>Close</Button>
                </DialogActions>
            </Dialog>
        </Box>
    )
}

export default TradingJournalPanel
