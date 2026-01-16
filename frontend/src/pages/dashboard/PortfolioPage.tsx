/**
 * Portfolio Page
 * 
 * User's personal portfolio dashboard with:
 * - Holdings table with P&L tracking
 * - Import from CSV
 * - Add stocks manually
 * - AI-powered signals for holdings
 * - Risk management metrics
 */

import { useState } from 'react'
import { Box, Typography, Grid, Button, Dialog, DialogTitle, DialogContent, DialogActions, TextField, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, IconButton, Alert, Tooltip } from '@mui/material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import AddIcon from '@mui/icons-material/Add'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import DeleteIcon from '@mui/icons-material/Delete'
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet'
import PeopleIcon from '@mui/icons-material/People'
import HistoryIcon from '@mui/icons-material/History'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import NotificationsActiveIcon from '@mui/icons-material/NotificationsActive'
import apiClient from '../../services/api'
import MetricCard from '../../components/MetricCard'
import { SectionCard, SignalBadge } from '../../components/SignalComponents'
import ImportPortfolioModal from '../../components/ImportPortfolioModal'
import { RiskManagementPanel } from '../../components/RiskManagementPanel'
import { LiveMarketHeader } from '../../components/LiveMarketHeader'
import { CorrelationHeatmap } from '../../components/CorrelationHeatmap'
import { usePortfolio } from '../../context'
import '../../styles/premium.css'

// API extension for portfolio
const portfolioApi = {
    getSummary: () => apiClient.get('/portfolio/summary').then((r: any) => r.data),
    getSignals: () => apiClient.get('/portfolio/signals').then((r: any) => r.data),
    addHolding: (data: any) => apiClient.post('/portfolio/add', data).then((r: any) => r.data),
    deleteHolding: (id: string) => apiClient.delete(`/portfolio/${id}`).then((r: any) => r.data),
    importCSV: (csvContent: string) => apiClient.post('/portfolio/import', { csv_content: csvContent }).then((r: any) => r.data),
    // Transaction endpoints
    getTransactions: (symbol?: string) => apiClient.get('/portfolio/transactions', { params: { symbol } }).then((r: any) => r.data),
    getTimeline: () => apiClient.get('/portfolio/transactions/timeline').then((r: any) => r.data),
    importTransactions: (csvContent: string) => apiClient.post('/portfolio/transactions/import', { csv_content: csvContent }).then((r: any) => r.data),
    addTransaction: (data: any) => apiClient.post('/portfolio/transactions/add', data).then((r: any) => r.data),
}

// Add Holding Modal
function AddHoldingModal({ open, onClose, onSuccess }: { open: boolean; onClose: () => void; onSuccess: () => void }) {
    const [symbol, setSymbol] = useState('')
    const [shares, setShares] = useState('')
    const [avgCost, setAvgCost] = useState('')
    const [error, setError] = useState('')

    const queryClient = useQueryClient()

    const mutation = useMutation({
        mutationFn: portfolioApi.addHolding,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['portfolio'] })
            onSuccess()
            onClose()
            setSymbol('')
            setShares('')
            setAvgCost('')
        },
        onError: (err: any) => {
            setError(err.response?.data?.error || 'Failed to add holding')
        }
    })

    const handleSubmit = () => {
        if (!symbol || !shares || !avgCost) {
            setError('All fields are required')
            return
        }
        mutation.mutate({
            symbol: symbol.toUpperCase(),
            shares: parseFloat(shares),
            avg_cost: parseFloat(avgCost),
        })
    }

    return (
        <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
            <DialogTitle sx={{ background: 'linear-gradient(135deg, #0a0b14, #1a1b2e)', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                Add Stock to Portfolio
            </DialogTitle>
            <DialogContent sx={{ background: '#0a0b14', pt: 3 }}>
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                <TextField
                    fullWidth
                    label="Symbol"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                    placeholder="e.g., AAPL"
                    sx={{ mb: 2, mt: 1 }}
                />
                <TextField
                    fullWidth
                    label="Number of Shares"
                    type="number"
                    value={shares}
                    onChange={(e) => setShares(e.target.value)}
                    placeholder="e.g., 100"
                    sx={{ mb: 2 }}
                />
                <TextField
                    fullWidth
                    label="Average Cost per Share"
                    type="number"
                    value={avgCost}
                    onChange={(e) => setAvgCost(e.target.value)}
                    placeholder="e.g., 150.50"
                    InputProps={{ startAdornment: <span style={{ marginRight: 4, color: '#94a3b8' }}>$</span> }}
                />
            </DialogContent>
            <DialogActions sx={{ background: '#0a0b14', p: 2 }}>
                <Button onClick={onClose} color="inherit">Cancel</Button>
                <Button onClick={handleSubmit} variant="contained" disabled={mutation.isPending}>
                    {mutation.isPending ? 'Adding...' : 'Add Stock'}
                </Button>
            </DialogActions>
        </Dialog>
    )
}

// Import CSV Modal
function ImportCSVModal({ open, onClose, onSuccess }: { open: boolean; onClose: () => void; onSuccess: () => void }) {
    const [csvContent, setCsvContent] = useState('')
    const [result, setResult] = useState<any>(null)
    const [error, setError] = useState('')

    const queryClient = useQueryClient()

    const mutation = useMutation({
        mutationFn: portfolioApi.importCSV,
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ['portfolio'] })
            setResult(data)
            onSuccess()
        },
        onError: (err: any) => {
            setError(err.response?.data?.error || 'Failed to import CSV')
        }
    })

    const handleImport = () => {
        if (!csvContent.trim()) {
            setError('Please paste CSV content')
            return
        }
        mutation.mutate(csvContent)
    }

    const sampleCSV = `Symbol,Shares,Avg Cost
AAPL,100,145.50
NVDA,50,280.00
MSFT,75,320.00`

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            <DialogTitle sx={{ background: 'linear-gradient(135deg, #0a0b14, #1a1b2e)', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                Import Portfolio from CSV
            </DialogTitle>
            <DialogContent sx={{ background: '#0a0b14', pt: 3 }}>
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                {result && (
                    <Alert severity="success" sx={{ mb: 2 }}>
                        Imported {result.imported?.length || 0} holdings. {result.errors?.length ? `${result.errors.length} errors.` : ''}
                    </Alert>
                )}
                <Typography sx={{ mb: 2, color: '#94a3b8', fontSize: '0.9rem' }}>
                    Paste your portfolio CSV with columns: Symbol, Shares, Avg Cost
                </Typography>
                <TextField
                    fullWidth
                    multiline
                    rows={8}
                    value={csvContent}
                    onChange={(e) => setCsvContent(e.target.value)}
                    placeholder={sampleCSV}
                    sx={{ fontFamily: 'monospace' }}
                />
                <Typography sx={{ mt: 2, color: '#64748b', fontSize: '0.8rem' }}>
                    Sample format:
                    <pre style={{ background: 'rgba(0,0,0,0.3)', padding: '8px', borderRadius: '4px', marginTop: '8px' }}>
                        {sampleCSV}
                    </pre>
                </Typography>
            </DialogContent>
            <DialogActions sx={{ background: '#0a0b14', p: 2 }}>
                <Button onClick={onClose} color="inherit">Cancel</Button>
                <Button onClick={handleImport} variant="contained" disabled={mutation.isPending}>
                    {mutation.isPending ? 'Importing...' : 'Import'}
                </Button>
            </DialogActions>
        </Dialog>
    )
}

// Import Transactions Modal - for full trading history
function ImportTransactionsModal({ open, onClose, onSuccess }: { open: boolean; onClose: () => void; onSuccess: () => void }) {
    const [csvContent, setCsvContent] = useState('')
    const [result, setResult] = useState<any>(null)
    const [error, setError] = useState('')

    const queryClient = useQueryClient()

    const mutation = useMutation({
        mutationFn: portfolioApi.importTransactions,
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ['portfolio'] })
            queryClient.invalidateQueries({ queryKey: ['transactions'] })
            setResult(data)
            onSuccess()
        },
        onError: (err: any) => {
            setError(err.response?.data?.error || 'Failed to import transactions')
        }
    })

    const handleImport = () => {
        if (!csvContent.trim()) {
            setError('Please paste transaction CSV content')
            return
        }
        mutation.mutate(csvContent)
    }

    const sampleCSV = `Symbol,Type,Date,Shares,Price,Notes
AAPL,BUY,2024-01-15,100,185.50,Initial purchase
AAPL,SELL,2024-06-20,50,195.25,Taking profits
NVDA,BUY,2024-02-01,25,650.00,AI play
MSFT,BUY,2024-03-10,30,410.00,
AAPL,DIVIDEND,2024-02-15,100,0.24,Q1 Dividend`

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            <DialogTitle sx={{ background: 'linear-gradient(135deg, #0a0b14, #1a1b2e)', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <HistoryIcon sx={{ color: '#f59e0b' }} />
                    Import Transaction History
                </Box>
            </DialogTitle>
            <DialogContent sx={{ background: '#0a0b14', pt: 3 }}>
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                {result && (
                    <Alert severity="success" sx={{ mb: 2 }}>
                        <strong>Success!</strong> Imported {result.imported?.length || 0} transactions.
                        {result.errors?.length > 0 && ` (${result.errors.length} errors)`}
                        <br />
                        Your holdings have been automatically recalculated.
                    </Alert>
                )}

                <Alert severity="info" sx={{ mb: 2, background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                    <Typography sx={{ color: '#e2e8f0' }}>
                        <strong>Import your complete trading history!</strong> Include all your BUY, SELL, and DIVIDEND transactions.
                        This enables P&L tracking and AI-powered lessons from your trades.
                    </Typography>
                </Alert>

                <Typography sx={{ mb: 2, color: '#94a3b8', fontSize: '0.9rem' }}>
                    Required columns: <strong>Symbol, Type, Date, Shares, Price</strong>
                </Typography>

                <TextField
                    fullWidth
                    multiline
                    rows={10}
                    value={csvContent}
                    onChange={(e) => setCsvContent(e.target.value)}
                    placeholder={sampleCSV}
                    sx={{ fontFamily: 'monospace', '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
                />

                <Box sx={{ mt: 2, p: 2, background: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem', mb: 1 }}>Sample format:</Typography>
                    <pre style={{ color: '#94a3b8', margin: 0, fontSize: '0.75rem', overflowX: 'auto' }}>
                        {sampleCSV}
                    </pre>
                </Box>

                <Typography sx={{ mt: 2, color: '#64748b', fontSize: '0.75rem' }}>
                    <strong>Supported Type values:</strong> BUY, SELL, DIVIDEND (or B, S, D)<br />
                    <strong>Date formats:</strong> YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY
                </Typography>
            </DialogContent>
            <DialogActions sx={{ background: '#0a0b14', p: 2, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                <Button onClick={onClose} color="inherit">Cancel</Button>
                <Button onClick={handleImport} variant="contained" disabled={mutation.isPending}
                    sx={{ background: 'linear-gradient(135deg, #f59e0b, #d97706)' }}>
                    {mutation.isPending ? 'Importing...' : 'Import Transactions'}
                </Button>
            </DialogActions>
        </Dialog>
    )
}

export default function PortfolioPage() {
    const [addModalOpen, setAddModalOpen] = useState(false)
    const [importModalOpen, setImportModalOpen] = useState(false)
    const [importTransactionsModalOpen, setImportTransactionsModalOpen] = useState(false)
    const [importFamousModalOpen, setImportFamousModalOpen] = useState(false)

    const queryClient = useQueryClient()

    // Portfolio context - to refresh global state after imports
    const { refreshPortfolio } = usePortfolio()

    const { data: portfolioData, isLoading, error } = useQuery({
        queryKey: ['portfolio', 'summary'],
        queryFn: portfolioApi.getSummary,
    })

    const { data: signalsData } = useQuery({
        queryKey: ['portfolio', 'signals'],
        queryFn: portfolioApi.getSignals,
    })

    const deleteMutation = useMutation({
        mutationFn: portfolioApi.deleteHolding,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['portfolio'] })
        }
    })

    const holdings = portfolioData?.holdings || []
    const summary = portfolioData?.summary || { total_holdings: 0, total_cost_basis: 0, total_current_value: 0, total_pnl: 0, total_pnl_pct: 0 }
    const signalsBySymbol: Record<string, any> = {}
    signalsData?.holdings_with_signals?.forEach((h: any) => {
        signalsBySymbol[h.symbol] = h.signal
    })

    return (
        <Box className="fade-in">
            {/* Live Market Data Header */}
            <Box sx={{ mx: -3, mt: -3, mb: 3 }}>
                <LiveMarketHeader />
            </Box>

            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#fff', fontSize: '1.75rem', mb: 0.5 }}>
                        My Portfolio
                    </Typography>
                    <Typography sx={{ color: '#64748b', fontSize: '0.9rem' }}>
                        Track your holdings, P&L, and get personalized AI signals
                    </Typography>
                </Box>
                <Box sx={{ display: 'flex', gap: 2 }}>
                    <Button
                        variant="outlined"
                        startIcon={<PeopleIcon />}
                        onClick={() => setImportFamousModalOpen(true)}
                        sx={{
                            borderColor: 'rgba(139, 92, 246, 0.5)',
                            color: '#a78bfa',
                            '&:hover': { borderColor: '#8b5cf6', background: 'rgba(139, 92, 246, 0.1)' }
                        }}
                    >
                        Import Famous Portfolio
                    </Button>
                    <Button
                        variant="outlined"
                        startIcon={<HistoryIcon />}
                        onClick={() => setImportTransactionsModalOpen(true)}
                        sx={{
                            borderColor: '#f59e0b',
                            color: '#f59e0b',
                            '&:hover': { borderColor: '#d97706', background: 'rgba(245, 158, 11, 0.1)' }
                        }}
                    >
                        Import Transactions
                    </Button>
                    <Button variant="outlined" startIcon={<UploadFileIcon />} onClick={() => setImportModalOpen(true)}>
                        Import Holdings
                    </Button>
                    <Button variant="contained" startIcon={<AddIcon />} onClick={() => setAddModalOpen(true)}>
                        Add Stock
                    </Button>
                </Box>
            </Box>

            {/* Summary Metrics */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={6} md={3} className="fade-in fade-in-delay-1">
                    <MetricCard
                        icon="chart"
                        iconColor="blue"
                        label="Portfolio Value"
                        value={`$${summary.total_current_value.toLocaleString()}`}
                        subText="Current value"
                    />
                </Grid>
                <Grid item xs={6} md={3} className="fade-in fade-in-delay-2">
                    <MetricCard
                        icon="chart"
                        iconColor="purple"
                        label="Cost Basis"
                        value={`$${summary.total_cost_basis.toLocaleString()}`}
                        subText="Total invested"
                    />
                </Grid>
                <Grid item xs={6} md={3} className="fade-in fade-in-delay-3">
                    <MetricCard
                        icon="trending-up"
                        iconColor={summary.total_pnl >= 0 ? 'green' : 'red'}
                        label="Total P&L"
                        value={`${summary.total_pnl >= 0 ? '+' : ''}$${summary.total_pnl.toLocaleString()}`}
                        subText={`${summary.total_pnl_pct >= 0 ? '+' : ''}${summary.total_pnl_pct.toFixed(2)}%`}
                        sentiment={summary.total_pnl >= 0 ? 'bullish' : 'bearish'}
                    />
                </Grid>
                <Grid item xs={6} md={3} className="fade-in fade-in-delay-4">
                    <MetricCard
                        icon="bar-chart"
                        iconColor="blue"
                        label="Holdings"
                        value={summary.total_holdings}
                        subText="Stocks in portfolio"
                    />
                </Grid>
            </Grid>

            {/* Holdings Table */}
            <SectionCard title="Holdings" icon={<AccountBalanceWalletIcon />} iconColor="#3b82f6">
                {isLoading ? (
                    <Typography sx={{ color: '#64748b', py: 4, textAlign: 'center' }}>Loading portfolio...</Typography>
                ) : error ? (
                    <Alert severity="error">Failed to load portfolio</Alert>
                ) : holdings.length === 0 ? (
                    <Box sx={{ py: 4, textAlign: 'center' }}>
                        <AccountBalanceWalletIcon sx={{ fontSize: 48, color: '#64748b', mb: 2 }} />
                        <Typography sx={{ color: '#94a3b8', mb: 2 }}>No holdings yet</Typography>
                        <Button variant="contained" startIcon={<AddIcon />} onClick={() => setAddModalOpen(true)}>
                            Add Your First Stock
                        </Button>
                    </Box>
                ) : (
                    <TableContainer>
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell>Symbol</TableCell>
                                    <TableCell align="right">Shares</TableCell>
                                    <TableCell align="right">Avg Cost</TableCell>
                                    <TableCell align="right">Current Price</TableCell>
                                    <TableCell align="right">P&L</TableCell>
                                    <TableCell align="center">Signal</TableCell>
                                    <TableCell align="center">Actions</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {holdings.map((holding: any) => {
                                    const signal = signalsBySymbol[holding.symbol]
                                    return (
                                        <TableRow key={holding.id} sx={{ '&:hover': { background: 'rgba(255,255,255,0.02)' } }}>
                                            <TableCell>
                                                <Typography sx={{ fontWeight: 700, color: '#fff' }}>{holding.symbol}</Typography>
                                            </TableCell>
                                            <TableCell align="right">{holding.shares}</TableCell>
                                            <TableCell align="right">${holding.avg_cost.toFixed(2)}</TableCell>
                                            <TableCell align="right">${holding.current_price?.toFixed(2) || '—'}</TableCell>
                                            <TableCell align="right">
                                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 1 }}>
                                                    {holding.pnl >= 0 ? (
                                                        <TrendingUpIcon sx={{ color: '#10b981', fontSize: 18 }} />
                                                    ) : (
                                                        <TrendingDownIcon sx={{ color: '#ef4444', fontSize: 18 }} />
                                                    )}
                                                    <Typography sx={{ color: holding.pnl >= 0 ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                                                        {holding.pnl >= 0 ? '+' : ''}${holding.pnl?.toFixed(2) || '0'}
                                                    </Typography>
                                                    <Typography sx={{ color: '#64748b', fontSize: '0.8rem' }}>
                                                        ({holding.pnl_pct >= 0 ? '+' : ''}{holding.pnl_pct?.toFixed(1) || '0'}%)
                                                    </Typography>
                                                </Box>
                                            </TableCell>
                                            <TableCell align="center">
                                                <Tooltip
                                                    title={
                                                        <Box sx={{ p: 0.5 }}>
                                                            <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.5 }}>
                                                                {signal?.signal_type || 'HOLD'} ({(signal?.confluence_score ? signal.confluence_score * 100 : 0).toFixed(0)}% Conf.)
                                                            </Typography>
                                                            <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                                                                {signal?.rationale || 'No rationale available'}
                                                            </Typography>
                                                        </Box>
                                                    }
                                                    arrow
                                                >
                                                    <Box sx={{ display: 'inline-block', cursor: 'help' }}>
                                                        <SignalBadge type={signal?.signal_type || 'HOLD'} />
                                                    </Box>
                                                </Tooltip>
                                            </TableCell>
                                            <TableCell align="center">
                                                <IconButton
                                                    size="small"
                                                    sx={{ color: '#60a5fa', mr: 1 }}
                                                    onClick={() => window.location.href = `/dashboard/charts?symbol=${holding.symbol}`}
                                                    title="View Chart"
                                                >
                                                    <ShowChartIcon fontSize="small" />
                                                </IconButton>
                                                <IconButton
                                                    size="small"
                                                    sx={{ color: '#f59e0b', mr: 1 }}
                                                    onClick={() => window.location.href = `/dashboard/alerts?symbol=${holding.symbol}`}
                                                    title="Set Alert"
                                                >
                                                    <NotificationsActiveIcon fontSize="small" />
                                                </IconButton>
                                                <IconButton
                                                    size="small"
                                                    sx={{ color: '#ef4444' }}
                                                    onClick={() => deleteMutation.mutate(holding.id)}
                                                    title="Remove Holding"
                                                >
                                                    <DeleteIcon fontSize="small" />
                                                </IconButton>
                                            </TableCell>
                                        </TableRow>
                                    )
                                })}
                            </TableBody>
                        </Table>
                    </TableContainer>
                )}
            </SectionCard>

            {/* Risk Management Panel */}
            {holdings.length > 0 && (
                <RiskManagementPanel
                    metrics={{
                        total_value: summary.total_current_value,
                        holdings: holdings.map((h: any) => ({
                            symbol: h.symbol,
                            shares: h.shares,
                            avg_cost: h.avg_cost,
                            current_price: h.current_price || h.avg_cost,
                            pnl: h.pnl || 0,
                            pnl_pct: h.pnl_pct || 0,
                        }))
                    }}
                />
            )}

            {/* Correlation Analysis */}
            {holdings.length >= 2 && (
                <Box sx={{ mt: 3 }}>
                    <CorrelationHeatmap symbols={holdings.map((h: any) => h.symbol)} />
                </Box>
            )}

            {/* Modals */}
            <AddHoldingModal
                open={addModalOpen}
                onClose={() => setAddModalOpen(false)}
                onSuccess={() => { refreshPortfolio() }}
            />
            <ImportCSVModal
                open={importModalOpen}
                onClose={() => setImportModalOpen(false)}
                onSuccess={() => { refreshPortfolio() }}
            />
            <ImportTransactionsModal
                open={importTransactionsModalOpen}
                onClose={() => setImportTransactionsModalOpen(false)}
                onSuccess={() => { refreshPortfolio() }}
            />
            <ImportPortfolioModal
                open={importFamousModalOpen}
                onClose={() => setImportFamousModalOpen(false)}
                onSuccess={() => {
                    queryClient.invalidateQueries({ queryKey: ['portfolio'] })
                    refreshPortfolio() // Refresh global portfolio context
                }}
            />
        </Box>
    )
}
