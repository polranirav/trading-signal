/**
 * Portfolio Page
 * 
 * User's personal portfolio dashboard with:
 * - Holdings table with P&L tracking
 * - Import from CSV
 * - Add stocks manually
 * - AI-powered signals for holdings
 */

import { useState } from 'react'
import { Box, Typography, Grid, Button, Dialog, DialogTitle, DialogContent, DialogActions, TextField, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, IconButton, Alert } from '@mui/material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import AddIcon from '@mui/icons-material/Add'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import TrendingDownIcon from '@mui/icons-material/TrendingDown'
import DeleteIcon from '@mui/icons-material/Delete'
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet'
import PeopleIcon from '@mui/icons-material/People'
import apiClient from '../../services/api'
import MetricCard from '../../components/MetricCard'
import { SectionCard, SignalBadge } from '../../components/SignalComponents'
import ImportPortfolioModal from '../../components/ImportPortfolioModal'
import '../../styles/premium.css'

// API extension for portfolio
const portfolioApi = {
    getSummary: () => apiClient.get('/portfolio/summary').then((r: any) => r.data),
    getSignals: () => apiClient.get('/portfolio/signals').then((r: any) => r.data),
    addHolding: (data: any) => apiClient.post('/portfolio/add', data).then((r: any) => r.data),
    deleteHolding: (id: string) => apiClient.delete(`/portfolio/${id}`).then((r: any) => r.data),
    importCSV: (csvContent: string) => apiClient.post('/portfolio/import', { csv_content: csvContent }).then((r: any) => r.data),
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

export default function PortfolioPage() {
    const [addModalOpen, setAddModalOpen] = useState(false)
    const [importModalOpen, setImportModalOpen] = useState(false)
    const [importFamousModalOpen, setImportFamousModalOpen] = useState(false)

    const queryClient = useQueryClient()

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
                    <Button variant="outlined" startIcon={<UploadFileIcon />} onClick={() => setImportModalOpen(true)}>
                        Import CSV
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
                                                <SignalBadge type={signal?.signal_type || 'HOLD'} />
                                            </TableCell>
                                            <TableCell align="center">
                                                <IconButton
                                                    size="small"
                                                    sx={{ color: '#ef4444' }}
                                                    onClick={() => deleteMutation.mutate(holding.id)}
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

            {/* Modals */}
            <AddHoldingModal
                open={addModalOpen}
                onClose={() => setAddModalOpen(false)}
                onSuccess={() => { }}
            />
            <ImportCSVModal
                open={importModalOpen}
                onClose={() => setImportModalOpen(false)}
                onSuccess={() => { }}
            />
            <ImportPortfolioModal
                open={importFamousModalOpen}
                onClose={() => setImportFamousModalOpen(false)}
                onSuccess={() => queryClient.invalidateQueries({ queryKey: ['portfolio'] })}
            />
        </Box>
    )
}
