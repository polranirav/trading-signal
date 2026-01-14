/**
 * Import Portfolio Modal
 * 
 * Modal for importing portfolios from:
 * - Famous Investors (Buffett, Ackman, etc.)
 * - Brokerage CSV exports
 * - Custom SEC URLs
 */

import { useState, useEffect } from 'react'
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    Box,
    Typography,
    Grid,
    Tabs,
    Tab,
    TextField,
    Alert,
    CircularProgress,
    Chip,
    Divider,
} from '@mui/material'
import TrendingUpIcon from '@mui/icons-material/TrendingUp'
import AccountBalanceIcon from '@mui/icons-material/AccountBalance'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import apiClient from '../services/api'

interface InvestorData {
    id: string
    name: string
    fund: string
    description: string
    avatar: string
}

interface HoldingPreview {
    symbol: string
    name: string
    shares: number
    value_usd: number
    pct_portfolio: number
}

interface ImportPortfolioModalProps {
    open: boolean
    onClose: () => void
    onSuccess: () => void
}

export default function ImportPortfolioModal({ open, onClose, onSuccess }: ImportPortfolioModalProps) {
    const [activeTab, setActiveTab] = useState(0)
    const [investors, setInvestors] = useState<InvestorData[]>([])
    const [selectedInvestor, setSelectedInvestor] = useState<string | null>(null)
    const [previewHoldings, setPreviewHoldings] = useState<HoldingPreview[]>([])
    const [loading, setLoading] = useState(false)
    const [importing, setImporting] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [success, setSuccess] = useState<string | null>(null)
    const [secUrl, setSecUrl] = useState('')
    const [csvContent, setCsvContent] = useState('')

    // Fetch famous investors on mount
    useEffect(() => {
        if (open) {
            fetchInvestors()
        }
    }, [open])

    const fetchInvestors = async () => {
        try {
            const response = await apiClient.get('/portfolio/famous-investors')
            setInvestors(response.data.investors || [])
        } catch (err: any) {
            console.error('Failed to fetch investors:', err)
            setError('Failed to load investors')
        }
    }

    const handleSelectInvestor = async (investorId: string) => {
        setSelectedInvestor(investorId)
        setLoading(true)
        setError(null)
        setPreviewHoldings([])

        try {
            const response = await apiClient.get(`/portfolio/preview-investor/${investorId}`)
            setPreviewHoldings(response.data.holdings || [])
        } catch (err: any) {
            setError(err.response?.data?.error || 'Failed to preview portfolio')
        } finally {
            setLoading(false)
        }
    }

    const handleImportInvestor = async () => {
        if (!selectedInvestor) return

        setImporting(true)
        setError(null)

        try {
            const response = await apiClient.post('/portfolio/import-investor', {
                investor_id: selectedInvestor
            })
            setSuccess(`Successfully imported ${response.data.count} holdings!`)
            setTimeout(() => {
                onSuccess()
                onClose()
            }, 1500)
        } catch (err: any) {
            setError(err.response?.data?.error || 'Import failed')
        } finally {
            setImporting(false)
        }
    }

    const handleImportSecUrl = async () => {
        if (!secUrl.trim()) return

        setImporting(true)
        setError(null)

        try {
            const response = await apiClient.post('/portfolio/import-investor', {
                sec_url: secUrl
            })
            setSuccess(`Successfully imported ${response.data.count} holdings!`)
            setTimeout(() => {
                onSuccess()
                onClose()
            }, 1500)
        } catch (err: any) {
            setError(err.response?.data?.error || 'Import failed')
        } finally {
            setImporting(false)
        }
    }

    const handleImportCsv = async () => {
        if (!csvContent.trim()) return

        setImporting(true)
        setError(null)

        try {
            const response = await apiClient.post('/portfolio/import', {
                csv_content: csvContent
            })
            setSuccess(`Successfully imported ${response.data.imported?.length || 0} holdings!`)
            setTimeout(() => {
                onSuccess()
                onClose()
            }, 1500)
        } catch (err: any) {
            setError(err.response?.data?.error || 'Import failed')
        } finally {
            setImporting(false)
        }
    }

    const handleClose = () => {
        setSelectedInvestor(null)
        setPreviewHoldings([])
        setError(null)
        setSuccess(null)
        setSecUrl('')
        setCsvContent('')
        onClose()
    }

    return (
        <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
            <DialogTitle
                sx={{
                    background: 'linear-gradient(135deg, #0a0b14, #1a1b2e)',
                    borderBottom: '1px solid rgba(255,255,255,0.1)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                }}
            >
                <TrendingUpIcon sx={{ color: '#3b82f6' }} />
                Import Portfolio
            </DialogTitle>

            <DialogContent sx={{ background: '#0a0b14', p: 0, minHeight: 400 }}>
                {success && (
                    <Alert severity="success" sx={{ m: 2 }}>{success}</Alert>
                )}
                {error && (
                    <Alert severity="error" sx={{ m: 2 }} onClose={() => setError(null)}>{error}</Alert>
                )}

                <Tabs
                    value={activeTab}
                    onChange={(_, v) => setActiveTab(v)}
                    sx={{
                        borderBottom: '1px solid rgba(255,255,255,0.1)',
                        px: 2,
                        '& .MuiTab-root': { color: '#94a3b8' },
                        '& .Mui-selected': { color: '#3b82f6' },
                    }}
                >
                    <Tab icon={<TrendingUpIcon />} label="Famous Investors" />
                    <Tab icon={<AccountBalanceIcon />} label="My Brokerage" />
                    <Tab icon={<UploadFileIcon />} label="CSV Import" />
                </Tabs>

                {/* Tab 0: Famous Investors */}
                {activeTab === 0 && (
                    <Box sx={{ p: 2 }}>
                        <Typography sx={{ color: '#64748b', mb: 2, fontSize: '0.9rem' }}>
                            Select a famous investor to import their portfolio holdings
                        </Typography>

                        <Grid container spacing={2}>
                            {investors.map((investor) => (
                                <Grid item xs={6} sm={4} md={3} key={investor.id}>
                                    <Box
                                        onClick={() => handleSelectInvestor(investor.id)}
                                        sx={{
                                            p: 2,
                                            borderRadius: 2,
                                            cursor: 'pointer',
                                            textAlign: 'center',
                                            background: selectedInvestor === investor.id
                                                ? 'rgba(59, 130, 246, 0.2)'
                                                : 'rgba(255,255,255,0.03)',
                                            border: selectedInvestor === investor.id
                                                ? '1px solid rgba(59, 130, 246, 0.5)'
                                                : '1px solid rgba(255,255,255,0.1)',
                                            transition: 'all 0.2s',
                                            '&:hover': {
                                                background: 'rgba(59, 130, 246, 0.1)',
                                                borderColor: 'rgba(59, 130, 246, 0.3)',
                                            }
                                        }}
                                    >
                                        <Typography sx={{ fontSize: '2rem', mb: 1 }}>
                                            {investor.avatar}
                                        </Typography>
                                        <Typography sx={{ color: '#fff', fontWeight: 600, fontSize: '0.9rem' }}>
                                            {investor.name}
                                        </Typography>
                                        <Typography sx={{ color: '#64748b', fontSize: '0.75rem' }}>
                                            {investor.fund}
                                        </Typography>
                                    </Box>
                                </Grid>
                            ))}
                        </Grid>

                        {/* Holdings Preview */}
                        {selectedInvestor && (
                            <Box sx={{ mt: 3 }}>
                                <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)', mb: 2 }} />
                                <Typography sx={{ color: '#fff', fontWeight: 600, mb: 2 }}>
                                    📊 Portfolio Preview
                                </Typography>

                                {loading ? (
                                    <Box sx={{ textAlign: 'center', py: 4 }}>
                                        <CircularProgress size={32} />
                                    </Box>
                                ) : (
                                    <Box sx={{ maxHeight: 200, overflow: 'auto' }}>
                                        {previewHoldings.map((h, i) => (
                                            <Box
                                                key={i}
                                                sx={{
                                                    display: 'flex',
                                                    justifyContent: 'space-between',
                                                    alignItems: 'center',
                                                    py: 1,
                                                    borderBottom: '1px solid rgba(255,255,255,0.05)'
                                                }}
                                            >
                                                <Box>
                                                    <Chip
                                                        label={h.symbol}
                                                        size="small"
                                                        sx={{
                                                            bgcolor: 'rgba(59, 130, 246, 0.2)',
                                                            color: '#3b82f6',
                                                            mr: 1,
                                                        }}
                                                    />
                                                    <Typography component="span" sx={{ color: '#94a3b8', fontSize: '0.85rem' }}>
                                                        {h.name}
                                                    </Typography>
                                                </Box>
                                                <Typography sx={{ color: '#10b981', fontWeight: 600 }}>
                                                    {h.pct_portfolio.toFixed(1)}%
                                                </Typography>
                                            </Box>
                                        ))}
                                    </Box>
                                )}
                            </Box>
                        )}
                    </Box>
                )}

                {/* Tab 1: Brokerage */}
                {activeTab === 1 && (
                    <Box sx={{ p: 3 }}>
                        <Typography sx={{ color: '#fff', fontWeight: 600, mb: 2 }}>
                            Import from SEC EDGAR URL
                        </Typography>
                        <Typography sx={{ color: '#64748b', fontSize: '0.9rem', mb: 2 }}>
                            Paste a SEC EDGAR 13F filing URL to import holdings
                        </Typography>
                        <TextField
                            fullWidth
                            placeholder="https://www.sec.gov/cgi-bin/browse-edgar?CIK=..."
                            value={secUrl}
                            onChange={(e) => setSecUrl(e.target.value)}
                            sx={{ mb: 3 }}
                        />

                        <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)', my: 3 }} />

                        <Typography sx={{ color: '#fff', fontWeight: 600, mb: 2 }}>
                            📁 Export from Your Broker
                        </Typography>
                        <Typography sx={{ color: '#64748b', fontSize: '0.85rem', lineHeight: 1.8 }}>
                            • <strong>Robinhood:</strong> Account → Statements → Download CSV<br />
                            • <strong>Fidelity:</strong> Positions → Export to CSV<br />
                            • <strong>TD Ameritrade:</strong> Positions → Export<br />
                            • <strong>Schwab:</strong> Positions → Download<br />
                            <br />
                            Then use the "CSV Import" tab to upload your file.
                        </Typography>
                    </Box>
                )}

                {/* Tab 2: CSV Import */}
                {activeTab === 2 && (
                    <Box sx={{ p: 3 }}>
                        <Typography sx={{ color: '#64748b', fontSize: '0.9rem', mb: 2 }}>
                            Paste your portfolio data in CSV format (Symbol, Shares, Avg Cost)
                        </Typography>
                        <TextField
                            fullWidth
                            multiline
                            rows={8}
                            placeholder={`Symbol,Shares,Avg Cost
AAPL,100,145.50
NVDA,50,280.00
MSFT,75,320.00`}
                            value={csvContent}
                            onChange={(e) => setCsvContent(e.target.value)}
                            sx={{ fontFamily: 'monospace' }}
                        />
                    </Box>
                )}
            </DialogContent>

            <DialogActions sx={{ background: '#0a0b14', p: 2, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                <Button onClick={handleClose} color="inherit">
                    Cancel
                </Button>

                {activeTab === 0 && selectedInvestor && (
                    <Button
                        onClick={handleImportInvestor}
                        variant="contained"
                        disabled={importing || !previewHoldings.length}
                    >
                        {importing ? 'Importing...' : `Import ${previewHoldings.length} Holdings`}
                    </Button>
                )}

                {activeTab === 1 && secUrl && (
                    <Button
                        onClick={handleImportSecUrl}
                        variant="contained"
                        disabled={importing}
                    >
                        {importing ? 'Importing...' : 'Import from SEC'}
                    </Button>
                )}

                {activeTab === 2 && csvContent && (
                    <Button
                        onClick={handleImportCsv}
                        variant="contained"
                        disabled={importing}
                    >
                        {importing ? 'Importing...' : 'Import CSV'}
                    </Button>
                )}
            </DialogActions>
        </Dialog>
    )
}
