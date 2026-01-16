/**
 * Data Sources API Keys Component
 * 
 * Premium UI for managing external API keys for signal intelligence.
 * Features:
 * - Beautiful card-based layout for each service
 * - Connection status indicators
 * - Secure input with visibility toggle
 * - Real-time validation feedback
 * - Signal categories enabled by each service
 */

import { useState } from 'react';
import {
    Box,
    Typography,
    Grid,
    Card,
    CardContent,
    TextField,
    Button,
    IconButton,
    Chip,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    CircularProgress,
    Tooltip,
    InputAdornment,
    Collapse,
    Alert,
    Link,
} from '@mui/material';
import {
    Visibility,
    VisibilityOff,
    CheckCircle,
    Error as ErrorIcon,
    Add as AddIcon,
    Delete as DeleteIcon,
    Refresh as RefreshIcon,
    Launch as LaunchIcon,
    TrendingUp,
    Analytics,
    Newspaper,
    BarChart,
    AccountBalance,
    Hexagon,
    Psychology,
    CloudDone,
    CloudOff,
    Security,
    InfoOutlined,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../../services/api';

interface ApiService {
    id: string;
    name: string;
    description: string;
    url: string;
    free_tier: string;
    signals: string[];
    icon: string;
    color: string;
    connected: boolean;
    is_valid: boolean | null;
    last_validated: string | null;
}

const getServiceIcon = (iconName: string, color: string) => {
    const iconProps = { sx: { fontSize: 32, color } };
    switch (iconName) {
        case 'trending_up': return <TrendingUp {...iconProps} />;
        case 'analytics': return <Analytics {...iconProps} />;
        case 'newspaper': return <Newspaper {...iconProps} />;
        case 'bar_chart': return <BarChart {...iconProps} />;
        case 'account_balance': return <AccountBalance {...iconProps} />;
        case 'hexagon': return <Hexagon {...iconProps} />;
        case 'psychology': return <Psychology {...iconProps} />;
        default: return <Analytics {...iconProps} />;
    }
};

export default function DataSourcesApiKeys() {
    const queryClient = useQueryClient();
    const [selectedService, setSelectedService] = useState<ApiService | null>(null);
    const [apiKeyInput, setApiKeyInput] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [dialogOpen, setDialogOpen] = useState(false);

    // Fetch API keys status
    const { data: apiKeysData, isLoading } = useQuery({
        queryKey: ['user-api-keys'],
        queryFn: () => api.getUserApiKeys(),
    });

    // Save API key mutation
    const saveMutation = useMutation({
        mutationFn: ({ service, apiKey }: { service: string; apiKey: string }) =>
            api.saveUserApiKey(service, apiKey),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['user-api-keys'] });
            setDialogOpen(false);
            setApiKeyInput('');
            setSelectedService(null);
        },
    });

    // Delete API key mutation
    const deleteMutation = useMutation({
        mutationFn: (service: string) => api.deleteUserApiKey(service),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['user-api-keys'] });
        },
    });

    // Test API key mutation
    const testMutation = useMutation({
        mutationFn: (service: string) => api.testUserApiKey(service),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['user-api-keys'] });
        },
    });

    const services: ApiService[] = apiKeysData?.services || [];
    const connectedCount = apiKeysData?.connected_count || 0;
    const totalSignals = apiKeysData?.total_signals_enabled || 0;

    const handleOpenDialog = (service: ApiService) => {
        setSelectedService(service);
        setApiKeyInput('');
        setShowPassword(false);
        setDialogOpen(true);
    };

    const handleSaveKey = () => {
        if (selectedService && apiKeyInput.trim()) {
            saveMutation.mutate({ service: selectedService.id, apiKey: apiKeyInput.trim() });
        }
    };

    const handleDeleteKey = (serviceId: string) => {
        if (window.confirm('Are you sure you want to disconnect this service?')) {
            deleteMutation.mutate(serviceId);
        }
    };

    const handleTestKey = (serviceId: string) => {
        testMutation.mutate(serviceId);
    };

    return (
        <Box>
            {/* Header */}
            <Box sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                mb: 3,
            }}>
                <Box>
                    <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Security sx={{ color: '#8b5cf6' }} />
                        Data Source API Keys
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                        Connect your own API keys to unlock real-time market data and advanced signals
                    </Typography>
                </Box>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                    <Chip
                        icon={<CloudDone sx={{ fontSize: 16 }} />}
                        label={`${connectedCount}/${services.length} Connected`}
                        color={connectedCount > 0 ? 'success' : 'default'}
                        variant="outlined"
                    />
                    <Chip
                        icon={<Analytics sx={{ fontSize: 16 }} />}
                        label={`${totalSignals} Signals Enabled`}
                        color="primary"
                        variant="outlined"
                    />
                </Box>
            </Box>

            {/* Info Banner */}
            <Alert
                severity="info"
                icon={<InfoOutlined />}
                sx={{
                    mb: 3,
                    bgcolor: 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    '& .MuiAlert-message': { width: '100%' }
                }}
            >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body2">
                        Your API keys are encrypted and stored securely. Each service provides different market data for signal analysis.
                    </Typography>
                </Box>
            </Alert>

            {isLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
                    <CircularProgress />
                </Box>
            ) : (
                <Grid container spacing={3}>
                    {services.map((service) => (
                        <Grid item xs={12} md={6} lg={4} key={service.id}>
                            <Card sx={{
                                height: '100%',
                                background: service.connected
                                    ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(16, 185, 129, 0.02) 100%)'
                                    : 'rgba(30, 41, 59, 0.5)',
                                border: `1px solid ${service.connected ? 'rgba(16, 185, 129, 0.3)' : 'rgba(255, 255, 255, 0.08)'}`,
                                borderRadius: 3,
                                transition: 'all 0.3s ease',
                                '&:hover': {
                                    transform: 'translateY(-4px)',
                                    boxShadow: `0 8px 30px ${service.color}20`,
                                    border: `1px solid ${service.color}50`,
                                },
                            }}>
                                <CardContent sx={{ p: 3 }}>
                                    {/* Service Header */}
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                            <Box sx={{
                                                p: 1.5,
                                                borderRadius: 2,
                                                bgcolor: `${service.color}15`,
                                                display: 'flex',
                                            }}>
                                                {getServiceIcon(service.icon, service.color)}
                                            </Box>
                                            <Box>
                                                <Typography variant="subtitle1" fontWeight={600}>
                                                    {service.name}
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    {service.free_tier}
                                                </Typography>
                                            </Box>
                                        </Box>
                                        <Tooltip title={service.connected ? (service.is_valid ? 'Connected & Valid' : 'Connected (validation pending)') : 'Not Connected'}>
                                            {service.connected ? (
                                                service.is_valid === true ? (
                                                    <CheckCircle sx={{ color: '#10b981', fontSize: 24 }} />
                                                ) : service.is_valid === false ? (
                                                    <ErrorIcon sx={{ color: '#ef4444', fontSize: 24 }} />
                                                ) : (
                                                    <CloudDone sx={{ color: '#f59e0b', fontSize: 24 }} />
                                                )
                                            ) : (
                                                <CloudOff sx={{ color: '#64748b', fontSize: 24 }} />
                                            )}
                                        </Tooltip>
                                    </Box>

                                    {/* Description */}
                                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2, minHeight: 40 }}>
                                        {service.description}
                                    </Typography>

                                    {/* Signals Enabled */}
                                    <Box sx={{ mb: 3 }}>
                                        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                                            Signals Provided:
                                        </Typography>
                                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                            {service.signals.slice(0, 4).map((signal) => (
                                                <Chip
                                                    key={signal}
                                                    label={signal}
                                                    size="small"
                                                    sx={{
                                                        fontSize: '0.65rem',
                                                        height: 22,
                                                        bgcolor: service.connected ? `${service.color}20` : 'rgba(255,255,255,0.05)',
                                                        color: service.connected ? service.color : '#64748b',
                                                        border: `1px solid ${service.connected ? service.color : 'transparent'}20`,
                                                    }}
                                                />
                                            ))}
                                            {service.signals.length > 4 && (
                                                <Chip
                                                    label={`+${service.signals.length - 4}`}
                                                    size="small"
                                                    sx={{
                                                        fontSize: '0.65rem',
                                                        height: 22,
                                                        bgcolor: 'rgba(255,255,255,0.05)',
                                                        color: '#94a3b8',
                                                    }}
                                                />
                                            )}
                                        </Box>
                                    </Box>

                                    {/* Actions */}
                                    <Box sx={{ display: 'flex', gap: 1, justifyContent: 'space-between', alignItems: 'center' }}>
                                        {service.connected ? (
                                            <>
                                                <Box sx={{ display: 'flex', gap: 1 }}>
                                                    <Tooltip title="Test Connection">
                                                        <IconButton
                                                            size="small"
                                                            onClick={() => handleTestKey(service.id)}
                                                            disabled={testMutation.isPending}
                                                            sx={{
                                                                bgcolor: 'rgba(59, 130, 246, 0.1)',
                                                                '&:hover': { bgcolor: 'rgba(59, 130, 246, 0.2)' }
                                                            }}
                                                        >
                                                            <RefreshIcon sx={{ fontSize: 18, color: '#3b82f6' }} />
                                                        </IconButton>
                                                    </Tooltip>
                                                    <Tooltip title="Update Key">
                                                        <IconButton
                                                            size="small"
                                                            onClick={() => handleOpenDialog(service)}
                                                            sx={{
                                                                bgcolor: 'rgba(245, 158, 11, 0.1)',
                                                                '&:hover': { bgcolor: 'rgba(245, 158, 11, 0.2)' }
                                                            }}
                                                        >
                                                            <AddIcon sx={{ fontSize: 18, color: '#f59e0b' }} />
                                                        </IconButton>
                                                    </Tooltip>
                                                    <Tooltip title="Disconnect">
                                                        <IconButton
                                                            size="small"
                                                            onClick={() => handleDeleteKey(service.id)}
                                                            disabled={deleteMutation.isPending}
                                                            sx={{
                                                                bgcolor: 'rgba(239, 68, 68, 0.1)',
                                                                '&:hover': { bgcolor: 'rgba(239, 68, 68, 0.2)' }
                                                            }}
                                                        >
                                                            <DeleteIcon sx={{ fontSize: 18, color: '#ef4444' }} />
                                                        </IconButton>
                                                    </Tooltip>
                                                </Box>
                                                <Chip
                                                    label="Connected"
                                                    size="small"
                                                    sx={{
                                                        bgcolor: 'rgba(16, 185, 129, 0.15)',
                                                        color: '#10b981',
                                                        fontWeight: 600,
                                                    }}
                                                />
                                            </>
                                        ) : (
                                            <>
                                                <Link
                                                    href={service.url}
                                                    target="_blank"
                                                    rel="noopener"
                                                    sx={{
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        gap: 0.5,
                                                        fontSize: '0.75rem',
                                                        color: '#94a3b8',
                                                        textDecoration: 'none',
                                                        '&:hover': { color: service.color }
                                                    }}
                                                >
                                                    Get API Key <LaunchIcon sx={{ fontSize: 14 }} />
                                                </Link>
                                                <Button
                                                    variant="contained"
                                                    size="small"
                                                    startIcon={<AddIcon />}
                                                    onClick={() => handleOpenDialog(service)}
                                                    sx={{
                                                        bgcolor: service.color,
                                                        '&:hover': { bgcolor: service.color, filter: 'brightness(1.2)' },
                                                        textTransform: 'none',
                                                        fontWeight: 600,
                                                    }}
                                                >
                                                    Connect
                                                </Button>
                                            </>
                                        )}
                                    </Box>
                                </CardContent>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            )}

            {/* Add/Update API Key Dialog */}
            <Dialog
                open={dialogOpen}
                onClose={() => setDialogOpen(false)}
                maxWidth="sm"
                fullWidth
                PaperProps={{
                    sx: {
                        bgcolor: '#1e293b',
                        borderRadius: 3,
                        border: '1px solid rgba(255,255,255,0.1)',
                    }
                }}
            >
                <DialogTitle sx={{ pb: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        {selectedService && getServiceIcon(selectedService.icon, selectedService.color)}
                        <Box>
                            <Typography variant="h6">
                                {selectedService?.connected ? 'Update' : 'Connect'} {selectedService?.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                                Enter your API key to enable {selectedService?.signals.length} signals
                            </Typography>
                        </Box>
                    </Box>
                </DialogTitle>
                <DialogContent>
                    {saveMutation.isError && (
                        <Alert severity="error" sx={{ mb: 2 }}>
                            Failed to save API key. Please check your key and try again.
                        </Alert>
                    )}

                    <TextField
                        fullWidth
                        label="API Key"
                        type={showPassword ? 'text' : 'password'}
                        value={apiKeyInput}
                        onChange={(e) => setApiKeyInput(e.target.value)}
                        placeholder={`Enter your ${selectedService?.name} API key`}
                        sx={{ mt: 2 }}
                        InputProps={{
                            endAdornment: (
                                <InputAdornment position="end">
                                    <IconButton
                                        onClick={() => setShowPassword(!showPassword)}
                                        edge="end"
                                    >
                                        {showPassword ? <VisibilityOff /> : <Visibility />}
                                    </IconButton>
                                </InputAdornment>
                            ),
                        }}
                    />

                    <Box sx={{ mt: 2, p: 2, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 2 }}>
                        <Typography variant="body2" color="text.secondary">
                            <strong>How to get your API key:</strong>
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                            1. Visit{' '}
                            <Link href={selectedService?.url} target="_blank" rel="noopener" sx={{ color: selectedService?.color }}>
                                {selectedService?.name}
                            </Link>
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            2. Create a free account (usually includes {selectedService?.free_tier})
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            3. Navigate to your dashboard/settings and generate an API key
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            4. Copy and paste the key above
                        </Typography>
                    </Box>

                    <Collapse in={selectedService?.signals && selectedService.signals.length > 0}>
                        <Box sx={{ mt: 2 }}>
                            <Typography variant="caption" color="text.secondary">
                                Signals you'll unlock:
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                                {selectedService?.signals.map((signal) => (
                                    <Chip
                                        key={signal}
                                        label={signal}
                                        size="small"
                                        sx={{
                                            bgcolor: `${selectedService?.color}20`,
                                            color: selectedService?.color,
                                            fontSize: '0.7rem',
                                        }}
                                    />
                                ))}
                            </Box>
                        </Box>
                    </Collapse>
                </DialogContent>
                <DialogActions sx={{ p: 2.5, pt: 0 }}>
                    <Button onClick={() => setDialogOpen(false)} sx={{ color: '#94a3b8' }}>
                        Cancel
                    </Button>
                    <Button
                        variant="contained"
                        onClick={handleSaveKey}
                        disabled={!apiKeyInput.trim() || saveMutation.isPending}
                        sx={{
                            bgcolor: selectedService?.color || '#3b82f6',
                            '&:hover': { bgcolor: selectedService?.color || '#3b82f6', filter: 'brightness(1.2)' },
                        }}
                    >
                        {saveMutation.isPending ? (
                            <>
                                <CircularProgress size={16} sx={{ mr: 1, color: 'white' }} />
                                Validating...
                            </>
                        ) : (
                            selectedService?.connected ? 'Update Key' : 'Connect & Validate'
                        )}
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}
