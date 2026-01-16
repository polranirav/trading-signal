/**
 * Account Page
 * 
 * User profile, subscription, and API key management.
 */

import { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Chip,
  Divider,
  CircularProgress,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../../services/api';
import { useAuthStore } from '../../store/authStore';
import { format } from 'date-fns';
import DataSourcesApiKeys from '../../components/account/DataSourcesApiKeys';

export default function AccountPage() {
  const { user } = useAuthStore();
  const queryClient = useQueryClient();
  const [apiKeyName, setApiKeyName] = useState('');
  const [newApiKeyDialog, setNewApiKeyDialog] = useState(false);
  const [newApiKey, setNewApiKey] = useState<string | null>(null);
  const [fullName, setFullName] = useState(user?.full_name || '');
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const { data: accountData } = useQuery({
    queryKey: ['account'],
    queryFn: () => api.account.getAccount(),
  });

  const { data: apiKeysData, isLoading: keysLoading, refetch: refetchKeys } = useQuery({
    queryKey: ['api-keys'],
    queryFn: () => api.account.getApiKeys(),
  });

  const { data: subscriptionData, isLoading: subscriptionLoading } = useQuery({
    queryKey: ['subscription'],
    queryFn: () => api.subscription.getSubscription(),
  });

  const updateAccountMutation = useMutation({
    mutationFn: (data: { full_name?: string }) => api.account.updateAccount(data),
    onSuccess: () => {
      setMessage({ type: 'success', text: 'Account updated successfully' });
      queryClient.invalidateQueries({ queryKey: ['account'] });
      useAuthStore.getState().fetchUser();
    },
    onError: (error: any) => {
      setMessage({ type: 'error', text: error.response?.data?.message || 'Failed to update account' });
    },
  });

  const createApiKeyMutation = useMutation({
    mutationFn: (name?: string) => api.account.createApiKey(name),
    onSuccess: (response) => {
      const apiKey = response.data?.api_key || null;
      setNewApiKey(apiKey);
      setApiKeyName('');
      refetchKeys();
    },
    onError: (error: any) => {
      setMessage({ type: 'error', text: error.response?.data?.message || 'Failed to create API key' });
      setNewApiKeyDialog(false);
    },
  });

  const deleteApiKeyMutation = useMutation({
    mutationFn: (keyId: string) => api.account.deleteApiKey(keyId),
    onSuccess: () => {
      setMessage({ type: 'success', text: 'API key deleted successfully' });
      refetchKeys();
    },
    onError: (error: any) => {
      setMessage({ type: 'error', text: error.response?.data?.message || 'Failed to delete API key' });
    },
  });

  useEffect(() => {
    if (accountData?.data?.user?.full_name) {
      setFullName(accountData.data.user.full_name);
    }
  }, [accountData]);

  const handleUpdateAccount = () => {
    updateAccountMutation.mutate({ full_name: fullName || undefined });
  };

  const handleCreateApiKey = () => {
    createApiKeyMutation.mutate(apiKeyName || undefined);
  };

  const handleDeleteApiKey = (keyId: string) => {
    if (window.confirm('Are you sure you want to delete this API key?')) {
      deleteApiKeyMutation.mutate(keyId);
    }
  };

  const handleCloseNewKeyDialog = () => {
    setNewApiKeyDialog(false);
    setApiKeyName('');
    setNewApiKey(null);
  };

  const apiKeys = apiKeysData?.data?.api_keys || [];
  const subscription = subscriptionData?.data?.subscription;

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom fontWeight={700}>
        Account Settings
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
        Manage your profile, subscription, and API keys.
      </Typography>

      {message && (
        <Alert severity={message.type} sx={{ mb: 3 }} onClose={() => setMessage(null)}>
          {message.text}
        </Alert>
      )}

      {/* Profile */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Profile Information
          </Typography>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Email"
                value={user?.email || ''}
                disabled
                helperText="Email cannot be changed"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Full Name"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
              />
            </Grid>
            <Grid item xs={12}>
              <Button
                variant="contained"
                onClick={handleUpdateAccount}
                disabled={updateAccountMutation.isPending}
              >
                {updateAccountMutation.isPending ? 'Updating...' : 'Update Profile'}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Subscription */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Subscription
          </Typography>
          {subscriptionLoading ? (
            <CircularProgress />
          ) : subscription ? (
            <Box sx={{ mt: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" color="text.secondary">
                    Tier
                  </Typography>
                  <Chip label={subscription.tier.toUpperCase()} color="primary" sx={{ mt: 1 }} />
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" color="text.secondary">
                    Status
                  </Typography>
                  <Chip label={subscription.status} color={subscription.status === 'active' ? 'success' : 'default'} sx={{ mt: 1 }} />
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" color="text.secondary">
                    Limits
                  </Typography>
                  <Typography variant="body1" sx={{ mt: 1 }}>
                    {subscription.limits.max_signals_per_day} signals/day
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          ) : (
            <Typography color="text.secondary">No subscription found</Typography>
          )}
        </CardContent>
      </Card>

      {/* API Keys */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              API Keys
            </Typography>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setNewApiKeyDialog(true)}
            >
              Create API Key
            </Button>
          </Box>
          <Divider sx={{ mb: 2 }} />
          {keysLoading ? (
            <CircularProgress />
          ) : apiKeys.length > 0 ? (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Key Prefix</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Last Used</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {apiKeys.map((key) => (
                    <TableRow key={key.id}>
                      <TableCell>{key.name || 'Unnamed'}</TableCell>
                      <TableCell>
                        <code>{key.key_prefix}...</code>
                      </TableCell>
                      <TableCell>
                        {key.created_at ? format(new Date(key.created_at), 'MMM d, yyyy') : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {key.last_used ? format(new Date(key.last_used), 'MMM d, yyyy') : 'Never'}
                      </TableCell>
                      <TableCell>
                        <IconButton
                          color="error"
                          size="small"
                          onClick={() => handleDeleteApiKey(key.id)}
                          disabled={deleteApiKeyMutation.isPending}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Typography color="text.secondary">No API keys found</Typography>
          )}
        </CardContent>
      </Card>

      {/* Data Source API Keys - External Services */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <DataSourcesApiKeys />
        </CardContent>
      </Card>

      {/* New API Key Dialog */}
      <Dialog open={newApiKeyDialog} onClose={handleCloseNewKeyDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Create API Key</DialogTitle>
        <DialogContent>
          {newApiKey ? (
            <Alert severity="success" sx={{ mb: 2 }}>
              API key created successfully! Copy it now - you won't be able to see it again.
            </Alert>
          ) : null}
          {newApiKey ? (
            <TextField
              fullWidth
              label="API Key"
              value={newApiKey}
              multiline
              rows={3}
              InputProps={{
                readOnly: true,
              }}
              sx={{ mt: 2 }}
            />
          ) : (
            <TextField
              fullWidth
              label="Name (optional)"
              value={apiKeyName}
              onChange={(e) => setApiKeyName(e.target.value)}
              placeholder="e.g., Production Key"
              sx={{ mt: 2 }}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseNewKeyDialog}>
            {newApiKey ? 'Close' : 'Cancel'}
          </Button>
          {!newApiKey && (
            <Button
              variant="contained"
              onClick={handleCreateApiKey}
              disabled={createApiKeyMutation.isPending}
            >
              {createApiKeyMutation.isPending ? 'Creating...' : 'Create'}
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
}
