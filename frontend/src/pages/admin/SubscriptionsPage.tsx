/**
 * Subscriptions Management Page
 * 
 * List and manage user subscriptions.
 */

import { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TextField,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminService } from '../../services/admin';
import EditIcon from '@mui/icons-material/Edit';
import type { AdminSubscription } from '../../types/admin';

export default function SubscriptionsPage() {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(50);
  const [selectedSubscription, setSelectedSubscription] = useState<AdminSubscription | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [formData, setFormData] = useState({ tier: '', status: '' });
  
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ['admin-subscriptions', page, pageSize],
    queryFn: () => adminService.listSubscriptions({ page: page + 1, page_size: pageSize }),
  });

  const updateMutation = useMutation({
    mutationFn: (data: { tier?: string; status?: string }) =>
      adminService.updateSubscription(selectedSubscription!.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-subscriptions'] });
      setEditDialogOpen(false);
    },
  });

  const handleEdit = (subscription: AdminSubscription) => {
    setSelectedSubscription(subscription);
    setFormData({
      tier: subscription.tier,
      status: subscription.status,
    });
    setEditDialogOpen(true);
  };

  const handleUpdate = () => {
    if (selectedSubscription) {
      updateMutation.mutate(formData);
    }
  };

  const getStatusColor = (status: string) => {
    if (status === 'active') return 'success';
    if (status === 'cancelled' || status === 'expired') return 'error';
    return 'default';
  };

  const subscriptions = data?.data?.subscriptions || [];
  const total = data?.data?.total || 0;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" component="h1" fontWeight={700}>
            Subscriptions Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage user subscriptions
          </Typography>
        </Box>
      </Box>

      <Card>
        <CardContent>
          {isLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          ) : error ? (
            <Alert severity="error">Failed to load subscriptions</Alert>
          ) : (
            <>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>User Email</TableCell>
                      <TableCell>Tier</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Period Start</TableCell>
                      <TableCell>Period End</TableCell>
                      <TableCell align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {subscriptions.map((subscription) => (
                      <TableRow key={subscription.id}>
                        <TableCell>{subscription.user_email || '-'}</TableCell>
                        <TableCell>
                          <Chip label={subscription.tier} size="small" variant="outlined" />
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={subscription.status}
                            color={getStatusColor(subscription.status) as any}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          {subscription.current_period_start
                            ? new Date(subscription.current_period_start).toLocaleDateString()
                            : '-'}
                        </TableCell>
                        <TableCell>
                          {subscription.current_period_end
                            ? new Date(subscription.current_period_end).toLocaleDateString()
                            : '-'}
                        </TableCell>
                        <TableCell align="right">
                          <IconButton size="small" onClick={() => handleEdit(subscription)}>
                            <EditIcon />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              <TablePagination
                component="div"
                count={total}
                page={page}
                onPageChange={(_, newPage) => setPage(newPage)}
                rowsPerPage={pageSize}
                onRowsPerPageChange={(e) => {
                  setPageSize(parseInt(e.target.value, 10));
                  setPage(0);
                }}
                rowsPerPageOptions={[25, 50, 100]}
              />
            </>
          )}
        </CardContent>
      </Card>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Subscription</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Tier</InputLabel>
              <Select
                value={formData.tier}
                label="Tier"
                onChange={(e) => setFormData({ ...formData, tier: e.target.value })}
              >
                <MenuItem value="free">Free</MenuItem>
                <MenuItem value="essential">Essential</MenuItem>
                <MenuItem value="advanced">Advanced</MenuItem>
                <MenuItem value="premium">Premium</MenuItem>
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>Status</InputLabel>
              <Select
                value={formData.status}
                label="Status"
                onChange={(e) => setFormData({ ...formData, status: e.target.value })}
              >
                <MenuItem value="active">Active</MenuItem>
                <MenuItem value="cancelled">Cancelled</MenuItem>
                <MenuItem value="expired">Expired</MenuItem>
                <MenuItem value="trial">Trial</MenuItem>
                <MenuItem value="past_due">Past Due</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleUpdate} variant="contained" disabled={updateMutation.isPending}>
            {updateMutation.isPending ? 'Saving...' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
