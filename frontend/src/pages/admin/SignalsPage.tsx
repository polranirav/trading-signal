/**
 * Signals Management Page
 * 
 * List and manage trading signals.
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
} from '@mui/material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminService } from '../../services/admin';
import DeleteIcon from '@mui/icons-material/Delete';
import type { AdminSignal } from '../../types/admin';

export default function SignalsPage() {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(50);
  const [search, setSearch] = useState('');
  const [selectedSignal, setSelectedSignal] = useState<AdminSignal | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ['admin-signals', page, pageSize, search],
    queryFn: () => adminService.listSignals({ page: page + 1, page_size: pageSize, symbol: search || undefined }),
  });

  const deleteMutation = useMutation({
    mutationFn: () => adminService.deleteSignal(selectedSignal!.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-signals'] });
      setDeleteDialogOpen(false);
    },
  });

  const handleDelete = (signal: AdminSignal) => {
    setSelectedSignal(signal);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (selectedSignal) {
      deleteMutation.mutate();
    }
  };

  const getSignalColor = (signalType: string) => {
    if (signalType.includes('BUY')) return 'success';
    if (signalType.includes('SELL')) return 'error';
    return 'default';
  };

  const signals = data?.data?.signals || [];
  const total = data?.data?.total || 0;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" component="h1" fontWeight={700}>
            Signals Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            View and manage trading signals
          </Typography>
        </Box>
      </Box>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <TextField
            fullWidth
            placeholder="Search signals by symbol..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(0);
            }}
          />
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          {isLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          ) : error ? (
            <Alert severity="error">Failed to load signals</Alert>
          ) : (
            <>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Confidence</TableCell>
                      <TableCell>Price</TableCell>
                      <TableCell>Created</TableCell>
                      <TableCell align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {signals.map((signal) => (
                      <TableRow key={signal.id}>
                        <TableCell>
                          <Typography fontWeight={600}>{signal.symbol}</Typography>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={signal.signal_type}
                            color={getSignalColor(signal.signal_type) as any}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          {signal.confluence_score
                            ? `${(signal.confluence_score * 100).toFixed(1)}%`
                            : '-'}
                        </TableCell>
                        <TableCell>
                          {signal.price_at_signal ? `$${signal.price_at_signal.toFixed(2)}` : '-'}
                        </TableCell>
                        <TableCell>
                          {signal.created_at ? new Date(signal.created_at).toLocaleDateString() : '-'}
                        </TableCell>
                        <TableCell align="right">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleDelete(signal)}
                          >
                            <DeleteIcon />
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

      {/* Delete Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Signal</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete signal for <strong>{selectedSignal?.symbol}</strong>? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained" disabled={deleteMutation.isPending}>
            {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
