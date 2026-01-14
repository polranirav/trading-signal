/**
 * Audit Logs Page
 * 
 * View audit trail of admin actions.
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
  Chip,
  Alert,
  CircularProgress,
  TextField,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { adminService } from '../../services/admin';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import type { AuditLog } from '../../types/admin';

export default function AuditLogsPage() {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(50);
  const [actionFilter, setActionFilter] = useState('');

  const { data, isLoading, error } = useQuery({
    queryKey: ['admin-audit-logs', page, pageSize, actionFilter],
    queryFn: () =>
      adminService.listAuditLogs({
        page: page + 1,
        page_size: pageSize,
        action: actionFilter || undefined,
      }),
  });

  const logs = data?.data?.logs || [];
  const total = data?.data?.total || 0;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" component="h1" fontWeight={700}>
            Audit Logs
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Complete audit trail of all admin actions
          </Typography>
        </Box>
      </Box>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <TextField
            fullWidth
            placeholder="Filter by action..."
            value={actionFilter}
            onChange={(e) => {
              setActionFilter(e.target.value);
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
            <Alert severity="error">Failed to load audit logs</Alert>
          ) : (
            <>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Timestamp</TableCell>
                      <TableCell>Admin</TableCell>
                      <TableCell>Action</TableCell>
                      <TableCell>Resource</TableCell>
                      <TableCell>IP Address</TableCell>
                      <TableCell>Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {logs.map((log) => (
                      <TableRow key={log.id}>
                        <TableCell>
                          {log.created_at ? new Date(log.created_at).toLocaleString() : '-'}
                        </TableCell>
                        <TableCell>{log.admin_email}</TableCell>
                        <TableCell>
                          <Typography variant="body2" fontWeight={600}>
                            {log.action}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {log.resource_type}
                            {log.resource_id && ` (${log.resource_id.substring(0, 8)}...)`}
                          </Typography>
                        </TableCell>
                        <TableCell>{log.ip_address || '-'}</TableCell>
                        <TableCell>
                          <Chip
                            icon={log.success ? <CheckCircleIcon /> : <CancelIcon />}
                            label={log.success ? 'Success' : 'Failed'}
                            color={log.success ? 'success' : 'error'}
                            size="small"
                          />
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
    </Box>
  );
}
