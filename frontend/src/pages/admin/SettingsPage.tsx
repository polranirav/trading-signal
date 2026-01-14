/**
 * Settings Management Page
 * 
 * Manage system settings.
 */

import { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
} from '@mui/material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminService } from '../../services/admin';
import EditIcon from '@mui/icons-material/Edit';
import type { SystemSetting } from '../../types/admin';

export default function SettingsPage() {
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [selectedSetting, setSelectedSetting] = useState<SystemSetting | null>(null);
  const [formData, setFormData] = useState({ value: '', value_type: 'string', description: '', category: '', is_public: false });
  
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ['admin-settings'],
    queryFn: () => adminService.listSettings(),
  });

  const updateMutation = useMutation({
    mutationFn: (data: { value: any; value_type: string; description?: string; category?: string; is_public?: boolean }) =>
      adminService.setSetting(selectedSetting!.key, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-settings'] });
      setEditDialogOpen(false);
    },
  });

  const handleEdit = (setting: SystemSetting) => {
    setSelectedSetting(setting);
    setFormData({
      value: typeof setting.value === 'string' ? setting.value : JSON.stringify(setting.value),
      value_type: setting.value_type,
      description: setting.description || '',
      category: setting.category || '',
      is_public: setting.is_public || false,
    });
    setEditDialogOpen(true);
  };

  const handleUpdate = () => {
    if (selectedSetting) {
      let value = formData.value;
      if (formData.value_type === 'json') {
        try {
          value = JSON.parse(value);
        } catch (e) {
          alert('Invalid JSON');
          return;
        }
      } else if (formData.value_type === 'int') {
        value = parseInt(value, 10);
      } else if (formData.value_type === 'float') {
        value = parseFloat(value);
      } else if (formData.value_type === 'bool') {
        value = value === 'true' || value === '1';
      }
      
      updateMutation.mutate({
        value,
        value_type: formData.value_type,
        description: formData.description,
        category: formData.category,
        is_public: formData.is_public,
      });
    }
  };

  const settings = data?.data?.settings || [];

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" component="h1" fontWeight={700}>
            System Settings
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage system-wide configuration
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
            <Alert severity="error">Failed to load settings</Alert>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Key</TableCell>
                    <TableCell>Value</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Category</TableCell>
                    <TableCell>Public</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {settings.map((setting) => (
                    <TableRow key={setting.key}>
                      <TableCell>
                        <Typography fontWeight={600}>{setting.key}</Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {typeof setting.value === 'object' ? JSON.stringify(setting.value) : String(setting.value)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip label={setting.value_type} size="small" variant="outlined" />
                      </TableCell>
                      <TableCell>{setting.category || '-'}</TableCell>
                      <TableCell>
                        <Chip
                          label={setting.is_public ? 'Yes' : 'No'}
                          color={setting.is_public ? 'success' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell align="right">
                        <IconButton size="small" onClick={() => handleEdit(setting)}>
                          <EditIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Edit Setting</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              label="Key"
              value={selectedSetting?.key || ''}
              disabled
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Value"
              value={formData.value}
              onChange={(e) => setFormData({ ...formData, value: e.target.value })}
              multiline
              rows={formData.value_type === 'json' ? 4 : 1}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Description"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Category"
              value={formData.category}
              onChange={(e) => setFormData({ ...formData, category: e.target.value })}
              sx={{ mb: 2 }}
            />
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
