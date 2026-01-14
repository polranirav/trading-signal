/**
 * Testimonials Component
 * 
 * Displays user testimonials for social proof.
 */

import { Box, Typography, Grid, Card, CardContent, Avatar, Rating } from '@mui/material';
import { useState } from 'react';

interface Testimonial {
  id: string;
  name: string;
  role: string;
  content: string;
  rating: number;
  avatar?: string;
}

const testimonials: Testimonial[] = [
  {
    id: '1',
    name: 'Sarah Chen',
    role: 'Day Trader',
    content:
      'The AI signals have completely transformed my trading. I\'ve seen a 34% increase in returns since using Trading Signals Pro. The risk metrics help me make better decisions.',
    rating: 5,
  },
  {
    id: '2',
    name: 'Michael Rodriguez',
    role: 'Swing Trader',
    content:
      'Finally, a platform that combines technical analysis with sentiment data. The signals are accurate and the interface is clean. Highly recommend!',
    rating: 5,
  },
  {
    id: '3',
    name: 'David Kim',
    role: 'Portfolio Manager',
    content:
      'Using this platform has saved me hours of research. The ML models catch patterns I would have missed. My portfolio performance has improved significantly.',
    rating: 5,
  },
  {
    id: '4',
    name: 'Emily Watson',
    role: 'Retail Investor',
    content:
      'As someone new to trading, the clear explanations and risk metrics helped me understand what I\'m getting into. The signals are reliable and well-explained.',
    rating: 5,
  },
];

export default function Testimonials() {
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map((n) => n[0])
      .join('')
      .toUpperCase();
  };

  return (
    <Box sx={{ py: 8, bgcolor: 'background.default' }}>
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h3" component="h2" gutterBottom fontWeight={700}>
          What Our Users Say
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
          Join thousands of traders who trust Trading Signals Pro for their trading decisions
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {testimonials.map((testimonial) => (
          <Grid item xs={12} md={6} key={testimonial.id}>
            <Card
              sx={{
                height: '100%',
                transition: 'transform 0.2s, box-shadow 0.2s',
                transform: hoveredId === testimonial.id ? 'translateY(-4px)' : 'none',
                boxShadow: hoveredId === testimonial.id ? 4 : 1,
              }}
              onMouseEnter={() => setHoveredId(testimonial.id)}
              onMouseLeave={() => setHoveredId(null)}
            >
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar
                    sx={{
                      bgcolor: 'primary.main',
                      width: 56,
                      height: 56,
                      mr: 2,
                      fontSize: '1.25rem',
                      fontWeight: 600,
                    }}
                  >
                    {getInitials(testimonial.name)}
                  </Avatar>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" fontWeight={600}>
                      {testimonial.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {testimonial.role}
                    </Typography>
                    <Rating value={testimonial.rating} readOnly size="small" sx={{ mt: 0.5 }} />
                  </Box>
                </Box>
                <Typography variant="body1" color="text.primary" sx={{ fontStyle: 'italic' }}>
                  "{testimonial.content}"
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
