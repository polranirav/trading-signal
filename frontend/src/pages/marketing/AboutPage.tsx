/**
 * About Page
 * 
 * Company/product information and mission.
 */

import { Box, Container, Typography, Grid, Card, CardContent, Button } from '@mui/material';
import { Link } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';

export default function AboutPage() {
  return (
    <Box>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }} component={Link} to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            Trading Signals Pro
          </Typography>
          <Button color="inherit" component={Link} to="/features">
            Features
          </Button>
          <Button color="inherit" component={Link} to="/pricing">
            Pricing
          </Button>
          <Button color="inherit" component={Link} to="/login">
            Login
          </Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 8 }}>
        <Typography variant="h3" component="h1" gutterBottom fontWeight={700} align="center">
          About Trading Signals Pro
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 6 }} align="center">
          AI-powered trading signals for informed investment decisions.
        </Typography>

        {/* Mission */}
        <Card sx={{ mb: 6 }}>
          <CardContent>
            <Typography variant="h5" component="h2" gutterBottom fontWeight={700}>
              Our Mission
            </Typography>
            <Typography variant="body1" paragraph>
              Trading Signals Pro empowers traders and investors with cutting-edge AI technology to make
              better-informed trading decisions. We combine advanced machine learning, technical analysis,
              and sentiment analysis to deliver high-quality trading signals with comprehensive risk metrics.
            </Typography>
            <Typography variant="body1">
              Our goal is to democratize access to professional-grade trading tools, making sophisticated
              market analysis available to traders of all levels.
            </Typography>
          </CardContent>
        </Card>

        {/* Technology */}
        <Box sx={{ mb: 6 }}>
          <Typography variant="h5" component="h2" gutterBottom fontWeight={700} sx={{ mb: 3 }}>
            Technology
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Machine Learning
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Advanced deep learning models including LSTMs, Temporal Fusion Transformers, and ensemble
                    methods trained on vast amounts of market data.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Sentiment Analysis
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    State-of-the-art FinBERT models and GPT-4 integration for real-time market sentiment
                    analysis from news, social media, and financial reports.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Risk Management
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Comprehensive risk metrics including Value at Risk (VaR), Conditional VaR (CVaR),
                    and dynamic position sizing based on Kelly Criterion principles.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Real-Time Processing
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    High-performance infrastructure ensuring fast signal generation and delivery,
                    with scalable architecture for growing user bases.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>

        {/* Trust & Security */}
        <Card sx={{ mb: 6 }}>
          <CardContent>
            <Typography variant="h5" component="h2" gutterBottom fontWeight={700}>
              Trust & Security
            </Typography>
            <Typography variant="body1" paragraph>
              We take security and data privacy seriously. All user data is encrypted in transit and at rest.
              We never store your trading credentials or execute trades on your behalf - we provide signals
              and analysis, you make the decisions.
            </Typography>
            <Typography variant="body1">
              Our API uses industry-standard authentication methods, and we regularly audit our systems
              for security vulnerabilities.
            </Typography>
          </CardContent>
        </Card>

        {/* Support */}
        <Card sx={{ mb: 6 }}>
          <CardContent>
            <Typography variant="h5" component="h2" gutterBottom fontWeight={700}>
              Support
            </Typography>
            <Typography variant="body1" paragraph>
              Need help? We're here for you. All plans include email support, and our Essential and Advanced
              plans include priority support with faster response times.
            </Typography>
            <Typography variant="body1">
              Visit our API documentation for technical integration help, or contact us directly for
              account-related questions.
            </Typography>
          </CardContent>
        </Card>

        {/* CTA */}
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="h5" gutterBottom>
            Ready to Get Started?
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mt: 3 }}>
            <Button variant="contained" size="large" component={Link} to="/register">
              Start Free Trial
            </Button>
            <Button variant="outlined" size="large" component={Link} to="/pricing">
              View Pricing
            </Button>
          </Box>
        </Box>
      </Container>
    </Box>
  );
}
