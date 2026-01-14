"""
Authentication layouts for the Dash application.

Provides login and register page layouts.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_login_page():
    """Create the login page layout."""
    return html.Div([
        # Background gradient
        html.Div(className="auth-background"),
        
        # Centered login card
        html.Div([
            # Logo
            html.Div([
                html.Div("TS", className="auth-logo-icon"),
                html.H2("Trading Signals Pro", className="auth-logo-text")
            ], className="auth-logo"),
            
            # Login form
            html.Div([
                html.H4("Welcome Back", className="text-white mb-4 text-center"),
                
                # Email input
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="fas fa-envelope")),
                    dbc.Input(
                        id="login-email",
                        type="email",
                        placeholder="Email address",
                        className="bg-dark text-white border-secondary"
                    )
                ], className="mb-3"),
                
                # Password input
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="fas fa-lock")),
                    dbc.Input(
                        id="login-password",
                        type="password",
                        placeholder="Password",
                        className="bg-dark text-white border-secondary"
                    )
                ], className="mb-3"),
                
                # Remember me checkbox
                dbc.Checkbox(
                    id="login-remember",
                    label="Remember me",
                    value=False,
                    className="mb-3 text-muted"
                ),
                
                # Error message placeholder
                html.Div(id="login-error", className="text-danger mb-3"),
                
                # Login button
                dbc.Button(
                    "Sign In",
                    id="login-button",
                    color="primary",
                    className="w-100 mb-3 btn-lg"
                ),
                
                # Forgot password link
                html.Div([
                    html.A("Forgot password?", href="#", className="text-muted small")
                ], className="text-center mb-4"),
                
                # Divider
                html.Div([
                    html.Hr(className="my-0"),
                    html.Span("or", className="auth-divider-text")
                ], className="auth-divider mb-4"),
                
                # Register link
                html.Div([
                    html.Span("Don't have an account? ", className="text-muted"),
                    dcc.Link("Sign up", href="/register", className="text-primary")
                ], className="text-center")
                
            ], className="auth-card glass-card p-4")
            
        ], className="auth-container")
        
    ], className="auth-page")


def create_register_page():
    """Create the registration page layout."""
    return html.Div([
        # Background gradient
        html.Div(className="auth-background"),
        
        # Centered register card
        html.Div([
            # Logo
            html.Div([
                html.Div("TS", className="auth-logo-icon"),
                html.H2("Trading Signals Pro", className="auth-logo-text")
            ], className="auth-logo"),
            
            # Register form
            html.Div([
                html.H4("Create Account", className="text-white mb-4 text-center"),
                
                # Full name input
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="fas fa-user")),
                    dbc.Input(
                        id="register-name",
                        type="text",
                        placeholder="Full name",
                        className="bg-dark text-white border-secondary"
                    )
                ], className="mb-3"),
                
                # Email input
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="fas fa-envelope")),
                    dbc.Input(
                        id="register-email",
                        type="email",
                        placeholder="Email address",
                        className="bg-dark text-white border-secondary"
                    )
                ], className="mb-3"),
                
                # Password input
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="fas fa-lock")),
                    dbc.Input(
                        id="register-password",
                        type="password",
                        placeholder="Password (min 8 characters)",
                        className="bg-dark text-white border-secondary"
                    )
                ], className="mb-3"),
                
                # Confirm password input
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="fas fa-lock")),
                    dbc.Input(
                        id="register-password-confirm",
                        type="password",
                        placeholder="Confirm password",
                        className="bg-dark text-white border-secondary"
                    )
                ], className="mb-3"),
                
                # Terms checkbox
                dbc.Checkbox(
                    id="register-terms",
                    label="I agree to the Terms of Service and Privacy Policy",
                    value=False,
                    className="mb-3 text-muted small"
                ),
                
                # Error message placeholder
                html.Div(id="register-error", className="text-danger mb-3"),
                
                # Register button
                dbc.Button(
                    "Create Account",
                    id="register-button",
                    color="primary",
                    className="w-100 mb-3 btn-lg"
                ),
                
                # Login link
                html.Div([
                    html.Span("Already have an account? ", className="text-muted"),
                    dcc.Link("Sign in", href="/login", className="text-primary")
                ], className="text-center")
                
            ], className="auth-card glass-card p-4")
            
        ], className="auth-container")
        
    ], className="auth-page")


def create_landing_page():
    """Create a marketing landing page for non-authenticated users."""
    return html.Div([
        # Navigation
        html.Nav([
            html.Div([
                # Logo
                html.Div([
                    html.Div("TS", className="logo-icon-sm"),
                    html.Span("Trading Signals Pro", className="ms-2 fw-bold")
                ], className="d-flex align-items-center"),
                
                # Nav Links
                html.Div([
                    dcc.Link("Features", href="#features", className="nav-link-landing"),
                    dcc.Link("Pricing", href="#pricing", className="nav-link-landing"),
                    dcc.Link("Login", href="/login", className="nav-link-landing"),
                    dcc.Link("Sign Up", href="/register", className="btn btn-primary ms-2")
                ], className="d-flex align-items-center gap-3")
            ], className="container d-flex justify-content-between align-items-center py-3")
        ], className="landing-nav"),
        
        # Hero Section
        html.Section([
            html.Div([
                html.H1("AI-Powered Trading Signals", className="hero-title"),
                html.P(
                    "Professional-grade trading signals powered by machine learning, "
                    "technical analysis, and FinBERT sentiment analysis.",
                    className="hero-subtitle"
                ),
                html.Div([
                    dcc.Link("Start Free Trial", href="/register", className="btn btn-primary btn-lg me-3"),
                    dcc.Link("Learn More", href="#features", className="btn btn-outline-light btn-lg")
                ], className="hero-cta"),
                html.Div([
                    html.Span("✓ No credit card required", className="me-4"),
                    html.Span("✓ 7-day free trial", className="me-4"),
                    html.Span("✓ Cancel anytime")
                ], className="hero-badges mt-4 text-muted")
            ], className="container text-center")
        ], className="hero-section"),
        
        # Features Section
        html.Section([
            html.Div([
                html.H2("Why Choose Trading Signals Pro?", className="text-center mb-5"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-brain feature-icon"),
                            html.H4("AI-Powered Analysis"),
                            html.P("FinBERT sentiment analysis + ML ensemble models for 62-65% accuracy.")
                        ], className="feature-card glass-card p-4 text-center h-100")
                    ], md=4, className="mb-4"),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-bolt feature-icon"),
                            html.H4("Real-Time Signals"),
                            html.P("Instant email alerts when high-confidence trading opportunities appear.")
                        ], className="feature-card glass-card p-4 text-center h-100")
                    ], md=4, className="mb-4"),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-shield-alt feature-icon"),
                            html.H4("Risk Management"),
                            html.P("Monte Carlo VaR, position sizing, and stop-loss recommendations.")
                        ], className="feature-card glass-card p-4 text-center h-100")
                    ], md=4, className="mb-4"),
                ])
            ], className="container")
        ], className="features-section py-5", id="features"),
        
        # CTA Section
        html.Section([
            html.Div([
                html.H2("Ready to Start Trading Smarter?", className="mb-4"),
                html.P("Join thousands of traders using AI-powered signals.", className="mb-4 text-muted"),
                dcc.Link("Create Free Account", href="/register", className="btn btn-primary btn-lg")
            ], className="container text-center")
        ], className="cta-section py-5"),
        
        # Footer
        html.Footer([
            html.Div([
                html.P("© 2026 Trading Signals Pro. All rights reserved.", className="text-muted mb-0")
            ], className="container text-center py-4")
        ])
        
    ], className="landing-page")
