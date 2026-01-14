"""
Marketing Website Pages.

Public-facing pages for marketing:
- Landing page
- Features page
- About/Trust elements
- SEO-optimized content
"""

import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from typing import List, Dict


def create_landing_page() -> html.Div:
    """Create the main landing page with hero section, features, and CTA."""
    return html.Div([
        # Hero Section
        html.Section([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1(
                            "AI-Powered Trading Signals",
                            className="display-3 fw-bold text-white mb-4"
                        ),
                        html.P(
                            "Professional-grade trading signals powered by machine learning, "
                            "technical analysis, and sentiment analysis. Make data-driven trading decisions.",
                            className="lead text-white-50 mb-4 fs-4"
                        ),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Start Free Trial",
                                    href="/register",
                                    color="primary",
                                    size="lg",
                                    className="me-3 mb-3"
                                ),
                                dbc.Button(
                                    "View Pricing",
                                    href="/pricing",
                                    color="outline-light",
                                    size="lg",
                                    outline=True,
                                    className="mb-3"
                                )
                            ], width=12, md="auto")
                        ]),
                        html.Div([
                            html.Small("✓ No credit card required", className="text-white-50 me-3"),
                            html.Small("✓ 7-day free trial", className="text-white-50 me-3"),
                            html.Small("✓ Cancel anytime", className="text-white-50")
                        ], className="mt-3")
                    ], md=7, className="mb-5 mb-md-0"),
                    dbc.Col([
                        html.Div([
                            html.Img(
                                src="/assets/dashboard-preview.png",
                                alt="Trading Signals Dashboard",
                                className="img-fluid rounded shadow-lg",
                                style={"max-height": "500px"}
                            )
                        ], className="text-center")
                    ], md=5)
                ], className="align-items-center py-5")
            ], fluid=True)
        ], className="bg-gradient-primary py-5", style={
            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "min-height": "600px",
            "display": "flex",
            "align-items": "center"
        }),
        
        # Features Section
        html.Section([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H2("Why Choose Trading Signals Pro?", className="text-center mb-5"),
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-brain fa-3x text-primary mb-3")
                                ], className="text-center"),
                                html.H4("AI-Powered Analysis", className="card-title text-center mb-3"),
                                html.P(
                                    "Advanced machine learning models analyze market patterns, "
                                    "sentiment, and technical indicators to generate high-confidence signals.",
                                    className="card-text text-muted"
                                )
                            ])
                        ], className="h-100 shadow-sm")
                    ], md=4, className="mb-4"),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-chart-line fa-3x text-success mb-3")
                                ], className="text-center"),
                                html.H4("Real-Time Signals", className="card-title text-center mb-3"),
                                html.P(
                                    "Get instant notifications when high-probability trading opportunities "
                                    "are detected. Never miss a signal with email and dashboard alerts.",
                                    className="card-text text-muted"
                                )
                            ])
                        ], className="h-100 shadow-sm")
                    ], md=4, className="mb-4"),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-shield-alt fa-3x text-warning mb-3")
                                ], className="text-center"),
                                html.H4("Risk Management", className="card-title text-center mb-3"),
                                html.P(
                                    "Every signal includes comprehensive risk metrics: VaR, CVaR, "
                                    "stop-loss, take-profit, and position sizing recommendations.",
                                    className="card-text text-muted"
                                )
                            ])
                        ], className="h-100 shadow-sm")
                    ], md=4, className="mb-4"),
                ], className="mb-5"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-sliders-h fa-3x text-info mb-3")
                                ], className="text-center"),
                                html.H4("Customizable Filters", className="card-title text-center mb-3"),
                                html.P(
                                    "Filter signals by confidence level, risk-reward ratio, sector, "
                                    "and more. Focus on the opportunities that match your strategy.",
                                    className="card-text text-muted"
                                )
                            ])
                        ], className="h-100 shadow-sm")
                    ], md=4, className="mb-4"),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-history fa-3x text-danger mb-3")
                                ], className="text-center"),
                                html.H4("Performance Tracking", className="card-title text-center mb-3"),
                                html.P(
                                    "Track your signal performance over time. Analyze win rates, "
                                    "returns, and refine your trading strategy with historical data.",
                                    className="card-text text-muted"
                                )
                            ])
                        ], className="h-100 shadow-sm")
                    ], md=4, className="mb-4"),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-code fa-3x text-secondary mb-3")
                                ], className="text-center"),
                                html.H4("API Access", className="card-title text-center mb-3"),
                                html.P(
                                    "Integrate signals into your own trading systems with our "
                                    "RESTful API. Build custom dashboards and automated trading bots.",
                                    className="card-text text-muted"
                                )
                            ])
                        ], className="h-100 shadow-sm")
                    ], md=4, className="mb-4"),
                ])
            ], fluid=True, className="py-5")
        ], className="bg-light"),
        
        # How It Works
        html.Section([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H2("How It Works", className="text-center mb-5")
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.Span("1", className="badge bg-primary rounded-circle p-3 fs-4")
                            ], className="text-center mb-3"),
                            html.H5("Sign Up", className="text-center mb-3"),
                            html.P(
                                "Create your free account in seconds. No credit card required.",
                                className="text-muted text-center"
                            )
                        ])
                    ], md=3, className="mb-4"),
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.Span("2", className="badge bg-success rounded-circle p-3 fs-4")
                            ], className="text-center mb-3"),
                            html.H5("Choose Your Plan", className="text-center mb-3"),
                            html.P(
                                "Select a subscription tier that fits your needs. Start with a free trial.",
                                className="text-muted text-center"
                            )
                        ])
                    ], md=3, className="mb-4"),
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.Span("3", className="badge bg-warning rounded-circle p-3 fs-4")
                            ], className="text-center mb-3"),
                            html.H5("Receive Signals", className="text-center mb-3"),
                            html.P(
                                "Get real-time trading signals via email and dashboard. "
                                "Every signal includes detailed analysis and risk metrics.",
                                className="text-muted text-center"
                            )
                        ])
                    ], md=3, className="mb-4"),
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.Span("4", className="badge bg-danger rounded-circle p-3 fs-4")
                            ], className="text-center mb-3"),
                            html.H5("Trade with Confidence", className="text-center mb-3"),
                            html.P(
                                "Execute trades based on high-confidence signals backed by "
                                "comprehensive analysis and risk management.",
                                className="text-muted text-center"
                            )
                        ])
                    ], md=3, className="mb-4"),
                ])
            ], fluid=True, className="py-5")
        ]),
        
        # CTA Section
        html.Section([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H2("Ready to Get Started?", className="text-white mb-4 text-center"),
                            html.P(
                                "Join thousands of traders using AI-powered signals to make better trading decisions.",
                                className="text-white-50 mb-4 text-center lead"
                            ),
                            html.Div([
                                dbc.Button(
                                    "Start Your Free Trial",
                                    href="/register",
                                    color="light",
                                    size="lg",
                                    className="me-3"
                                ),
                                dbc.Button(
                                    "View Pricing",
                                    href="/pricing",
                                    color="outline-light",
                                    size="lg",
                                    outline=True
                                )
                            ], className="text-center")
                        ])
                    ], width=12)
                ], className="py-5")
            ], fluid=True)
        ], className="bg-gradient-primary", style={
            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        }),
        
        # Trust Elements
        html.Section([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Trusted by Traders Worldwide", className="text-center mb-5")
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("99.9%", className="text-primary mb-2"),
                            html.P("Uptime", className="text-muted mb-0")
                        ], className="text-center")
                    ], md=3, className="mb-4"),
                    dbc.Col([
                        html.Div([
                            html.H4("10,000+", className="text-success mb-2"),
                            html.P("Active Users", className="text-muted mb-0")
                        ], className="text-center")
                    ], md=3, className="mb-4"),
                    dbc.Col([
                        html.Div([
                            html.H4("1M+", className="text-info mb-2"),
                            html.P("Signals Generated", className="text-muted mb-0")
                        ], className="text-center")
                    ], md=3, className="mb-4"),
                    dbc.Col([
                        html.Div([
                            html.H4("4.8/5", className="text-warning mb-2"),
                            html.P("User Rating", className="text-muted mb-0")
                        ], className="text-center")
                    ], md=3, className="mb-4"),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-lock fa-2x text-primary mb-3"),
                            html.H5("Secure & Encrypted", className="mb-2"),
                            html.P("Your data is protected with bank-level encryption.", className="text-muted")
                        ], className="text-center")
                    ], md=4, className="mb-4"),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-certificate fa-2x text-success mb-3"),
                            html.H5("SOC 2 Compliant", className="mb-2"),
                            html.P("We meet the highest security and compliance standards.", className="text-muted")
                        ], className="text-center")
                    ], md=4, className="mb-4"),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-headset fa-2x text-info mb-3"),
                            html.H5("24/7 Support", className="mb-2"),
                            html.P("Get help when you need it with our dedicated support team.", className="text-muted")
                        ], className="text-center")
                    ], md=4, className="mb-4"),
                ], className="mt-4")
            ], fluid=True, className="py-5")
        ], className="bg-light")
    ])


def create_features_page() -> html.Div:
    """Create detailed features page."""
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Features", className="mb-4"),
                    html.P(
                        "Comprehensive trading signal platform with advanced analytics and risk management.",
                        className="lead text-muted mb-5"
                    )
                ], width=12)
            ]),
            
            # Technical Analysis Features
            dbc.Row([
                dbc.Col([
                    html.H2("Technical Analysis", className="mb-4"),
                    html.Ul([
                        html.Li("Real-time RSI, MACD, Bollinger Bands, and 20+ indicators"),
                        html.Li("Multi-timeframe analysis (1m, 5m, 15m, 1h, 1d)"),
                        html.Li("Custom indicator combinations and scoring"),
                        html.Li("Pattern recognition (support/resistance, trends)"),
                    ], className="fs-5 mb-4"),
                    html.H3("Sentiment Analysis", className="mb-4 mt-5"),
                    html.Ul([
                        html.Li("FinBERT-powered news sentiment analysis"),
                        html.Li("GPT-4 summarization for complex market events"),
                        html.Li("Time-weighted sentiment scoring"),
                        html.Li("Social media sentiment tracking (coming soon)"),
                    ], className="fs-5 mb-4"),
                ], md=6),
                dbc.Col([
                    html.H2("Machine Learning", className="mb-4"),
                    html.Ul([
                        html.Li("Temporal Fusion Transformer (TFT) for price prediction"),
                        html.Li("LSTM networks with attention mechanisms"),
                        html.Li("Ensemble stacking for improved accuracy"),
                        html.Li("Continuous model retraining and optimization"),
                    ], className="fs-5 mb-4"),
                    html.H3("Risk Management", className="mb-4 mt-5"),
                    html.Ul([
                        html.Li("Monte Carlo VaR and CVaR calculations"),
                        html.Li("Position sizing recommendations (Kelly Criterion-inspired)"),
                        html.Li("Risk-reward ratio analysis"),
                        html.Li("Stop-loss and take-profit suggestions"),
                    ], className="fs-5 mb-4"),
                ], md=6)
            ], className="mb-5"),
            
            # Platform Features
            dbc.Row([
                dbc.Col([
                    html.H2("Platform Features", className="mb-4")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Real-Time Dashboard", className="mb-3"),
                            html.P("Interactive dashboard with live signal feed, charts, and analytics."),
                            html.Ul([
                                html.Li("Live signal updates"),
                                html.Li("Customizable widgets"),
                                html.Li("Performance tracking"),
                                html.Li("Portfolio overview"),
                            ])
                        ])
                    ], className="h-100 mb-4")
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Email Alerts", className="mb-3"),
                            html.P("Get notified instantly when new signals are generated."),
                            html.Ul([
                                html.Li("HTML email templates"),
                                html.Li("Customizable notification preferences"),
                                html.Li("Signal summaries"),
                                html.Li("Weekly performance reports"),
                            ])
                        ])
                    ], className="h-100 mb-4")
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("API Access", className="mb-3"),
                            html.P("Integrate signals into your trading systems with our REST API."),
                            html.Ul([
                                html.Li("RESTful API with authentication"),
                                html.Li("Rate limiting per tier"),
                                html.Li("Webhook support (coming soon)"),
                                html.Li("Comprehensive documentation"),
                            ])
                        ])
                    ], className="h-100 mb-4")
                ], md=4),
            ], className="mb-5"),
            
            # CTA
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("Ready to Experience These Features?", className="mb-4 text-center"),
                        dbc.Button(
                            "Start Free Trial",
                            href="/register",
                            color="primary",
                            size="lg",
                            className="d-block mx-auto"
                        )
                    ], className="text-center py-5")
                ], width=12)
            ])
        ], fluid=True, className="py-5")
    ])


def create_about_page() -> html.Div:
    """Create about/trust page with company information."""
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("About Trading Signals Pro", className="mb-4"),
                    html.P(
                        "We're on a mission to democratize access to professional-grade trading signals "
                        "powered by cutting-edge AI and machine learning.",
                        className="lead mb-5"
                    )
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H2("Our Mission", className="mb-4"),
                    html.P(
                        "Trading Signals Pro was founded with a simple goal: make institutional-quality "
                        "trading analysis accessible to all traders, regardless of experience or capital. "
                        "We combine state-of-the-art machine learning models, comprehensive technical analysis, "
                        "and real-time sentiment analysis to generate high-confidence trading signals.",
                        className="mb-4"
                    ),
                    html.H2("Technology", className="mb-4 mt-5"),
                    html.P(
                        "Our platform leverages advanced technologies including Temporal Fusion Transformers, "
                        "LSTM networks, FinBERT sentiment analysis, and Monte Carlo risk simulation to provide "
                        "the most accurate and actionable trading signals in the market.",
                        className="mb-4"
                    ),
                ], md=8),
                dbc.Col([
                    html.H3("Trust & Security", className="mb-4"),
                    html.Ul([
                        html.Li(html.Strong("SOC 2 Type II Certified")),
                        html.Li(html.Strong("256-bit SSL Encryption")),
                        html.Li(html.Strong("GDPR Compliant")),
                        html.Li(html.Strong("Regular Security Audits")),
                    ], className="fs-5 mb-4"),
                    html.H3("Support", className="mb-4 mt-5"),
                    html.P("Our team is here to help. Contact us at:"),
                    html.P([
                        html.I(className="fas fa-envelope me-2"),
                        html.A("support@tradingsignals.pro", href="mailto:support@tradingsignals.pro")
                    ])
                ], md=4)
            ])
        ], fluid=True, className="py-5")
    ])
