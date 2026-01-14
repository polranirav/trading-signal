"""
Account Management Pages.

Comprehensive user account settings, profile management, API keys,
usage statistics, and security preferences.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime
from typing import List, Dict, Optional

from src.logging_config import get_logger

logger = get_logger(__name__)


def create_account_page() -> html.Div:
    """Create comprehensive account management page."""
    return html.Div([
        # Header
        html.Div([
            html.H2("Account Settings", style={
                "fontSize": "1.75rem",
                "fontWeight": "700",
                "color": "#fff",
                "marginBottom": "4px"
            }),
            html.P("Manage your profile, API keys, and preferences", style={
                "color": "#64748b",
                "fontSize": "0.9rem",
                "marginBottom": "0"
            }),
        ], style={"marginBottom": "24px"}),

        # Main Content Grid
        dbc.Row([
            # Left Column - Profile & Settings
            dbc.Col([
                # Profile Information
                html.Div([
                    html.Div([
                        html.I(className="fas fa-user", style={"marginRight": "10px", "color": "#3b82f6"}),
                        html.Span("Profile Information", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Email Address", className="filter-label"),
                                dbc.Input(
                                    id="account-email",
                                    type="email",
                                    className="filter-input mb-3",
                                    disabled=True
                                )
                            ], xs=12, md=6),
                            dbc.Col([
                                html.Label("Full Name", className="filter-label"),
                                dbc.Input(
                                    id="account-name",
                                    type="text",
                                    className="filter-input mb-3"
                                )
                            ], xs=12, md=6),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Company (Optional)", className="filter-label"),
                                dbc.Input(
                                    id="account-company",
                                    type="text",
                                    className="filter-input mb-3"
                                )
                            ], xs=12, md=6),
                            dbc.Col([
                                html.Label("Phone (Optional)", className="filter-label"),
                                dbc.Input(
                                    id="account-phone",
                                    type="tel",
                                    className="filter-input mb-3"
                                )
                            ], xs=12, md=6),
                        ]),
                        dbc.Button(
                            [html.I(className="fas fa-save me-2"), "Update Profile"],
                            id="account-update-profile-btn",
                            color="primary",
                            className="w-100"
                        ),
                        html.Div(id="account-profile-feedback", className="mt-3")
                    ], className="section-body")
                ], className="section-card mb-4"),

                # Security Settings
                html.Div([
                    html.Div([
                        html.I(className="fas fa-shield-alt", style={"marginRight": "10px", "color": "#10b981"}),
                        html.Span("Security", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        html.Div([
                            html.Label("Current Password", className="filter-label"),
                            dbc.Input(
                                id="account-current-password",
                                type="password",
                                className="filter-input mb-3"
                            ),
                            html.Label("New Password", className="filter-label"),
                            dbc.Input(
                                id="account-new-password",
                                type="password",
                                className="filter-input mb-3"
                            ),
                            html.Label("Confirm New Password", className="filter-label"),
                            dbc.Input(
                                id="account-confirm-password",
                                type="password",
                                className="filter-input mb-3"
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-key me-2"), "Change Password"],
                                id="account-change-password-btn",
                                color="success",
                                className="w-100"
                            ),
                            html.Div(id="account-password-feedback", className="mt-3")
                        ]),
                        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "20px 0"}),
                        html.Div([
                            html.Label("Two-Factor Authentication", className="filter-label mb-3"),
                            dbc.Switch(
                                id="account-2fa-toggle",
                                label="Enable 2FA",
                                value=False,
                                className="mb-3"
                            ),
                            html.Small("Add an extra layer of security to your account", className="text-muted")
                        ])
                    ], className="section-body")
                ], className="section-card mb-4"),

                # Notification Preferences
                html.Div([
                    html.Div([
                        html.I(className="fas fa-bell", style={"marginRight": "10px", "color": "#f59e0b"}),
                        html.Span("Notifications", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        dbc.Checklist(
                            options=[
                                {"label": "📧 Email alerts for new signals", "value": "email_signals"},
                                {"label": "📊 Weekly performance summary", "value": "weekly_summary"},
                                {"label": "🔔 System announcements", "value": "announcements"},
                                {"label": "💰 Price alerts for watchlist", "value": "price_alerts"},
                                {"label": "📈 Backtest completion notifications", "value": "backtest_notifications"},
                            ],
                            id="account-notification-prefs",
                            className="mb-3"
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-save me-2"), "Save Preferences"],
                            id="account-save-prefs-btn",
                            color="primary",
                            className="w-100"
                        ),
                        html.Div(id="account-prefs-feedback", className="mt-3")
                    ], className="section-body")
                ], className="section-card")
            ], xs=12, lg=6, className="mb-4"),

            # Right Column - Subscription & API Keys
            dbc.Col([
                # Subscription Info
                html.Div([
                    html.Div([
                        html.I(className="fas fa-crown", style={"marginRight": "10px", "color": "#fbbf24"}),
                        html.Span("Subscription", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div(id="account-subscription-info", className="section-body")
                ], className="section-card mb-4"),

                # Usage Statistics
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-pie", style={"marginRight": "10px", "color": "#8b5cf6"}),
                        html.Span("Usage Statistics", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div(id="account-usage-stats", className="section-body")
                ], className="section-card mb-4"),

                # API Keys
                html.Div([
                    html.Div([
                        html.I(className="fas fa-key", style={"marginRight": "10px", "color": "#06b6d4"}),
                        html.Span("API Keys", style={"fontWeight": "600", "color": "#fff"}),
                        dbc.Button(
                            [html.I(className="fas fa-plus me-2"), "Create Key"],
                            id="account-create-api-key-btn",
                            color="success",
                            size="sm",
                            className="ms-auto"
                        )
                    ], className="section-header", style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                    html.Div(id="account-api-keys-list", className="section-body"),
                    
                    # Create API Key Modal
                    dbc.Modal([
                        dbc.ModalHeader([
                            html.I(className="fas fa-key me-2"),
                            "Create New API Key"
                        ]),
                        dbc.ModalBody([
                            html.Label("Key Name (Optional)", className="filter-label"),
                            dbc.Input(
                                id="api-key-name-input",
                                type="text",
                                placeholder="e.g., Trading Bot, Mobile App",
                                className="filter-input mb-3"
                            ),
                            html.Label("Expiration (Days)", className="filter-label"),
                            dbc.Input(
                                id="api-key-expiry-input",
                                type="number",
                                value=365,
                                min=1,
                                max=3650,
                                className="filter-input mb-3"
                            ),
                            html.Div([
                                html.Small([
                                    html.I(className="fas fa-info-circle me-2"),
                                    "Your API key will be shown only once. Make sure to save it securely."
                                ], className="text-muted")
                            ])
                        ]),
                        dbc.ModalFooter([
                            dbc.Button("Cancel", id="api-key-modal-close", outline=True, className="me-2"),
                            dbc.Button("Create Key", id="api-key-create-btn", color="primary")
                        ])
                    ], id="api-key-modal", is_open=False)
                ], className="section-card mb-4"),

                # Danger Zone
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle", style={"marginRight": "10px", "color": "#ef4444"}),
                        html.Span("Danger Zone", style={"fontWeight": "600", "color": "#ef4444"})
                    ], className="section-header"),
                    html.Div([
                        html.P(
                            "Permanently delete your account and all associated data. "
                            "This action cannot be undone.",
                            className="text-muted mb-3"
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-trash me-2"), "Delete Account"],
                            id="account-delete-btn",
                            color="danger",
                            outline=True,
                            className="w-100"
                        )
                    ], className="section-body")
                ], className="section-card")
            ], xs=12, lg=6, className="mb-4")
        ]),
        
        # Hidden Stores
        dcc.Store(id="account-data-store", data=None),
    ], className="account-page")


def create_api_docs_page() -> html.Div:
    """Create comprehensive API documentation page."""
    return html.Div([
        # Header
        html.Div([
            html.H2("API Documentation", style={
                "fontSize": "1.75rem",
                "fontWeight": "700",
                "color": "#fff",
                "marginBottom": "4px"
            }),
            html.P("Integrate Trading Signals into your applications with our RESTful API", style={
                "color": "#64748b",
                "fontSize": "0.9rem",
                "marginBottom": "0"
            }),
        ], style={"marginBottom": "24px"}),

        # Quick Start
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-rocket", style={"marginRight": "10px", "color": "#10b981"}),
                        html.Span("Quick Start", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        html.P("Get started in 3 simple steps:", className="mb-3"),
                        html.Ol([
                            html.Li("Create an API key in your Account settings"),
                            html.Li("Include the key in your request headers"),
                            html.Li("Start making API calls")
                        ], className="mb-4"),
                        html.Div([
                            html.Label("Base URL", className="filter-label"),
                            html.Pre([
                                html.Code("https://api.tradingsignals.pro/api/v1")
                            ], className="bg-dark text-light p-3 rounded mb-3"),
                            html.Label("Authentication Header", className="filter-label"),
                            html.Pre([
                                html.Code("X-API-Key: your-api-key-here")
                            ], className="bg-dark text-light p-3 rounded")
                        ])
                    ], className="section-body")
                ], className="section-card mb-4")
            ], xs=12)
        ]),

        # Endpoints
        dbc.Row([
            dbc.Col([
                html.H3("Endpoints", style={"color": "#fff", "marginBottom": "20px"}),
                
                # Get Signals
                html.Div([
                    html.Div([
                        html.Span("GET", className="badge bg-success me-2"),
                        html.Span("/api/v1/signals", style={"fontFamily": "monospace", "fontSize": "1.1rem", "fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        html.P("Retrieve the latest trading signals with optional filtering.", className="mb-3"),
                        html.Div([
                            html.H6("Query Parameters:", className="text-muted mb-2"),
                            html.Ul([
                                html.Li([html.Code("limit"), " - Number of signals (default: 20, max: 100)"]),
                                html.Li([html.Code("symbol"), " - Filter by stock symbol (e.g., AAPL)"]),
                                html.Li([html.Code("min_confidence"), " - Minimum confluence score (0.0-1.0)"]),
                                html.Li([html.Code("signal_type"), " - Filter by type (BUY, SELL, HOLD, etc.)"]),
                                html.Li([html.Code("start_date"), " - Start date (ISO 8601 format)"]),
                                html.Li([html.Code("end_date"), " - End date (ISO 8601 format)"]),
                            ], className="mb-3")
                        ]),
                        html.Div([
                            html.H6("Example Request:", className="text-muted mb-2"),
                            html.Pre([
                                html.Code("""curl -X GET \\
  -H "X-API-Key: your-api-key" \\
  "https://api.tradingsignals.pro/api/v1/signals?limit=10&symbol=AAPL&min_confidence=0.7"
""")
                            ], className="bg-dark text-light p-3 rounded mb-3")
                        ]),
                        html.Div([
                            html.H6("Example Response:", className="text-muted mb-2"),
                            html.Pre([
                                html.Code("""{
  "success": true,
  "data": {
    "signals": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "symbol": "AAPL",
        "signal_type": "STRONG_BUY",
        "confluence_score": 0.85,
        "technical_score": 0.82,
        "sentiment_score": 0.88,
        "ml_score": 0.85,
        "created_at": "2026-01-13T10:00:00Z",
        "price_at_signal": 185.50,
        "risk_reward_ratio": 2.5,
        "var_95": 0.03,
        "cvar_95": 0.05,
        "max_drawdown": 0.08,
        "sharpe_ratio": 1.2,
        "suggested_position_size": 0.15,
        "technical_rationale": "Strong bullish momentum with RSI at 65...",
        "sentiment_rationale": "Positive sentiment from recent news...",
        "risk_warning": "Moderate risk - monitor for reversal signals"
      }
    ],
    "total": 10,
    "page": 1,
    "limit": 10
  }
}""")
                            ], className="bg-dark text-light p-3 rounded")
                        ])
                    ], className="section-body")
                ], className="section-card mb-4"),

                # Get Signal by ID
                html.Div([
                    html.Div([
                        html.Span("GET", className="badge bg-success me-2"),
                        html.Span("/api/v1/signals/{signal_id}", style={"fontFamily": "monospace", "fontSize": "1.1rem", "fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        html.P("Get detailed information about a specific signal.", className="mb-3"),
                        html.Div([
                            html.H6("Example Request:", className="text-muted mb-2"),
                            html.Pre([
                                html.Code("""curl -X GET \\
  -H "X-API-Key: your-api-key" \\
  "https://api.tradingsignals.pro/api/v1/signals/550e8400-e29b-41d4-a716-446655440000"
""")
                            ], className="bg-dark text-light p-3 rounded")
                        ])
                    ], className="section-body")
                ], className="section-card mb-4"),

                # Get Historical Signals
                html.Div([
                    html.Div([
                        html.Span("GET", className="badge bg-success me-2"),
                        html.Span("/api/v1/signals/history", style={"fontFamily": "monospace", "fontSize": "1.1rem", "fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        html.P("Retrieve historical signals with performance data.", className="mb-3"),
                        html.Div([
                            html.H6("Query Parameters:", className="text-muted mb-2"),
                            html.Ul([
                                html.Li([html.Code("symbol"), " - Stock symbol (required)"]),
                                html.Li([html.Code("start_date"), " - Start date (ISO 8601, required)"]),
                                html.Li([html.Code("end_date"), " - End date (ISO 8601, required)"]),
                                html.Li([html.Code("include_performance"), " - Include execution data (true/false)"]),
                            ], className="mb-3")
                        ])
                    ], className="section-body")
                ], className="section-card mb-4"),

                # Get Market Data
                html.Div([
                    html.Div([
                        html.Span("GET", className="badge bg-info me-2"),
                        html.Span("/api/v1/market/{symbol}", style={"fontFamily": "monospace", "fontSize": "1.1rem", "fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        html.P("Get current market data and technical indicators for a symbol.", className="mb-3"),
                        html.Div([
                            html.H6("Example Request:", className="text-muted mb-2"),
                            html.Pre([
                                html.Code("""curl -X GET \\
  -H "X-API-Key: your-api-key" \\
  "https://api.tradingsignals.pro/api/v1/market/AAPL"
""")
                            ], className="bg-dark text-light p-3 rounded")
                        ])
                    ], className="section-body")
                ], className="section-card mb-4"),

                # Webhooks
                html.Div([
                    html.Div([
                        html.Span("POST", className="badge bg-warning me-2"),
                        html.Span("/api/v1/webhooks", style={"fontFamily": "monospace", "fontSize": "1.1rem", "fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        html.P("Register a webhook to receive real-time signal notifications.", className="mb-3"),
                        html.Div([
                            html.H6("Request Body:", className="text-muted mb-2"),
                            html.Pre([
                                html.Code("""{
  "url": "https://your-server.com/webhook",
  "events": ["signal.created", "signal.updated"],
  "filters": {
    "min_confidence": 0.7,
    "symbols": ["AAPL", "MSFT", "GOOGL"]
  }
}""")
                            ], className="bg-dark text-light p-3 rounded")
                        ])
                    ], className="section-body")
                ], className="section-card mb-4")
            ], xs=12)
        ]),

        # Rate Limits & Status Codes
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-tachometer-alt", style={"marginRight": "10px", "color": "#f59e0b"}),
                        html.Span("Rate Limits & Status Codes", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        html.P("API requests are rate-limited based on your subscription tier:", className="mb-3"),
                        html.Div([
                            html.Table([
                                html.Thead([
                                    html.Tr([
                                        html.Th("Tier", style={"color": "#94a3b8"}),
                                        html.Th("Requests/Day", style={"color": "#94a3b8"}),
                                        html.Th("Requests/Minute", style={"color": "#94a3b8"}),
                                        html.Th("Concurrent", style={"color": "#94a3b8"})
                                    ])
                                ]),
                                html.Tbody([
                                    html.Tr([
                                        html.Td("Free", style={"color": "#e2e8f0"}),
                                        html.Td("100", style={"color": "#e2e8f0"}),
                                        html.Td("10", style={"color": "#e2e8f0"}),
                                        html.Td("2", style={"color": "#e2e8f0"})
                                    ]),
                                    html.Tr([
                                        html.Td("Essential", style={"color": "#e2e8f0"}),
                                        html.Td("1,000", style={"color": "#e2e8f0"}),
                                        html.Td("60", style={"color": "#e2e8f0"}),
                                        html.Td("5", style={"color": "#e2e8f0"})
                                    ]),
                                    html.Tr([
                                        html.Td("Advanced", style={"color": "#e2e8f0"}),
                                        html.Td("10,000", style={"color": "#e2e8f0"}),
                                        html.Td("300", style={"color": "#e2e8f0"}),
                                        html.Td("10", style={"color": "#e2e8f0"})
                                    ]),
                                    html.Tr([
                                        html.Td("Premium", style={"color": "#e2e8f0"}),
                                        html.Td("Unlimited", style={"color": "#e2e8f0"}),
                                        html.Td("1,000", style={"color": "#e2e8f0"}),
                                        html.Td("50", style={"color": "#e2e8f0"})
                                    ]),
                                ])
                            ], className="table table-dark table-striped mb-4", style={"width": "100%"})
                        ]),
                        html.Div([
                            html.H6("Rate Limit Headers:", className="text-muted mb-2"),
                            html.Pre([
                                html.Code("""X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1640995200""")
                            ], className="bg-dark text-light p-3 rounded mb-3")
                        ]),
                        html.Div([
                            html.H6("HTTP Status Codes:", className="text-muted mb-2"),
                            html.Ul([
                                html.Li([html.Code("200"), " - Success"]),
                                html.Li([html.Code("400"), " - Bad Request (invalid parameters)"]),
                                html.Li([html.Code("401"), " - Unauthorized (invalid API key)"]),
                                html.Li([html.Code("403"), " - Forbidden (rate limit exceeded)"]),
                                html.Li([html.Code("404"), " - Not Found"]),
                                html.Li([html.Code("429"), " - Too Many Requests"]),
                                html.Li([html.Code("500"), " - Internal Server Error"]),
                            ])
                        ])
                    ], className="section-body")
                ], className="section-card mb-4")
            ], xs=12)
        ]),

        # SDKs & Libraries
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-code", style={"marginRight": "10px", "color": "#8b5cf6"}),
                        html.Span("SDKs & Libraries", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        html.P("Official SDKs and community libraries:", className="mb-3"),
                        html.Div([
                            html.Div([
                                html.H6("Python", className="mb-2"),
                                html.Pre([
                                    html.Code("""pip install tradingsignals-sdk

from tradingsignals import TradingSignalsClient

client = TradingSignalsClient(api_key="your-key")
signals = client.get_signals(limit=10, symbol="AAPL")""")
                                ], className="bg-dark text-light p-3 rounded mb-3")
                            ]),
                            html.Div([
                                html.H6("JavaScript/Node.js", className="mb-2"),
                                html.Pre([
                                    html.Code("""npm install @tradingsignals/sdk

const { TradingSignalsClient } = require('@tradingsignals/sdk');

const client = new TradingSignalsClient('your-key');
const signals = await client.getSignals({ limit: 10, symbol: 'AAPL' });""")
                                ], className="bg-dark text-light p-3 rounded mb-3")
                            ]),
                            html.Div([
                                html.H6("cURL Examples", className="mb-2"),
                                html.P("All endpoints support standard HTTP requests. See examples above.", className="text-muted")
                            ])
                        ])
                    ], className="section-body")
                ], className="section-card")
            ], xs=12)
        ]),

        # Support
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-life-ring", style={"marginRight": "10px", "color": "#06b6d4"}),
                        html.Span("Need Help?", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        html.P("Having trouble with the API? We're here to help:", className="mb-3"),
                        html.Ul([
                            html.Li([html.I(className="fas fa-envelope me-2"), "Email: ", html.A("support@tradingsignals.pro", href="mailto:support@tradingsignals.pro")]),
                            html.Li([html.I(className="fas fa-comments me-2"), "Discord: ", html.A("Join our community", href="#", target="_blank")]),
                            html.Li([html.I(className="fas fa-book me-2"), "Documentation: ", html.A("Full API Reference", href="#", target="_blank")]),
                        ], className="mb-0")
                    ], className="section-body")
                ], className="section-card")
            ], xs=12, className="mt-4")
        ])
    ], className="api-docs-page")
