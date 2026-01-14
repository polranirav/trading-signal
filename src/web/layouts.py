"""
Premium Layout Module.

Defines the structure of the application:
- Sidebar Navigation
- Top Ticker Bar
- Main Content Area
- Page Layouts (Overview, Analysis, Performance, Backtest)
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

# Live dashboard components
from src.web.components.dashboard_components import (
    create_live_status_bar
)

def create_sidebar():
    """Create the persistent sidebar navigation."""
    return html.Div([
        # Logo Area
        html.Div([
            html.Div("TS", className="logo-icon"),
            html.H3("TradeSignals", className="mb-0", style={"fontSize": "18px"})
        ], className="sidebar-header"),
        
        # Navigation
        html.Nav([
            dcc.Link([
                html.I(className="fas fa-chart-line me-2"),
                "Overview"
            ], href="/overview", className="nav-link", id="nav-overview"),
            
            dcc.Link([
                html.I(className="fas fa-list me-2"),
                "Stocks"
            ], href="/stocks", className="nav-link", id="nav-stocks"),
            
            dcc.Link([
                html.I(className="fas fa-search-dollar me-2"),
                "Analysis"
            ], href="/analysis", className="nav-link", id="nav-analysis"),
            
            dcc.Link([
                html.I(className="fas fa-history me-2"),
                "History"
            ], href="/history", className="nav-link", id="nav-history"),
            
            dcc.Link([
                html.I(className="fas fa-chart-bar me-2"),
                "Performance"
            ], href="/performance", className="nav-link", id="nav-performance"),
            
            dcc.Link([
                html.I(className="fas fa-flask me-2"),
                "Backtest"
            ], href="/backtest", className="nav-link", id="nav-backtest"),
            
            dcc.Link([
                html.I(className="fas fa-code me-2"),
                "API Docs"
            ], href="/api-docs", className="nav-link", id="nav-api-docs"),
            
            dcc.Link([
                html.I(className="fas fa-user me-2"),
                "Account"
            ], href="/account", className="nav-link", id="nav-account"),
        ], className="d-flex flex-column"),
        
        # Bottom Status Area
        html.Div([
            html.Small("System Status", className="text-muted d-block mb-2"),
            html.Div([
                html.Div(className="notification-dot bg-success pulse"),
                html.Span("Online", className="text-success ms-2")
            ], className="d-flex align-items-center mb-1"),
            html.Small("v1.0.0 Enterprise", className="text-muted")
        ], style={"marginTop": "auto"})
        
    ], className="sidebar", id="sidebar")

def create_ticker_tape():
    """Create the scrolling market ticker."""
    items = [
        {"symbol": "SPY", "price": "478.20", "change": "+0.45%"},
        {"symbol": "BTC", "price": "45,230", "change": "+2.1%"},
        {"symbol": "ETH", "price": "2,405", "change": "+1.8%"},
        {"symbol": "VIX", "price": "13.45", "change": "-4.2%"},
        {"symbol": "NDX", "price": "16,840", "change": "+0.8%"},
        {"symbol": "EUR/USD", "price": "1.095", "change": "-0.1%"},
        {"symbol": "GOLD", "price": "2,045", "change": "+0.3%"},
    ]
    
    ticker_items = []
    for item in items:
        color_class = "text-success" if "+" in item["change"] else "text-danger"
        ticker_items.extend([
            html.Span(item["symbol"], className="fw-bold ms-4"),
            html.Span(item["price"], className="ms-2"),
            html.Span(item["change"], className=f"{color_class} ms-2")
        ])
    
    return html.Div([
        html.Div([
            html.Div(ticker_items + ticker_items, className="ticker-move")
        ], className="ticker-wrap")
    ], className="top-bar-ticker")

def create_layout():
    """Create the main application layout."""
    return html.Div([
        dcc.Location(id="url"),
        dcc.Store(id="sidebar-state", data=False),  # Track sidebar state
        
        # Mobile Menu Button
        html.Button(html.I(className="fas fa-bars"), id="menu-toggle", className="menu-toggle"),
        
        # Sidebar & Overlay
        create_sidebar(),
        html.Div(id="sidebar-overlay", className="sidebar-overlay"),
        
        html.Div([
            create_ticker_tape(),
            # Live Status Bar
            create_live_status_bar(),
            html.Div(id="page-content", className="mt-4"),
            # Interval for live status updates (every 60 seconds)
            dcc.Interval(
                id='interval-live-status',
                interval=60*1000,  # 60 seconds
                n_intervals=0
            )
        ], className="main-content")
    ], className="app-container")


# ============================================================================
# PAGE LAYOUTS
# ============================================================================

def create_overview_page():
    """Create the premium overview page with modern design."""
    return html.Div([
        # Header Section
        html.Div([
            html.H2("Market Overview", style={
                "fontSize": "1.75rem",
                "fontWeight": "700",
                "color": "#fff",
                "marginBottom": "4px"
            }),
            html.P("Real-time signal monitoring and market sentiment", style={
                "color": "#64748b",
                "fontSize": "0.9rem",
                "marginBottom": "0"
            }),
        ], style={"marginBottom": "24px"}),

        # Metrics Row - 4 Cards
        dbc.Row([
            # Market Sentiment
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-line")
                    ], className="metric-icon blue"),
                    html.Div("SENTIMENT", className="metric-label"),
                    html.Div(id="summary-sentiment", className="metric-value"),
                    html.Div(id="summary-stocks", className="metric-sub")
                ], className="metric-card", id="sentiment-card")
            ], xs=6, md=3, className="mb-3"),
            
            # Buy Signals
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-arrow-trend-up")
                    ], className="metric-icon green"),
                    html.Div("BUY SIGNALS", className="metric-label"),
                    html.Div(id="summary-buys", className="metric-value", style={"color": "#10b981"}),
                    html.Div("Active opportunities", className="metric-sub")
                ], className="metric-card")
            ], xs=6, md=3, className="mb-3"),
            
            # Sell Signals
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-arrow-trend-down")
                    ], className="metric-icon red"),
                    html.Div("SELL SIGNALS", className="metric-label"),
                    html.Div(id="summary-sells", className="metric-value", style={"color": "#ef4444"}),
                    html.Div("Risk warnings", className="metric-sub")
                ], className="metric-card")
            ], xs=6, md=3, className="mb-3"),
            
            # System Status
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-server")
                    ], className="metric-icon purple"),
                    html.Div("SYSTEM", className="metric-label"),
                    html.Div([
                        html.Span("●", style={
                            "color": "#10b981",
                            "marginRight": "8px",
                            "fontSize": "0.8rem",
                            "animation": "pulse 2s infinite"
                        }),
                        html.Span("Online", style={"fontSize": "1.25rem", "fontWeight": "600"})
                    ]),
                    html.Div(id="last-update-time", className="metric-sub")
                ], className="metric-card")
            ], xs=6, md=3, className="mb-3"),
        ], className="mb-4"),

        # Main Content Grid
        dbc.Row([
            # Left Column - Predictions & Opportunities
            dbc.Col([
                # Price Predictions Section
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-chart-bar", style={"marginRight": "10px", "color": "#3b82f6"}),
                            "Price Predictions"
                        ], className="section-title"),
                        # Timeframe Buttons
                        html.Div([
                            html.Button("1D", id="pred-1d", className="timeframe-btn", n_clicks=0),
                            html.Button("7D", id="pred-7d", className="timeframe-btn", n_clicks=0),
                            html.Button("15D", id="pred-15d", className="timeframe-btn", n_clicks=0),
                            html.Button("30D", id="pred-30d", className="timeframe-btn active", n_clicks=0),
                        ], className="timeframe-group")
                    ], className="section-header"),
                    html.Div([
                        # Symbol Input
                        html.Div([
                            dcc.Input(
                                id="prediction-symbol",
                                value="AAPL",
                                placeholder="Symbol",
                                className="symbol-input"
                            ),
                            html.Button("Update", id="update-prediction-btn", className="update-btn")
                        ], className="symbol-input-group"),
                        # Predictions Content
                        html.Div(id="prediction-points-content"),
                        # Meta Info
                        html.Div(id="prediction-meta", style={
                            "marginTop": "16px",
                            "paddingTop": "12px",
                            "borderTop": "1px solid rgba(255,255,255,0.05)",
                            "fontSize": "0.8rem",
                            "color": "#64748b"
                        })
                    ], className="section-body")
                ], className="section-card mb-4"),
                
                # Top Opportunities Section
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-fire", style={"marginRight": "10px", "color": "#f59e0b"}),
                            "Top Opportunities"
                        ], className="section-title")
                    ], className="section-header"),
                    html.Div(id="top-opportunities-list", className="section-body")
                ], className="section-card")
            ], xs=12, lg=7, className="mb-4"),

            # Right Column - Data Sources & Signal Feed
            dbc.Col([
                # Data Sources Section
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-database", style={"marginRight": "10px", "color": "#8b5cf6"}),
                            "Data Sources"
                        ], className="section-title")
                    ], className="section-header"),
                    html.Div([
                        html.Div(id="data-sources-list"),
                        html.Div([
                            html.Button("Refresh", id="refresh-data-sources-btn", className="update-btn", style={"marginRight": "8px"}),
                            html.Button("Health Check", id="health-check-btn", style={
                                "background": "transparent",
                                "border": "1px solid rgba(255,255,255,0.2)",
                                "borderRadius": "8px",
                                "padding": "8px 16px",
                                "color": "#94a3b8",
                                "fontSize": "0.85rem",
                                "cursor": "pointer"
                            })
                        ], style={"marginTop": "16px"}),
                        html.Div(id="health-check-output", style={"marginTop": "12px"})
                    ], className="section-body")
                ], className="section-card mb-4"),
                
                # Signal Feed Section
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-broadcast-tower", style={"marginRight": "10px", "color": "#10b981"}),
                            "Signal Feed"
                        ], className="section-title")
                    ], className="section-header"),
                    html.Div(id="all-signals-table", className="section-body", style={"maxHeight": "400px", "overflowY": "auto"})
                ], className="section-card")
            ], xs=12, lg=5, className="mb-4"),
        ]),

        # Hidden Stores
        dcc.Store(id="prediction-timeframe", data="30"),
        
        # Refresh Interval
        dcc.Interval(id="refresh-interval", interval=60*1000, n_intervals=0)
    ])


def create_analysis_page(symbol=None):
    """Create the premium analysis page with searchable stock selector."""
    # Default symbol value
    try:
        default_symbol = symbol.upper() if symbol and isinstance(symbol, str) else "AAPL"
    except Exception:
        default_symbol = "AAPL"
    
    return html.Div([
        # Header with integrated search
        dbc.Row([
            dbc.Col([
                html.H2("Deep Dive Analysis", style={
                    "fontSize": "1.75rem",
                    "fontWeight": "700",
                    "color": "#fff",
                    "marginBottom": "4px"
                }),
                html.P("Technical, Fundamental, and AI-driven insights.", style={
                    "color": "#64748b",
                    "fontSize": "0.9rem",
                    "marginBottom": "0"
                }),
            ], xs=12, lg=6, className="mb-3 mb-lg-0"),
            
            # Search Section - Clean & Compact
            dbc.Col([
                html.Div([
                    # Current Symbol Display
                    html.Div([
                        html.Span("Analyzing:", style={
                            "color": "#64748b",
                            "fontSize": "0.8rem",
                            "marginRight": "8px"
                        }),
                        html.Span(id="current-symbol-display", children=default_symbol, style={
                            "color": "#3b82f6",
                            "fontSize": "1.25rem",
                            "fontWeight": "700"
                        })
                    ], style={"marginBottom": "8px"}),
                    
                    # Search Input
                    dbc.InputGroup([
                        dbc.Input(
                            id="analysis-symbol-input",
                            placeholder="Search any stock (e.g., AAPL, TSLA, MSFT...)",
                            value="",
                            type="text",
                            debounce=True,
                            className="stock-search-input"
                        ),
                        dbc.Button(
                            html.I(className="fas fa-search"),
                            id="analysis-search-btn",
                            color="primary",
                            className="stock-search-btn"
                        )
                    ], className="stock-search-group")
                ], className="d-flex flex-column align-items-end")
            ], xs=12, lg=6, className="d-flex justify-content-lg-end align-items-center")
        ], className="mb-4"),
        
        # Recent/Suggested Stocks - Horizontal scrollable
        html.Div([
            html.Div([
                html.I(className="fas fa-history", style={"marginRight": "8px", "color": "#64748b"}),
                html.Span("Quick Access", style={"color": "#64748b", "fontSize": "0.8rem", "fontWeight": "500"})
            ], style={"marginRight": "16px", "whiteSpace": "nowrap"}),
            html.Div(id="recent-stocks-list", className="recent-stocks-scroll")
        ], className="recent-stocks-bar mb-4"),

        # Main Analysis Grid
        dbc.Row([
            # Left: Price & Indicators Charts
            dbc.Col([
                # Price Chart
                html.Div([
                    dcc.Loading(
                        id="price-chart-loading",
                        type="default",
                        color="#3b82f6",
                        children=[
                            dcc.Graph(
                                id="price-chart",
                                style={'height': '500px'},
                                config={'responsive': True, 'displayModeBar': True},
                                figure=get_empty_chart_figure("Loading price data...")
                            )
                        ]
                    )
                ], className="analysis-chart-card mb-3"),
                
                # Indicators Chart
                html.Div([
                    dcc.Loading(
                        id="indicators-chart-loading",
                        type="default",
                        color="#3b82f6",
                        children=[
                            dcc.Graph(
                                id="indicators-chart",
                                style={'height': '300px'},
                                config={'responsive': True, 'displayModeBar': True},
                                figure=get_empty_chart_figure("Loading indicators...")
                            )
                        ]
                    )
                ], className="analysis-chart-card")
            ], xs=12, lg=8, className="mb-3 mb-lg-0"),

            # Right: AI Insights & Score
            dbc.Col([
                # Confluence Score Card
                dcc.Loading(
                    id="confluence-loading",
                    type="circle",
                    color="#3b82f6",
                    children=[
                        html.Div(id="confluence-result", className="analysis-score-card mb-3")
                    ]
                ),
                
                # Score Breakdown
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-pie", style={"marginRight": "8px", "color": "#8b5cf6"}),
                        html.Span("SCORE BREAKDOWN", style={"fontSize": "0.75rem", "letterSpacing": "0.5px", "color": "#94a3b8"})
                    ], style={"marginBottom": "16px"}),
                    dcc.Loading(
                        id="breakdown-loading",
                        type="default",
                        color="#3b82f6",
                        children=[
                            dcc.Graph(
                                id="score-breakdown",
                                style={'height': '220px'},
                                config={'responsive': True, 'displayModeBar': False},
                                figure=get_empty_pie_figure()
                            )
                        ]
                    )
                ], className="analysis-card mb-3"),
                
                # Risk Management Card
                html.Div([
                    html.Div([
                        html.I(className="fas fa-shield-alt", style={"marginRight": "8px", "color": "#ef4444"}),
                        html.Span("RISK MANAGEMENT", style={"fontSize": "0.75rem", "letterSpacing": "0.5px", "color": "#94a3b8"})
                    ], style={"marginBottom": "16px"}),
                    dcc.Loading(
                        id="risk-loading",
                        type="circle",
                        color="#3b82f6",
                        children=[
                            html.Div(id="risk-metrics")
                        ]
                    )
                ], className="analysis-card")
            ], xs=12, lg=4)
        ]),
        
        # Store for current symbol and recent symbols
        dcc.Store(id="current-analysis-symbol", data=default_symbol),
        dcc.Store(id="recent-symbols-store", data=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]),
        
        # Auto-load trigger
        dcc.Interval(id="analysis-load-trigger", interval=100, n_intervals=0, max_intervals=1)
    ], className="analysis-page")


def get_empty_chart_figure(message="Loading..."):
    """Create an empty chart figure with dark background and loading message."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#64748b")
    )
    fig.update_layout(
        plot_bgcolor='rgba(15, 23, 42, 0.9)',
        paper_bgcolor='rgba(15, 23, 42, 0.9)',
        font=dict(color='#94a3b8'),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


def get_empty_pie_figure():
    """Create an empty pie chart figure with dark background."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_annotation(
        text="Analyzing...",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#64748b")
    )
    fig.update_layout(
        plot_bgcolor='rgba(15, 23, 42, 0.9)',
        paper_bgcolor='rgba(15, 23, 42, 0.9)',
        font=dict(color='#94a3b8'),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    return fig


def create_performance_page():
    """Create the performance page."""
    return html.Div([
        html.H2("System Performance", className="mb-4"),
        dbc.Row([
            dbc.Col([html.Div("Coming Soon: Advanced Portfolio Metrics", className="glass-card p-5 text-center text-muted")], width=12)
        ])
    ])

def create_backtest_page():
    """Create the backtest page."""
    from src.web.backtest import create_backtest_page as create_backtest_page_impl
    return create_backtest_page_impl()
