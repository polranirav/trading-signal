"""
Strategy Backtesting Page Layout.

Provides Simple and Advanced modes for backtesting trading strategies
with walk-forward validation and comprehensive performance analytics.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from typing import Optional

from src.logging_config import get_logger

logger = get_logger(__name__)


def create_backtest_page() -> html.Div:
    """Create the comprehensive backtesting dashboard."""
    
    # Default date range (1 year ago to today)
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365)
    
    return html.Div([
        # Header
        html.Div([
            html.H2("Strategy Backtesting", style={
                "fontSize": "1.75rem",
                "fontWeight": "700",
                "color": "#fff",
                "marginBottom": "4px"
            }),
            html.P("Test trading strategies with walk-forward validation to avoid overfitting", style={
                "color": "#64748b",
                "fontSize": "0.9rem",
                "marginBottom": "0"
            }),
        ], style={"marginBottom": "24px"}),
        
        # Mode Toggle
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="fas fa-bolt me-2"), "Simple"],
                        id="backtest-mode-simple",
                        color="primary",
                        outline=False,
                        className="backtest-mode-btn active"
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-cog me-2"), "Advanced"],
                        id="backtest-mode-advanced",
                        color="secondary",
                        outline=True,
                        className="backtest-mode-btn"
                    )
                ], className="backtest-mode-toggle mb-4")
            ], width=12)
        ]),
        
        # Main Content
        dbc.Row([
            # Left Column - Configuration
            dbc.Col([
                # Simple Mode Panel
                html.Div(id="backtest-simple-panel", className="backtest-config-panel", children=[
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-sliders-h", style={"marginRight": "10px", "color": "#3b82f6"}),
                            html.Span("Quick Backtest", style={"fontWeight": "600", "color": "#fff"})
                        ], className="section-header"),
                        html.Div([
                            # Symbol Selector
                            html.Div([
                                html.Label("Symbol", className="filter-label"),
                                dbc.Input(
                                    id="backtest-symbol",
                                    placeholder="e.g., AAPL",
                                    value="AAPL",
                                    className="filter-input mb-3"
                                )
                            ]),
                            
                            # Date Range
                            html.Div([
                                html.Label("Date Range", className="filter-label"),
                                dbc.Select(
                                    id="backtest-date-range",
                                    options=[
                                        {"label": "Last 1 Year", "value": "1y"},
                                        {"label": "Last 2 Years", "value": "2y"},
                                        {"label": "Last 5 Years", "value": "5y"},
                                        {"label": "Custom", "value": "custom"},
                                    ],
                                    value="1y",
                                    className="filter-select mb-3"
                                )
                            ]),
                            
                            # Custom Date Range (hidden by default)
                            html.Div(id="backtest-custom-dates", style={"display": "none"}, children=[
                                html.Label("Start Date", className="filter-label"),
                                dbc.Input(
                                    id="backtest-start-date",
                                    type="date",
                                    value=default_start.strftime("%Y-%m-%d"),
                                    className="mb-3"
                                ),
                                html.Label("End Date", className="filter-label"),
                                dbc.Input(
                                    id="backtest-end-date",
                                    type="date",
                                    value=default_end.strftime("%Y-%m-%d"),
                                    className="mb-3"
                                )
                            ]),
                            
                            # Strategy Selector
                            html.Div([
                                html.Label("Strategy", className="filter-label"),
                                dbc.Select(
                                    id="backtest-strategy",
                                    options=[
                                        {"label": "📊 Test My Signals", "value": "my_signals"},
                                        {"label": "📈 RSI Mean Reversion", "value": "rsi"},
                                        {"label": "📉 Moving Average Crossover", "value": "ma_crossover"},
                                        {"label": "🚀 Momentum Strategy", "value": "momentum"},
                                    ],
                                    value="rsi",
                                    className="filter-select mb-3"
                                )
                            ]),
                            
                            # Run Button
                            dbc.Button(
                                [html.I(className="fas fa-play me-2"), "Run Backtest"],
                                id="backtest-run-btn",
                                color="primary",
                                className="w-100 mb-3",
                                size="lg"
                            ),
                            
                            # Loading Indicator
                            dcc.Loading(
                                id="backtest-loading",
                                type="circle",
                                color="#3b82f6",
                                children=html.Div(id="backtest-loading-msg")
                            )
                        ], className="section-body")
                    ], className="section-card")
                ]),
                
                # Advanced Mode Panel (hidden by default)
                html.Div(id="backtest-advanced-panel", style={"display": "none"}, className="backtest-config-panel", children=[
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-cogs", style={"marginRight": "10px", "color": "#8b5cf6"}),
                            html.Span("Strategy Builder", style={"fontWeight": "600", "color": "#fff"})
                        ], className="section-header"),
                        html.Div([
                            # Strategy Type
                            html.Div([
                                html.Label("Strategy Type", className="filter-label"),
                                dbc.Select(
                                    id="backtest-advanced-strategy",
                                    options=[
                                        {"label": "RSI Mean Reversion", "value": "rsi"},
                                        {"label": "MA Crossover", "value": "ma_crossover"},
                                        {"label": "Momentum", "value": "momentum"},
                                    ],
                                    value="rsi",
                                    className="filter-select mb-3"
                                )
                            ]),
                            
                            # Entry Rules
                            html.Div([
                                html.H6("Entry Rules", className="text-muted mb-3"),
                                html.Div([
                                    html.Label("RSI Oversold Level", className="filter-label"),
                                    dcc.Slider(
                                        id="backtest-oversold",
                                        min=20, max=40, step=1, value=30,
                                        marks={20: "20", 30: "30", 40: "40"},
                                        className="mb-3"
                                    ),
                                    html.Label("RSI Overbought Level", className="filter-label"),
                                    dcc.Slider(
                                        id="backtest-overbought",
                                        min=60, max=80, step=1, value=70,
                                        marks={60: "60", 70: "70", 80: "80"},
                                        className="mb-3"
                                    )
                                ], id="backtest-rsi-params"),
                                
                                html.Div([
                                    html.Label("Fast MA Period", className="filter-label"),
                                    dbc.Input(id="backtest-fast-ma", type="number", value=50, className="mb-3"),
                                    html.Label("Slow MA Period", className="filter-label"),
                                    dbc.Input(id="backtest-slow-ma", type="number", value=200, className="mb-3")
                                ], id="backtest-ma-params", style={"display": "none"}),
                                
                                html.Div([
                                    html.Label("Momentum Lookback", className="filter-label"),
                                    dbc.Input(id="backtest-momentum-lookback", type="number", value=20, className="mb-3"),
                                    html.Label("Momentum Threshold (%)", className="filter-label"),
                                    dbc.Input(id="backtest-momentum-threshold", type="number", value=5, step=0.5, className="mb-3")
                                ], id="backtest-momentum-params", style={"display": "none"})
                            ]),
                            
                            # Exit Rules
                            html.Div([
                                html.H6("Exit Rules", className="text-muted mb-3"),
                                html.Label("Stop Loss (%)", className="filter-label"),
                                dcc.Slider(
                                    id="backtest-stop-loss",
                                    min=1, max=10, step=0.5, value=5,
                                    marks={1: "1%", 5: "5%", 10: "10%"},
                                    className="mb-3"
                                ),
                                html.Label("Take Profit (%)", className="filter-label"),
                                dcc.Slider(
                                    id="backtest-take-profit",
                                    min=5, max=20, step=0.5, value=10,
                                    marks={5: "5%", 10: "10%", 20: "20%"},
                                    className="mb-3"
                                ),
                                html.Label("Max Hold Days", className="filter-label"),
                                dbc.Input(id="backtest-hold-days", type="number", value=20, className="mb-3")
                            ]),
                            
                            # Walk-Forward Settings
                            html.Div([
                                html.H6("Walk-Forward Settings", className="text-muted mb-3"),
                                html.Label("Training Period (years)", className="filter-label"),
                                dcc.Slider(
                                    id="backtest-train-years",
                                    min=1, max=5, step=1, value=3,
                                    marks={1: "1y", 3: "3y", 5: "5y"},
                                    className="mb-3"
                                ),
                                html.Label("Test Period (months)", className="filter-label"),
                                dcc.Slider(
                                    id="backtest-test-months",
                                    min=1, max=12, step=1, value=3,
                                    marks={1: "1m", 3: "3m", 6: "6m", 12: "12m"},
                                    className="mb-3"
                                )
                            ]),
                            
                            # Run Button
                            dbc.Button(
                                [html.I(className="fas fa-play me-2"), "Run Advanced Backtest"],
                                id="backtest-run-advanced-btn",
                                color="primary",
                                className="w-100 mb-3",
                                size="lg"
                            )
                        ], className="section-body")
                    ], className="section-card")
                ])
            ], xs=12, lg=4, className="mb-4"),
            
            # Right Column - Results
            dbc.Col([
                # Results Panel (initially hidden)
                html.Div(id="backtest-results-panel", style={"display": "none"}, children=[
                    # Key Metrics
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([html.I(className="fas fa-chart-line")], className="metric-icon blue"),
                                html.Div("TOTAL RETURN", className="metric-label"),
                                html.Div(id="backtest-total-return", className="metric-value"),
                                html.Div("Cumulative", className="metric-sub")
                            ], className="metric-card")
                        ], xs=6, md=3, className="mb-3"),
                        dbc.Col([
                            html.Div([
                                html.Div([html.I(className="fas fa-calendar-alt")], className="metric-icon green"),
                                html.Div("ANNUAL RETURN", className="metric-label"),
                                html.Div(id="backtest-annual-return", className="metric-value"),
                                html.Div("Annualized", className="metric-sub")
                            ], className="metric-card")
                        ], xs=6, md=3, className="mb-3"),
                        dbc.Col([
                            html.Div([
                                html.Div([html.I(className="fas fa-shield-alt")], className="metric-icon purple"),
                                html.Div("SHARPE RATIO", className="metric-label"),
                                html.Div(id="backtest-sharpe", className="metric-value"),
                                html.Div("Risk-adjusted", className="metric-sub")
                            ], className="metric-card")
                        ], xs=6, md=3, className="mb-3"),
                        dbc.Col([
                            html.Div([
                                html.Div([html.I(className="fas fa-arrow-down")], className="metric-icon red"),
                                html.Div("MAX DRAWDOWN", className="metric-label"),
                                html.Div(id="backtest-max-dd", className="metric-value"),
                                html.Div("Peak to trough", className="metric-sub")
                            ], className="metric-card")
                        ], xs=6, md=3, className="mb-3"),
                    ], className="mb-4"),
                    
                    # Additional Metrics
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([html.I(className="fas fa-bullseye")], className="metric-icon green"),
                                html.Div("WIN RATE", className="metric-label"),
                                html.Div(id="backtest-win-rate", className="metric-value"),
                                html.Div("Profitable trades", className="metric-sub")
                            ], className="metric-card")
                        ], xs=6, md=3, className="mb-3"),
                        dbc.Col([
                            html.Div([
                                html.Div([html.I(className="fas fa-balance-scale")], className="metric-icon purple"),
                                html.Div("PROFIT FACTOR", className="metric-label"),
                                html.Div(id="backtest-profit-factor", className="metric-value"),
                                html.Div("Gross profit/loss", className="metric-sub")
                            ], className="metric-card")
                        ], xs=6, md=3, className="mb-3"),
                        dbc.Col([
                            html.Div([
                                html.Div([html.I(className="fas fa-exchange-alt")], className="metric-icon blue"),
                                html.Div("TOTAL TRADES", className="metric-label"),
                                html.Div(id="backtest-total-trades", className="metric-value"),
                                html.Div("Executed", className="metric-sub")
                            ], className="metric-card")
                        ], xs=6, md=3, className="mb-3"),
                        dbc.Col([
                            html.Div([
                                html.Div([html.I(className="fas fa-exclamation-triangle")], className="metric-icon orange"),
                                html.Div("OVERFITTING", className="metric-label"),
                                html.Div(id="backtest-overfitting", className="metric-value"),
                                html.Div("Naive/Walk-forward", className="metric-sub")
                            ], className="metric-card")
                        ], xs=6, md=3, className="mb-3"),
                    ], className="mb-4"),
                    
                    # Charts
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Span("Equity Curve", style={"fontWeight": "600", "color": "#fff"})
                                ], className="section-header"),
                                html.Div([
                                    dcc.Graph(id="backtest-equity-chart", style={"height": "350px"}, config={"displayModeBar": False})
                                ], className="section-body")
                            ], className="section-card mb-4")
                        ], xs=12, className="mb-4"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Span("Drawdown Chart", style={"fontWeight": "600", "color": "#fff"})
                                ], className="section-header"),
                                html.Div([
                                    dcc.Graph(id="backtest-drawdown-chart", style={"height": "300px"}, config={"displayModeBar": False})
                                ], className="section-body")
                            ], className="section-card mb-4")
                        ], xs=12, md=6, className="mb-4"),
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Span("Returns Distribution", style={"fontWeight": "600", "color": "#fff"})
                                ], className="section-header"),
                                html.Div([
                                    dcc.Graph(id="backtest-returns-dist", style={"height": "300px"}, config={"displayModeBar": False})
                                ], className="section-body")
                            ], className="section-card mb-4")
                        ], xs=12, md=6, className="mb-4"),
                    ]),
                    
                    # Period Performance
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Span("Period Performance", style={"fontWeight": "600", "color": "#fff"})
                                ], className="section-header"),
                                html.Div([
                                    dcc.Graph(id="backtest-period-chart", style={"height": "300px"}, config={"displayModeBar": False})
                                ], className="section-body")
                            ], className="section-card mb-4")
                        ], xs=12, className="mb-4"),
                    ]),
                    
                    # Trade List
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Span("Trade Log", style={"fontWeight": "600", "color": "#fff"}),
                                    dbc.Button(
                                        [html.I(className="fas fa-download me-2"), "Export"],
                                        id="backtest-export-btn",
                                        color="secondary",
                                        size="sm",
                                        className="ms-auto"
                                    )
                                ], className="section-header", style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                                html.Div([
                                    html.Div(id="backtest-trade-list", className="trade-list-container")
                                ], className="section-body")
                            ], className="section-card")
                        ], xs=12)
                    ]),
                    
                    # Action Buttons
                    dbc.Row([
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button(
                                    [html.I(className="fas fa-save me-2"), "Save Results"],
                                    id="backtest-save-btn",
                                    color="success",
                                    outline=True
                                ),
                                dbc.Button(
                                    [html.I(className="fas fa-redo me-2"), "Run Again"],
                                    id="backtest-rerun-btn",
                                    color="primary",
                                    outline=True
                                )
                            ], className="w-100 mt-3")
                        ], xs=12)
                    ])
                ]),
                
                # Empty State
                html.Div(id="backtest-empty-state", children=[
                    html.Div([
                        html.I(className="fas fa-chart-bar", style={"fontSize": "4rem", "color": "#64748b", "marginBottom": "20px"}),
                        html.H4("Ready to Backtest", style={"color": "#94a3b8", "marginBottom": "10px"}),
                        html.P("Configure your strategy and click 'Run Backtest' to see results", style={"color": "#64748b"})
                    ], style={"textAlign": "center", "padding": "60px 20px"})
                ])
            ], xs=12, lg=8, className="mb-4")
        ]),
        
        # Hidden Stores
        dcc.Store(id="backtest-results-store", data=None),
        dcc.Store(id="backtest-mode-store", data="simple"),
        dcc.Store(id="backtest-config-store", data={}),
        
    ], className="backtest-page")
