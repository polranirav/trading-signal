"""
Signal History & Intelligence Dashboard.

A comprehensive view of trading signals with:
- Signal timeline with performance tracking
- Market sentiment trends
- Symbol performance leaderboard
- Signal accuracy analytics
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from src.logging_config import get_logger

logger = get_logger(__name__)


def create_history_page() -> html.Div:
    """Create the Signal Intelligence Dashboard."""
    return html.Div([
        # Header
        html.Div([
            html.H2("Signal Intelligence", style={
                "fontSize": "1.75rem",
                "fontWeight": "700",
                "color": "#fff",
                "marginBottom": "4px"
            }),
            html.P("Track signal performance, market trends, and discover trading patterns.", style={
                "color": "#64748b",
                "fontSize": "0.9rem",
                "marginBottom": "0"
            }),
        ], style={"marginBottom": "24px"}),

        # Key Metrics Row
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([html.I(className="fas fa-signal")], className="metric-icon blue"),
                    html.Div("TOTAL SIGNALS", className="metric-label"),
                    html.Div(id="history-total-signals", className="metric-value"),
                    html.Div("All time", className="metric-sub")
                ], className="metric-card")
            ], xs=6, md=3, className="mb-3"),
            
            dbc.Col([
                html.Div([
                    html.Div([html.I(className="fas fa-bullseye")], className="metric-icon green"),
                    html.Div("ACCURACY", className="metric-label"),
                    html.Div(id="history-accuracy", className="metric-value"),
                    html.Div("Win rate", className="metric-sub")
                ], className="metric-card")
            ], xs=6, md=3, className="mb-3"),
            
            dbc.Col([
                html.Div([
                    html.Div([html.I(className="fas fa-chart-line")], className="metric-icon purple"),
                    html.Div("AVG RETURN", className="metric-label"),
                    html.Div(id="history-avg-return", className="metric-value"),
                    html.Div("Per signal", className="metric-sub")
                ], className="metric-card")
            ], xs=6, md=3, className="mb-3"),
            
            dbc.Col([
                html.Div([
                    html.Div([html.I(className="fas fa-fire")], className="metric-icon red"),
                    html.Div("HOT STREAK", className="metric-label"),
                    html.Div(id="history-hot-streak", className="metric-value"),
                    html.Div("Consecutive wins", className="metric-sub")
                ], className="metric-card")
            ], xs=6, md=3, className="mb-3"),
        ], className="mb-4"),

        # Main Content Grid
        dbc.Row([
            # Left Column - Filters & Signal List
            dbc.Col([
                # Smart Filters Card
                html.Div([
                    html.Div([
                        html.I(className="fas fa-filter", style={"marginRight": "10px", "color": "#3b82f6"}),
                        html.Span("Smart Filters", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        # Search
                        dbc.InputGroup([
                            dbc.Input(
                                id="history-symbol-filter",
                                placeholder="Search symbol...",
                                type="text",
                                className="filter-input"
                            ),
                            dbc.Button(html.I(className="fas fa-search"), id="history-search-btn", className="filter-search-btn")
                        ], className="mb-3"),
                        
                        # Quick Filters
                        html.Div([
                            html.Label("Signal Type", className="filter-label"),
                            dbc.RadioItems(
                                id="history-type-filter",
                                options=[
                                    {"label": "All", "value": "all"},
                                    {"label": "🟢 Buy", "value": "BUY"},
                                    {"label": "🔴 Sell", "value": "SELL"},
                                    {"label": "🟡 Hold", "value": "HOLD"},
                                ],
                                value="all",
                                inline=True,
                                className="signal-type-radio mb-3"
                            )
                        ]),
                        
                        # Time Range
                        html.Div([
                            html.Label("Time Period", className="filter-label"),
                            dbc.Select(
                                id="history-time-filter",
                                options=[
                                    {"label": "Today", "value": "1"},
                                    {"label": "Last 7 days", "value": "7"},
                                    {"label": "Last 30 days", "value": "30"},
                                    {"label": "Last 90 days", "value": "90"},
                                    {"label": "All time", "value": "all"},
                                ],
                                value="30",
                                className="filter-select mb-3"
                            )
                        ]),
                        
                        # Confidence Slider
                        html.Div([
                            html.Label("Min Confidence", className="filter-label"),
                            dcc.Slider(
                                id="history-confidence-filter",
                                min=0,
                                max=100,
                                step=10,
                                value=40,
                                marks={0: '0%', 50: '50%', 100: '100%'},
                                className="confidence-slider"
                            )
                        ], className="mb-3"),
                        
                        dbc.Button([
                            html.I(className="fas fa-sync-alt me-2"),
                            "Apply Filters"
                        ], id="history-filter-button", color="primary", className="w-100")
                    ], className="section-body")
                ], className="section-card mb-4"),
                
                # Signal Timeline
                html.Div([
                    html.Div([
                        html.I(className="fas fa-history", style={"marginRight": "10px", "color": "#8b5cf6"}),
                        html.Span("Recent Signals", style={"fontWeight": "600", "color": "#fff"}),
                        html.Span(id="history-count-badge", className="ms-2")
                    ], className="section-header"),
                    html.Div(id="history-signals-list", className="section-body", style={"maxHeight": "500px", "overflowY": "auto"})
                ], className="section-card")
                
            ], xs=12, lg=5, className="mb-4"),
            
            # Right Column - Analytics
            dbc.Col([
                # Performance Chart
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-area", style={"marginRight": "10px", "color": "#10b981"}),
                        html.Span("Signal Performance", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        dcc.Loading(
                            dcc.Graph(id="history-performance-chart", style={"height": "280px"}, config={"displayModeBar": False}),
                            type="circle",
                            color="#3b82f6"
                        )
                    ], className="section-body")
                ], className="section-card mb-4"),
                
                # Two Column Stats
                dbc.Row([
                    dbc.Col([
                        # Signal Distribution
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-pie-chart", style={"marginRight": "10px", "color": "#f59e0b"}),
                                html.Span("Distribution", style={"fontWeight": "600", "color": "#fff", "fontSize": "0.9rem"})
                            ], className="section-header"),
                            html.Div([
                                dcc.Loading(
                                    dcc.Graph(id="history-distribution-chart", style={"height": "200px"}, config={"displayModeBar": False}),
                                    type="circle",
                                    color="#3b82f6"
                                )
                            ], className="section-body")
                        ], className="section-card")
                    ], xs=12, md=6, className="mb-4"),
                    
                    dbc.Col([
                        # Top Performers
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-trophy", style={"marginRight": "10px", "color": "#fbbf24"}),
                                html.Span("Top Signals", style={"fontWeight": "600", "color": "#fff", "fontSize": "0.9rem"})
                            ], className="section-header"),
                            html.Div(id="history-top-performers", className="section-body")
                        ], className="section-card")
                    ], xs=12, md=6, className="mb-4"),
                ]),
                
                # Market Insights
                html.Div([
                    html.Div([
                        html.I(className="fas fa-lightbulb", style={"marginRight": "10px", "color": "#06b6d4"}),
                        html.Span("Market Insights", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div(id="history-market-insights", className="section-body")
                ], className="section-card")
            ], xs=12, lg=7)
        ]),
        
        # Hidden stores
        dcc.Store(id="history-filters-store", data={}),
        dcc.Store(id="history-data-store", data=[]),
        
    ], className="history-page")


def create_performance_page() -> html.Div:
    """Create detailed performance tracking page."""
    return html.Div([
        html.Div([
            html.H2("Performance Analytics", style={
                "fontSize": "1.75rem",
                "fontWeight": "700",
                "color": "#fff",
                "marginBottom": "4px"
            }),
            html.P("Deep dive into signal accuracy, returns, and trading patterns.", style={
                "color": "#64748b",
                "fontSize": "0.9rem"
            }),
        ], style={"marginBottom": "24px"}),
        
        # Performance Metrics Row
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([html.I(className="fas fa-check-circle")], className="metric-icon green"),
                    html.Div("WIN RATE", className="metric-label"),
                    html.Div(id="perf-win-rate", className="metric-value", style={"color": "#10b981"}),
                    html.Div("Success ratio", className="metric-sub")
                ], className="metric-card")
            ], xs=6, md=3, className="mb-3"),
            
            dbc.Col([
                html.Div([
                    html.Div([html.I(className="fas fa-dollar-sign")], className="metric-icon blue"),
                    html.Div("TOTAL P&L", className="metric-label"),
                    html.Div(id="perf-total-pnl", className="metric-value"),
                    html.Div("Realized gains", className="metric-sub")
                ], className="metric-card")
            ], xs=6, md=3, className="mb-3"),
            
            dbc.Col([
                html.Div([
                    html.Div([html.I(className="fas fa-balance-scale")], className="metric-icon purple"),
                    html.Div("RISK/REWARD", className="metric-label"),
                    html.Div(id="perf-risk-reward", className="metric-value"),
                    html.Div("Average ratio", className="metric-sub")
                ], className="metric-card")
            ], xs=6, md=3, className="mb-3"),
            
            dbc.Col([
                html.Div([
                    html.Div([html.I(className="fas fa-shield-alt")], className="metric-icon red"),
                    html.Div("MAX DRAWDOWN", className="metric-label"),
                    html.Div(id="perf-max-dd", className="metric-value", style={"color": "#ef4444"}),
                    html.Div("Peak to trough", className="metric-sub")
                ], className="metric-card")
            ], xs=6, md=3, className="mb-3"),
        ], className="mb-4"),
        
        # Charts Row
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("Cumulative Returns", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        dcc.Graph(id="perf-cumulative-chart", style={"height": "350px"}, config={"displayModeBar": False})
                    ], className="section-body")
                ], className="section-card")
            ], xs=12, lg=8, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("Returns Distribution", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div([
                        dcc.Graph(id="perf-returns-dist", style={"height": "350px"}, config={"displayModeBar": False})
                    ], className="section-body")
                ], className="section-card")
            ], xs=12, lg=4, className="mb-4"),
        ]),
        
        # Symbol Breakdown
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("Performance by Symbol", style={"fontWeight": "600", "color": "#fff"})
                    ], className="section-header"),
                    html.Div(id="perf-symbol-breakdown", className="section-body")
                ], className="section-card")
            ], xs=12)
        ]),
        
        dcc.Store(id="performance-data-store"),
    ], className="performance-page")


def format_signal_item(signal) -> html.Div:
    """Format a single signal as a list item."""
    # Signal type styling
    type_colors = {
        "STRONG_BUY": ("#10b981", "↗"),
        "BUY": ("#34d399", "↑"),
        "HOLD": ("#f59e0b", "→"),
        "SELL": ("#f87171", "↓"),
        "STRONG_SELL": ("#ef4444", "↘"),
    }
    color, icon = type_colors.get(signal.signal_type, ("#64748b", "•"))
    
    # Confidence level
    conf = signal.confluence_score * 100 if signal.confluence_score else 0
    conf_class = "high" if conf >= 70 else "medium" if conf >= 50 else "low"
    
    # Time formatting
    if signal.created_at:
        time_str = signal.created_at.strftime("%b %d, %H:%M")
    else:
        time_str = "Unknown"
    
    return html.Div([
        # Header Row
        html.Div([
            html.Div([
                html.Span(signal.symbol, style={"fontWeight": "700", "color": "#fff", "fontSize": "1rem"}),
                html.Span(f" {icon} {signal.signal_type}", style={"color": color, "marginLeft": "8px", "fontSize": "0.85rem", "fontWeight": "600"})
            ]),
            html.Span(time_str, style={"color": "#64748b", "fontSize": "0.75rem"})
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"}),
        
        # Metrics Row
        html.Div([
            html.Div([
                html.Span("Confidence: ", style={"color": "#64748b", "fontSize": "0.8rem"}),
                html.Span(f"{conf:.0f}%", style={"color": "#fff", "fontWeight": "600", "fontSize": "0.8rem"})
            ]),
            html.Div([
                html.Span("Price: ", style={"color": "#64748b", "fontSize": "0.8rem"}),
                html.Span(f"${signal.price_at_signal:.2f}" if signal.price_at_signal else "N/A", style={"color": "#fff", "fontWeight": "500", "fontSize": "0.8rem"})
            ])
        ], style={"display": "flex", "gap": "20px"}),
        
        # Confidence Bar
        html.Div([
            html.Div(style={"width": f"{conf}%", "height": "3px", "background": color, "borderRadius": "2px"})
        ], style={"background": "rgba(255,255,255,0.05)", "borderRadius": "2px", "marginTop": "8px"})
        
    ], className="signal-list-item", style={
        "padding": "14px",
        "background": "rgba(0,0,0,0.2)",
        "borderRadius": "10px",
        "marginBottom": "10px",
        "borderLeft": f"3px solid {color}"
    })


def calculate_performance_metrics(signals) -> Dict:
    """Calculate comprehensive performance metrics."""
    if not signals:
        return {
            "total_signals": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_return": 0.0,
            "hot_streak": 0,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
        }
    
    total = len(signals)
    executed = [s for s in signals if s.is_executed and s.realized_pnl_pct is not None]
    
    buy_count = sum(1 for s in signals if "BUY" in (s.signal_type or ""))
    sell_count = sum(1 for s in signals if "SELL" in (s.signal_type or ""))
    hold_count = sum(1 for s in signals if s.signal_type == "HOLD")
    
    if not executed:
        return {
            "total_signals": total,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_return": 0.0,
            "hot_streak": 0,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
            "executed_count": 0
        }
    
    wins = [s for s in executed if s.realized_pnl_pct > 0]
    win_rate = len(wins) / len(executed) * 100 if executed else 0
    
    returns = [s.realized_pnl_pct * 100 for s in executed]
    avg_return = sum(returns) / len(returns) if returns else 0
    total_return = sum(returns)
    
    # Calculate hot streak (consecutive wins)
    hot_streak = 0
    current_streak = 0
    sorted_executed = sorted(executed, key=lambda s: s.created_at or datetime.min, reverse=True)
    for s in sorted_executed:
        if s.realized_pnl_pct > 0:
            current_streak += 1
            hot_streak = max(hot_streak, current_streak)
        else:
            current_streak = 0
    
    return {
        "total_signals": total,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "total_return": total_return,
        "hot_streak": hot_streak,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "executed_count": len(executed)
    }
