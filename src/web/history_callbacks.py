"""
Callbacks for Signal History & Intelligence Dashboard.
"""

import dash
from dash import html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict

from src.logging_config import get_logger

logger = get_logger(__name__)


def register_history_callbacks(app: dash.Dash):
    """Register callbacks for history and performance pages."""
    
    @app.callback(
        [
            Output("history-total-signals", "children"),
            Output("history-accuracy", "children"),
            Output("history-avg-return", "children"),
            Output("history-hot-streak", "children"),
            Output("history-signals-list", "children"),
            Output("history-count-badge", "children"),
            Output("history-performance-chart", "figure"),
            Output("history-distribution-chart", "figure"),
            Output("history-top-performers", "children"),
            Output("history-market-insights", "children"),
            Output("history-filters-store", "data"),
        ],
        [
            Input("history-filter-button", "n_clicks"),
            Input("history-search-btn", "n_clicks"),
            Input("url", "pathname"),
        ],
        [
            State("history-symbol-filter", "value"),
            State("history-type-filter", "value"),
            State("history-time-filter", "value"),
            State("history-confidence-filter", "value"),
            State("history-filters-store", "data"),
        ]
    )
    def update_history_dashboard(n_clicks, search_clicks, pathname, symbol, signal_type, time_range, min_confidence, stored_filters):
        """Update the entire history dashboard."""
        # Only run on history page
        if pathname != "/history":
            return (dash.no_update,) * 11
        
        # Default empty outputs
        empty_fig = create_empty_figure("No data available")
        default_outputs = (
            "0", "0%", "0%", "0",
            html.Div("No signals found", style={"color": "#64748b", "textAlign": "center", "padding": "40px"}),
            dbc.Badge("0", color="secondary", pill=True),
            empty_fig, empty_fig,
            html.Div("No data", style={"color": "#64748b", "textAlign": "center", "padding": "20px"}),
            html.Div("No insights available", style={"color": "#64748b", "textAlign": "center", "padding": "20px"}),
            stored_filters or {}
        )
        
        try:
            from src.data.persistence import get_database
            from src.web.history import format_signal_item, calculate_performance_metrics
            
            db = get_database()
            
            # Use default values
            symbol = symbol.upper().strip() if symbol else None
            signal_type = signal_type if signal_type else "all"
            time_range = time_range if time_range else "30"
            min_confidence = min_confidence / 100.0 if min_confidence else 0.4
            
            # Calculate date range
            if time_range == "all":
                start_date = None
            else:
                days = int(time_range)
                start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get signals from database
            signals = db.get_latest_signals(limit=500, min_confidence=min_confidence)
            
            # Filter by date
            if start_date:
                signals = [s for s in signals if s.created_at and s.created_at >= start_date]
            
            # Filter by symbol
            if symbol:
                signals = [s for s in signals if s.symbol == symbol]
            
            # Filter by signal type
            if signal_type and signal_type != "all":
                if signal_type == "BUY":
                    signals = [s for s in signals if "BUY" in (s.signal_type or "")]
                elif signal_type == "SELL":
                    signals = [s for s in signals if "SELL" in (s.signal_type or "")]
                elif signal_type == "HOLD":
                    signals = [s for s in signals if s.signal_type == "HOLD"]
            
            if not signals:
                return default_outputs
            
            # Sort by date
            signals = sorted(signals, key=lambda s: s.created_at or datetime.min, reverse=True)
            
            # Calculate metrics
            metrics = calculate_performance_metrics(signals)
            
            # Format signal list (show top 20)
            signal_items = [format_signal_item(s) for s in signals[:20]]
            
            # Count badge
            count_badge = dbc.Badge(f"{len(signals)}", color="primary", pill=True)
            
            # Performance chart
            perf_fig = create_performance_chart(signals)
            
            # Distribution chart
            dist_fig = create_distribution_chart(metrics)
            
            # Top performers
            top_performers = create_top_performers(signals)
            
            # Market insights
            insights = create_market_insights(signals, metrics)
            
            # Format metrics for display
            total_str = str(metrics['total_signals'])
            accuracy_str = f"{metrics['win_rate']:.1f}%" if metrics['win_rate'] > 0 else "N/A"
            avg_return_str = f"{metrics['avg_return']:+.1f}%" if metrics['avg_return'] != 0 else "0%"
            hot_streak_str = str(metrics['hot_streak'])
            
            # Store filters
            new_filters = {
                "symbol": symbol,
                "signal_type": signal_type,
                "time_range": time_range,
                "min_confidence": min_confidence * 100
            }
            
            return (
                total_str, accuracy_str, avg_return_str, hot_streak_str,
                signal_items, count_badge,
                perf_fig, dist_fig,
                top_performers, insights,
                new_filters
            )
            
        except Exception as e:
            logger.error(f"Error updating history dashboard: {e}", exc_info=True)
            error_msg = html.Div([
                html.I(className="fas fa-exclamation-triangle", style={"color": "#f59e0b", "marginRight": "8px"}),
                f"Error loading data: {str(e)[:100]}"
            ], style={"color": "#f59e0b", "padding": "20px"})
            return (
                "Error", "-", "-", "-",
                error_msg, dbc.Badge("!", color="danger", pill=True),
                create_empty_figure("Error"), create_empty_figure("Error"),
                error_msg, error_msg,
                stored_filters or {}
            )
    
    # Performance page callbacks
    @app.callback(
        [
            Output("perf-win-rate", "children"),
            Output("perf-total-pnl", "children"),
            Output("perf-risk-reward", "children"),
            Output("perf-max-dd", "children"),
            Output("perf-cumulative-chart", "figure"),
            Output("perf-returns-dist", "figure"),
            Output("perf-symbol-breakdown", "children"),
        ],
        [Input("url", "pathname")]
    )
    def update_performance_page(pathname):
        """Update performance page metrics."""
        if pathname != "/performance":
            return (dash.no_update,) * 7
        
        empty_fig = create_empty_figure("No performance data")
        default_outputs = (
            "N/A", "$0", "N/A", "N/A",
            empty_fig, empty_fig,
            html.Div("No data available", style={"color": "#64748b", "textAlign": "center", "padding": "40px"})
        )
        
        try:
            from src.data.persistence import get_database
            
            db = get_database()
            signals = db.get_latest_signals(limit=1000)
            executed = [s for s in signals if s.is_executed and s.realized_pnl_pct is not None]
            
            if not executed:
                return default_outputs
            
            # Calculate metrics
            wins = [s for s in executed if s.realized_pnl_pct > 0]
            win_rate = len(wins) / len(executed) * 100 if executed else 0
            
            returns = [s.realized_pnl_pct * 100 for s in executed]
            total_pnl = sum(returns)
            
            # Cumulative returns for max drawdown
            cumulative = []
            running = 0
            peak = 0
            max_dd = 0
            for r in returns:
                running += r
                cumulative.append(running)
                peak = max(peak, running)
                dd = (peak - running) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            # Cumulative chart
            sorted_signals = sorted(executed, key=lambda s: s.created_at or datetime.min)
            dates = [s.created_at for s in sorted_signals]
            cum_returns = []
            cum = 0
            for s in sorted_signals:
                cum += s.realized_pnl_pct * 100
                cum_returns.append(cum)
            
            cum_fig = go.Figure()
            cum_fig.add_trace(go.Scatter(
                x=dates, y=cum_returns,
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)',
                line=dict(color='#3b82f6', width=2),
                name='Cumulative Return'
            ))
            cum_fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
            cum_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True, title='Return %'),
                showlegend=False
            )
            
            # Returns distribution
            dist_fig = go.Figure()
            dist_fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=20,
                marker_color='#8b5cf6',
                opacity=0.7
            ))
            dist_fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
            dist_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(title='Return %', gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(title='Count', gridcolor='rgba(255,255,255,0.05)'),
                showlegend=False
            )
            
            # Symbol breakdown
            symbol_stats = {}
            for s in executed:
                if s.symbol not in symbol_stats:
                    symbol_stats[s.symbol] = {"count": 0, "wins": 0, "total_return": 0}
                symbol_stats[s.symbol]["count"] += 1
                if s.realized_pnl_pct > 0:
                    symbol_stats[s.symbol]["wins"] += 1
                symbol_stats[s.symbol]["total_return"] += s.realized_pnl_pct * 100
            
            # Sort by total return
            sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]["total_return"], reverse=True)[:10]
            
            breakdown_items = []
            for sym, stats in sorted_symbols:
                wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
                ret_color = "#10b981" if stats["total_return"] > 0 else "#ef4444"
                breakdown_items.append(
                    html.Div([
                        html.Div([
                            html.Span(sym, style={"fontWeight": "700", "color": "#fff"}),
                            html.Span(f" ({stats['count']} trades)", style={"color": "#64748b", "fontSize": "0.85rem"})
                        ]),
                        html.Div([
                            html.Span(f"Win: {wr:.0f}%", style={"marginRight": "16px", "color": "#94a3b8"}),
                            html.Span(f"{stats['total_return']:+.1f}%", style={"fontWeight": "700", "color": ret_color})
                        ])
                    ], style={
                        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                        "padding": "12px 0", "borderBottom": "1px solid rgba(255,255,255,0.05)"
                    })
                )
            
            breakdown = html.Div(breakdown_items) if breakdown_items else html.Div("No data", style={"color": "#64748b"})
            
            return (
                f"{win_rate:.1f}%",
                f"${total_pnl:.0f}" if abs(total_pnl) < 1000 else f"${total_pnl/1000:.1f}K",
                "N/A",  # Risk/reward calculation would need more data
                f"{max_dd:.1f}%",
                cum_fig, dist_fig,
                breakdown
            )
            
        except Exception as e:
            logger.error(f"Performance page error: {e}", exc_info=True)
            return default_outputs


def create_empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#64748b")
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


def create_performance_chart(signals) -> go.Figure:
    """Create a performance overview chart."""
    if not signals:
        return create_empty_figure("No signals")
    
    # Group by date
    date_counts = {}
    for s in signals:
        if s.created_at:
            date_key = s.created_at.strftime("%Y-%m-%d")
            if date_key not in date_counts:
                date_counts[date_key] = {"buy": 0, "sell": 0, "hold": 0}
            if "BUY" in (s.signal_type or ""):
                date_counts[date_key]["buy"] += 1
            elif "SELL" in (s.signal_type or ""):
                date_counts[date_key]["sell"] += 1
            else:
                date_counts[date_key]["hold"] += 1
    
    if not date_counts:
        return create_empty_figure("No dated signals")
    
    dates = sorted(date_counts.keys())
    buys = [date_counts[d]["buy"] for d in dates]
    sells = [date_counts[d]["sell"] for d in dates]
    holds = [date_counts[d]["hold"] for d in dates]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=buys, name='Buy', marker_color='#10b981'))
    fig.add_trace(go.Bar(x=dates, y=sells, name='Sell', marker_color='#ef4444'))
    fig.add_trace(go.Bar(x=dates, y=holds, name='Hold', marker_color='#f59e0b'))
    
    fig.update_layout(
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            bgcolor='rgba(0,0,0,0)', font=dict(size=10)
        ),
        hovermode='x unified'
    )
    return fig


def create_distribution_chart(metrics: Dict) -> go.Figure:
    """Create signal distribution pie chart."""
    values = [metrics.get("buy_count", 0), metrics.get("sell_count", 0), metrics.get("hold_count", 0)]
    labels = ["Buy", "Sell", "Hold"]
    colors = ["#10b981", "#ef4444", "#f59e0b"]
    
    if sum(values) == 0:
        return create_empty_figure("No signals")
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker=dict(colors=colors),
        textinfo='percent',
        textposition='outside',
        textfont=dict(size=11, color='#fff')
    )])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        annotations=[dict(text=f"{sum(values)}", x=0.5, y=0.5, font_size=20, font_color='#fff', showarrow=False)]
    )
    return fig


def create_top_performers(signals) -> html.Div:
    """Create top performing signals list."""
    # Get signals with high confidence
    high_conf = sorted(signals, key=lambda s: s.confluence_score or 0, reverse=True)[:5]
    
    if not high_conf:
        return html.Div("No top performers", style={"color": "#64748b"})
    
    items = []
    for s in high_conf:
        conf = s.confluence_score * 100 if s.confluence_score else 0
        type_color = "#10b981" if "BUY" in (s.signal_type or "") else "#ef4444" if "SELL" in (s.signal_type or "") else "#f59e0b"
        
        items.append(html.Div([
            html.Div([
                html.Span(s.symbol, style={"fontWeight": "600", "color": "#fff"}),
                html.Span(f" {s.signal_type}", style={"color": type_color, "fontSize": "0.8rem", "marginLeft": "6px"})
            ]),
            html.Span(f"{conf:.0f}%", style={"color": "#10b981", "fontWeight": "600"})
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "padding": "8px 0", "borderBottom": "1px solid rgba(255,255,255,0.05)"
        }))
    
    return html.Div(items)


def create_market_insights(signals, metrics: Dict) -> html.Div:
    """Generate market insights from signals."""
    insights = []
    
    # Sentiment insight
    buy_pct = metrics["buy_count"] / metrics["total_signals"] * 100 if metrics["total_signals"] > 0 else 0
    sell_pct = metrics["sell_count"] / metrics["total_signals"] * 100 if metrics["total_signals"] > 0 else 0
    
    if buy_pct > 60:
        sentiment = "Bullish"
        sentiment_color = "#10b981"
        sentiment_icon = "fas fa-arrow-trend-up"
    elif sell_pct > 60:
        sentiment = "Bearish"
        sentiment_color = "#ef4444"
        sentiment_icon = "fas fa-arrow-trend-down"
    else:
        sentiment = "Mixed"
        sentiment_color = "#f59e0b"
        sentiment_icon = "fas fa-arrows-left-right"
    
    insights.append(html.Div([
        html.I(className=sentiment_icon, style={"color": sentiment_color, "marginRight": "10px"}),
        html.Span("Market Sentiment: ", style={"color": "#94a3b8"}),
        html.Span(sentiment, style={"color": sentiment_color, "fontWeight": "600"})
    ], style={"marginBottom": "12px"}))
    
    # Top symbol
    symbol_counts = {}
    for s in signals:
        symbol_counts[s.symbol] = symbol_counts.get(s.symbol, 0) + 1
    if symbol_counts:
        top_symbol = max(symbol_counts, key=symbol_counts.get)
        insights.append(html.Div([
            html.I(className="fas fa-star", style={"color": "#fbbf24", "marginRight": "10px"}),
            html.Span("Most Active: ", style={"color": "#94a3b8"}),
            html.Span(f"{top_symbol} ({symbol_counts[top_symbol]} signals)", style={"color": "#fff", "fontWeight": "600"})
        ], style={"marginBottom": "12px"}))
    
    # Accuracy insight
    if metrics["win_rate"] > 0:
        acc_text = "Excellent" if metrics["win_rate"] >= 70 else "Good" if metrics["win_rate"] >= 55 else "Moderate"
        acc_color = "#10b981" if metrics["win_rate"] >= 55 else "#f59e0b"
        insights.append(html.Div([
            html.I(className="fas fa-bullseye", style={"color": acc_color, "marginRight": "10px"}),
            html.Span("Signal Quality: ", style={"color": "#94a3b8"}),
            html.Span(f"{acc_text} ({metrics['win_rate']:.0f}% accuracy)", style={"color": acc_color, "fontWeight": "600"})
        ], style={"marginBottom": "12px"}))
    
    # Recent trend
    recent = signals[:10] if len(signals) >= 10 else signals
    recent_buys = sum(1 for s in recent if "BUY" in (s.signal_type or ""))
    recent_sells = sum(1 for s in recent if "SELL" in (s.signal_type or ""))
    
    if recent_buys > recent_sells:
        trend = "Buying momentum"
        trend_color = "#10b981"
    elif recent_sells > recent_buys:
        trend = "Selling pressure"
        trend_color = "#ef4444"
    else:
        trend = "Balanced"
        trend_color = "#f59e0b"
    
    insights.append(html.Div([
        html.I(className="fas fa-chart-line", style={"color": "#3b82f6", "marginRight": "10px"}),
        html.Span("Recent Trend: ", style={"color": "#94a3b8"}),
        html.Span(trend, style={"color": trend_color, "fontWeight": "600"})
    ]))
    
    return html.Div(insights)
