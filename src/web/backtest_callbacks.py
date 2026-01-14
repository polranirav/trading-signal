"""
Callbacks for Strategy Backtesting Dashboard.

Handles mode switching, backtest execution, results display, and visualizations.
"""

import dash
from dash import Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from src.logging_config import get_logger
from src.analytics.backtesting import WalkForwardBacktester
from src.data.ingestion import MarketDataClient
from src.web.backtest_strategies import create_strategy_from_config
from src.data.persistence import get_database

logger = get_logger(__name__)


def register_backtest_callbacks(app: dash.Dash):
    """Register all backtest-related callbacks."""
    
    # Mode Toggle
    @app.callback(
        [
            Output("backtest-mode-store", "data"),
            Output("backtest-simple-panel", "style"),
            Output("backtest-advanced-panel", "style"),
            Output("backtest-mode-simple", "outline"),
            Output("backtest-mode-advanced", "outline"),
            Output("backtest-mode-simple", "className"),
            Output("backtest-mode-advanced", "className"),
        ],
        [
            Input("backtest-mode-simple", "n_clicks"),
            Input("backtest-mode-advanced", "n_clicks"),
        ],
        [State("backtest-mode-store", "data")]
    )
    def toggle_backtest_mode(simple_clicks, advanced_clicks, current_mode):
        """Toggle between Simple and Advanced modes."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return "simple", {}, {"display": "none"}, False, True, "backtest-mode-btn active", "backtest-mode-btn"
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id == "backtest-mode-simple":
            return (
                "simple",
                {},
                {"display": "none"},
                False,
                True,
                "backtest-mode-btn active",
                "backtest-mode-btn"
            )
        else:
            return (
                "advanced",
                {"display": "none"},
                {},
                True,
                False,
                "backtest-mode-btn",
                "backtest-mode-btn active"
            )
    
    # Show/hide custom date picker
    @app.callback(
        Output("backtest-custom-dates", "style"),
        Input("backtest-date-range", "value")
    )
    def toggle_custom_dates(date_range):
        """Show custom date picker when 'custom' is selected."""
        if date_range == "custom":
            return {}
        return {"display": "none"}
    
    # Show/hide strategy parameters based on strategy type
    @app.callback(
        [
            Output("backtest-rsi-params", "style"),
            Output("backtest-ma-params", "style"),
            Output("backtest-momentum-params", "style"),
        ],
        Input("backtest-advanced-strategy", "value")
    )
    def show_strategy_params(strategy_type):
        """Show relevant parameters based on strategy type."""
        if strategy_type == "rsi":
            return {}, {"display": "none"}, {"display": "none"}
        elif strategy_type == "ma_crossover":
            return {"display": "none"}, {}, {"display": "none"}
        else:  # momentum
            return {"display": "none"}, {"display": "none"}, {}
    
    # Main Backtest Execution (Simple Mode)
    @app.callback(
        [
            Output("backtest-results-store", "data"),
            Output("backtest-results-panel", "style"),
            Output("backtest-empty-state", "style"),
            Output("backtest-loading-msg", "children"),
        ],
        Input("backtest-run-btn", "n_clicks"),
        [
            State("backtest-symbol", "value"),
            State("backtest-date-range", "value"),
            State("backtest-start-date", "date"),
            State("backtest-end-date", "date"),
            State("backtest-strategy", "value"),
        ]
    )
    def run_simple_backtest(n_clicks, symbol, date_range, start_date, end_date, strategy_type):
        """Execute simple mode backtest."""
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, ""
        
        try:
            # Calculate date range
            end = datetime.now()
            if date_range == "custom":
                if not start_date or not end_date:
                    return dash.no_update, dash.no_update, dash.no_update, html.Div("Please select custom dates", className="text-danger")
                try:
                    start = datetime.strptime(start_date, "%Y-%m-%d")
                    end = datetime.strptime(end_date, "%Y-%m-%d")
                except ValueError:
                    return dash.no_update, dash.no_update, dash.no_update, html.Div("Invalid date format", className="text-danger")
            elif date_range == "1y":
                start = end - timedelta(days=365)
            elif date_range == "2y":
                start = end - timedelta(days=730)
            elif date_range == "5y":
                start = end - timedelta(days=1825)
            else:
                start = end - timedelta(days=365)
            
            if not symbol:
                return dash.no_update, dash.no_update, dash.no_update, html.Div("Please enter a symbol", className="text-danger")
            
            symbol = symbol.upper().strip()
            
            # Show loading
            loading_msg = html.Div([
                html.I(className="fas fa-spinner fa-spin me-2"),
                f"Running backtest for {symbol}..."
            ], className="text-info")
            
            # Fetch data
            client = MarketDataClient()
            days_needed = (end - start).days
            data = client.fetch_daily_candles(symbol, days=days_needed + 50)  # Extra buffer
            
            if data is None or len(data) == 0:
                return dash.no_update, dash.no_update, dash.no_update, html.Div("Failed to fetch data. Please try again.", className="text-danger")
            
            # Filter to date range
            data = data[(data['time'] >= start) & (data['time'] <= end)]
            
            if len(data) < 100:
                return dash.no_update, dash.no_update, dash.no_update, html.Div("Insufficient data. Need at least 100 days.", className="text-danger")
            
            # Create strategy config
            config = {
                "strategy_type": strategy_type,
                "symbol": symbol,
                "start_date": start,
                "end_date": end,
            }
            
            # Create strategy
            strategy = create_strategy_from_config(config)
            
            # Run backtest
            backtester = WalkForwardBacktester(train_years=3, test_months=3)
            result = backtester.run_backtest(strategy, data, f"{strategy_type}_{symbol}")
            
            # Convert to dict for storage
            result_dict = result.to_dict()
            result_dict['trades'] = []
            result_dict['periods'] = []
            result_dict['equity_curve'] = []
            result_dict['drawdowns'] = []
            
            # Collect all trades
            for period in result.periods:
                period_dict = {
                    "period_number": period.period_number,
                    "total_return": period.total_return,
                    "num_trades": period.num_trades,
                    "win_rate": period.win_rate,
                    "sharpe_ratio": period.sharpe_ratio,
                    "max_drawdown": period.max_drawdown,
                }
                result_dict['periods'].append(period_dict)
                
                for trade in period.trades:
                    if trade.exit_price:
                        trade_dict = {
                            "symbol": trade.symbol,
                            "entry_date": trade.entry_date.isoformat() if trade.entry_date else None,
                            "exit_date": trade.exit_date.isoformat() if trade.exit_date else None,
                            "entry_price": trade.entry_price,
                            "exit_price": trade.exit_price,
                            "pnl": trade.pnl,
                            "pnl_percent": trade.pnl_percent,
                        }
                        result_dict['trades'].append(trade_dict)
            
            # Calculate equity curve
            if result_dict['trades']:
                trades_df = pd.DataFrame(result_dict['trades'])
                trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                trades_df = trades_df.sort_values('entry_date')
                cumulative = trades_df['pnl'].cumsum()
                result_dict['equity_curve'] = {
                    "dates": trades_df['entry_date'].dt.strftime('%Y-%m-%d').tolist(),
                    "values": (1 + cumulative).tolist()
                }
                
                # Calculate drawdowns
                peak = cumulative.expanding().max()
                drawdown = (cumulative - peak) / (1 + peak)
                result_dict['drawdowns'] = {
                    "dates": trades_df['entry_date'].dt.strftime('%Y-%m-%d').tolist(),
                    "values": drawdown.tolist()
                }
            
            # Store results
            result_dict['symbol'] = symbol
            result_dict['start_date'] = start.isoformat()
            result_dict['end_date'] = end.isoformat()
            result_dict['strategy_name'] = f"{strategy_type}_{symbol}"
            
            return (
                result_dict,
                {},
                {"display": "none"},
                html.Div("Backtest complete!", className="text-success")
            )
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}", exc_info=True)
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                html.Div(f"Error: {str(e)[:100]}", className="text-danger")
            )
    
    # Advanced Mode Backtest
    @app.callback(
        [
            Output("backtest-results-store", "data", allow_duplicate=True),
            Output("backtest-results-panel", "style", allow_duplicate=True),
            Output("backtest-empty-state", "style", allow_duplicate=True),
        ],
        Input("backtest-run-advanced-btn", "n_clicks"),
        [
            State("backtest-symbol", "value"),
            State("backtest-date-range", "value"),
            State("backtest-start-date", "date"),
            State("backtest-end-date", "date"),
            State("backtest-advanced-strategy", "value"),
            State("backtest-oversold", "value"),
            State("backtest-overbought", "value"),
            State("backtest-fast-ma", "value"),
            State("backtest-slow-ma", "value"),
            State("backtest-momentum-lookback", "value"),
            State("backtest-momentum-threshold", "value"),
            State("backtest-stop-loss", "value"),
            State("backtest-take-profit", "value"),
            State("backtest-hold-days", "value"),
            State("backtest-train-years", "value"),
            State("backtest-test-months", "value"),
        ],
        prevent_initial_call=True
    )
    def run_advanced_backtest(n_clicks, symbol, date_range, start_date, end_date,
                              strategy_type, oversold, overbought, fast_ma, slow_ma,
                              momentum_lookback, momentum_threshold, stop_loss, take_profit,
                              hold_days, train_years, test_months):
        """Execute advanced mode backtest with custom parameters."""
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update
        
        try:
            # Calculate date range (same as simple mode)
            end = datetime.now()
            if date_range == "custom":
                if not start_date or not end_date:
                    return dash.no_update, dash.no_update, dash.no_update
                try:
                    start = datetime.strptime(start_date, "%Y-%m-%d")
                    end = datetime.strptime(end_date, "%Y-%m-%d")
                except ValueError:
                    return dash.no_update, dash.no_update, dash.no_update
            elif date_range == "1y":
                start = end - timedelta(days=365)
            elif date_range == "2y":
                start = end - timedelta(days=730)
            elif date_range == "5y":
                start = end - timedelta(days=1825)
            else:
                start = end - timedelta(days=365)
            
            if not symbol:
                return dash.no_update, dash.no_update, dash.no_update
            
            symbol = symbol.upper().strip()
            
            # Fetch data
            client = MarketDataClient()
            days_needed = (end - start).days
            data = client.fetch_daily_candles(symbol, days=days_needed + 50)
            
            if data is None or len(data) == 0:
                return dash.no_update, dash.no_update, dash.no_update
            
            data = data[(data['time'] >= start) & (data['time'] <= end)]
            
            if len(data) < 100:
                return dash.no_update, dash.no_update, dash.no_update
            
            # Create strategy config with advanced parameters
            config = {
                "strategy_type": strategy_type,
                "symbol": symbol,
                "start_date": start,
                "end_date": end,
                "oversold": oversold,
                "overbought": overbought,
                "fast_period": fast_ma,
                "slow_period": slow_ma,
                "momentum_lookback": momentum_lookback,
                "momentum_threshold": momentum_threshold / 100.0,  # Convert % to decimal
                "stop_loss": stop_loss / 100.0,
                "take_profit": take_profit / 100.0,
                "hold_days": hold_days,
            }
            
            # Create strategy
            strategy = create_strategy_from_config(config)
            
            # Run backtest with custom walk-forward settings
            backtester = WalkForwardBacktester(train_years=train_years, test_months=test_months)
            result = backtester.run_backtest(strategy, data, f"{strategy_type}_{symbol}_advanced")
            
            # Convert to dict (same as simple mode)
            result_dict = result.to_dict()
            result_dict['trades'] = []
            result_dict['periods'] = []
            result_dict['equity_curve'] = []
            result_dict['drawdowns'] = []
            
            for period in result.periods:
                period_dict = {
                    "period_number": period.period_number,
                    "total_return": period.total_return,
                    "num_trades": period.num_trades,
                    "win_rate": period.win_rate,
                    "sharpe_ratio": period.sharpe_ratio,
                    "max_drawdown": period.max_drawdown,
                }
                result_dict['periods'].append(period_dict)
                
                for trade in period.trades:
                    if trade.exit_price:
                        trade_dict = {
                            "symbol": trade.symbol,
                            "entry_date": trade.entry_date.isoformat() if trade.entry_date else None,
                            "exit_date": trade.exit_date.isoformat() if trade.exit_date else None,
                            "entry_price": trade.entry_price,
                            "exit_price": trade.exit_price,
                            "pnl": trade.pnl,
                            "pnl_percent": trade.pnl_percent,
                        }
                        result_dict['trades'].append(trade_dict)
            
            # Calculate equity curve and drawdowns
            if result_dict['trades']:
                trades_df = pd.DataFrame(result_dict['trades'])
                trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                trades_df = trades_df.sort_values('entry_date')
                cumulative = trades_df['pnl'].cumsum()
                result_dict['equity_curve'] = {
                    "dates": trades_df['entry_date'].dt.strftime('%Y-%m-%d').tolist(),
                    "values": (1 + cumulative).tolist()
                }
                
                peak = cumulative.expanding().max()
                drawdown = (cumulative - peak) / (1 + peak)
                result_dict['drawdowns'] = {
                    "dates": trades_df['entry_date'].dt.strftime('%Y-%m-%d').tolist(),
                    "values": drawdown.tolist()
                }
            
            result_dict['symbol'] = symbol
            result_dict['start_date'] = start.isoformat()
            result_dict['end_date'] = end.isoformat()
            result_dict['strategy_name'] = f"{strategy_type}_{symbol}_advanced"
            
            return result_dict, {}, {"display": "none"}
            
        except Exception as e:
            logger.error(f"Error running advanced backtest: {e}", exc_info=True)
            return dash.no_update, dash.no_update, dash.no_update
    
    # Update Results Display
    @app.callback(
        [
            Output("backtest-total-return", "children"),
            Output("backtest-annual-return", "children"),
            Output("backtest-sharpe", "children"),
            Output("backtest-max-dd", "children"),
            Output("backtest-win-rate", "children"),
            Output("backtest-profit-factor", "children"),
            Output("backtest-total-trades", "children"),
            Output("backtest-overfitting", "children"),
        ],
        Input("backtest-results-store", "data")
    )
    def update_backtest_metrics(results):
        """Update performance metrics from results."""
        if not results:
            return (dash.no_update,) * 8
        
        try:
            total_return = results.get('oos_total_return', 0)
            annual_return = results.get('oos_annual_return', 0)
            sharpe = results.get('oos_sharpe_ratio', 0)
            max_dd = results.get('oos_max_drawdown', 0)
            win_rate = results.get('oos_win_rate', 0)
            profit_factor = results.get('oos_profit_factor', 0)
            overfitting = results.get('overfitting_ratio', 0)
            
            # Count total trades
            total_trades = len(results.get('trades', []))
            
            # Format with colors
            total_return_str = f"{total_return:+.2f}%"
            total_return_color = "text-success" if total_return > 0 else "text-danger"
            
            annual_return_str = f"{annual_return:+.2f}%"
            annual_return_color = "text-success" if annual_return > 0 else "text-danger"
            
            sharpe_str = f"{sharpe:.2f}"
            sharpe_color = "text-success" if sharpe > 1.0 else "text-warning" if sharpe > 0.5 else "text-danger"
            
            max_dd_str = f"{max_dd:.2f}%"
            max_dd_color = "text-danger"
            
            win_rate_str = f"{win_rate:.1f}%"
            win_rate_color = "text-success" if win_rate > 50 else "text-danger"
            
            profit_factor_str = f"{profit_factor:.2f}"
            profit_factor_color = "text-success" if profit_factor > 1.5 else "text-warning" if profit_factor > 1.0 else "text-danger"
            
            overfitting_str = f"{overfitting:.2f}x"
            overfitting_color = "text-danger" if overfitting > 1.5 else "text-warning" if overfitting > 1.2 else "text-success"
            
            return (
                html.Span(total_return_str, className=total_return_color),
                html.Span(annual_return_str, className=annual_return_color),
                html.Span(sharpe_str, className=sharpe_color),
                html.Span(max_dd_str, className=max_dd_color),
                html.Span(win_rate_str, className=win_rate_color),
                html.Span(profit_factor_str, className=profit_factor_color),
                str(total_trades),
                html.Span(overfitting_str, className=overfitting_color),
            )
        except Exception as e:
            logger.error(f"Error updating metrics: {e}", exc_info=True)
            return ("Error",) * 8
    
    # Update Charts
    @app.callback(
        [
            Output("backtest-equity-chart", "figure"),
            Output("backtest-drawdown-chart", "figure"),
            Output("backtest-returns-dist", "figure"),
            Output("backtest-period-chart", "figure"),
            Output("backtest-trade-list", "children"),
        ],
        Input("backtest-results-store", "data")
    )
    def update_backtest_charts(results):
        """Update all charts from results."""
        if not results:
            empty_fig = create_empty_chart("No data")
            return empty_fig, empty_fig, empty_fig, empty_fig, html.Div()
        
        try:
            # Equity Curve
            equity_fig = create_equity_chart(results)
            
            # Drawdown Chart
            drawdown_fig = create_drawdown_chart(results)
            
            # Returns Distribution
            returns_fig = create_returns_distribution(results)
            
            # Period Performance
            period_fig = create_period_chart(results)
            
            # Trade List
            trade_list = create_trade_list(results)
            
            return equity_fig, drawdown_fig, returns_fig, period_fig, trade_list
            
        except Exception as e:
            logger.error(f"Error updating charts: {e}", exc_info=True)
            empty_fig = create_empty_chart("Error loading charts")
            return empty_fig, empty_fig, empty_fig, empty_fig, html.Div("Error loading trades", className="text-danger")


def create_empty_chart(message: str) -> go.Figure:
    """Create an empty chart with a message."""
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
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(color='#94a3b8')
    )
    return fig


def create_equity_chart(results: Dict) -> go.Figure:
    """Create equity curve chart."""
    equity_data = results.get('equity_curve', {})
    if not equity_data or not equity_data.get('dates'):
        return create_empty_chart("No equity data")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_data['dates'],
        y=equity_data['values'],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)',
        line=dict(color='#3b82f6', width=2),
        name='Equity'
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True, title='Date'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True, title='Cumulative Return'),
        showlegend=False,
        hovermode='x unified'
    )
    return fig


def create_drawdown_chart(results: Dict) -> go.Figure:
    """Create drawdown chart."""
    drawdown_data = results.get('drawdowns', {})
    if not drawdown_data or not drawdown_data.get('dates'):
        return create_empty_chart("No drawdown data")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_data['dates'],
        y=[d * 100 for d in drawdown_data['values']],  # Convert to percentage
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(color='#ef4444', width=2),
        name='Drawdown'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True, title='Date'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True, title='Drawdown (%)'),
        showlegend=False,
        hovermode='x unified'
    )
    return fig


def create_returns_distribution(results: Dict) -> go.Figure:
    """Create returns distribution histogram."""
    trades = results.get('trades', [])
    if not trades:
        return create_empty_chart("No trade data")
    
    returns = [t.get('pnl_percent', 0) for t in trades if t.get('pnl_percent') is not None]
    if not returns:
        return create_empty_chart("No returns data")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=20,
        marker_color='#8b5cf6',
        opacity=0.7,
        name='Returns'
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444", line_width=1)
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(title='Return (%)', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Frequency', gridcolor='rgba(255,255,255,0.05)'),
        showlegend=False
    )
    return fig


def create_period_chart(results: Dict) -> go.Figure:
    """Create period-by-period performance chart."""
    periods = results.get('periods', [])
    if not periods:
        return create_empty_chart("No period data")
    
    period_nums = [p['period_number'] for p in periods]
    returns = [p['total_return'] * 100 for p in periods]  # Convert to percentage
    colors = ['#10b981' if r > 0 else '#ef4444' for r in returns]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=period_nums,
        y=returns,
        marker_color=colors,
        name='Period Return'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(title='Period', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Return (%)', gridcolor='rgba(255,255,255,0.05)'),
        showlegend=False
    )
    return fig


def create_trade_list(results: Dict) -> html.Div:
    """Create trade list display."""
    trades = results.get('trades', [])
    if not trades:
        return html.Div("No trades to display", className="text-muted")
    
    trade_items = []
    for i, trade in enumerate(trades[:100]):  # Limit to 100 trades
        pnl = trade.get('pnl_percent', 0)
        pnl_color = "#10b981" if pnl > 0 else "#ef4444"
        
        entry_date = trade.get('entry_date', '')[:10] if trade.get('entry_date') else 'N/A'
        exit_date = trade.get('exit_date', '')[:10] if trade.get('exit_date') else 'N/A'
        
        trade_items.append(
            html.Div([
                html.Div([
                    html.Span(f"#{i+1}", style={"fontWeight": "600", "color": "#64748b", "marginRight": "12px"}),
                    html.Span(f"{entry_date} → {exit_date}", style={"color": "#94a3b8", "fontSize": "0.85rem"})
                ]),
                html.Div([
                    html.Span(f"${trade.get('entry_price', 0):.2f}", style={"marginRight": "8px", "color": "#94a3b8"}),
                    html.Span("→", style={"marginRight": "8px", "color": "#64748b"}),
                    html.Span(f"${trade.get('exit_price', 0):.2f}", style={"marginRight": "12px", "color": "#94a3b8"}),
                    html.Span(f"{pnl:+.2f}%", style={"fontWeight": "600", "color": pnl_color})
                ])
            ], style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "padding": "10px 14px",
                "background": "rgba(0,0,0,0.2)",
                "borderRadius": "8px",
                "marginBottom": "8px",
                "borderLeft": f"3px solid {pnl_color}"
            })
        )
    
    return html.Div(trade_items, className="trade-list-container")
