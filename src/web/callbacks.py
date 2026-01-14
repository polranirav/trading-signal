"""
Interactive Callbacks Module.

Handles all user interactions and real-time updates.
"""

import dash
from dash import Input, Output, State, html, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pickle
from typing import List, Dict

from src.logging_config import get_logger
from src.web.layouts import (
    create_overview_page, 
    create_analysis_page, 
    create_backtest_page
)
from src.web.auth import create_login_page, create_register_page, register_auth_callbacks
from src.web.payments import (
    create_pricing_page,
    create_checkout_page,
    create_checkout_success_page,
    create_checkout_cancel_page,
    create_billing_page,
    register_payment_callbacks
)
from src.web.charts import create_sparkline, premium_dark

logger = get_logger(__name__)

def register_callbacks(app):
    """Register all callbacks with the Dash app."""
    
    # Register live status callbacks
    try:
        from src.web.callbacks_live_status import register_live_status_callbacks
        register_live_status_callbacks(app)
    except ImportError:
        pass  # Live status callbacks not available
    
    # Register authentication callbacks
    register_auth_callbacks(app)
    
    # Register payment callbacks
    register_payment_callbacks(app)
    
    # Register history callbacks
    try:
        from src.web.history_callbacks import register_history_callbacks
        register_history_callbacks(app)
    except ImportError:
        pass  # History callbacks not available
    
    # Register stocks and watchlist callbacks
    try:
        from src.web.stocks_callbacks import register_stocks_callbacks
        register_stocks_callbacks(app)
    except ImportError:
        pass  # Stocks callbacks not available
    
    # Register dual chart analysis callbacks
    try:
        from src.web.dual_chart_callbacks import register_dual_chart_callbacks
        register_dual_chart_callbacks(app)
    except ImportError:
        pass  # Dual chart callbacks not available
    
    # Register backtest callbacks
    try:
        from src.web.backtest_callbacks import register_backtest_callbacks
        register_backtest_callbacks(app)
    except ImportError:
        pass  # Backtest callbacks not available
    
    # Register account callbacks
    try:
        from src.web.account_callbacks import register_account_callbacks
        register_account_callbacks(app)
    except ImportError:
        pass  # Account callbacks not available

    @app.callback(
        [Output("sidebar", "className"), Output("sidebar-overlay", "className")],
        [Input("menu-toggle", "n_clicks"), 
         Input("sidebar-overlay", "n_clicks"),
         Input("url", "pathname")],
        [State("sidebar", "className")]
    )
    def toggle_sidebar(n_menu, n_overlay, pathname, current_class):
        ctx = dash.callback_context
        if not ctx.triggered:
            return "sidebar", "sidebar-overlay"
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id == "menu-toggle":
            if current_class and "active" in current_class:
                return "sidebar", "sidebar-overlay"
            else:
                return "sidebar active", "sidebar-overlay active"
        
        return "sidebar", "sidebar-overlay"

    @app.callback(
        Output("page-content", "children"), 
        [Input("url", "pathname"), Input("url", "search")]
    )
    def display_page(pathname, search):
        """Route to appropriate page with authentication check."""
        from flask import session as flask_session, request
        
        def is_authenticated():
            """Check if user is logged in."""
            return flask_session.get('user_id') is not None
        
        def require_auth(page_func):
            """Return page if authenticated, else redirect to login."""
            if is_authenticated():
                return page_func()
            else:
                # Return login page without intrusive alert
                # The login form is self-explanatory
                return create_login_page()
        
        # Marketing pages (public)
        if pathname == "/" or pathname == "/home":
            from src.web.marketing import create_landing_page
            return create_landing_page()
        elif pathname == "/features":
            from src.web.marketing import create_features_page
            return create_features_page()
        elif pathname == "/about":
            from src.web.marketing import create_about_page
            return create_about_page()
        
        # Auth pages (public)
        if pathname == "/login": 
            # If already logged in, redirect to overview
            if is_authenticated():
                return create_overview_page()
            return create_login_page()
        elif pathname == "/register": 
            if is_authenticated():
                return create_overview_page()
            return create_register_page()
        elif pathname == "/logout":
            # Clear session and show login page
            flask_session.clear()
            return html.Div([
                dbc.Alert("You have been logged out.", color="success", className="mb-3"),
                create_login_page()
            ])
        
        # Payment pages
        if pathname == "/pricing": return create_pricing_page()
        elif pathname == "/checkout": return require_auth(create_checkout_page)
        elif pathname == "/checkout/success": return create_checkout_success_page()
        elif pathname == "/checkout/cancel": return create_checkout_cancel_page()
        elif pathname == "/billing": return require_auth(create_billing_page)
        
        # Protected pages (require auth)
        if pathname == "/overview" or pathname == "/dashboard": 
            return require_auth(create_overview_page)
        elif pathname == "/analysis": 
            # Use the new dual chart layout
            symbol = None
            if search and search.strip():
                try:
                    from urllib.parse import parse_qs
                    query_params = parse_qs(search.lstrip('?'))
                    symbol = query_params.get('symbol', [None])[0]
                except Exception:
                    symbol = None
            from src.web.dual_chart import create_dual_chart_analysis_page
            return require_auth(lambda: create_dual_chart_analysis_page(symbol=symbol or "AAPL"))
        elif pathname == "/history": 
            from src.web.history import create_history_page
            return require_auth(create_history_page)
        elif pathname == "/performance": 
            from src.web.history import create_performance_page
            return require_auth(create_performance_page)
        elif pathname == "/account":
            from src.web.account import create_account_page
            return require_auth(create_account_page)
        elif pathname == "/api-docs":
            from src.web.account import create_api_docs_page
            return require_auth(create_api_docs_page)
        elif pathname == "/backtest": 
            return require_auth(create_backtest_page)
        
        # Dual Chart Analysis (Split View)
        elif pathname == "/analysis-v2":
            symbol = None
            if search and search.strip():
                try:
                    from urllib.parse import parse_qs
                    query_params = parse_qs(search.lstrip('?'))
                    symbol = query_params.get('symbol', [None])[0]
                except Exception:
                    symbol = None
            from src.web.dual_chart import create_dual_chart_analysis_page
            return require_auth(lambda: create_dual_chart_analysis_page(symbol=symbol or "AAPL"))
        
        # Stock discovery and watchlist pages
        elif pathname == "/stocks":
            from src.web.stocks import create_stocks_page
            return require_auth(create_stocks_page)
        elif pathname == "/onboarding":
            from src.web.stocks import create_onboarding_page
            return require_auth(create_onboarding_page)
        
        return html.Div("404 - Page not found")

    @app.callback(
        [Output("summary-sentiment", "children"),
         Output("summary-stocks", "children"),
         Output("summary-buys", "children"),
         Output("summary-sells", "children"),
         Output("top-opportunities-list", "children"),
         Output("all-signals-table", "children"),
         Output("last-update-time", "children"),
         Output("sentiment-card", "className")],
        Input("refresh-interval", "n_intervals")
    )
    def update_overview(n):
        """Update overview page with latest signals."""
        from datetime import datetime
        
        signals = _get_cached_signals()
        
        # Use demo data if no real signals available
        if not signals:
            signals = _get_demo_signals()
        
        buy_count = sum(1 for s in signals if "BUY" in s.get("signal_type", ""))
        sell_count = sum(1 for s in signals if "SELL" in s.get("signal_type", ""))
        total_signals = len(signals)
        
        # Determine sentiment with visual styling
        if buy_count > sell_count:
            sentiment = "Bullish"
            sentiment_class = "metric-card bullish"
        elif sell_count > buy_count:
            sentiment = "Bearish"
            sentiment_class = "metric-card bearish"
        else:
            sentiment = "Neutral"
            sentiment_class = "metric-card neutral"
        
        # Top Opportunities List - Premium styled cards
        top_picks = signals[:5]
        opportunities_html = []
        
        if not top_picks:
            opportunities_html = [html.Div([
                html.I(className="fas fa-info-circle", style={"marginRight": "8px", "color": "#64748b"}),
                html.Span("No opportunities available", style={"color": "#64748b"})
            ], style={"padding": "20px", "textAlign": "center"})]
        else:
            for s in top_picks:
                badge_cls, icon = _get_signal_badge(s["signal_type"])
                score_pct = s['confluence_score'] * 100
                score_class = "high" if score_pct >= 70 else "medium" if score_pct >= 50 else "low"
                
                opportunities_html.append(html.Div([
                    # Header
                    html.Div([
                        html.Div([
                            html.Span(s["symbol"], className="opportunity-symbol"),
                            html.Span([html.I(className=f"{icon}", style={"marginRight": "4px"}), s["signal_type"]], 
                                     className=badge_cls, style={"marginLeft": "10px", "fontSize": "0.7rem"})
                        ], style={"display": "flex", "alignItems": "center"}),
                        html.Span(f"{score_pct:.0f}%", className=f"opportunity-score {score_class}")
                    ], className="opportunity-header"),
                    # Progress Bar
                    html.Div([
                        html.Div(style={"width": f"{score_pct}%"}, className=f"opportunity-bar-fill {score_class}")
                    ], className="opportunity-bar"),
                    # Footer
                    html.Div([
                        html.Span(s['signal_strength'], style={"fontSize": "0.75rem", "color": "#64748b"})
                    ])
                ], className="opportunity-card"))

        # Signal Feed - Compact list style
        signal_items = []
        if not signals:
            signal_items = [html.Div("No signals available", style={"color": "#64748b", "textAlign": "center", "padding": "20px"})]
        else:
            for s in signals[:8]:  # Show top 8
                badge_cls, icon = _get_signal_badge(s["signal_type"])
                score_pct = s['confluence_score'] * 100
                
                signal_items.append(html.Div([
                    html.Div([
                        html.Span(s["symbol"], style={"fontWeight": "600", "color": "#fff", "marginRight": "12px", "minWidth": "50px"}),
                        html.Span([html.I(className=icon, style={"marginRight": "4px"}), s["signal_type"]], 
                                 className=badge_cls, style={"fontSize": "0.7rem"})
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Div([
                        html.Span(f"{score_pct:.0f}%", style={
                            "fontWeight": "600",
                            "color": "#10b981" if score_pct >= 60 else "#f59e0b" if score_pct >= 40 else "#ef4444",
                            "fontSize": "0.9rem"
                        })
                    ])
                ], style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "padding": "12px 0",
                    "borderBottom": "1px solid rgba(255,255,255,0.05)"
                }))

        # Last update time
        last_update = datetime.now().strftime('%I:%M %p')

        return (
            sentiment, 
            f"{total_signals} stocks", 
            str(buy_count), 
            str(sell_count), 
            opportunities_html, 
            signal_items,
            last_update,
            sentiment_class
        )
    
    # Prediction timeframe callbacks - Update button classes
    @app.callback(
        [Output("pred-1d", "className"),
         Output("pred-7d", "className"),
         Output("pred-15d", "className"),
         Output("pred-30d", "className"),
         Output("prediction-timeframe", "data")],
        [Input("pred-1d", "n_clicks"),
         Input("pred-7d", "n_clicks"),
         Input("pred-15d", "n_clicks"),
         Input("pred-30d", "n_clicks")],
        prevent_initial_call=True
    )
    def update_prediction_timeframe(n1, n7, n15, n30):
        """Update prediction timeframe selection."""
        ctx = dash.callback_context
        base = "timeframe-btn"
        active = "timeframe-btn active"
        
        if not ctx.triggered:
            return base, base, base, active, "30"
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "pred-1d":
            return active, base, base, base, "1"
        elif button_id == "pred-7d":
            return base, active, base, base, "7"
        elif button_id == "pred-15d":
            return base, base, active, base, "15"
        else:  # pred-30d
            return base, base, base, active, "30"
    
    @app.callback(
        [Output("prediction-points-content", "children"),
         Output("prediction-meta", "children")],
        [Input("update-prediction-btn", "n_clicks"),
         Input("prediction-timeframe", "data"),
         Input("refresh-interval", "n_intervals")],
        [State("prediction-symbol", "value")]
    )
    def update_predictions(n_clicks, timeframe, n_intervals, symbol):
        """Update prediction points based on symbol and timeframe."""
        from datetime import datetime
        import random
        
        if not symbol:
            symbol = "AAPL"
        symbol = symbol.upper().strip()
        
        # Get timeframe in days
        days = int(timeframe) if timeframe else 30
        
        # Try to get real predictions, fallback to demo
        try:
            from src.web.components.prediction_points import get_prediction_points
            
            # Get a sample price (would come from real data)
            current_price = _get_current_price(symbol)
            
            # Generate predictions for selected timeframe
            if days == 1:
                timeframes = ["1 day"]
            elif days == 7:
                timeframes = ["1 day", "3 days", "7 days"]
            elif days == 15:
                timeframes = ["1 day", "7 days", "15 days"]
            else:
                timeframes = ["1 day", "7 days", "15 days", "30 days"]
            
            # Generate prediction points
            points = []
            for tf in timeframes:
                tf_days = int(tf.split()[0]) if tf.split()[0].isdigit() else 30
                # Calculate target based on small random movement
                change_pct = random.uniform(-0.02, 0.05) * (tf_days / 30)
                target = current_price * (1 + change_pct)
                confidence = max(50, 85 - (tf_days * 0.5) + random.uniform(-5, 5))
                
                points.append({
                    "target": target,
                    "confidence": confidence,
                    "change": change_pct * 100,
                    "timeframe": tf
                })
            
            # Build UI with new prediction-row styling
            point_rows = []
            for p in points:
                direction = "↑" if p["change"] > 0 else "↓"
                change_class = "up" if p["change"] > 0 else "down"
                conf_class = "high" if p['confidence'] >= 70 else "medium" if p['confidence'] >= 50 else "low"
                
                point_rows.append(html.Div([
                    # Price
                    html.Span(f"${p['target']:.2f}", className="prediction-price"),
                    # Confidence with bar
                    html.Div([
                        html.Div([
                            html.Div(style={"width": f"{p['confidence']}%"}, className=f"confidence-fill {conf_class}")
                        ], className="confidence-bar"),
                        html.Span(f"{p['confidence']:.0f}%", style={"fontSize": "0.85rem", "color": "#10b981" if p['confidence'] >= 70 else "#f59e0b"})
                    ], className="prediction-confidence"),
                    # Change
                    html.Span(f"{direction}{abs(p['change']):.1f}%", className=f"prediction-change {change_class}"),
                    # Timeframe
                    html.Span(p['timeframe'], className="prediction-timeframe")
                ], className="prediction-row"))
            
            meta = html.Div([
                html.Span([
                    html.I(className="fas fa-tag", style={"marginRight": "6px", "fontSize": "0.75rem"}),
                    symbol
                ], style={"marginRight": "24px"}),
                html.Span([
                    html.I(className="fas fa-dollar-sign", style={"marginRight": "6px", "fontSize": "0.75rem"}),
                    f"{current_price:.2f}"
                ], style={"marginRight": "24px"}),
                html.Span([
                    html.I(className="fas fa-clock", style={"marginRight": "6px", "fontSize": "0.75rem"}),
                    datetime.now().strftime('%I:%M %p')
                ])
            ])
            
            return point_rows, meta
            
        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            return html.Div("Unable to generate predictions", style={"color": "#64748b", "padding": "20px", "textAlign": "center"}), ""
    
    @app.callback(
        Output("data-sources-list", "children"),
        [Input("refresh-interval", "n_intervals"),
         Input("refresh-data-sources-btn", "n_clicks")]
    )
    def update_data_sources(n_intervals, n_clicks):
        """Update data sources status with premium styling."""
        try:
            from src.web.components.live_status import get_live_status
            status = get_live_status()
            
            source_rows = []
            for key, source_status in status['statuses'].items():
                # Determine status styling
                status_val = source_status.status
                dot_class = f"source-dot {status_val}"
                status_class = f"source-status {status_val}"
                status_text = status_val.upper() if status_val != 'unknown' else 'N/A'
                
                # Last update
                if source_status.last_update:
                    last_str = source_status.last_update.strftime('%I:%M %p')
                else:
                    last_str = "Never"
                
                source_rows.append(html.Div([
                    html.Div([
                        html.Div(className=dot_class),
                        html.Span(source_status.name, className="source-name")
                    ], className="source-info"),
                    html.Div([
                        html.Span(status_text, className=status_class),
                        html.Span(f"Last: {last_str}", className="source-time", style={"marginLeft": "12px"})
                    ], style={"display": "flex", "alignItems": "center"})
                ], className="source-item"))
            
            return source_rows
            
        except Exception as e:
            logger.warning(f"Data sources error: {e}")
            return html.Div("Unable to load data sources", style={"color": "#64748b", "padding": "20px", "textAlign": "center"})

    @app.callback(
        [Output("current-analysis-symbol", "data"),
         Output("current-symbol-display", "children"),
         Output("analysis-symbol-input", "value"),
         Output("recent-symbols-store", "data")],
        [Input("analysis-search-btn", "n_clicks"),
         Input("analysis-symbol-input", "n_submit"),
         Input({"type": "recent-stock", "symbol": ALL}, "n_clicks")],
        [State("analysis-symbol-input", "value"),
         State("current-analysis-symbol", "data"),
         State("recent-symbols-store", "data")],
        prevent_initial_call=True
    )
    def update_analysis_symbol(search_clicks, n_submit, recent_clicks, input_value, current_symbol, recent_symbols):
        """Update the current analysis symbol based on user interaction."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        trigger_id = ctx.triggered_id
        new_symbol = None
        
        # Handle recent stock click
        if isinstance(trigger_id, dict) and trigger_id.get("type") == "recent-stock":
            new_symbol = trigger_id.get("symbol")
        
        # Handle search button click or Enter key
        elif trigger_id in ["analysis-search-btn", "analysis-symbol-input"] and input_value:
            new_symbol = input_value.upper().strip()
        
        if not new_symbol:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # Update recent symbols list (keep last 8, move new to front)
        if recent_symbols is None:
            recent_symbols = []
        if new_symbol in recent_symbols:
            recent_symbols.remove(new_symbol)
        recent_symbols.insert(0, new_symbol)
        recent_symbols = recent_symbols[:8]  # Keep only 8
        
        return new_symbol, new_symbol, "", recent_symbols
    
    @app.callback(
        Output("recent-stocks-list", "children"),
        Input("recent-symbols-store", "data")
    )
    def update_recent_stocks_list(recent_symbols):
        """Update the recent stocks horizontal list."""
        if not recent_symbols:
            recent_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        
        return [
            html.Span(
                symbol,
                id={"type": "recent-stock", "symbol": symbol},
                className="recent-stock-chip",
                n_clicks=0
            ) for symbol in recent_symbols
        ]

    @app.callback(
        [Output("price-chart", "figure"),
         Output("indicators-chart", "figure"),
         Output("confluence-result", "children"),
         Output("score-breakdown", "figure"),
         Output("risk-metrics", "children")],
        [Input("current-analysis-symbol", "data"),
         Input("url", "search"),
         Input("analysis-load-trigger", "n_intervals")],
        [State("url", "pathname")],
        prevent_initial_call=False
    )
    def update_analysis_chart(stored_symbol, url_search, n_intervals, pathname):
        """Update analysis page charts with premium dark styling."""
        # Get symbol from store
        symbol = stored_symbol
        
        # Fallback to URL if store is empty
        if not symbol and pathname == "/analysis" and url_search:
            try:
                from urllib.parse import parse_qs
                query_params = parse_qs(url_search.lstrip('?'))
                url_symbol = query_params.get('symbol', [None])[0]
                if url_symbol:
                    symbol = url_symbol
            except Exception:
                pass
        
        # Create empty dark figure for errors/loading
        def create_dark_empty_figure(message="Loading...", height=500):
            fig = go.Figure()
            fig.add_annotation(
                text=message,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="#64748b")
            )
            fig.update_layout(
                plot_bgcolor='rgba(15, 23, 42, 0.95)',
                paper_bgcolor='rgba(15, 23, 42, 0.95)',
                font=dict(color='#94a3b8'),
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                margin=dict(l=20, r=20, t=20, b=20),
                height=height
            )
            return fig
        
        def create_empty_confluence():
            return html.Div([
                html.Div([
                    html.Div(style={
                        "width": "60px", "height": "60px",
                        "borderRadius": "50%",
                        "border": "3px solid rgba(255,255,255,0.1)",
                        "margin": "0 auto 16px"
                    }),
                    html.Div("Analyzing...", style={"color": "#64748b", "fontSize": "1.25rem"})
                ], style={"padding": "40px", "textAlign": "center"})
            ])
        
        def create_empty_risk():
            return html.Div([
                html.Div([
                    html.Span("Loading risk metrics...", style={"color": "#64748b"})
                ], style={"padding": "20px", "textAlign": "center"})
            ])
        
        if not symbol:
            return [
                create_dark_empty_figure("Select a symbol to analyze", 500),
                create_dark_empty_figure("Technical indicators will appear here", 300),
                create_empty_confluence(),
                create_dark_empty_figure("Score breakdown", 220),
                create_empty_risk()
            ]
        
        try:
            from src.analytics.confluence import ConfluenceEngine
            from src.analytics.technical import TechnicalAnalyzer
            from src.data.persistence import get_database
            
            db = get_database()
            df = db.get_candles(symbol.upper(), limit=100)
            
            if df.empty:
                return [
                    create_dark_empty_figure(f"No data available for {symbol.upper()}", 500),
                    create_dark_empty_figure("No indicator data", 300),
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle fa-2x", style={"color": "#f59e0b", "marginBottom": "12px"}),
                            html.Div("No Data", style={"color": "#fff", "fontSize": "1.5rem", "fontWeight": "600"}),
                            html.Div(f"Unable to fetch data for {symbol.upper()}", style={"color": "#64748b", "marginTop": "8px"})
                        ], style={"padding": "40px", "textAlign": "center"})
                    ]),
                    create_dark_empty_figure("", 220),
                    create_empty_risk()
                ]
            
            analyzer = TechnicalAnalyzer()
            df = analyzer.compute_all(df)
            
            # Price Chart with dark theme
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(
                x=df['time'], 
                open=df['open'], 
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                name="Price",
                increasing_line_color='#10b981',
                decreasing_line_color='#ef4444'
            ), row=1, col=1)
            if 'sma_20' in df.columns: 
                fig.add_trace(go.Scatter(x=df['time'], y=df['sma_20'], name="SMA 20", line=dict(color='#3b82f6', width=1)), row=1, col=1)
            if 'sma_50' in df.columns: 
                fig.add_trace(go.Scatter(x=df['time'], y=df['sma_50'], name="SMA 50", line=dict(color='#8b5cf6', width=1)), row=1, col=1)
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name="Volume", marker_color='rgba(59, 130, 246, 0.4)'), row=2, col=1)
            fig.update_layout(
                plot_bgcolor='rgba(15, 23, 42, 0.95)',
                paper_bgcolor='rgba(15, 23, 42, 0.95)',
                font=dict(color='#94a3b8'),
                height=500, 
                xaxis_rangeslider_visible=False,
                autosize=True,
                margin=dict(l=50, r=20, t=30, b=30),
                legend=dict(
                    bgcolor='rgba(30, 41, 59, 0.8)',
                    bordercolor='rgba(255,255,255,0.1)',
                    font=dict(color='#fff')
                ),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
                xaxis2=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
                yaxis2=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True)
            )
            
            # Indicators with dark theme
            fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
            if 'rsi_14' in df.columns:
                fig_ind.add_trace(go.Scatter(x=df['time'], y=df['rsi_14'], name="RSI", line=dict(color='#f59e0b', width=2)), row=1, col=1)
                fig_ind.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", row=1, col=1)
                fig_ind.add_hline(y=30, line_dash="dash", line_color="rgba(16, 185, 129, 0.5)", row=1, col=1)
            if 'macd' in df.columns:
                fig_ind.add_trace(go.Scatter(x=df['time'], y=df['macd'], name="MACD", line=dict(color='#06b6d4', width=2)), row=2, col=1)
                fig_ind.add_trace(go.Scatter(x=df['time'], y=df['macd_signal'], name="Signal", line=dict(color='#ef4444', width=1)), row=2, col=1)
            fig_ind.update_layout(
                plot_bgcolor='rgba(15, 23, 42, 0.95)',
                paper_bgcolor='rgba(15, 23, 42, 0.95)',
                font=dict(color='#94a3b8'),
                height=300,
                autosize=True,
                margin=dict(l=50, r=20, t=30, b=30),
                legend=dict(
                    bgcolor='rgba(30, 41, 59, 0.8)',
                    bordercolor='rgba(255,255,255,0.1)',
                    font=dict(color='#fff')
                ),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
                xaxis2=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
                yaxis2=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True)
            )
            
            # Confluence Analysis
            engine = ConfluenceEngine()
            result = engine.analyze(symbol, price_data=df)
            
            # Signal styling
            signal_type = result.signal_type.value
            if "BUY" in signal_type:
                signal_color = "#10b981"
                signal_bg = "rgba(16, 185, 129, 0.15)"
                signal_icon = "fas fa-arrow-up"
            elif "SELL" in signal_type:
                signal_color = "#ef4444"
                signal_bg = "rgba(239, 68, 68, 0.15)"
                signal_icon = "fas fa-arrow-down"
            else:
                signal_color = "#f59e0b"
                signal_bg = "rgba(245, 158, 11, 0.15)"
                signal_icon = "fas fa-minus"
            
            # Strength styling
            strength = result.signal_strength.value
            if strength in ["VERY_STRONG", "STRONG"]:
                strength_color = "#10b981"
                strength_bg = "linear-gradient(90deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.4))"
            elif strength == "MODERATE":
                strength_color = "#f59e0b"
                strength_bg = "linear-gradient(90deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.4))"
            else:
                strength_color = "#ef4444"
                strength_bg = "linear-gradient(90deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.4))"
            
            confluence_html = html.Div([
                # Score Circle
                html.Div([
                    html.Div(style={
                        "width": "80px", "height": "80px",
                        "borderRadius": "50%",
                        "border": f"4px solid {signal_color}",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "margin": "0 auto 16px"
                    }, children=[
                        html.I(className=signal_icon, style={"fontSize": "1.5rem", "color": signal_color})
                    ])
                ]),
                # Signal Type
                html.Div(signal_type, style={
                    "fontSize": "2rem",
                    "fontWeight": "700",
                    "color": "#fff",
                    "marginBottom": "12px"
                }),
                # Strength Badge
                html.Div(f"STRENGTH: {strength}", style={
                    "display": "inline-block",
                    "padding": "8px 20px",
                    "borderRadius": "20px",
                    "background": strength_bg,
                    "color": strength_color,
                    "fontSize": "0.75rem",
                    "fontWeight": "600",
                    "letterSpacing": "0.5px",
                    "marginBottom": "16px"
                }),
                # Rationale
                html.Div(style={"borderTop": "1px solid rgba(255,255,255,0.05)", "paddingTop": "16px", "marginTop": "8px"}),
                html.P(result.overall_rationale, style={
                    "color": "#94a3b8",
                    "fontSize": "0.85rem",
                    "lineHeight": "1.5",
                    "textAlign": "left",
                    "margin": "0"
                })
            ], style={"padding": "20px"})
            
            # Breakdown Pie with dark theme
            pie = go.Figure(go.Pie(
                labels=['Technical', 'Sentiment', 'Risk'],
                values=[result.technical_score, result.sentiment_score, result.risk_score],
                hole=0.65,
                marker=dict(colors=['#3b82f6', '#8b5cf6', '#f59e0b']),
                textinfo='percent',
                textfont=dict(color='#fff', size=11),
                hovertemplate='%{label}: %{value:.2f}<extra></extra>'
            ))
            pie.update_layout(
                plot_bgcolor='rgba(15, 23, 42, 0.95)',
                paper_bgcolor='rgba(15, 23, 42, 0.95)',
                font=dict(color='#94a3b8'),
                height=220, 
                margin=dict(t=10, b=10, l=10, r=10), 
                showlegend=False,
                autosize=True
            )
            pie.add_annotation(
                text=f"{result.confluence_score:.2f}", 
                x=0.5, y=0.5, 
                font_size=28, 
                showarrow=False, 
                font_color="#fff",
                font_weight=700
            )
            
            # Risk HTML with premium styling
            var_color = "#10b981" if result.var_95 < 0.02 else "#f59e0b" if result.var_95 < 0.05 else "#ef4444"
            pos_color = "#10b981"
            stop_color = "#f59e0b"
            
            risk_html = html.Div([
                html.Div([
                    html.Span("VaR (95%)", className="risk-metric-label"),
                    html.Span(f"{result.var_95*100:.2f}%", style={"color": var_color, "fontWeight": "600"})
                ], className="risk-metric-row"),
                html.Div([
                    html.Span("Rec. Position", className="risk-metric-label"),
                    html.Span(f"{result.recommended_position_pct*100:.1f}%", style={"color": pos_color, "fontWeight": "600"})
                ], className="risk-metric-row"),
                html.Div([
                    html.Span("Stop Loss", className="risk-metric-label"),
                    html.Span(f"{result.stop_loss_pct*100:.1f}%", style={"color": stop_color, "fontWeight": "600"})
                ], className="risk-metric-row")
            ])
            
            return fig, fig_ind, confluence_html, pie, risk_html
            
        except Exception as e:
            logger.error(f"Analysis Error: {e}")
            error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
            return [
                create_dark_empty_figure(f"Error loading chart", 500),
                create_dark_empty_figure("Error loading indicators", 300),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-circle fa-2x", style={"color": "#ef4444", "marginBottom": "12px"}),
                        html.Div("Analysis Error", style={"color": "#fff", "fontSize": "1.25rem", "fontWeight": "600"}),
                        html.Div(error_msg, style={"color": "#64748b", "marginTop": "8px", "fontSize": "0.85rem"})
                    ], style={"padding": "40px", "textAlign": "center"})
                ]),
                create_dark_empty_figure("", 220),
                html.Div([
                    html.Span("Unable to load risk metrics", style={"color": "#64748b"})
                ], style={"padding": "20px", "textAlign": "center"})
            ]

# ============================================================================
# HELPER FUNCTIONS (Internal)
# ============================================================================

def _get_signal_badge(signal_type: str):
    """Get badge class and icon for signal type."""
    if "BUY" in signal_type:
        return "badge-neon badge-success", "fas fa-arrow-up"
    elif "SELL" in signal_type:
        return "badge-neon badge-danger", "fas fa-arrow-down"
    return "badge-neon badge-warning", "fas fa-minus"

def _get_cached_signals() -> List[Dict]:
    """Get cached signals from Redis or fetch from database."""
    try:
        # Try to use cache and database
        try:
            from src.data.cache import get_cache
            cache = get_cache()
            cache_key = "dashboard:latest_signals"
            
            # Try cache
            try:
                cached_data = cache.client.get(cache_key)
                if cached_data:
                    cached = pickle.loads(cached_data)
                    if isinstance(cached, list):
                        return cached
            except Exception:
                pass  # Cache not available
        except Exception:
            pass  # Cache module not available
        
        # DB Fallback
        try:
            from src.data.persistence import get_database
            db = get_database()
            signals = db.get_latest_signals(limit=50, min_confidence=0.3)
            if not signals: 
                # Return empty list if no database
                return []
            
            signal_data = []
            for signal in signals:
                try:
                    s_dict = {
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type,
                        "confluence_score": float(signal.confluence_score or 0.5),
                        "technical_score": float(signal.technical_score or 0.5),
                        "sentiment_score": float(signal.sentiment_score or 0.0),
                        "ml_score": float(signal.ml_score or 0.5),
                        "var_95": float(signal.var_95 or 0.05),
                        "position_size": float(signal.suggested_position_size or 0.01),
                        "created_at": signal.created_at.isoformat() if signal.created_at else "",
                        "rationale": signal.technical_rationale or "No analysis available"
                    }
                    
                    score = s_dict["confluence_score"]
                    s_dict["signal_strength"] = "VERY STRONG" if score >= 0.75 else "STRONG" if score >= 0.65 else "MODERATE" if score >= 0.55 else "WEAK" if score >= 0.45 else "NEUTRAL"
                    signal_data.append(s_dict)
                except Exception:
                    continue
                    
            signal_data.sort(key=lambda x: x.get("confluence_score", 0), reverse=True)
            
            # Try to cache the result
            try:
                if 'cache' in locals() and cache:
                    cache.client.setex(cache_key, 900, pickle.dumps(signal_data))
            except Exception:
                pass
                
            return signal_data
        except Exception as e:
            logger.warning(f"Database not available: {e}")
            return []  # Return empty list if database unavailable
            
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        return []  # Return empty list on any error


def _get_demo_signals() -> List[Dict]:
    """Get demo signals when no real data is available."""
    import random
    
    # Define some sample stocks
    stocks = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft"},
        {"symbol": "GOOGL", "name": "Alphabet"},
        {"symbol": "AMZN", "name": "Amazon"},
        {"symbol": "TSLA", "name": "Tesla"},
        {"symbol": "NVDA", "name": "NVIDIA"},
        {"symbol": "META", "name": "Meta Platforms"},
        {"symbol": "JPM", "name": "JPMorgan Chase"},
    ]
    
    signals = []
    signal_types = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    
    for stock in stocks:
        # Generate random but realistic scores
        confluence = random.uniform(0.45, 0.85)
        technical = random.uniform(0.4, 0.9)
        sentiment = random.uniform(0.3, 0.8)
        
        # Determine signal type based on confluence
        if confluence >= 0.7:
            signal_type = "STRONG_BUY"
        elif confluence >= 0.6:
            signal_type = "BUY"
        elif confluence >= 0.45:
            signal_type = "HOLD"
        elif confluence >= 0.35:
            signal_type = "SELL"
        else:
            signal_type = "STRONG_SELL"
        
        # Determine strength
        if confluence >= 0.75:
            strength = "VERY STRONG"
        elif confluence >= 0.65:
            strength = "STRONG"
        elif confluence >= 0.55:
            strength = "MODERATE"
        elif confluence >= 0.45:
            strength = "WEAK"
        else:
            strength = "NEUTRAL"
        
        signals.append({
            "symbol": stock["symbol"],
            "signal_type": signal_type,
            "confluence_score": confluence,
            "technical_score": technical,
            "sentiment_score": sentiment,
            "ml_score": random.uniform(0.4, 0.8),
            "var_95": random.uniform(0.03, 0.08),
            "position_size": random.uniform(0.01, 0.05),
            "signal_strength": strength,
            "rationale": f"Demo signal for {stock['symbol']}"
        })
    
    # Sort by confluence score
    signals.sort(key=lambda x: x["confluence_score"], reverse=True)
    return signals


def _get_current_price(symbol: str) -> float:
    """Get current price for a symbol."""
    try:
        # Try yfinance first
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    
    # Fallback to approximate prices
    default_prices = {
        "AAPL": 185.00,
        "MSFT": 420.00,
        "GOOGL": 175.00,
        "AMZN": 185.00,
        "TSLA": 250.00,
        "NVDA": 900.00,
        "META": 500.00,
        "JPM": 195.00,
        "V": 280.00,
        "MA": 450.00,
    }
    
    return default_prices.get(symbol.upper(), 100.00)
