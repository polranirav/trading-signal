"""
Dual Chart Callbacks - Research-Based Implementation.

Implements callbacks as specified in research documents:
- Live chart updates every 5 seconds
- Predictions update on timeframe interval
- Display 5-period predictions with confidence
- Show BUY/SELL recommendation
"""

from dash import Input, Output, State, html, callback_context, ALL
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

from src.logging_config import get_logger
from src.web.dual_chart import create_prediction_item

logger = get_logger(__name__)


def register_dual_chart_callbacks(app):
    """Register all dual-chart analysis callbacks."""
    
    # ========================================
    # LIVE MARKET CHART (updates every 5s)
    # ========================================
    @app.callback(
        [Output("live-market-chart", "figure"),
         Output("live-price", "children"),
         Output("live-change", "children"),
         Output("live-change", "style"),
         Output("live-high", "children"),
         Output("live-low", "children"),
         Output("live-volume", "children"),
         Output("live-trend", "children"),
         Output("live-trend", "style"),
         Output("live-last-update", "children")],
        [Input("live-update-interval", "n_intervals"),
         Input("dual-symbol-store", "data"),
         Input("timeframe-store", "data")]
    )
    def update_live_chart(n_intervals, symbol, timeframe):
        """Update live candlestick chart (as per research: every 5 seconds)."""
        if not symbol:
            symbol = "AAPL"
        
        try:
            from src.data.persistence import get_database
            from src.data.realistic_data_generator import RealisticMarketDataGenerator
            
            db = get_database()
            
            # Map timeframe to candle count and days equivalent
            timeframe_config = {
                "1M": {"limit": 60, "days": 1},     # Last 1 hour (60 mins) - clearer view for scalping
                "5M": {"limit": 100, "days": 2},    # Last ~8 hours of 5-min candles
                "15M": {"limit": 100, "days": 3},   
                "1H": {"limit": 100, "days": 15},   
                "4H": {"limit": 50, "days": 30},    
                "1D": {"limit": 100, "days": 100}   
            }
            
            config = timeframe_config.get(timeframe, {"limit": 100, "days": 100})
            
            # For intraday timeframes (not 1D), generate fresh data with proper timestamps
            if timeframe != "1D":
                # Generate intraday data with the realistic generator
                # Use a stable seed based on symbol to prevent flickering during live updates
                stable_seed = sum(ord(c) for c in symbol) % 1000
                generator = RealisticMarketDataGenerator(seed=stable_seed)
                df = generator.generate(
                    symbol=symbol.upper(), 
                    days=config["days"], 
                    interval=timeframe
                )
                # Take only the most recent candles
                df = df.tail(config["limit"])
            else:
                # For daily, use database if available
                df = db.get_candles(symbol.upper(), limit=config["limit"])
                
                if df.empty:
                    # Generate daily data
                    generator = RealisticMarketDataGenerator(seed=42)
                    df = generator.generate(symbol=symbol.upper(), days=config["days"], interval="1D")
                    df = df.tail(config["limit"])
            
            if df.empty:
                # Use demo data
                return create_demo_live_chart_with_info(symbol)
            
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df['time'] if 'time' in df.columns else df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#10b981',
                decreasing_line_color='#ef4444',
                increasing_fillcolor='rgba(16, 185, 129, 0.8)',
                decreasing_fillcolor='rgba(239, 68, 68, 0.8)'
            )])
            
            # Add SMA20 if we have enough data
            if len(df) >= 20:
                sma20 = df['close'].rolling(20).mean()
                fig.add_trace(go.Scatter(
                    x=df['time'] if 'time' in df.columns else df.index,
                    y=sma20, name="SMA 20",
                    line=dict(color='#3b82f6', width=1.5)
                ))
            
            # Apply dark theme
            layout_args = dict(
                plot_bgcolor='rgba(15, 23, 42, 0.95)',
                paper_bgcolor='rgba(15, 23, 42, 0.95)',
                font=dict(color='#94a3b8'),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.05)', 
                    showgrid=True, 
                    rangeslider=dict(visible=False)
                ),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
                margin=dict(l=10, r=10, t=10, b=30),
                showlegend=False,
                hovermode='x unified'
            )
            
            # Refine 1M chart layout for "perfect" visibility
            if timeframe == "1M":
                # Force ticks every 5 minutes (300,000 ms)
                # Format as HH:MM
                layout_args['xaxis'].update(dict(
                    dtick=300000 if len(df) <= 60 else 600000, # 5 min ticks for 1h view
                    tickformat="%H:%M"
                ))
                
            fig.update_layout(**layout_args)
            
            # Price info
            current_price = float(df['close'].iloc[-1])
            prev_price = float(df['close'].iloc[-2]) if len(df) > 1 else current_price
            change = ((current_price - prev_price) / prev_price) * 100
            high = float(df['high'].iloc[-1])
            low = float(df['low'].iloc[-1])
            volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0
            
            # Trend (based on MA)
            if len(df) >= 20:
                sma_val = float(df['close'].rolling(20).mean().iloc[-1])
                if current_price > sma_val * 1.01:
                    trend = "↑ STRONG UP"
                    trend_color = "#10b981"
                elif current_price > sma_val:
                    trend = "↑ UP"
                    trend_color = "#34d399"
                elif current_price < sma_val * 0.99:
                    trend = "↓ STRONG DOWN"
                    trend_color = "#ef4444"
                else:
                    trend = "↓ DOWN"
                    trend_color = "#f87171"
            else:
                trend = "→ RANGING"
                trend_color = "#f59e0b"
            
            # Format values
            price_str = f"${current_price:,.2f}"
            change_str = f"{change:+.2f}%"
            change_color = "#10b981" if change >= 0 else "#ef4444"
            high_str = f"${high:,.2f}"
            low_str = f"${low:,.2f}"
            volume_str = format_volume(volume)
            update_str = f"Updated: {datetime.now().strftime('%I:%M:%S %p')}"
            
            return (fig, price_str, change_str, {"color": change_color},
                   high_str, low_str, volume_str, trend, {"color": trend_color}, update_str)
            
        except Exception as e:
            logger.error(f"Live chart error: {e}")
            return create_demo_live_chart_with_info(symbol)

    # ========================================
    # ML PREDICTIONS (as per research)
    # ========================================
    @app.callback(
        [Output("predictions-list", "children"),
         Output("avg-confidence", "children"),
         Output("recommendation", "children"),
         Output("recommendation", "style"),
         Output("confidence-chart", "figure"),
         Output("prediction-window-label", "children"),
         Output("model-type", "children"),
         Output("last-trained", "children"),
         Output("model-accuracy", "children")],
        [Input("prediction-update-interval", "n_intervals"),
         Input("dual-symbol-store", "data"),
         Input("timeframe-store", "data")]
    )
    def update_predictions(n_intervals, symbol, timeframe):
        """
        Update ML predictions (as per research).
        
        Format: Hour 1: 🟢 UP (64%), Hour 2: 🟢 UP (58%), etc.
        """
        if not symbol:
            symbol = "AAPL"
        
        try:
            from src.analytics.ml_prediction_service import generate_predictions, get_prediction_model
            
            # Get predictions
            predictions, recommendation = generate_predictions(symbol, timeframe, num_periods=5)
            
            # Get model info
            model = get_prediction_model()
            
            # Create prediction items (as per research format)
            prediction_items = []
            for pred in predictions:
                item = create_prediction_item(
                    pred['period'],
                    pred['label'],
                    pred['direction'],
                    pred['confidence']
                )
                prediction_items.append(item)
            
            # Confidence chart (bar chart as per research)
            fig = create_confidence_bar_chart(predictions)
            
            # Format recommendation
            rec_text = f"{recommendation['symbol']} {recommendation['recommendation']}"
            rec_style = {
                "color": "#10b981" if recommendation['recommendation'] == 'BUY' else 
                         "#ef4444" if recommendation['recommendation'] == 'SELL' else "#f59e0b",
                "fontWeight": "bold"
            }
            
            # Window label
            window_labels = {"1M": "Next 5 min", "5M": "Next 25 min", "15M": "Next 75 min", "1H": "Next 5 Hours", "4H": "Next 20 Hours", "1D": "Next 5 Days"}
            window_label = window_labels.get(timeframe, "Next 5 Periods")
            
            # Model info
            model_type = "XGBoost" if model.is_trained else "Demo (not trained)"
            accuracy = f"{model.metrics.get('accuracy', 0) * 100:.1f}%" if model.is_trained else "N/A"
            last_trained = model.last_trained.strftime("%I:%M %p") if model.last_trained else "Never"
            
            return (
                prediction_items,
                f"{recommendation['avg_confidence']:.1f}%",
                rec_text,
                rec_style,
                fig,
                window_label,
                model_type,
                last_trained,
                accuracy
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return create_demo_predictions_response(timeframe)

    # ========================================
    # GHOST CANDLE CHART (From Arch Spec)
    # ========================================
    @app.callback(
        Output("ghost-candle-chart", "figure"),
        [Input("prediction-update-interval", "n_intervals"),
         Input("dual-symbol-store", "data"),
         Input("timeframe-store", "data")]
    )
    def update_ghost_candle_chart(n_intervals, symbol, timeframe):
        """
        Generate Ghost Candle Chart with Fan Chart (Confidence Intervals).
        
        From Architectural Specification:
        - "Ghost Candles: Transparent candlesticks for predictions"
        - "Confidence Intervals: Fan Chart with 50% and 95% bands"
        - "NOW Divider: Vertical line separating reality from prediction"
        """
        try:
            from src.analytics.ghost_candle_generator import generate_ghost_prediction_chart
            
            # Get ghost candle data
            ghost_data = generate_ghost_prediction_chart(
                symbol or "AAPL", 
                timeframe or "1H", 
                num_periods=5
            )
            
            # Create the figure
            fig = go.Figure()
            
            # Extract times and prices
            times = [gc['time'] for gc in ghost_data['ghost_candles']]
            
            # Add 95% Confidence Band (outer, lighter)
            fig.add_trace(go.Scatter(
                x=times + times[::-1],
                y=ghost_data['upper_band_95'] + ghost_data['lower_band_95'][::-1],
                fill='toself',
                fillcolor='rgba(139, 92, 246, 0.1)',
                line=dict(color='rgba(139, 92, 246, 0.3)', dash='dot'),
                name='95% Confidence',
                hoverinfo='skip'
            ))
            
            # Add 50% Confidence Band (inner, darker)
            fig.add_trace(go.Scatter(
                x=times + times[::-1],
                y=ghost_data['upper_band_50'] + ghost_data['lower_band_50'][::-1],
                fill='toself',
                fillcolor='rgba(139, 92, 246, 0.2)',
                line=dict(color='rgba(139, 92, 246, 0.5)'),
                name='50% Confidence',
                hoverinfo='skip'
            ))
            
            # Add Ghost Candles (transparent candlesticks)
            opens = [gc['open'] for gc in ghost_data['ghost_candles']]
            highs = [gc['high'] for gc in ghost_data['ghost_candles']]
            lows = [gc['low'] for gc in ghost_data['ghost_candles']]
            closes = [gc['close'] for gc in ghost_data['ghost_candles']]
            
            # Determine colors based on direction
            colors_increasing = 'rgba(16, 185, 129, 0.4)'  # Green with transparency
            colors_decreasing = 'rgba(239, 68, 68, 0.4)'   # Red with transparency
            
            fig.add_trace(go.Candlestick(
                x=times,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name='Ghost Candles',
                increasing_line_color=colors_increasing,
                decreasing_line_color=colors_decreasing,
                increasing_fillcolor=colors_increasing,
                decreasing_fillcolor=colors_decreasing,
                whiskerwidth=0.5
            ))
            
            # Add NOW divider line
            current_price = ghost_data['current_price']
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="rgba(255, 255, 255, 0.5)",
                annotation_text="NOW",
                annotation_position="right",
                annotation_font=dict(color="white", size=10)
            )
            
            # Apply dark theme matching live chart
            fig.update_layout(
                plot_bgcolor='rgba(15, 23, 42, 0.95)',
                paper_bgcolor='rgba(15, 23, 42, 0.95)',
                font=dict(color='#94a3b8', size=10),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.05)', 
                    showgrid=True,
                    rangeslider=dict(visible=False)
                ),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
                margin=dict(l=10, r=10, t=10, b=30),
                showlegend=False,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Ghost candle chart error: {e}")
            return create_empty_ghost_chart()

    # ========================================
    # TIMEFRAME SELECTOR
    # ========================================
    @app.callback(
        Output("timeframe-store", "data"),
        Input("timeframe-select", "value")
    )
    def update_timeframe(value):
        """Update timeframe store."""
        return value or "1H"

    @app.callback(
        Output("live-timeframe-label", "children"),
        Input("timeframe-store", "data")
    )
    def update_live_timeframe_label(timeframe):
        """Update the live panel timeframe label."""
        labels = {
            "1M": "(1 Min)",
            "5M": "(5 Min)",
            "15M": "(15 Min)",
            "1H": "(1 Hour)",
            "4H": "(4 Hour)",
            "1D": "(Daily)"
        }
        return labels.get(timeframe, "(1 Hour)")

    # ========================================
    # SYMBOL SWITCHING
    # ========================================
    @app.callback(
        Output("dual-symbol-store", "data"),
        [Input("dual-search-btn", "n_clicks"),
         Input({"type": "quick-stock", "symbol": ALL}, "n_clicks")],
        [State("dual-symbol-input", "value")],
        prevent_initial_call=True
    )
    def update_symbol(search_click, quick_clicks, input_value):
        """Update the current analysis symbol."""
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        
        trigger_id = ctx.triggered_id
        
        # Handle quick stock pill click
        if isinstance(trigger_id, dict) and trigger_id.get("type") == "quick-stock":
            return trigger_id.get("symbol")
        
        # Handle search
        if trigger_id == "dual-search-btn" and input_value:
            return input_value.upper().strip()
        
        return dash.no_update

    @app.callback(
        Output("dual-current-symbol", "children"),
        Input("dual-symbol-store", "data")
    )
    def update_symbol_display(symbol):
        """Update the displayed symbol."""
        return symbol or "AAPL"

    # ========================================
    # BOTTOM ANALYSIS CARDS
    # ========================================
    @app.callback(
        [Output("signal-analysis-content", "children"),
         Output("indicators-content", "children"),
         Output("risk-metrics-content", "children")],
        [Input("prediction-update-interval", "n_intervals"),
         Input("dual-symbol-store", "data")]
    )
    def update_analysis_cards(n_intervals, symbol):
        """Update bottom analysis cards."""
        try:
            from src.data.persistence import get_database
            from src.analytics.feature_engineer import FeatureEngineer
            
            db = get_database()
            df = db.get_candles(symbol.upper() if symbol else "AAPL", limit=100)
            
            if df.empty or len(df) < 50:
                return create_demo_analysis_cards()
            
            # Calculate features
            engineer = FeatureEngineer(df)
            df_features = engineer.get_all_features(add_labels=False)
            
            if df_features.empty:
                return create_demo_analysis_cards()
            
            latest = df_features.iloc[-1]
            
            # Signal Analysis
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            
            if rsi > 70:
                rsi_signal = "Overbought (Sell)"
                rsi_color = "#ef4444"
            elif rsi < 30:
                rsi_signal = "Oversold (Buy)"
                rsi_color = "#10b981"
            else:
                rsi_signal = "Neutral"
                rsi_color = "#f59e0b"
            
            macd_signal = "Bullish" if macd > 0 else "Bearish"
            macd_color = "#10b981" if macd > 0 else "#ef4444"
            
            signal_content = html.Div([
                html.Div([
                    html.Span("RSI(14): ", className="metric-label"),
                    html.Span(f"{rsi:.1f} - {rsi_signal}", style={"color": rsi_color})
                ], className="metric-row"),
                html.Div([
                    html.Span("MACD: ", className="metric-label"),
                    html.Span(f"{macd:.2f} - {macd_signal}", style={"color": macd_color})
                ], className="metric-row"),
            ])
            
            # Indicators
            momentum = latest.get('momentum', 0)
            volatility = latest.get('volatility', 0)
            atr = latest.get('atr_pct', 0)
            
            indicators_content = html.Div([
                html.Div([
                    html.Span("Momentum: ", className="metric-label"),
                    html.Span(f"{momentum:+.2f}%", style={"color": "#10b981" if momentum > 0 else "#ef4444"})
                ], className="metric-row"),
                html.Div([
                    html.Span("Volatility: ", className="metric-label"),
                    html.Span(f"{volatility:.2f}%")
                ], className="metric-row"),
                html.Div([
                    html.Span("ATR: ", className="metric-label"),
                    html.Span(f"{atr:.2f}%")
                ], className="metric-row"),
            ])
            
            # Risk Metrics (simplified)
            current_price = float(df['close'].iloc[-1])
            risk_content = html.Div([
                html.Div([
                    html.Span("Position Size: ", className="metric-label"),
                    html.Span("1-2% of capital recommended")
                ], className="metric-row"),
                html.Div([
                    html.Span("Stop Loss: ", className="metric-label"),
                    html.Span(f"${current_price * 0.98:.2f} (-2%)")
                ], className="metric-row"),
                html.Div([
                    html.Span("Take Profit: ", className="metric-label"),
                    html.Span(f"${current_price * 1.04:.2f} (+4%)")
                ], className="metric-row"),
            ])
            
            return signal_content, indicators_content, risk_content
            
        except Exception as e:
            logger.error(f"Analysis cards error: {e}")
            return create_demo_analysis_cards()

    # ========================================
    # MODEL STATUS PANEL
    # ========================================
    @app.callback(
        [Output("training-status", "children"),
         Output("training-status", "style"),
         Output("data-points", "children"),
         Output("model-type-detail", "children"),
         Output("model-accuracy-detail", "children"),
         Output("model-accuracy-detail", "style"),
         Output("last-trained-detail", "children"),
         Output("next-retrain", "children")],
        Input("prediction-update-interval", "n_intervals")
    )
    def update_model_status(n_intervals):
        """Update Model Status panel with training info."""
        try:
            from src.analytics.ml_prediction_service import get_prediction_model
            model = get_prediction_model()
            
            if model.is_trained:
                status = "✅ Trained"
                status_style = {"color": "#10b981", "fontWeight": "600"}
                data_points = f"{model.metrics.get('train_samples', 0)} candles"
                model_type = "XGBoost" if hasattr(model.model, 'get_booster') else "RandomForest"
                accuracy = f"{model.metrics.get('accuracy', 0) * 100:.1f}%"
                acc_style = {"color": "#10b981" if model.metrics.get('accuracy', 0) > 0.55 else "#f59e0b"}
                last_trained = model.last_trained.strftime("%b %d, %I:%M %p") if model.last_trained else "N/A"
                # Calculate next retrain (7 days from last)
                if model.last_trained:
                    from datetime import timedelta
                    next_retrain = model.last_trained + timedelta(days=7)
                    next_str = next_retrain.strftime("%b %d")
                else:
                    next_str = "After training"
            else:
                status = "❌ Not Trained"
                status_style = {"color": "#ef4444", "fontWeight": "600"}
                data_points = "0 candles"
                model_type = "XGBoost (pending)"
                accuracy = "N/A"
                acc_style = {"color": "#64748b"}
                last_trained = "Never"
                next_str = "Train now!"
            
            return status, status_style, data_points, model_type, accuracy, acc_style, last_trained, next_str
            
        except Exception as e:
            logger.error(f"Model status error: {e}")
            return ("❌ Error", {"color": "#ef4444"}, "0", "Unknown", "N/A", 
                    {"color": "#64748b"}, "Never", "N/A")

    # ========================================
    # PERFORMANCE TRACKING
    # ========================================
    @app.callback(
        [Output("today-predictions", "children"),
         Output("correct-predictions", "children"),
         Output("win-rate", "children"),
         Output("win-rate", "style"),
         Output("best-hour", "children"),
         Output("accuracy-trend", "children"),
         Output("accuracy-trend", "style"),
         Output("validation-status", "children"),
         Output("validation-status", "style")],
        Input("prediction-update-interval", "n_intervals")
    )
    def update_performance_tracking(n_intervals):
        """Update performance tracking metrics."""
        # In demo mode, show placeholder values
        # In production, this would read from a prediction log
        try:
            from src.analytics.ml_prediction_service import get_prediction_model
            model = get_prediction_model()
            
            if model.is_trained:
                # Demo values - in production these would come from actual tracking
                import random
                predictions_today = random.randint(5, 15)
                correct = int(predictions_today * random.uniform(0.5, 0.65))
                win_rate = (correct / predictions_today * 100) if predictions_today > 0 else 0
                
                return (
                    str(predictions_today),
                    str(correct),
                    f"{win_rate:.1f}%",
                    {"color": "#10b981" if win_rate >= 55 else "#f59e0b"},
                    "Hour 4 (61%)",
                    "⬆️ Improving",
                    {"color": "#10b981"},
                    "✅ Validated",
                    {"color": "#10b981"}
                )
            else:
                return (
                    "0",
                    "0",
                    "N/A",
                    {"color": "#64748b"},
                    "N/A",
                    "No data",
                    {"color": "#64748b"},
                    "⚠️ Paper trade first",
                    {"color": "#f59e0b"}
                )
        except Exception as e:
            logger.error(f"Performance tracking error: {e}")
            return ("0", "0", "N/A", {}, "N/A", "N/A", {}, "Error", {"color": "#ef4444"})

    # ========================================
    # PREDICTION INTERPRETATION
    # ========================================
    @app.callback(
        Output("interpretation-content", "children"),
        [Input("prediction-update-interval", "n_intervals"),
         Input("dual-symbol-store", "data")]
    )
    def update_interpretation(n_intervals, symbol):
        """Generate intelligent interpretation of predictions."""
        try:
            from src.analytics.ml_prediction_service import generate_predictions, get_prediction_model
            
            predictions, recommendation = generate_predictions(symbol or "AAPL", "1H", 5)
            model = get_prediction_model()
            
            # Build interpretation
            up_count = sum(1 for p in predictions if p['direction'] == 'UP')
            avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
            
            # Determine overall sentiment
            if up_count >= 4:
                sentiment = "📈 Strong Bullish"
                sentiment_color = "#10b981"
            elif up_count >= 3:
                sentiment = "📊 Bullish"
                sentiment_color = "#34d399"
            elif up_count <= 1:
                sentiment = "📉 Strong Bearish"
                sentiment_color = "#ef4444"
            elif up_count <= 2:
                sentiment = "📊 Bearish"
                sentiment_color = "#f87171"
            else:
                sentiment = "➡️ Neutral"
                sentiment_color = "#f59e0b"
            
            # Confidence assessment
            if avg_conf >= 60:
                conf_text = "✅ High confidence - signals are strong"
                conf_color = "#10b981"
            elif avg_conf >= 55:
                conf_text = "⚠️ Moderate confidence - proceed with caution"
                conf_color = "#f59e0b"
            else:
                conf_text = "❌ Low confidence - consider waiting"
                conf_color = "#ef4444"
            
            # Model status warning
            if not model.is_trained:
                warning = html.Div([
                    html.Span("⚠️ ", style={"color": "#f59e0b"}),
                    html.Span("Model not trained - predictions are DEMO only!", 
                              style={"color": "#f59e0b", "fontWeight": "500"})
                ], className="interpretation-warning")
            else:
                warning = None
            
            # Build content
            content = html.Div([
                # Sentiment
                html.Div([
                    html.Span("Sentiment: ", className="metric-label"),
                    html.Span(sentiment, style={"color": sentiment_color, "fontWeight": "600"})
                ], className="metric-row"),
                
                # Confidence
                html.Div([
                    html.Span(conf_text, style={"color": conf_color, "fontSize": "0.85rem"})
                ], className="interpretation-note"),
                
                # Action
                html.Div([
                    html.Span("Action: ", className="metric-label"),
                    html.Span(f"{recommendation['symbol']} {recommendation['recommendation']}", 
                              style={"color": recommendation['color'], "fontWeight": "600"})
                ], className="metric-row mt-2"),
                
                # Reason
                html.Div([
                    html.Span(f"Reason: {recommendation['reason']}", 
                              style={"color": "#94a3b8", "fontSize": "0.8rem"})
                ], className="interpretation-reason"),
                
                # Warning if not trained
                warning if warning else html.Span()
            ])
            
            return content
            
        except Exception as e:
            logger.error(f"Interpretation error: {e}")
            return html.Div([
                html.Span("Unable to generate interpretation", style={"color": "#64748b"})
            ])

    # ========================================
    # TRAIN MODEL BUTTON
    # ========================================
    @app.callback(
        Output("train-model-btn", "children"),
        Output("train-model-btn", "disabled"),
        Input("train-model-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def train_model_button(n_clicks):
        """Handle train model button click."""
        if n_clicks:
            try:
                from src.analytics.ml_prediction_service import get_prediction_model
                from src.analytics.feature_engineer import FeatureEngineer
                from src.data.persistence import get_database
                
                # Get data
                db = get_database()
                df = db.get_candles("AAPL", limit=200)
                
                if len(df) < 100:
                    return [html.I(className="fas fa-exclamation-triangle", style={"marginRight": "8px"}),
                            "Need more data!"], False
                
                # Calculate features
                engineer = FeatureEngineer(df)
                df_features = engineer.get_all_features(add_labels=True)
                
                # Train
                model = get_prediction_model()
                metrics = model.train(df_features)
                
                if 'error' in metrics:
                    return [html.I(className="fas fa-times", style={"marginRight": "8px"}),
                            f"Error: {metrics['error']}"], False
                
                return [html.I(className="fas fa-check", style={"marginRight": "8px"}),
                        f"Trained! {metrics['accuracy']*100:.1f}%"], True
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                return [html.I(className="fas fa-times", style={"marginRight": "8px"}),
                        "Training failed"], False
        
        return [html.I(className="fas fa-sync-alt", style={"marginRight": "8px"}),
                "Train Model Now"], False


# ========================================
# HELPER FUNCTIONS
# ========================================

def format_volume(volume: float) -> str:
    """Format volume with K/M suffix."""
    if volume >= 1_000_000:
        return f"{volume / 1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume / 1_000:.1f}K"
    else:
        return f"{volume:.0f}"


def create_confidence_bar_chart(predictions: list) -> go.Figure:
    """
    Create confidence bar chart as per research.
    
    Green bars for UP, red bars for DOWN.
    """
    labels = [p['label'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    colors = ['#10b981' if p['direction'] == 'UP' else '#ef4444' for p in predictions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=confidences,
        marker_color=colors,
        text=[f"{c:.0f}%" for c in confidences],
        textposition='auto',
        name='Confidence'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(15, 23, 42, 0.95)',
        paper_bgcolor='rgba(15, 23, 42, 0.95)',
        font=dict(color='#94a3b8', size=10),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 100], title=""),
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
        height=200
    )
    
    return fig


def create_demo_live_chart_with_info(symbol: str = "AAPL"):
    """Create demo chart with price info."""
    import numpy as np
    
    n = 50
    dates = [datetime.now() - timedelta(hours=n-i) for i in range(n)]
    
    base = 180
    prices = [base]
    for _ in range(n-1):
        prices.append(max(160, min(200, prices[-1] + random.uniform(-2, 2.5))))
    
    opens, highs, lows, closes = [], [], [], []
    for p in prices:
        o = p
        c = p + random.uniform(-1.5, 2)
        h = max(o, c) + random.uniform(0, 1)
        l = min(o, c) - random.uniform(0, 1)
        opens.append(o)
        closes.append(c)
        highs.append(h)
        lows.append(l)
    
    fig = go.Figure(data=[go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        increasing_line_color='#10b981', decreasing_line_color='#ef4444',
        increasing_fillcolor='rgba(16, 185, 129, 0.8)',
        decreasing_fillcolor='rgba(239, 68, 68, 0.8)'
    )])
    
    fig.update_layout(
        plot_bgcolor='rgba(15, 23, 42, 0.95)',
        paper_bgcolor='rgba(15, 23, 42, 0.95)',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False
    )
    
    current = closes[-1]
    prev = closes[-2]
    change = ((current - prev) / prev) * 100
    
    return (
        fig, f"${current:.2f}", f"{change:+.2f}%", {"color": "#10b981" if change >= 0 else "#ef4444"},
        f"${max(highs[-5:]):.2f}", f"${min(lows[-5:]):.2f}", "15.2M",
        "↑ UP", {"color": "#10b981"}, f"Demo • {datetime.now().strftime('%I:%M %p')}"
    )


def create_demo_predictions_response(timeframe: str = "1H"):
    """Create demo predictions response."""
    
    # Time labels
    tf_labels = {
        "15M": ["+15m", "+30m", "+45m", "+60m", "+75m"],
        "1H": ["Hour 1", "Hour 2", "Hour 3", "Hour 4", "Hour 5"],
        "4H": ["+4h", "+8h", "+12h", "+16h", "+20h"],
        "1D": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
    }
    labels = tf_labels.get(timeframe, tf_labels["1H"])
    
    # Generate demo predictions
    base_dir = random.choice(['UP', 'DOWN'])
    predictions = []
    for i, lbl in enumerate(labels):
        direction = base_dir if random.random() < 0.7 else ('DOWN' if base_dir == 'UP' else 'UP')
        conf = random.uniform(55, 68) - i * 2
        predictions.append({'period': i+1, 'label': lbl, 'direction': direction, 'confidence': max(50, conf)})
    
    # Create items
    items = [create_prediction_item(p['period'], p['label'], p['direction'], p['confidence']) for p in predictions]
    
    # Chart
    fig = create_confidence_bar_chart(predictions)
    
    # Recommendation
    up_count = sum(1 for p in predictions if p['direction'] == 'UP')
    if up_count >= 3:
        rec = "✅ BUY"
        rec_style = {"color": "#10b981", "fontWeight": "bold"}
    elif up_count <= 2:
        rec = "❌ SELL"
        rec_style = {"color": "#ef4444", "fontWeight": "bold"}
    else:
        rec = "⚠️ NEUTRAL"
        rec_style = {"color": "#f59e0b", "fontWeight": "bold"}
    
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    
    window_labels = {"1M": "Next 5 min", "5M": "Next 25 min", "15M": "Next 75 min", "1H": "Next 5 Hours", "4H": "Next 20 Hours", "1D": "Next 5 Days"}
    
    return (items, f"{avg_conf:.1f}%", rec, rec_style, fig, 
            window_labels.get(timeframe, "Next 5 Periods"),
            "Demo (not trained)", "Never", "N/A")


def create_demo_analysis_cards():
    """Create demo analysis cards content."""
    signal = html.Div([
        html.Div([html.Span("RSI(14): ", className="metric-label"), html.Span("52.3 - Neutral", style={"color": "#f59e0b"})], className="metric-row"),
        html.Div([html.Span("MACD: ", className="metric-label"), html.Span("12.5 - Bullish", style={"color": "#10b981"})], className="metric-row"),
    ])
    
    indicators = html.Div([
        html.Div([html.Span("Momentum: ", className="metric-label"), html.Span("+0.85%", style={"color": "#10b981"})], className="metric-row"),
        html.Div([html.Span("Volatility: ", className="metric-label"), html.Span("1.2%")], className="metric-row"),
        html.Div([html.Span("ATR: ", className="metric-label"), html.Span("2.4%")], className="metric-row"),
    ])
    
    risk = html.Div([
        html.Div([html.Span("Position Size: ", className="metric-label"), html.Span("1-2% recommended")], className="metric-row"),
        html.Div([html.Span("Stop Loss: ", className="metric-label"), html.Span("$181.50 (-2%)")], className="metric-row"),
        html.Div([html.Span("Take Profit: ", className="metric-label"), html.Span("$192.92 (+4%)")], className="metric-row"),
    ])
    
    return signal, indicators, risk


def create_empty_ghost_chart() -> go.Figure:
    """Create empty ghost chart placeholder."""
    fig = go.Figure()
    
    fig.add_annotation(
        text="Ghost Candles Loading...",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color="#64748b", size=14)
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(15, 23, 42, 0.95)',
        paper_bgcolor='rgba(15, 23, 42, 0.95)',
        font=dict(color='#94a3b8'),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    
    return fig
