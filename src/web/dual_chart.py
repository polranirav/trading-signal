"""
Dual Chart Analysis Layout - Research-Based Redesign.

Implements the exact layout from research documents:
- LEFT: Live Market (candlesticks, price, volume, trend)
- RIGHT: ML Predictions (5 periods with confidence, recommendation)

Layout matches SYSTEM_ARCHITECTURE_FLOWCHART.md visual design.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_dual_chart_analysis_page(symbol: str = "AAPL"):
    """
    Create the research-based dual-chart analysis page.
    
    Layout from research:
    ┌─────────────────┬───────────────────────────────────────┐
    │  LEFT: LIVE     │  RIGHT: PREDICTIONS (Next 5 Hours)   │
    │  Candlestick    │  Hour 1: UP (64%)                    │
    │  Chart          │  Hour 2: UP (58%)                    │
    │  Price/Volume   │  ...                                 │
    │  Trend          │  Avg Confidence: 59%                 │
    │                 │  Recommendation: BUY ✅              │
    └─────────────────┴───────────────────────────────────────┘
    """
    
    return html.Div([
        # Header Section
        html.Div([
            html.Div([
                html.H2("📊 Dual-Chart Prediction Dashboard", className="analysis-title"),
                html.P("Live Market Data + ML Predictions", className="analysis-subtitle")
            ]),
            # Asset & Timeframe Selectors
            html.Div([
                # Current Symbol Display
                html.Div([
                    html.Span("Analyzing: ", className="analyzing-label"),
                    html.Span(symbol, id="dual-current-symbol", className="analyzing-symbol")
                ], className="current-symbol-display"),
                
                # Symbol Search
                dbc.InputGroup([
                    dbc.Input(
                        id="dual-symbol-input",
                        placeholder="Search stock...",
                        type="text",
                        className="symbol-search-input"
                    ),
                    dbc.Button(
                        html.I(className="fas fa-search"),
                        id="dual-search-btn",
                        className="symbol-search-btn"
                    )
                ], className="symbol-search-group"),
                
                # Timeframe Selector (as per research - includes scalping intervals)
                html.Div([
                    html.Label("Timeframe:", className="tf-label"),
                    dcc.Dropdown(
                        id='timeframe-select',
                        options=[
                            {'label': '1 Min (Scalp)', 'value': '1M'},
                            {'label': '5 Min (Day)', 'value': '5M'},
                            {'label': '15 Min', 'value': '15M'},
                            {'label': '1 Hour', 'value': '1H'},
                            {'label': '4 Hours', 'value': '4H'},
                            {'label': '1 Day', 'value': '1D'}
                        ],
                        value='5M',  # Default to 5M for day traders
                        clearable=False,
                        className="timeframe-dropdown"
                    )
                ], className="timeframe-selector")
            ], className="header-right-v2")
        ], className="analysis-header-v2"),

        # Quick Access Pills
        html.Div([
            html.Div([
                html.I(className="fas fa-history", style={"marginRight": "8px"}),
                "Quick Access"
            ], className="quick-access-label"),
            html.Div([
                html.Span("AAPL", className="quick-stock-pill active", id={"type": "quick-stock", "symbol": "AAPL"}),
                html.Span("MSFT", className="quick-stock-pill", id={"type": "quick-stock", "symbol": "MSFT"}),
                html.Span("GOOGL", className="quick-stock-pill", id={"type": "quick-stock", "symbol": "GOOGL"}),
                html.Span("TSLA", className="quick-stock-pill", id={"type": "quick-stock", "symbol": "TSLA"}),
                html.Span("NVDA", className="quick-stock-pill", id={"type": "quick-stock", "symbol": "NVDA"}),
            ], className="quick-stocks-list")
        ], className="quick-access-bar"),

        # Main Dual Chart Section (Research Layout)
        html.Div([
            # LEFT SIDE: Live Market Data
            html.Div([
                # Header
                html.Div([
                    html.Div([
                        html.Span("📍", style={"marginRight": "8px"}),
                        html.Span("LIVE MARKET DATA", className="panel-title"),
                        html.Span(id="live-timeframe-label", className="timeframe-badge", 
                                  style={"marginLeft": "12px", "opacity": "0.7", "fontSize": "0.75rem"})
                    ], className="panel-header-left"),
                    html.Span("Updates every 5s", className="update-badge")
                ], className="panel-header live"),
                
                # Price Info Row (as per research layout)
                html.Div([
                    html.Div([
                        html.Div([
                            html.Span("💰 Price", className="info-label"),
                            html.Span(id="live-price", className="price-value")
                        ], className="info-item"),
                        html.Div([
                            html.Span("📈 Change", className="info-label"),
                            html.Span(id="live-change", className="change-value")
                        ], className="info-item"),
                    ], className="info-row-left"),
                    html.Div([
                        html.Div([
                            html.Span("⬆️ High", className="info-label"),
                            html.Span(id="live-high", className="info-value")
                        ], className="info-item"),
                        html.Div([
                            html.Span("⬇️ Low", className="info-label"),
                            html.Span(id="live-low", className="info-value")
                        ], className="info-item"),
                    ], className="info-row-right"),
                ], className="price-info-container", id="live-price-info"),
                
                # Candlestick Chart
                html.Div([
                    dcc.Loading(
                        type="circle",
                        color="#3b82f6",
                        children=[
                            dcc.Graph(
                                id="live-market-chart",
                                style={'height': '350px'},
                                config={'responsive': True, 'displayModeBar': False}
                            )
                        ]
                    )
                ], className="chart-container-v2"),
                
                # Footer Info
                html.Div([
                    html.Div([
                        html.Span("📊 Volume: ", className="footer-label"),
                        html.Span(id="live-volume", className="footer-value")
                    ]),
                    html.Div([
                        html.Span("Trend: ", className="footer-label"),
                        html.Span(id="live-trend", className="trend-indicator")
                    ]),
                    html.Div(id="live-last-update", className="last-update-text")
                ], className="panel-footer")
            ], className="dual-panel live-panel-v2"),

            # RIGHT SIDE: ML Predictions
            html.Div([
                # Header
                html.Div([
                    html.Div([
                        html.Span("🎯", style={"marginRight": "8px"}),
                        html.Span("AI SCENARIO", className="panel-title"),  # Per spec: "AI SCENARIO"
                        html.Span(id="prediction-window-label", className="prediction-window-badge")
                    ], className="panel-header-left"),
                    html.Div([
                        html.Span("Model: ", className="model-label"),
                        html.Span(id="model-type", className="model-type-value")
                    ], className="model-info")
                ], className="panel-header prediction"),
                
                # Ghost Candle Chart (The "Probabilistic Future" from spec)
                html.Div([
                    html.Div([
                        html.Span("👻 Ghost Candles ", className="ghost-chart-label"),
                        html.Span("(Confidence Intervals)", className="ghost-chart-sublabel")
                    ], className="ghost-chart-header"),
                    dcc.Graph(
                        id="ghost-candle-chart",
                        style={'height': '200px'},
                        config={'responsive': True, 'displayModeBar': False}
                    )
                ], className="ghost-chart-container"),
                
                # Predictions List (as per research: Hour 1: UP (64%), etc.)
                html.Div([
                    html.Div(id="predictions-list", className="predictions-list")
                ], className="predictions-container"),
                
                # Average Confidence
                html.Div([
                    html.Span("📊 Average Confidence: ", className="avg-conf-label"),
                    html.Span(id="avg-confidence", className="avg-conf-value")
                ], className="avg-confidence-row"),
                
                # Recommendation (BUY/SELL/NEUTRAL as per research)
                html.Div([
                    html.Span("Recommendation: ", className="rec-label"),
                    html.Span(id="recommendation", className="recommendation-badge")
                ], className="recommendation-row"),
                
                # Confidence Chart (bar visualization)
                html.Div([
                    dcc.Graph(
                        id="confidence-chart",
                        style={'height': '200px'},
                        config={'responsive': True, 'displayModeBar': False}
                    )
                ], className="confidence-chart-container"),
                
                # Footer
                html.Div([
                    html.Div([
                        html.Span("Last trained: ", className="footer-label"),
                        html.Span(id="last-trained", className="footer-value")
                    ]),
                    html.Div([
                        html.Span("Accuracy: ", className="footer-label"),
                        html.Span(id="model-accuracy", className="accuracy-value")
                    ])
                ], className="panel-footer")
            ], className="dual-panel prediction-panel-v2"),
        ], className="dual-chart-container-v2"),

        # Bottom Analysis Cards
        dbc.Row([
            # Signal Analysis
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-signal", style={"marginRight": "10px", "color": "#3b82f6"}),
                        html.Span("SIGNAL ANALYSIS", className="card-section-title")
                    ], className="analysis-card-header"),
                    html.Div(id="signal-analysis-content", className="analysis-card-body")
                ], className="analysis-bottom-card")
            ], xs=12, lg=4, className="mb-3"),
            
            # Technical Indicators
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-bar", style={"marginRight": "10px", "color": "#8b5cf6"}),
                        html.Span("INDICATORS", className="card-section-title")
                    ], className="analysis-card-header"),
                    html.Div(id="indicators-content", className="analysis-card-body")
                ], className="analysis-bottom-card")
            ], xs=12, lg=4, className="mb-3"),
            
            # Risk Management
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-shield-alt", style={"marginRight": "10px", "color": "#ef4444"}),
                        html.Span("RISK METRICS", className="card-section-title")
                    ], className="analysis-card-header"),
                    html.Div(id="risk-metrics-content", className="analysis-card-body")
                ], className="analysis-bottom-card")
            ], xs=12, lg=4, className="mb-3"),
        ], className="mt-4"),
        
        # MODEL STATUS & PERFORMANCE ROW
        dbc.Row([
            # Model Status Panel
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-brain", style={"marginRight": "10px", "color": "#a78bfa"}),
                        html.Span("MODEL STATUS", className="card-section-title")
                    ], className="analysis-card-header"),
                    html.Div([
                        html.Div([
                            html.Span("Training Status: ", className="metric-label"),
                            html.Span(id="training-status", className="metric-value")
                        ], className="metric-row"),
                        html.Div([
                            html.Span("Data Points: ", className="metric-label"),
                            html.Span(id="data-points", className="metric-value")
                        ], className="metric-row"),
                        html.Div([
                            html.Span("Model Type: ", className="metric-label"),
                            html.Span(id="model-type-detail", className="metric-value")
                        ], className="metric-row"),
                        html.Div([
                            html.Span("Accuracy: ", className="metric-label"),
                            html.Span(id="model-accuracy-detail", className="metric-value")
                        ], className="metric-row"),
                        html.Div([
                            html.Span("Last Trained: ", className="metric-label"),
                            html.Span(id="last-trained-detail", className="metric-value")
                        ], className="metric-row"),
                        html.Div([
                            html.Span("Next Retrain: ", className="metric-label"),
                            html.Span(id="next-retrain", className="metric-value")
                        ], className="metric-row"),
                        # Train Button
                        dbc.Button([
                            html.I(className="fas fa-sync-alt", style={"marginRight": "8px"}),
                            "Train Model Now"
                        ], id="train-model-btn", color="primary", size="sm", className="mt-3 w-100")
                    ], className="analysis-card-body", id="model-status-content")
                ], className="analysis-bottom-card model-status-card")
            ], xs=12, lg=4, className="mb-3"),
            
            # Performance Tracking
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-trophy", style={"marginRight": "10px", "color": "#f59e0b"}),
                        html.Span("PERFORMANCE TRACKING", className="card-section-title")
                    ], className="analysis-card-header"),
                    html.Div([
                        html.Div([
                            html.Span("Today's Predictions: ", className="metric-label"),
                            html.Span(id="today-predictions", className="metric-value")
                        ], className="metric-row"),
                        html.Div([
                            html.Span("Correct: ", className="metric-label"),
                            html.Span(id="correct-predictions", className="metric-value")
                        ], className="metric-row"),
                        html.Div([
                            html.Span("Win Rate: ", className="metric-label"),
                            html.Span(id="win-rate", className="metric-value")
                        ], className="metric-row"),
                        html.Div([
                            html.Span("Best Hour: ", className="metric-label"),
                            html.Span(id="best-hour", className="metric-value")
                        ], className="metric-row"),
                        html.Div([
                            html.Span("Trend: ", className="metric-label"),
                            html.Span(id="accuracy-trend", className="metric-value")
                        ], className="metric-row"),
                        html.Div([
                            html.Span("Status: ", className="metric-label"),
                            html.Span(id="validation-status", className="metric-value")
                        ], className="metric-row"),
                    ], className="analysis-card-body")
                ], className="analysis-bottom-card performance-card")
            ], xs=12, lg=4, className="mb-3"),
            
            # Prediction Interpretation
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-lightbulb", style={"marginRight": "10px", "color": "#10b981"}),
                        html.Span("PREDICTION INTERPRETATION", className="card-section-title")
                    ], className="analysis-card-header"),
                    html.Div(id="interpretation-content", className="analysis-card-body")
                ], className="analysis-bottom-card interpretation-card")
            ], xs=12, lg=4, className="mb-3"),
        ], className="mt-3"),

        # Hidden Stores
        dcc.Store(id="dual-symbol-store", data=symbol),
        dcc.Store(id="timeframe-store", data="5M"),  # Default to 5M for day traders
        
        # Update Intervals (as per research: 5s for live, 1h for predictions)
        dcc.Interval(id="live-update-interval", interval=5*1000, n_intervals=0),  # 5 seconds
        dcc.Interval(id="prediction-update-interval", interval=60*1000, n_intervals=0)  # 1 minute (for demo)
    ], className="dual-analysis-page-v2")


def create_prediction_item(period: int, label: str, direction: str, confidence: float) -> html.Div:
    """
    Create a single prediction item.
    
    As per research format: "Hour 1: 🟢 UP (64%)"
    """
    # Direction symbol and color
    if direction == "UP":
        symbol = "🟢"
        color = "#10b981"  # Green
    else:
        symbol = "🔴"
        color = "#ef4444"  # Red
    
    # Confidence color
    if confidence >= 60:
        conf_color = "#10b981"  # High confidence - green
    elif confidence >= 55:
        conf_color = "#f59e0b"  # Medium confidence - amber
    else:
        conf_color = "#64748b"  # Low confidence - gray
    
    return html.Div([
        html.Span(f"{symbol} ", className="direction-symbol"),
        html.Span(f"{label}: ", className="period-label"),
        html.Span(f"{direction} ", className="direction-text", style={"color": color}),
        html.Span(f"({confidence:.0f}%)", className="confidence-text", style={"color": conf_color})
    ], className="prediction-item")
