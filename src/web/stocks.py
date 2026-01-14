"""
Premium Stock Discovery Page.

Provides actionable trading intelligence:
- Market overview with index performance
- Top movers (gainers/losers)
- AI-powered signal scores on every stock
- Real-time price data and mini charts
- Advanced filtering by signal, sector, volume
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_stocks_page():
    """Create the premium stock discovery page with real trading value."""
    return html.Div([
        # Page Header
        html.Div([
            html.Div([
                html.H2("Stock Discovery", style={
                    "fontSize": "1.75rem",
                    "fontWeight": "700",
                    "color": "#fff",
                    "marginBottom": "4px"
                }),
                html.P("Find high-confidence trading opportunities with AI-powered signals", style={
                    "color": "#64748b",
                    "fontSize": "0.9rem",
                    "marginBottom": "0"
                }),
            ]),
            # Quick Stats
            html.Div([
                html.Div([
                    html.Span("🟢", style={"marginRight": "6px"}),
                    html.Span("Market Open", style={"color": "#10b981", "fontWeight": "500"})
                ], style={"marginRight": "24px"}),
                html.Div(id="market-time", style={"color": "#64748b", "fontSize": "0.85rem"})
            ], style={"display": "flex", "alignItems": "center"})
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "marginBottom": "24px"
        }),

        # Market Indices Bar
        html.Div([
            html.Div(id="market-indices-bar", className="market-indices-bar")
        ], className="section-card mb-4"),

        # Top Movers Section
        dbc.Row([
            # Top Gainers
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-arrow-trend-up", style={"marginRight": "10px", "color": "#10b981"}),
                            "Top Gainers"
                        ], className="section-title")
                    ], className="section-header"),
                    html.Div(id="top-gainers-list", className="section-body")
                ], className="section-card h-100")
            ], xs=12, md=6, lg=4, className="mb-4"),
            
            # Top Losers
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-arrow-trend-down", style={"marginRight": "10px", "color": "#ef4444"}),
                            "Top Losers"
                        ], className="section-title")
                    ], className="section-header"),
                    html.Div(id="top-losers-list", className="section-body")
                ], className="section-card h-100")
            ], xs=12, md=6, lg=4, className="mb-4"),
            
            # Hot Signals (AI Picks)
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-bolt", style={"marginRight": "10px", "color": "#f59e0b"}),
                            "AI Hot Picks"
                        ], className="section-title"),
                        html.Span("High Confidence", style={
                            "fontSize": "0.7rem",
                            "background": "rgba(245, 158, 11, 0.15)",
                            "color": "#fbbf24",
                            "padding": "4px 8px",
                            "borderRadius": "4px"
                        })
                    ], className="section-header"),
                    html.Div(id="hot-signals-list", className="section-body")
                ], className="section-card h-100")
            ], xs=12, lg=4, className="mb-4"),
        ]),

        # Advanced Filters Section
        html.Div([
            html.Div([
                html.H5([
                    html.I(className="fas fa-filter", style={"marginRight": "10px", "color": "#3b82f6"}),
                    "Filter Stocks"
                ], className="section-title")
            ], className="section-header"),
            html.Div([
                dbc.Row([
                    # Search Group (Combined Input + Button)
                    dbc.Col([
                        html.Label("Search", className="filter-label"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="stock-search-input",
                                placeholder="Symbol or name...",
                                type="text",
                                className="filter-input"
                            ),
                            dbc.Button([
                                html.I(className="fas fa-search", style={"marginRight": "8px"}),
                                "Search"
                            ], id="stock-search-btn", className="filter-search-btn")
                        ])
                    ], xs=12, md=4, className="mb-3"),
                    
                    # Signal Type Filter
                    dbc.Col([
                        html.Label("Signal Type", className="filter-label"),
                        dbc.Select(
                            id="signal-filter",
                            options=[
                                {"label": "All Signals", "value": ""},
                                {"label": "🟢 Strong Buy", "value": "STRONG_BUY"},
                                {"label": "🟢 Buy", "value": "BUY"},
                                {"label": "🟡 Hold", "value": "HOLD"},
                                {"label": "🔴 Sell", "value": "SELL"},
                                {"label": "🔴 Strong Sell", "value": "STRONG_SELL"},
                            ],
                            value="",
                            className="filter-select"
                        )
                    ], xs=6, md=2, className="mb-3"),
                    
                    # Sector Filter
                    dbc.Col([
                        html.Label("Sector", className="filter-label"),
                        dbc.Select(
                            id="sector-filter",
                            options=[
                                {"label": "All Sectors", "value": ""},
                                {"label": "Technology", "value": "Technology"},
                                {"label": "Healthcare", "value": "Healthcare"},
                                {"label": "Financial", "value": "Financial Services"},
                                {"label": "Consumer", "value": "Consumer Cyclical"},
                                {"label": "Energy", "value": "Energy"},
                                {"label": "Industrial", "value": "Industrials"},
                            ],
                            value="",
                            className="filter-select"
                        )
                    ], xs=6, md=2, className="mb-3"),
                    
                    # Sort By
                    dbc.Col([
                        html.Label("Sort By", className="filter-label"),
                        dbc.Select(
                            id="sort-filter",
                            options=[
                                {"label": "AI Score (High→Low)", "value": "score_desc"},
                                {"label": "AI Score (Low→High)", "value": "score_asc"},
                                {"label": "Price Change (%)", "value": "change_desc"},
                                {"label": "Volume", "value": "volume_desc"},
                                {"label": "Name (A-Z)", "value": "alpha"},
                            ],
                            value="score_desc",
                            className="filter-select"
                        )
                    ], xs=6, md=2, className="mb-3"),
                    
                    # Min Score Filter
                    dbc.Col([
                        html.Label("Min AI Score", className="filter-label"),
                        dbc.Select(
                            id="min-score-filter",
                            options=[
                                {"label": "Any Score", "value": "0"},
                                {"label": "> 50%", "value": "50"},
                                {"label": "> 60%", "value": "60"},
                                {"label": "> 70%", "value": "70"},
                                {"label": "> 80%", "value": "80"},
                            ],
                            value="0",
                            className="filter-select"
                        )
                    ], xs=6, md=2, className="mb-3"),
                ])
            ], className="section-body")
        ], className="section-card mb-4"),

        # Results Summary
        html.Div([
            html.Div(id="results-summary", style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "16px"
            }),
        ]),

        # Stock Cards Grid
        html.Div(id="stocks-grid", className="mb-4"),
        
        # Load More / Pagination
        html.Div([
            dbc.Button([
                html.I(className="fas fa-plus", style={"marginRight": "8px"}),
                "Load More Stocks"
            ], id="load-more-stocks", className="load-more-btn")
        ], className="text-center mb-4"),
        
        # Hidden stores
        dcc.Store(id="stocks-page-store", data={"page": 1, "per_page": 12}),
        
        # Refresh interval for live updates
        dcc.Interval(id="stocks-refresh-interval", interval=60*1000, n_intervals=0),
    ])


def create_premium_stock_card(
    symbol: str,
    name: str,
    sector: str,
    price: float = 0,
    change_pct: float = 0,
    volume: str = "0",
    ai_score: float = 0,
    signal: str = "HOLD",
    in_watchlist: bool = False,
    mini_chart_data: list = None
):
    """Create a premium stock card with real trading data."""
    
    # Determine signal styling
    signal_config = {
        "STRONG_BUY": {"color": "#10b981", "bg": "rgba(16, 185, 129, 0.15)", "icon": "fas fa-arrow-up", "text": "Strong Buy"},
        "BUY": {"color": "#34d399", "bg": "rgba(52, 211, 153, 0.15)", "icon": "fas fa-arrow-up", "text": "Buy"},
        "HOLD": {"color": "#f59e0b", "bg": "rgba(245, 158, 11, 0.15)", "icon": "fas fa-minus", "text": "Hold"},
        "SELL": {"color": "#f87171", "bg": "rgba(248, 113, 113, 0.15)", "icon": "fas fa-arrow-down", "text": "Sell"},
        "STRONG_SELL": {"color": "#ef4444", "bg": "rgba(239, 68, 68, 0.15)", "icon": "fas fa-arrow-down", "text": "Strong Sell"},
    }
    
    sig = signal_config.get(signal, signal_config["HOLD"])
    change_color = "#10b981" if change_pct >= 0 else "#ef4444"
    change_icon = "fa-caret-up" if change_pct >= 0 else "fa-caret-down"
    
    # Score color
    score_pct = ai_score * 100
    if score_pct >= 70:
        score_color = "#10b981"
    elif score_pct >= 50:
        score_color = "#f59e0b"
    else:
        score_color = "#ef4444"
    
    # Watchlist star
    star_class = "fas fa-star" if in_watchlist else "far fa-star"
    star_color = "#f59e0b" if in_watchlist else "#64748b"
    
    # Sector badge colors
    sector_colors = {
        "Technology": "#3b82f6",
        "Healthcare": "#10b981",
        "Financial Services": "#8b5cf6",
        "Consumer Cyclical": "#f59e0b",
        "Energy": "#ef4444",
        "Industrials": "#64748b",
    }
    sector_color = sector_colors.get(sector, "#64748b")
    
    return html.Div([
        # Card Header
        html.Div([
            html.Div([
                html.Span(symbol, style={
                    "fontSize": "1.25rem",
                    "fontWeight": "700",
                    "color": "#fff"
                }),
                html.I(
                    id={"type": "watchlist-star", "symbol": symbol},
                    className=star_class,
                    n_clicks=0,
                    style={"color": star_color, "cursor": "pointer", "fontSize": "1rem"}
                )
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
            html.Div(name, style={
                "fontSize": "0.8rem",
                "color": "#94a3b8",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
                "maxWidth": "180px"
            }),
        ], style={"marginBottom": "12px"}),
        
        # Price Section
        html.Div([
            html.Div([
                html.Span(f"${price:.2f}" if price else "—", style={
                    "fontSize": "1.5rem",
                    "fontWeight": "700",
                    "color": "#fff"
                }),
                html.Div([
                    html.I(className=f"fas {change_icon}", style={"marginRight": "4px"}),
                    html.Span(f"{abs(change_pct):.2f}%")
                ], style={
                    "color": change_color,
                    "fontSize": "0.9rem",
                    "fontWeight": "600",
                    "marginLeft": "10px"
                })
            ], style={"display": "flex", "alignItems": "baseline"}),
        ], style={"marginBottom": "12px"}),
        
        # Mini Chart Placeholder
        html.Div([
            html.Div(style={
                "height": "40px",
                "background": f"linear-gradient(to right, transparent, {change_color}20)",
                "borderRadius": "4px",
                "position": "relative",
                "overflow": "hidden"
            }, children=[
                # Simulated trend line
                html.Div(style={
                    "position": "absolute",
                    "bottom": "50%" if change_pct >= 0 else "30%",
                    "left": "0",
                    "right": "0",
                    "height": "2px",
                    "background": f"linear-gradient(to right, transparent, {change_color})",
                    "transform": f"rotate({'−2deg' if change_pct >= 0 else '2deg'})"
                })
            ])
        ], style={"marginBottom": "12px"}),
        
        # AI Score & Signal
        html.Div([
            # AI Score
            html.Div([
                html.Div("AI Score", style={"fontSize": "0.7rem", "color": "#64748b", "marginBottom": "4px"}),
                html.Div([
                    html.Div(style={
                        "width": "100%",
                        "height": "6px",
                        "background": "rgba(255,255,255,0.1)",
                        "borderRadius": "3px",
                        "overflow": "hidden"
                    }, children=[
                        html.Div(style={
                            "width": f"{score_pct}%",
                            "height": "100%",
                            "background": score_color,
                            "borderRadius": "3px"
                        })
                    ]),
                    html.Span(f"{score_pct:.0f}%", style={
                        "fontSize": "0.85rem",
                        "fontWeight": "600",
                        "color": score_color,
                        "marginLeft": "8px"
                    })
                ], style={"display": "flex", "alignItems": "center"})
            ], style={"flex": "1", "marginRight": "16px"}),
            
            # Signal Badge
            html.Div([
                html.I(className=sig["icon"], style={"marginRight": "4px"}),
                sig["text"]
            ], style={
                "background": sig["bg"],
                "color": sig["color"],
                "padding": "6px 12px",
                "borderRadius": "6px",
                "fontSize": "0.75rem",
                "fontWeight": "600"
            })
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
        
        # Metrics Row
        html.Div([
            html.Div([
                html.Span("Vol", style={"color": "#64748b", "fontSize": "0.7rem"}),
                html.Span(volume, style={"color": "#fff", "fontSize": "0.8rem", "fontWeight": "500", "marginLeft": "4px"})
            ]),
            html.Div([
                html.Span(sector[:10], style={
                    "background": f"{sector_color}20",
                    "color": sector_color,
                    "padding": "2px 8px",
                    "borderRadius": "4px",
                    "fontSize": "0.7rem"
                })
            ])
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "paddingTop": "12px",
            "borderTop": "1px solid rgba(255,255,255,0.05)"
        }),
        
        # Analyze Button
        html.A([
            html.I(className="fas fa-chart-line", style={"marginRight": "8px"}),
            "Deep Analysis"
        ], href=f"/analysis?symbol={symbol}", className="stock-card-btn")
        
    ], className="premium-stock-card")


def create_mover_item(symbol: str, name: str, price: float, change_pct: float, volume: str = "0"):
    """Create a compact mover item for gainers/losers lists."""
    change_color = "#10b981" if change_pct >= 0 else "#ef4444"
    change_icon = "fa-caret-up" if change_pct >= 0 else "fa-caret-down"
    
    return html.A([
        html.Div([
            html.Div([
                html.Span(symbol, style={"fontWeight": "600", "color": "#fff", "marginRight": "8px"}),
                html.Span(name[:15] + "..." if len(name) > 15 else name, style={"color": "#64748b", "fontSize": "0.8rem"})
            ]),
            html.Div([
                html.Span(f"${price:.2f}", style={"color": "#fff", "marginRight": "12px", "fontSize": "0.9rem"}),
                html.Div([
                    html.I(className=f"fas {change_icon}", style={"marginRight": "4px"}),
                    html.Span(f"{abs(change_pct):.2f}%")
                ], style={"color": change_color, "fontWeight": "600", "fontSize": "0.9rem"})
            ], style={"display": "flex", "alignItems": "center"})
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"})
    ], href=f"/analysis?symbol={symbol}", className="mover-item")


def create_hot_signal_item(symbol: str, signal: str, score: float, price: float, change_pct: float):
    """Create a hot signal item for AI picks."""
    signal_config = {
        "STRONG_BUY": {"color": "#10b981", "text": "Strong Buy"},
        "BUY": {"color": "#34d399", "text": "Buy"},
        "STRONG_SELL": {"color": "#ef4444", "text": "Strong Sell"},
        "SELL": {"color": "#f87171", "text": "Sell"},
    }
    sig = signal_config.get(signal, {"color": "#f59e0b", "text": "Hold"})
    score_pct = score * 100
    
    return html.A([
        html.Div([
            html.Div([
                html.Span(symbol, style={"fontWeight": "700", "color": "#fff", "fontSize": "1rem"}),
                html.Span(sig["text"], style={
                    "background": f"{sig['color']}20",
                    "color": sig["color"],
                    "padding": "2px 8px",
                    "borderRadius": "4px",
                    "fontSize": "0.7rem",
                    "marginLeft": "8px"
                })
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Div([
                    html.Span("Score: ", style={"color": "#64748b", "fontSize": "0.8rem"}),
                    html.Span(f"{score_pct:.0f}%", style={"color": "#10b981", "fontWeight": "600"})
                ]),
                html.Span(f"${price:.2f}", style={"color": "#94a3b8", "fontSize": "0.85rem", "marginLeft": "16px"})
            ], style={"display": "flex", "alignItems": "center"})
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"})
    ], href=f"/analysis?symbol={symbol}", className="mover-item hot-signal")


def create_market_index(name: str, value: float, change: float, change_pct: float):
    """Create a market index display."""
    color = "#10b981" if change >= 0 else "#ef4444"
    icon = "fa-caret-up" if change >= 0 else "fa-caret-down"
    
    return html.Div([
        html.Div(name, style={"color": "#64748b", "fontSize": "0.75rem", "marginBottom": "2px"}),
        html.Div([
            html.Span(f"{value:,.2f}", style={"color": "#fff", "fontWeight": "600", "marginRight": "8px"}),
            html.Span([
                html.I(className=f"fas {icon}", style={"marginRight": "4px"}),
                f"{abs(change_pct):.2f}%"
            ], style={"color": color, "fontSize": "0.85rem"})
        ])
    ], className="market-index-item")


# Legacy functions for compatibility
def create_stock_card(symbol: str, name: str, sector: str, in_watchlist: bool = False, signal: str = None):
    """Legacy stock card - redirects to premium version."""
    return create_premium_stock_card(
        symbol=symbol,
        name=name,
        sector=sector,
        price=0,
        change_pct=0,
        volume="0",
        ai_score=0.5,
        signal=signal or "HOLD",
        in_watchlist=in_watchlist
    )


def create_watchlist_widget():
    """Create the watchlist widget for the overview page sidebar."""
    return html.Div([
        html.Div([
            html.H5("📋 My Watchlist", className="mb-0"),
            dcc.Link(
                html.I(className="fas fa-plus"),
                href="/stocks",
                className="text-primary"
            )
        ], className="d-flex justify-content-between align-items-center mb-3"),
        
        html.Div(id="watchlist-items"),
        
        html.Div([
            html.P("No stocks in your watchlist yet.", className="text-muted mb-2"),
            dcc.Link("Browse stocks →", href="/stocks", className="text-primary")
        ], id="watchlist-empty", style={"display": "none"})
        
    ], className="glass-card p-3")


def create_watchlist_item(symbol: str, name: str, signal: str = None, score: float = None):
    """Create a single watchlist item row."""
    signal_colors = {
        "STRONG_BUY": "success",
        "BUY": "success",
        "HOLD": "warning", 
        "SELL": "danger",
        "STRONG_SELL": "danger"
    }
    signal_color = signal_colors.get(signal, "secondary")
    
    return html.Div([
        html.Div([
            html.Span(symbol, className="fw-bold"),
            html.Small(f" {name[:15]}..." if len(name) > 15 else f" {name}", className="text-muted")
        ]),
        html.Div([
            dbc.Badge(signal or "—", color=signal_color, className="me-2"),
            html.Small(f"{int(score*100)}%" if score else "—", className="text-muted")
        ])
    ], className="d-flex justify-content-between align-items-center py-2 border-bottom border-secondary")


def create_onboarding_page():
    """Create the onboarding wizard for first-time users."""
    return html.Div([
        html.Div(className="auth-background"),
        
        html.Div([
            html.Div([
                html.Div(className="onboarding-progress-dot active"),
                html.Div(className="onboarding-progress-line"),
                html.Div(className="onboarding-progress-dot"),
                html.Div(className="onboarding-progress-line"),
                html.Div(className="onboarding-progress-dot"),
            ], className="onboarding-progress mb-5"),
            
            html.Div([
                html.Div([
                    html.Div("TS", className="auth-logo-icon mb-3"),
                    html.H2("Welcome to Trading Signals Pro!", className="text-white mb-3"),
                    html.P(
                        "Let's set up your personalized trading dashboard. "
                        "This will only take a minute.",
                        className="text-muted mb-4"
                    ),
                    dbc.Button("Get Started →", id="onboarding-start", color="primary", size="lg")
                ], className="text-center")
            ], id="onboarding-step-1", className="glass-card p-5"),
            
            html.Div([
                html.H4("What sectors interest you?", className="text-white mb-4"),
                html.P("Select the sectors you want to track:", className="text-muted mb-4"),
                
                dbc.Checklist(
                    id="onboarding-sectors",
                    options=[
                        {"label": "🖥️ Technology", "value": "Technology"},
                        {"label": "💊 Healthcare", "value": "Healthcare"},
                        {"label": "🏦 Financial Services", "value": "Financial Services"},
                        {"label": "🛒 Consumer", "value": "Consumer Cyclical"},
                        {"label": "📡 Communication", "value": "Communication Services"},
                        {"label": "⚡ Energy", "value": "Energy"},
                    ],
                    value=["Technology"],
                    className="sector-checklist mb-4",
                    inline=True
                ),
                
                html.Div([
                    dbc.Button("← Back", id="onboarding-back-1", color="secondary", outline=True, className="me-2"),
                    dbc.Button("Next →", id="onboarding-next-1", color="primary")
                ], className="text-end")
            ], id="onboarding-step-2", className="glass-card p-4", style={"display": "none"}),
            
            html.Div([
                html.H4("Pick some stocks to track", className="text-white mb-4"),
                html.P("Select at least 3 stocks to add to your watchlist:", className="text-muted mb-4"),
                
                dbc.Checklist(
                    id="onboarding-stocks",
                    options=[
                        {"label": "AAPL - Apple", "value": "AAPL"},
                        {"label": "MSFT - Microsoft", "value": "MSFT"},
                        {"label": "GOOGL - Alphabet", "value": "GOOGL"},
                        {"label": "AMZN - Amazon", "value": "AMZN"},
                        {"label": "NVDA - NVIDIA", "value": "NVDA"},
                        {"label": "META - Meta", "value": "META"},
                        {"label": "TSLA - Tesla", "value": "TSLA"},
                        {"label": "JPM - JPMorgan", "value": "JPM"},
                    ],
                    value=["AAPL", "MSFT", "GOOGL"],
                    className="stock-checklist mb-4"
                ),
                
                html.Div([
                    dbc.Button("← Back", id="onboarding-back-2", color="secondary", outline=True, className="me-2"),
                    dbc.Button("Complete Setup ✓", id="onboarding-complete", color="success")
                ], className="text-end")
            ], id="onboarding-step-3", className="glass-card p-4", style={"display": "none"}),
            
        ], className="onboarding-container")
        
    ], className="onboarding-page")
