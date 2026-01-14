"""
Premium Stock Discovery Callbacks.

Handles:
- Market indices display
- Top movers (gainers/losers)
- AI hot picks
- Advanced stock filtering and search
- Watchlist management
- Real-time price data
"""

from dash import Input, Output, State, html, callback_context, ALL, MATCH
import dash_bootstrap_components as dbc
import dash
import random
from datetime import datetime

from src.logging_config import get_logger
from src.web.stocks import (
    create_premium_stock_card,
    create_mover_item,
    create_hot_signal_item,
    create_market_index,
    create_watchlist_item
)
from src.web.watchlist_service import (
    get_all_stocks,
    get_user_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    is_in_watchlist,
    complete_onboarding,
    POPULAR_STOCKS
)

logger = get_logger(__name__)


# Sample market data (would come from real API)
def get_market_indices():
    """Get market index data."""
    # In production, this would call a real API
    return [
        {"name": "S&P 500", "value": 5234.18, "change": 45.32, "change_pct": 0.87},
        {"name": "NASDAQ", "value": 16428.82, "change": 128.64, "change_pct": 0.79},
        {"name": "DOW", "value": 39412.75, "change": 312.48, "change_pct": 0.80},
        {"name": "VIX", "value": 13.45, "change": -0.82, "change_pct": -5.75},
        {"name": "10Y Treasury", "value": 4.32, "change": 0.03, "change_pct": 0.70},
    ]


def get_stock_data_with_prices():
    """Get stock data with simulated real-time prices."""
    stocks = get_all_stocks()
    
    # Simulated price data (would come from real API)
    price_data = {
        "AAPL": {"price": 185.92, "change_pct": 1.24, "volume": "52.3M", "ai_score": 0.78, "signal": "STRONG_BUY"},
        "MSFT": {"price": 425.22, "change_pct": 0.89, "volume": "18.2M", "ai_score": 0.82, "signal": "STRONG_BUY"},
        "GOOGL": {"price": 175.98, "change_pct": -0.45, "volume": "21.5M", "ai_score": 0.65, "signal": "BUY"},
        "AMZN": {"price": 186.13, "change_pct": 2.15, "volume": "45.1M", "ai_score": 0.71, "signal": "BUY"},
        "TSLA": {"price": 248.50, "change_pct": -2.87, "volume": "98.7M", "ai_score": 0.42, "signal": "HOLD"},
        "NVDA": {"price": 924.79, "change_pct": 3.42, "volume": "38.4M", "ai_score": 0.88, "signal": "STRONG_BUY"},
        "META": {"price": 505.95, "change_pct": 1.67, "volume": "14.2M", "ai_score": 0.75, "signal": "BUY"},
        "JPM": {"price": 198.45, "change_pct": 0.32, "volume": "8.9M", "ai_score": 0.58, "signal": "HOLD"},
        "V": {"price": 282.30, "change_pct": 0.78, "volume": "5.6M", "ai_score": 0.69, "signal": "BUY"},
        "JNJ": {"price": 147.82, "change_pct": -0.23, "volume": "6.2M", "ai_score": 0.52, "signal": "HOLD"},
    }
    
    # Add price data to stocks
    enriched_stocks = []
    for stock in stocks:
        symbol = stock["symbol"]
        if symbol in price_data:
            stock.update(price_data[symbol])
        else:
            # Generate random data for other stocks
            stock.update({
                "price": round(random.uniform(50, 500), 2),
                "change_pct": round(random.uniform(-5, 5), 2),
                "volume": f"{random.randint(1, 50)}.{random.randint(0, 9)}M",
                "ai_score": round(random.uniform(0.35, 0.85), 2),
                "signal": random.choice(["STRONG_BUY", "BUY", "HOLD", "SELL"])
            })
        enriched_stocks.append(stock)
    
    return enriched_stocks


def register_stocks_callbacks(app):
    """Register all stock-related callbacks."""
    
    @app.callback(
        Output("market-indices-bar", "children"),
        Input("stocks-refresh-interval", "n_intervals"),
        prevent_initial_call=False
    )
    def update_market_indices(n):
        """Update market indices display."""
        indices = get_market_indices()
        
        return html.Div([
            create_market_index(idx["name"], idx["value"], idx["change"], idx["change_pct"])
            for idx in indices
        ], style={
            "display": "flex",
            "justifyContent": "space-around",
            "padding": "12px 20px",
            "flexWrap": "wrap",
            "gap": "16px"
        })
    
    @app.callback(
        Output("top-gainers-list", "children"),
        Input("stocks-refresh-interval", "n_intervals"),
        prevent_initial_call=False
    )
    def update_top_gainers(n):
        """Update top gainers list."""
        stocks = get_stock_data_with_prices()
        
        # Sort by positive change
        gainers = sorted(
            [s for s in stocks if s.get("change_pct", 0) > 0],
            key=lambda x: x.get("change_pct", 0),
            reverse=True
        )[:5]
        
        if not gainers:
            return html.Div("No gainers today", style={"color": "#64748b", "padding": "20px", "textAlign": "center"})
        
        return html.Div([
            create_mover_item(
                symbol=s["symbol"],
                name=s["name"],
                price=s.get("price", 0),
                change_pct=s.get("change_pct", 0),
                volume=s.get("volume", "0")
            )
            for s in gainers
        ])
    
    @app.callback(
        Output("top-losers-list", "children"),
        Input("stocks-refresh-interval", "n_intervals"),
        prevent_initial_call=False
    )
    def update_top_losers(n):
        """Update top losers list."""
        stocks = get_stock_data_with_prices()
        
        # Sort by negative change
        losers = sorted(
            [s for s in stocks if s.get("change_pct", 0) < 0],
            key=lambda x: x.get("change_pct", 0)
        )[:5]
        
        if not losers:
            return html.Div("No losers today", style={"color": "#64748b", "padding": "20px", "textAlign": "center"})
        
        return html.Div([
            create_mover_item(
                symbol=s["symbol"],
                name=s["name"],
                price=s.get("price", 0),
                change_pct=s.get("change_pct", 0),
                volume=s.get("volume", "0")
            )
            for s in losers
        ])
    
    @app.callback(
        Output("hot-signals-list", "children"),
        Input("stocks-refresh-interval", "n_intervals"),
        prevent_initial_call=False
    )
    def update_hot_signals(n):
        """Update AI hot picks list."""
        stocks = get_stock_data_with_prices()
        
        # Filter for high-confidence signals
        hot_picks = sorted(
            [s for s in stocks if s.get("ai_score", 0) >= 0.70 and s.get("signal") in ["STRONG_BUY", "BUY"]],
            key=lambda x: x.get("ai_score", 0),
            reverse=True
        )[:5]
        
        if not hot_picks:
            return html.Div("No hot picks right now", style={"color": "#64748b", "padding": "20px", "textAlign": "center"})
        
        return html.Div([
            create_hot_signal_item(
                symbol=s["symbol"],
                signal=s.get("signal", "HOLD"),
                score=s.get("ai_score", 0),
                price=s.get("price", 0),
                change_pct=s.get("change_pct", 0)
            )
            for s in hot_picks
        ])
    
    @app.callback(
        Output("market-time", "children"),
        Input("stocks-refresh-interval", "n_intervals"),
        prevent_initial_call=False
    )
    def update_market_time(n):
        """Update market time display."""
        now = datetime.now()
        return f"Last updated: {now.strftime('%I:%M:%S %p')}"
    
    @app.callback(
        [Output("stocks-grid", "children"),
         Output("results-summary", "children")],
        [Input("stock-search-btn", "n_clicks"),
         Input("signal-filter", "value"),
         Input("sector-filter", "value"),
         Input("sort-filter", "value"),
         Input("min-score-filter", "value"),
         Input("stocks-refresh-interval", "n_intervals")],
        [State("stock-search-input", "value")],
        prevent_initial_call=False
    )
    def update_stocks_grid(n_clicks, signal_filter, sector, sort_by, min_score, n_intervals, search):
        """Update the stock grid based on filters."""
        stocks = get_stock_data_with_prices()
        
        # Apply filters
        filtered = stocks.copy()
        
        # Search filter
        if search:
            search_lower = search.lower()
            filtered = [s for s in filtered if search_lower in s["symbol"].lower() or search_lower in s["name"].lower()]
        
        # Signal filter
        if signal_filter:
            filtered = [s for s in filtered if s.get("signal") == signal_filter]
        
        # Sector filter
        if sector:
            filtered = [s for s in filtered if s.get("sector") == sector]
        
        # Min score filter
        min_score_val = int(min_score) / 100 if min_score else 0
        if min_score_val > 0:
            filtered = [s for s in filtered if s.get("ai_score", 0) >= min_score_val]
        
        # Sort
        if sort_by == "score_desc":
            filtered = sorted(filtered, key=lambda x: x.get("ai_score", 0), reverse=True)
        elif sort_by == "score_asc":
            filtered = sorted(filtered, key=lambda x: x.get("ai_score", 0))
        elif sort_by == "change_desc":
            filtered = sorted(filtered, key=lambda x: x.get("change_pct", 0), reverse=True)
        elif sort_by == "volume_desc":
            # Parse volume string for sorting
            def parse_volume(v):
                v = str(v).replace("M", "").replace("B", "000").replace("K", "")
                try:
                    return float(v)
                except:
                    return 0
            filtered = sorted(filtered, key=lambda x: parse_volume(x.get("volume", "0")), reverse=True)
        elif sort_by == "alpha":
            filtered = sorted(filtered, key=lambda x: x["symbol"])
        
        # Build summary
        total = len(filtered)
        buy_signals = len([s for s in filtered if "BUY" in s.get("signal", "")])
        sell_signals = len([s for s in filtered if "SELL" in s.get("signal", "")])
        avg_score = sum(s.get("ai_score", 0) for s in filtered) / total if total > 0 else 0
        
        summary = html.Div([
            html.Div([
                html.Span(f"{total} stocks", style={"fontWeight": "600", "color": "#fff", "marginRight": "20px"}),
                html.Span([
                    html.Span("🟢 ", style={"fontSize": "0.8rem"}),
                    f"{buy_signals} Buy"
                ], style={"color": "#10b981", "marginRight": "16px"}),
                html.Span([
                    html.Span("🔴 ", style={"fontSize": "0.8rem"}),
                    f"{sell_signals} Sell"
                ], style={"color": "#ef4444", "marginRight": "16px"}),
                html.Span([
                    html.Span("Avg Score: ", style={"color": "#64748b"}),
                    f"{avg_score*100:.0f}%"
                ], style={"color": "#f59e0b"})
            ]),
            html.Div([
                html.Span("View: ", style={"color": "#64748b", "marginRight": "8px"}),
                html.Span("Grid", style={"color": "#3b82f6", "fontWeight": "500"})
            ])
        ])
        
        if not filtered:
            return html.Div([
                html.I(className="fas fa-search fa-3x", style={"color": "#64748b", "marginBottom": "16px"}),
                html.P("No stocks match your criteria", style={"color": "#94a3b8"}),
                html.P("Try adjusting your filters", style={"color": "#64748b", "fontSize": "0.85rem"})
            ], style={"textAlign": "center", "padding": "60px"}), summary
        
        # Create stock cards grid
        cards = []
        for stock in filtered[:12]:  # Limit to 12 for initial load
            in_watchlist = is_in_watchlist(stock["symbol"])
            cards.append(
                dbc.Col([
                    create_premium_stock_card(
                        symbol=stock["symbol"],
                        name=stock["name"],
                        sector=stock.get("sector", "Unknown"),
                        price=stock.get("price", 0),
                        change_pct=stock.get("change_pct", 0),
                        volume=stock.get("volume", "0"),
                        ai_score=stock.get("ai_score", 0.5),
                        signal=stock.get("signal", "HOLD"),
                        in_watchlist=in_watchlist
                    )
                ], xs=12, sm=6, md=4, lg=3, className="mb-4")
            )
        
        return dbc.Row(cards), summary

    @app.callback(
        Output({"type": "watchlist-star", "symbol": MATCH}, "className"),
        Input({"type": "watchlist-star", "symbol": MATCH}, "n_clicks"),
        State({"type": "watchlist-star", "symbol": MATCH}, "className"),
        prevent_initial_call=True
    )
    def toggle_watchlist(n_clicks, current_class):
        """Toggle a stock in/out of watchlist when star is clicked."""
        if not n_clicks:
            return dash.no_update
        
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        
        triggered_id = ctx.triggered_id
        if isinstance(triggered_id, dict):
            symbol = triggered_id.get("symbol")
        else:
            return dash.no_update
        
        if "fas fa-star" in str(current_class):
            remove_from_watchlist(symbol)
            return "far fa-star"
        else:
            add_to_watchlist(symbol)
            return "fas fa-star"

    @app.callback(
        [Output("watchlist-items", "children"),
         Output("watchlist-empty", "style")],
        Input("refresh-interval", "n_intervals"),
        prevent_initial_call=False
    )
    def update_watchlist_widget(n):
        """Update the watchlist widget on the overview page."""
        watchlist = get_user_watchlist()
        
        if not watchlist:
            return [], {"display": "block"}
        
        items = [
            create_watchlist_item(
                symbol=item["symbol"],
                name=item["name"],
                signal=item.get("signal_type"),
                score=item.get("confluence_score")
            )
            for item in watchlist[:10]
        ]
        
        return items, {"display": "none"}

    # Onboarding callbacks
    @app.callback(
        [Output("onboarding-step-1", "style"),
         Output("onboarding-step-2", "style"),
         Output("onboarding-step-3", "style")],
        [Input("onboarding-start", "n_clicks"),
         Input("onboarding-next-1", "n_clicks"),
         Input("onboarding-back-1", "n_clicks"),
         Input("onboarding-back-2", "n_clicks")],
        prevent_initial_call=True
    )
    def navigate_onboarding(start, next1, back1, back2):
        """Navigate between onboarding steps."""
        ctx = callback_context
        if not ctx.triggered:
            return {"display": "block"}, {"display": "none"}, {"display": "none"}
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id == "onboarding-start" or trigger_id == "onboarding-next-1":
            if trigger_id == "onboarding-start":
                return {"display": "none"}, {"display": "block"}, {"display": "none"}
            else:
                return {"display": "none"}, {"display": "none"}, {"display": "block"}
        elif trigger_id == "onboarding-back-1":
            return {"display": "block"}, {"display": "none"}, {"display": "none"}
        elif trigger_id == "onboarding-back-2":
            return {"display": "none"}, {"display": "block"}, {"display": "none"}
        
        return {"display": "block"}, {"display": "none"}, {"display": "none"}

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("onboarding-complete", "n_clicks"),
        [State("onboarding-sectors", "value"),
         State("onboarding-stocks", "value")],
        prevent_initial_call=True
    )
    def complete_onboarding_flow(n_clicks, sectors, stocks):
        """Complete onboarding and redirect to dashboard."""
        if not n_clicks:
            return dash.no_update
        
        success = complete_onboarding(sectors=sectors, symbols=stocks)
        
        if success:
            logger.info("Onboarding completed successfully")
            return "/overview"
        
        return dash.no_update
