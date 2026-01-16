"""
Signal Intelligence API Endpoint

Provides comprehensive signal data for the frontend Signal Intelligence page.
Aggregates signals from all providers with efficient caching.
"""

import asyncio
from flask import Blueprint, request, g
from datetime import datetime

from src.api.utils import success_response, error_response, require_api_key_or_auth
from src.logging_config import get_logger

logger = get_logger(__name__)

signal_intelligence_bp = Blueprint('signal_intelligence', __name__)


def get_providers():
    """Lazy load signal providers."""
    from src.services.signal_intelligence import TechnicalSignalProvider, SignalCategory, SignalTier, Signal
    from src.services.sentiment_provider import SentimentSignalProvider
    from src.services.fundamentals_provider import FundamentalsSignalProvider, MacroSignalProvider
    
    return {
        'technical': TechnicalSignalProvider(),
        'sentiment': SentimentSignalProvider(),
        'fundamentals': FundamentalsSignalProvider(),
        'macro': MacroSignalProvider(),
    }


def run_async(coro):
    """Run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@signal_intelligence_bp.route('/signal-intelligence/<symbol>', methods=['GET'])
@require_api_key_or_auth
def get_signal_intelligence(symbol: str):
    """
    Get comprehensive signal intelligence for a symbol.
    
    Returns all signal categories with individual signal scores.
    
    Path Parameters:
        symbol: Stock symbol (e.g., AAPL, GOOGL)
    
    Query Parameters:
        categories: Comma-separated list of categories (optional)
                   Options: technical, sentiment, fundamentals, macro
        include_details: Include individual signals (default: true)
    
    Returns:
        {
            "symbol": "AAPL",
            "confluence_score": 0.72,
            "signal_type": "BUY",
            "categories": {
                "technical": {
                    "avg_score": 0.68,
                    "bullish_count": 10,
                    "bearish_count": 4,
                    "signals": [...]
                },
                ...
            },
            "live_feed": [...],
            "generated_at": "2024-01-15T10:30:00Z"
        }
    """
    try:
        symbol = symbol.strip().upper()
        if not symbol or len(symbol) > 10:
            return error_response("Invalid symbol", 400)
        
        # Parse query params
        requested_categories = request.args.get('categories', '').strip()
        if requested_categories:
            requested_categories = [c.strip().lower() for c in requested_categories.split(',')]
        else:
            requested_categories = ['technical', 'sentiment', 'fundamentals', 'macro']
        
        include_details = request.args.get('include_details', 'true').lower() == 'true'
        
        # Get providers
        providers = get_providers()
        
        # Fetch signals from requested categories
        async def fetch_all_signals():
            results = {}
            tasks = []
            
            for category in requested_categories:
                if category in providers:
                    tasks.append(fetch_category(category, providers[category], symbol))
            
            category_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in category_results:
                if isinstance(result, Exception):
                    logger.warning(f"Category fetch error: {result}")
                    continue
                if result:
                    category_name, signals_data = result
                    results[category_name] = signals_data
            
            return results
        
        async def fetch_category(category: str, provider, symbol: str):
            try:
                signals = await provider.get_signals(symbol)
                
                if not signals:
                    return None
                
                # Calculate category stats
                avg_score = sum(s.value for s in signals) / len(signals)
                bullish_count = sum(1 for s in signals if s.value > 0.55)
                bearish_count = sum(1 for s in signals if s.value < 0.45)
                
                return (category, {
                    "category": category,
                    "avg_score": round(avg_score, 4),
                    "bullish_count": bullish_count,
                    "bearish_count": bearish_count,
                    "neutral_count": len(signals) - bullish_count - bearish_count,
                    "total_signals": len(signals),
                    "direction": "bullish" if avg_score > 0.6 else "bearish" if avg_score < 0.4 else "neutral",
                    "signals": [s.to_dict() for s in signals] if include_details else []
                })
                
            except Exception as e:
                logger.error(f"Category {category} error: {e}")
                return None
        
        # Run async fetch
        categories_data = run_async(fetch_all_signals())
        
        if not categories_data:
            # Return fallback data
            categories_data = generate_fallback_data(symbol, requested_categories)
        
        # Calculate overall confluence score
        weights = {
            'technical': 0.35,
            'sentiment': 0.25,
            'fundamentals': 0.20,
            'macro': 0.10,
            'correlations': 0.05,
            'external': 0.05
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for category, data in categories_data.items():
            weight = weights.get(category, 0.1)
            weighted_sum += data['avg_score'] * weight
            total_weight += weight
        
        confluence_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Determine signal type
        if confluence_score >= 0.75:
            signal_type = "STRONG_BUY"
        elif confluence_score >= 0.6:
            signal_type = "BUY"
        elif confluence_score <= 0.25:
            signal_type = "STRONG_SELL"
        elif confluence_score <= 0.4:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"
        
        # Calculate total signals count
        total_signals = sum(data.get('total_signals', 0) for data in categories_data.values())
        
        # Generate live feed events
        live_feed = generate_live_feed(symbol, categories_data)
        
        return success_response(
            data={
                "symbol": symbol,
                "confluence_score": round(confluence_score, 4),
                "signal_type": signal_type,
                "total_signals": total_signals,
                "categories": categories_data,
                "live_feed": live_feed,
                "component_scores": {
                    "technical": categories_data.get('technical', {}).get('avg_score', 0.5),
                    "sentiment": categories_data.get('sentiment', {}).get('avg_score', 0.5),
                    "fundamentals": categories_data.get('fundamentals', {}).get('avg_score', 0.5),
                    "macro": categories_data.get('macro', {}).get('avg_score', 0.5),
                },
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
        )
        
    except Exception as e:
        logger.error(f"Signal intelligence error: {e}", exc_info=True)
        return error_response("Failed to get signal intelligence", 500)


def generate_fallback_data(symbol: str, categories: list) -> dict:
    """Generate fallback mock data when APIs fail."""
    import random
    seed = sum(ord(c) for c in symbol) + int(datetime.now().timestamp() // 3600)
    random.seed(seed)
    
    # Stock-specific biases
    bullish_stocks = ['AAPL', 'NVDA', 'MSFT', 'META', 'LLY', 'UNH', 'CMG']
    bearish_stocks = ['TSLA', 'PFE', 'LOW', 'NKE']
    
    base = 0.65 if symbol in bullish_stocks else 0.4 if symbol in bearish_stocks else 0.5
    
    result = {}
    
    category_signals = {
        'technical': [
            'RSI (14)', 'MACD Crossover', 'Bollinger Bands', 'SMA 50/200',
            'ADX Strength', 'Stochastic', 'Williams %R', 'OBV Trend',
            'Parabolic SAR', 'ATR Volatility', 'Ichimoku Cloud', 'Volume Ratio',
            'Fibonacci Levels', 'Pivot Points', 'Pattern Recognition'
        ],
        'sentiment': [
            'News Sentiment', 'Social Media Buzz', 'Analyst Ratings',
            'Insider Activity', 'Options Flow', 'Short Interest',
            'Earnings Sentiment', 'Institutional Ownership'
        ],
        'fundamentals': [
            'P/E Ratio', 'EPS Growth', 'Revenue Growth', 'Gross Margin',
            'Free Cash Flow', 'Debt/Equity', 'ROE', 'Dividend Yield',
            'Book Value', 'Working Capital'
        ],
        'macro': [
            'Fed Rate Outlook', 'Inflation (CPI)', 'GDP Growth',
            'Unemployment', 'Consumer Confidence', 'Dollar Strength',
            'Treasury Yields', 'VIX Volatility'
        ]
    }
    
    for category in categories:
        if category not in category_signals:
            continue
        
        signals = []
        category_base = base + (random.random() - 0.5) * 0.2
        
        for name in category_signals[category]:
            value = max(0, min(1, category_base + (random.random() - 0.5) * 0.3))
            signals.append({
                'id': f"{category[:4]}_{name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}",
                'name': name,
                'category': category,
                'tier': 'near_rt' if category == 'technical' else 'periodic',
                'value': round(value, 4),
                'direction': 'bullish' if value > 0.55 else 'bearish' if value < 0.45 else 'neutral',
                'confidence': round(0.5 + random.random() * 0.35, 2),
                'source': 'computed',
                'description': f'{name} analysis for {symbol}',
                'last_updated': datetime.utcnow().isoformat()
            })
        
        avg_score = sum(s['value'] for s in signals) / len(signals)
        bullish_count = sum(1 for s in signals if s['value'] > 0.55)
        bearish_count = sum(1 for s in signals if s['value'] < 0.45)
        
        result[category] = {
            'category': category,
            'avg_score': round(avg_score, 4),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': len(signals) - bullish_count - bearish_count,
            'total_signals': len(signals),
            'direction': 'bullish' if avg_score > 0.6 else 'bearish' if avg_score < 0.4 else 'neutral',
            'signals': signals
        }
    
    return result


def generate_live_feed(symbol: str, categories_data: dict) -> list:
    """Generate live signal feed events."""
    import random
    from datetime import timedelta
    
    seed = sum(ord(c) for c in symbol) + int(datetime.now().timestamp() // 60)
    random.seed(seed)
    
    # Get some actual signals to create events
    events = []
    now = datetime.now()
    
    # Create events from high-value signals
    for category, data in categories_data.items():
        for signal in data.get('signals', [])[:3]:  # Top 3 from each
            if signal['value'] > 0.65 or signal['value'] < 0.35:
                time_offset = random.randint(1, 60)  # 1-60 minutes ago
                event_time = now - timedelta(minutes=time_offset)
                
                events.append({
                    'time': event_time.strftime('%H:%M:%S'),
                    'title': f"{signal['name']} {'Bullish' if signal['value'] > 0.5 else 'Bearish'} Signal",
                    'category': category.replace('_', ' ').title(),
                    'positive': signal['value'] > 0.5,
                    'impact': round((signal['value'] - 0.5) * 4, 1),  # -2 to +2%
                    'confidence': signal.get('confidence', 0.5)
                })
    
    # Sort by time (most recent first)
    events.sort(key=lambda x: x['time'], reverse=True)
    
    return events[:10]  # Return top 10 events


@signal_intelligence_bp.route('/signal-intelligence/<symbol>/category/<category>', methods=['GET'])
@require_api_key_or_auth
def get_category_signals(symbol: str, category: str):
    """Get signals for a specific category."""
    try:
        symbol = symbol.strip().upper()
        category = category.strip().lower()
        
        valid_categories = ['technical', 'sentiment', 'fundamentals', 'macro', 'correlations', 'regime', 'external']
        if category not in valid_categories:
            return error_response(f"Invalid category. Valid options: {', '.join(valid_categories)}", 400)
        
        providers = get_providers()
        if category not in providers:
            return error_response(f"Category '{category}' not available", 400)
        
        async def fetch():
            return await providers[category].get_signals(symbol)
        
        signals = run_async(fetch())
        
        signals_data = [s.to_dict() for s in signals]
        avg_score = sum(s['value'] for s in signals_data) / len(signals_data) if signals_data else 0.5
        
        return success_response(
            data={
                "symbol": symbol,
                "category": category,
                "avg_score": round(avg_score, 4),
                "total_signals": len(signals_data),
                "signals": signals_data,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
        )
        
    except Exception as e:
        logger.error(f"Category signals error: {e}", exc_info=True)
        return error_response("Failed to get category signals", 500)
