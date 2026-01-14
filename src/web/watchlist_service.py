"""
Watchlist Service.

Handles user watchlist operations:
- Add/remove stocks from watchlist
- Get user's watchlist
- Get popular stocks
- Search stocks
"""

from typing import List, Dict, Optional
from flask import session as flask_session

from src.logging_config import get_logger
from src.data.persistence import get_database

logger = get_logger(__name__)

# Pre-defined list of popular stocks (can be expanded from database)
POPULAR_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services"},
    {"symbol": "V", "name": "Visa Inc.", "sector": "Financial Services"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
    {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consumer Defensive"},
    {"symbol": "PG", "name": "Procter & Gamble Co.", "sector": "Consumer Defensive"},
    {"symbol": "UNH", "name": "UnitedHealth Group Inc.", "sector": "Healthcare"},
    {"symbol": "HD", "name": "Home Depot Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "MA", "name": "Mastercard Inc.", "sector": "Financial Services"},
    {"symbol": "DIS", "name": "Walt Disney Co.", "sector": "Communication Services"},
    {"symbol": "PYPL", "name": "PayPal Holdings Inc.", "sector": "Financial Services"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Communication Services"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
    {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
]

SECTORS = [
    "Technology",
    "Healthcare", 
    "Financial Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Communication Services",
    "Industrials",
    "Energy",
    "Utilities",
    "Real Estate",
    "Basic Materials",
]


def get_current_user_id() -> Optional[str]:
    """Get current user ID from session."""
    return flask_session.get('user_id')


def get_all_stocks(sector: str = None, search: str = None) -> List[Dict]:
    """
    Get all available stocks, optionally filtered.
    
    Args:
        sector: Filter by sector
        search: Search by symbol or name
    
    Returns:
        List of stock dictionaries
    """
    stocks = POPULAR_STOCKS.copy()
    
    # Try to get from database first
    try:
        db = get_database()
        with db.get_session() as session:
            from src.data.models import AssetMetadata
            
            query = session.query(AssetMetadata).filter(AssetMetadata.is_active == True)
            
            if sector:
                query = query.filter(AssetMetadata.sector == sector)
            if search:
                search_term = f"%{search.upper()}%"
                query = query.filter(
                    (AssetMetadata.symbol.ilike(search_term)) |
                    (AssetMetadata.name.ilike(search_term))
                )
            
            db_stocks = query.limit(100).all()
            
            if db_stocks:
                stocks = [
                    {"symbol": s.symbol, "name": s.name, "sector": s.sector}
                    for s in db_stocks
                ]
    except Exception as e:
        logger.warning(f"Could not fetch stocks from database: {e}")
    
    # Apply filters to fallback list
    if sector:
        stocks = [s for s in stocks if s.get("sector") == sector]
    if search:
        search = search.upper()
        stocks = [s for s in stocks if search in s["symbol"] or search in s["name"].upper()]
    
    return stocks


def get_user_watchlist(user_id: str = None) -> List[Dict]:
    """
    Get user's watchlist with signal data.
    
    Returns:
        List of watchlist items with stock info and latest signals
    """
    user_id = user_id or get_current_user_id()
    if not user_id:
        return []
    
    try:
        db = get_database()
        with db.get_session() as session:
            from src.auth.models import UserWatchlist
            
            watchlist_items = session.query(UserWatchlist).filter(
                UserWatchlist.user_id == user_id
            ).all()
            
            result = []
            for item in watchlist_items:
                # Get stock info
                stock_info = next(
                    (s for s in POPULAR_STOCKS if s["symbol"] == item.symbol),
                    {"symbol": item.symbol, "name": item.symbol, "sector": "Unknown"}
                )
                
                # Get latest signal for this stock
                signals = db.get_latest_signals(symbol=item.symbol, limit=1)
                signal_data = {}
                if signals:
                    signal = signals[0] if isinstance(signals, list) else signals
                    if hasattr(signal, '__dict__'):
                        signal_data = {
                            "signal_type": getattr(signal, 'signal_type', 'HOLD'),
                            "confluence_score": getattr(signal, 'confluence_score', 0.5),
                            "technical_score": getattr(signal, 'technical_score', 0.5),
                        }
                
                result.append({
                    "symbol": item.symbol,
                    "name": stock_info.get("name", item.symbol),
                    "sector": stock_info.get("sector", "Unknown"),
                    "added_at": item.added_at,
                    "alerts_enabled": item.alerts_enabled,
                    "notes": item.notes,
                    **signal_data
                })
            
            return result
            
    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        return []


def add_to_watchlist(symbol: str, user_id: str = None) -> bool:
    """Add a stock to user's watchlist."""
    user_id = user_id or get_current_user_id()
    if not user_id:
        return False
    
    try:
        db = get_database()
        with db.get_session() as session:
            from src.auth.models import UserWatchlist
            
            # Check if already exists
            existing = session.query(UserWatchlist).filter(
                UserWatchlist.user_id == user_id,
                UserWatchlist.symbol == symbol.upper()
            ).first()
            
            if existing:
                return True  # Already in watchlist
            
            # Add new entry
            watchlist_item = UserWatchlist(
                user_id=user_id,
                symbol=symbol.upper()
            )
            session.add(watchlist_item)
            
            logger.info(f"Added {symbol} to watchlist for user {user_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error adding to watchlist: {e}")
        return False


def remove_from_watchlist(symbol: str, user_id: str = None) -> bool:
    """Remove a stock from user's watchlist."""
    user_id = user_id or get_current_user_id()
    if not user_id:
        return False
    
    try:
        db = get_database()
        with db.get_session() as session:
            from src.auth.models import UserWatchlist
            
            deleted = session.query(UserWatchlist).filter(
                UserWatchlist.user_id == user_id,
                UserWatchlist.symbol == symbol.upper()
            ).delete()
            
            if deleted:
                logger.info(f"Removed {symbol} from watchlist for user {user_id}")
            return deleted > 0
            
    except Exception as e:
        logger.error(f"Error removing from watchlist: {e}")
        return False


def is_in_watchlist(symbol: str, user_id: str = None) -> bool:
    """Check if a stock is in user's watchlist."""
    user_id = user_id or get_current_user_id()
    if not user_id:
        return False
    
    try:
        db = get_database()
        with db.get_session() as session:
            from src.auth.models import UserWatchlist
            
            exists = session.query(UserWatchlist).filter(
                UserWatchlist.user_id == user_id,
                UserWatchlist.symbol == symbol.upper()
            ).first()
            
            return exists is not None
            
    except Exception as e:
        logger.error(f"Error checking watchlist: {e}")
        return False


def get_user_preferences(user_id: str = None) -> Optional[Dict]:
    """Get user preferences and onboarding state."""
    user_id = user_id or get_current_user_id()
    if not user_id:
        return None
    
    try:
        db = get_database()
        with db.get_session() as session:
            from src.auth.models import UserPreferences
            
            prefs = session.query(UserPreferences).filter(
                UserPreferences.user_id == user_id
            ).first()
            
            if prefs:
                return {
                    "onboarding_completed": prefs.onboarding_completed,
                    "preferred_sectors": prefs.preferred_sectors or [],
                    "risk_tolerance": prefs.risk_tolerance,
                    "theme": prefs.theme,
                }
            
            return None
            
    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        return None


def complete_onboarding(user_id: str = None, sectors: List[str] = None, symbols: List[str] = None) -> bool:
    """
    Complete user onboarding by saving preferences and initial watchlist.
    """
    user_id = user_id or get_current_user_id()
    if not user_id:
        return False
    
    try:
        db = get_database()
        with db.get_session() as session:
            from src.auth.models import UserPreferences, UserWatchlist
            
            # Create or update preferences
            prefs = session.query(UserPreferences).filter(
                UserPreferences.user_id == user_id
            ).first()
            
            if not prefs:
                prefs = UserPreferences(user_id=user_id)
                session.add(prefs)
            
            prefs.onboarding_completed = True
            prefs.preferred_sectors = sectors or []
            
            # Add initial stocks to watchlist
            if symbols:
                for symbol in symbols:
                    existing = session.query(UserWatchlist).filter(
                        UserWatchlist.user_id == user_id,
                        UserWatchlist.symbol == symbol.upper()
                    ).first()
                    
                    if not existing:
                        session.add(UserWatchlist(
                            user_id=user_id,
                            symbol=symbol.upper()
                        ))
            
            logger.info(f"Completed onboarding for user {user_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error completing onboarding: {e}")
        return False
