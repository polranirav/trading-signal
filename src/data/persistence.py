"""
Database persistence layer with SQLAlchemy.

Provides abstraction for database operations:
- CRUD operations for all models
- Bulk inserts with upsert
- Time-series optimized queries

Usage:
    from src.data.persistence import DatabaseManager
    db = DatabaseManager()
    db.save_candles(df)
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from contextlib import contextmanager

from src.config import settings
from src.logging_config import get_logger
from src.data.models import (
    Base, AssetMetadata, MarketCandle, NewsSentiment,
    TradeSignal, IndicatorCache, BacktestResult
)

# Import auth models to ensure they're registered
try:
    from src.auth.models import User, Subscription, APIKey, SubscriptionLimit
except ImportError:
    # Auth module not yet created, skip
    pass

# Import payment models to ensure they're registered
try:
    from src.payments.models import Payment, Invoice
except ImportError:
    # Payments module not yet created, skip
    pass

logger = get_logger(__name__)


class DatabaseManager:
    """
    Database operations manager.
    
    Provides clean interface for all database operations
    with proper connection handling and error logging.
    """
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.engine = create_engine(
            self.database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db.get_session() as session:
                session.query(...)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            
    def execute_query(self, query: str, params: dict = None):
        """
        Execute raw SQL query.
        
        Args:
            query: SQL query string with :named parameters
            params: Dict of named parameters (e.g., {"user_id": "123"})
            
        Returns:
            List of dicts for SELECT, None for others
        """
        with self.get_session() as session:
            # Handle text query
            stmt = text(query)
            
            # Execute with named parameters
            result = session.execute(stmt, params or {})
            
            # Try to fetch results if available
            try:
                if result.returns_rows:
                    return [dict(row._mapping) for row in result]
            except Exception:
                pass
            
            return None
    
    # ============ ASSET METADATA ============
    
    def get_asset(self, symbol: str) -> Optional[AssetMetadata]:
        """Get asset metadata by symbol."""
        with self.get_session() as session:
            return session.query(AssetMetadata).filter(
                AssetMetadata.symbol == symbol
            ).first()
    
    def get_active_assets(self) -> List[AssetMetadata]:
        """Get all active assets."""
        with self.get_session() as session:
            return session.query(AssetMetadata).filter(
                AssetMetadata.is_active == True
            ).all()
    
    def save_asset(self, asset_data: Dict) -> None:
        """Save or update asset metadata."""
        with self.get_session() as session:
            stmt = insert(AssetMetadata).values(**asset_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol'],
                set_={
                    'name': stmt.excluded.name,
                    'sector': stmt.excluded.sector,
                    'industry': stmt.excluded.industry,
                    'last_updated': datetime.utcnow()
                }
            )
            session.execute(stmt)
    
    # ============ MARKET DATA ============
    
    def save_candles(self, df: pd.DataFrame, symbol: str = None) -> int:
        """
        Save OHLCV candles to database.
        
        Args:
            df: DataFrame with columns: time, open, high, low, close, volume
            symbol: Optional symbol (if not in df)
        
        Returns:
            Number of rows saved
        """
        if df.empty:
            return 0
        
        # Ensure symbol is in dataframe
        if symbol and 'symbol' not in df.columns:
            df['symbol'] = symbol
        
        if 'symbol' not in df.columns:
            raise ValueError("DataFrame must have 'symbol' column")
        
        # Prepare data
        records = df.to_dict('records')
        
        with self.get_session() as session:
            # Use bulk insert with upsert
            for record in records:
                stmt = insert(MarketCandle).values(**record)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['time', 'symbol'],
                    set_={
                        'open': stmt.excluded.open,
                        'high': stmt.excluded.high,
                        'low': stmt.excluded.low,
                        'close': stmt.excluded.close,
                        'volume': stmt.excluded.volume,
                        'adj_close': stmt.excluded.adj_close
                    }
                )
                session.execute(stmt)
        
        return len(records)
    
    def get_candles(
        self,
        symbol: str,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Get OHLCV candles from database.
        
        Args:
            symbol: Stock ticker
            start_date: Optional start date
            end_date: Optional end date
            limit: Optional limit on number of rows
        
        Returns:
            DataFrame with columns: time, symbol, open, high, low, close, volume
        """
        with self.get_session() as session:
            query = session.query(MarketCandle).filter(
                MarketCandle.symbol == symbol
            )
            
            if start_date:
                query = query.filter(MarketCandle.time >= start_date)
            
            if end_date:
                query = query.filter(MarketCandle.time <= end_date)
            
            query = query.order_by(MarketCandle.time.desc())
            
            if limit:
                query = query.limit(limit)
            
            candles = query.all()
            
            if not candles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = [{
                'time': c.time,
                'symbol': c.symbol,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume,
                'adj_close': c.adj_close
            } for c in candles]
            
            df = pd.DataFrame(data)
            
            # Sort by time ascending
            df = df.sort_values('time')
            
            return df
    
    # ============ NEWS & SENTIMENT ============
    
    def save_news(self, news_data: Dict) -> str:
        """Save news article with sentiment."""
        with self.get_session() as session:
            news = NewsSentiment(**news_data)
            session.add(news)
            session.flush()
            return str(news.id)
    
    def get_news_for_symbol(
        self,
        symbol: str,
        days: int = 7
    ) -> List[NewsSentiment]:
        """Get news articles for a symbol."""
        since = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            return session.query(NewsSentiment).filter(
                NewsSentiment.symbol == symbol,
                NewsSentiment.published_at >= since
            ).order_by(
                NewsSentiment.published_at.desc()
            ).all()
    
    def get_recent_news(
        self,
        symbol: str,
        days: int = 90
    ) -> List[NewsSentiment]:
        """
        Get recent news articles for a symbol.
        
        Alias for get_news_for_symbol with different default days.
        Used by confluence engine and sentiment tasks.
        """
        return self.get_news_for_symbol(symbol, days=days)
    
    def get_recent_news(
        self,
        symbol: str,
        days: int = 90
    ) -> List[NewsSentiment]:
        """
        Get recent news articles for a symbol.
        
        Args:
            symbol: Stock ticker
            days: Number of days to look back (default: 90)
        
        Returns:
            List of NewsSentiment objects
        """
        since = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            news_items = session.query(NewsSentiment).filter(
                NewsSentiment.symbol == symbol,
                NewsSentiment.published_at >= since
            ).order_by(
                NewsSentiment.published_at.desc()
            ).all()
            
            # Expunge to detach from session
            for item in news_items:
                session.expunge(item)
            
            return news_items
    
    def get_sentiment_summary(
        self,
        symbol: str,
        days: int = 30
    ) -> Dict:
        """Get sentiment summary for a symbol."""
        since = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            news = session.query(NewsSentiment).filter(
                NewsSentiment.symbol == symbol,
                NewsSentiment.published_at >= since
            ).all()
            
            if not news:
                return {
                    'avg_sentiment': 0.0,
                    'count': 0,
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
            
            scores = [n.finbert_score for n in news if n.finbert_score is not None]
            labels = [n.finbert_label for n in news if n.finbert_label]
            
            return {
                'avg_sentiment': sum(scores) / len(scores) if scores else 0.0,
                'count': len(news),
                'positive': labels.count('positive'),
                'negative': labels.count('negative'),
                'neutral': labels.count('neutral')
            }
    
    # ============ TRADE SIGNALS ============
    
    def save_signal(self, signal_data: Dict) -> str:
        """Save a trade signal."""
        with self.get_session() as session:
            signal = TradeSignal(**signal_data)
            session.add(signal)
            session.flush()
            logger.info(
                "Saved signal",
                symbol=signal.symbol,
                type=signal.signal_type,
                confidence=signal.confluence_score
            )
            return str(signal.id)
    
    def get_latest_signals(
        self,
        limit: int = 20,
        min_confidence: float = None,
        user_id: Optional[str] = None
    ) -> List[TradeSignal]:
        """
        Get latest trade signals.
        
        Args:
            limit: Maximum number of signals
            min_confidence: Minimum confluence score
            user_id: Optional user ID filter
        """
        with self.get_session() as session:
            query = session.query(TradeSignal)
            
            if min_confidence:
                query = query.filter(
                    TradeSignal.confluence_score >= min_confidence
                )
            
            if user_id:
                query = query.filter(TradeSignal.user_id == user_id)
            
            signals = query.order_by(
                TradeSignal.created_at.desc()
            ).limit(limit).all()
            
            # Expunge to detach from session cleanly
            for signal in signals:
                session.expunge(signal)
            
            return signals
    
    def get_signals_for_symbol(
        self,
        symbol: str,
        days: int = 30
    ) -> List[TradeSignal]:
        """Get signals for a specific symbol."""
        since = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            return session.query(TradeSignal).filter(
                TradeSignal.symbol == symbol,
                TradeSignal.created_at >= since
            ).order_by(
                TradeSignal.created_at.desc()
            ).all()
    
    # ============ USERS (for notifications) ============
    
    def get_users_with_email_alerts(self, tier: str = None) -> List[User]:
        """
        Get users who should receive email alerts.
        
        Args:
            tier: Optional subscription tier filter
        
        Returns:
            List of User objects with active subscriptions and email alerts enabled
        """
        try:
            from src.auth.models import User, Subscription
        except ImportError:
            return []
        
        with self.get_session() as session:
            query = session.query(User).filter(
                User.is_active == True,
                User.email_verified == True
            ).join(Subscription).filter(
                Subscription.status == 'active'
            )
            
            if tier:
                query = query.filter(Subscription.tier == tier)
            
            users = query.all()
            return users
    # ============ INDICATOR CACHE ============
    
    def save_indicators(self, indicator_data: Dict) -> None:
        """Save computed indicators."""
        with self.get_session() as session:
            stmt = insert(IndicatorCache).values(**indicator_data)
            stmt = stmt.on_conflict_do_update(
                constraint='uq_indicator_symbol_date',
                set_={
                    'computed_at': datetime.utcnow(),
                    'rsi_14': stmt.excluded.rsi_14,
                    'rsi_signal': stmt.excluded.rsi_signal,
                    'macd': stmt.excluded.macd,
                    'macd_signal': stmt.excluded.macd_signal,
                    'macd_histogram': stmt.excluded.macd_histogram,
                    'sma_20': stmt.excluded.sma_20,
                    'sma_50': stmt.excluded.sma_50,
                    'sma_200': stmt.excluded.sma_200,
                    'bb_upper': stmt.excluded.bb_upper,
                    'bb_middle': stmt.excluded.bb_middle,
                    'bb_lower': stmt.excluded.bb_lower,
                    'atr': stmt.excluded.atr,
                    'technical_score': stmt.excluded.technical_score,
                }
            )
            session.execute(stmt)
    
    def get_latest_indicators(self, symbol: str) -> Optional[IndicatorCache]:
        """Get most recent indicators for a symbol."""
        with self.get_session() as session:
            return session.query(IndicatorCache).filter(
                IndicatorCache.symbol == symbol
            ).order_by(
                IndicatorCache.as_of_date.desc()
            ).first()
    
    # ============ BACKTEST RESULTS ============
    
    def save_backtest_result(self, result_data: Dict) -> str:
        """Save backtest result."""
        with self.get_session() as session:
            result = BacktestResult(**result_data)
            session.add(result)
            session.flush()
            return str(result.id)
    
    def get_backtest_summary(self, strategy_name: str) -> Dict:
        """Get aggregated backtest results for a strategy."""
        with self.get_session() as session:
            results = session.query(BacktestResult).filter(
                BacktestResult.strategy_name == strategy_name
            ).all()
            
            if not results:
                return {}
            
            returns = [r.total_return for r in results if r.total_return]
            sharpes = [r.sharpe_ratio for r in results if r.sharpe_ratio]
            drawdowns = [r.max_drawdown for r in results if r.max_drawdown]
            
            return {
                'num_periods': len(results),
                'avg_return': sum(returns) / len(returns) if returns else 0,
                'avg_sharpe': sum(sharpes) / len(sharpes) if sharpes else 0,
                'avg_drawdown': sum(drawdowns) / len(drawdowns) if drawdowns else 0,
                'win_rate': len([r for r in returns if r > 0]) / len(returns) if returns else 0
            }


# Singleton instance
_db_manager = None

def get_database() -> DatabaseManager:
    """Get the database manager singleton."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
