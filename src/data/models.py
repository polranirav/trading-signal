"""
SQLAlchemy ORM models for the trading signals database.

Tables:
- AssetMetadata: Static stock information
- MarketCandle: OHLCV time-series data (TimescaleDB hypertable)
- NewsSentiment: News articles with FinBERT scores
- TradeSignal: Generated trading signals with risk metrics
- IndicatorCache: Pre-computed technical indicators
- User, Subscription, APIKey, SubscriptionLimit: Authentication (from src.auth.models)
- Payment, Invoice: Payment processing (from src.payments.models)
"""

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Boolean, 
    Index, Text, create_engine, UniqueConstraint, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, DOUBLE_PRECISION
from sqlalchemy import DECIMAL
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()


# ============ ASSET METADATA ============
class AssetMetadata(Base):
    """
    Static information about tradeable assets.
    
    This table stores company/asset information that changes infrequently.
    Used for filtering, categorization, and display purposes.
    """
    __tablename__ = "asset_metadata"
    
    symbol = Column(String(10), primary_key=True, nullable=False)
    name = Column(String(255), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    country = Column(String(10), default="US")
    currency = Column(String(10), default="USD")
    market_cap = Column(Float)
    
    # Tracking
    is_active = Column(Boolean, default=True)
    date_added = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_asset_sector_industry', 'sector', 'industry'),
        Index('idx_asset_is_active', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Asset {self.symbol}: {self.name}>"


# ============ MARKET DATA ============
class MarketCandle(Base):
    """
    OHLCV candlestick data - optimized for time-series queries.
    
    This table will be converted to a TimescaleDB hypertable for
    efficient time-series queries and automatic data compression.
    """
    __tablename__ = "market_candles"
    
    time = Column(DateTime, primary_key=True, nullable=False)
    symbol = Column(String(10), primary_key=True, nullable=False)
    
    # OHLCV
    open = Column(DOUBLE_PRECISION, nullable=False)
    high = Column(DOUBLE_PRECISION, nullable=False)
    low = Column(DOUBLE_PRECISION, nullable=False)
    close = Column(DOUBLE_PRECISION, nullable=False)
    volume = Column(Integer)
    
    # Adjusted prices (for splits/dividends)
    adj_close = Column(DOUBLE_PRECISION)
    
    __table_args__ = (
        Index('idx_candles_symbol_time', 'symbol', 'time'),
        Index('idx_candles_time', 'time'),
    )
    
    def __repr__(self):
        return f"<Candle {self.symbol} {self.time}: O={self.open} C={self.close}>"


# ============ NEWS & SENTIMENT ============
class NewsSentiment(Base):
    """
    Financial news articles with sentiment analysis results.
    
    Stores news headlines, summaries, and FinBERT sentiment scores.
    Used for the sentiment layer of the trading signal engine.
    """
    __tablename__ = "news_sentiment"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Article metadata
    symbol = Column(String(10), nullable=False, index=True)
    published_at = Column(DateTime, nullable=False, index=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    
    # Content
    headline = Column(String(500), nullable=False)
    summary = Column(Text)
    source = Column(String(100))
    url = Column(String(500), unique=True)
    
    # FinBERT Sentiment Analysis
    finbert_score = Column(Float)  # -1.0 to 1.0
    finbert_label = Column(String(20))  # positive, neutral, negative
    finbert_confidence = Column(Float)  # Model confidence 0-1
    
    # GPT-4 Analysis (optional)
    gpt4_summary = Column(Text)
    gpt4_sentiment = Column(Float)
    
    __table_args__ = (
        Index('idx_news_symbol_published', 'symbol', 'published_at'),
        Index('idx_news_finbert_score', 'finbert_score'),
    )
    
    def __repr__(self):
        return f"<News {self.symbol} {self.published_at}: {self.headline[:50]}>"


# ============ TRADE SIGNALS ============
class TradeSignal(Base):
    """
    Generated trading signals - immutable audit log.
    
    Each signal captures the complete reasoning: technical scores,
    sentiment scores, confluence score, risk metrics, and rationale.
    """
    __tablename__ = "trade_signals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    
    # User association (for subscription limits)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    
    # Trade Details
    symbol = Column(String(10), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)  # BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    
    # Scoring Components (0-1 scale)
    technical_score = Column(Float)  # From technical indicators
    sentiment_score = Column(Float)  # From FinBERT/GPT-4
    ml_score = Column(Float)  # From ML models
    confluence_score = Column(Float)  # Final combined confidence
    
    # Risk Metrics
    var_95 = Column(Float)  # Value at Risk 95%
    cvar_95 = Column(Float)  # Conditional VaR 95%
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    suggested_position_size = Column(Float)  # % of portfolio
    risk_reward_ratio = Column(Float)  # Risk-reward ratio (RR = reward/risk)
    
    # White-Box Explanation
    technical_rationale = Column(Text)
    sentiment_rationale = Column(Text)
    risk_warning = Column(Text)
    
    # Price at signal generation
    price_at_signal = Column(Float)
    
    # Execution tracking
    is_executed = Column(Boolean, default=False)
    execution_price = Column(Float)
    execution_date = Column(DateTime)
    close_price = Column(Float)
    close_date = Column(DateTime)
    realized_pnl = Column(Float)
    realized_pnl_pct = Column(Float)
    
    __table_args__ = (
        Index('idx_signals_symbol_created', 'symbol', 'created_at'),
        Index('idx_signals_confluence', 'confluence_score'),
        Index('idx_signals_type', 'signal_type'),
        Index('idx_signals_user_id', 'user_id'),
    )
    
    def __repr__(self):
        return f"<Signal {self.symbol} {self.signal_type}: {self.confluence_score:.2f}>"


# ============ INDICATOR CACHE ============
class IndicatorCache(Base):
    """
    Pre-computed technical indicators for fast retrieval.
    
    Caches calculated indicators to avoid recomputation.
    Updated daily after market close.
    """
    __tablename__ = "indicator_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    symbol = Column(String(10), nullable=False)
    as_of_date = Column(DateTime, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)
    
    # Momentum Indicators
    rsi_14 = Column(Float)
    rsi_signal = Column(String(20))  # OVERBOUGHT, OVERSOLD, NEUTRAL
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    macd_crossover = Column(String(20))  # BULLISH, BEARISH, NONE
    
    # Trend Indicators
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    trend_signal = Column(String(20))  # UPTREND, DOWNTREND, SIDEWAYS
    adx = Column(Float)
    
    # Volatility Indicators
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_bandwidth = Column(Float)
    bb_position = Column(Float)  # 0-1, position within bands
    atr = Column(Float)
    
    # Volume Indicators
    obv = Column(Float)
    mfi = Column(Float)
    volume_sma_20 = Column(Float)
    volume_ratio = Column(Float)
    
    # Aggregated Scores
    technical_score = Column(Float)  # Combined 0-1 score
    
    # Sentiment (aggregated from news)
    sentiment_24h = Column(Float)
    sentiment_7d = Column(Float)
    news_count_24h = Column(Integer)
    news_count_7d = Column(Integer)
    
    __table_args__ = (
        Index('idx_indicator_symbol_date', 'symbol', 'as_of_date'),
        UniqueConstraint('symbol', 'as_of_date', name='uq_indicator_symbol_date'),
    )
    
    def __repr__(self):
        return f"<Indicators {self.symbol} {self.as_of_date}: RSI={self.rsi_14}>"


# ============ BACKTEST RESULTS ============
class BacktestResult(Base):
    """
    Walk-forward backtest results.
    
    Stores performance metrics for each test period
    to track strategy robustness over time.
    """
    __tablename__ = "backtest_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Test period
    strategy_name = Column(String(100), nullable=False)
    train_start = Column(DateTime, nullable=False)
    train_end = Column(DateTime, nullable=False)
    test_start = Column(DateTime, nullable=False)
    test_end = Column(DateTime, nullable=False)
    period_number = Column(Integer)
    
    # Performance metrics
    total_return = Column(Float)
    annual_return = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    
    # Trade statistics
    num_trades = Column(Integer)
    avg_trade_return = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    
    # Risk metrics
    var_95 = Column(Float)
    cvar_95 = Column(Float)
    
    __table_args__ = (
        Index('idx_backtest_strategy_period', 'strategy_name', 'test_start'),
    )
    
    def __repr__(self):
        return f"<Backtest {self.strategy_name} {self.test_start}: Sharpe={self.sharpe_ratio}>"


# ============ PORTFOLIO HOLDINGS ============
class PortfolioHolding(Base):
    """
    User's portfolio holdings.
    
    Tracks stocks the user owns, their cost basis, and current position.
    Used for personalized signal generation and P&L tracking.
    """
    __tablename__ = "portfolio_holdings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Position details
    symbol = Column(String(10), nullable=False, index=True)
    shares = Column(Float, nullable=False)
    avg_cost = Column(Float, nullable=False)  # Average cost per share
    
    # Metadata
    purchase_date = Column(DateTime)
    source = Column(String(50), default='manual')  # 'manual', 'csv', 'robinhood', 'schwab'
    notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Soft delete
    is_active = Column(Boolean, default=True)
    
    __table_args__ = (
        Index('idx_portfolio_user_symbol', 'user_id', 'symbol'),
        Index('idx_portfolio_user_active', 'user_id', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Holding {self.symbol}: {self.shares} shares @ ${self.avg_cost}>"


class PortfolioTransaction(Base):
    """
    Transaction history for portfolio holdings.
    
    Tracks buy/sell/dividend transactions for accurate P&L calculation.
    """
    __tablename__ = "portfolio_transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    holding_id = Column(UUID(as_uuid=True), ForeignKey("portfolio_holdings.id", ondelete="SET NULL"), nullable=True)
    
    # Transaction details
    symbol = Column(String(10), nullable=False, index=True)
    transaction_type = Column(String(20), nullable=False)  # 'buy', 'sell', 'dividend'
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float)  # shares * price
    
    # Metadata
    transaction_date = Column(DateTime, nullable=False)
    source = Column(String(50), default='manual')
    notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_transaction_user_date', 'user_id', 'transaction_date'),
        Index('idx_transaction_symbol', 'symbol'),
    )
    
    def __repr__(self):
        return f"<Transaction {self.transaction_type} {self.symbol}: {self.shares} @ ${self.price}>"


# ============ DATABASE INITIALIZATION ============
def get_engine(database_url: str = None):
    """Create SQLAlchemy engine."""
    from src.config import settings
    url = database_url or settings.DATABASE_URL
    return create_engine(url, echo=False)


def init_database(database_url: str = None):
    """
    Initialize database tables.
    
    Creates all tables defined in this module.
    For TimescaleDB hypertables, run init_timescaledb() after this.
    """
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully")
    return engine


def init_timescaledb(database_url: str = None):
    """
    Convert market_candles to TimescaleDB hypertable.
    
    Enables time-series optimizations:
    - Automatic partitioning by time
    - Compression for old data
    - Fast time-range queries
    """
    from sqlalchemy import text
    
    engine = get_engine(database_url)
    
    with engine.connect() as conn:
        # Enable TimescaleDB extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
        
        # Convert to hypertable
        try:
            conn.execute(text("""
                SELECT create_hypertable(
                    'market_candles', 
                    'time',
                    if_not_exists => TRUE
                )
            """))
            print("✓ Created hypertable for market_candles")
        except Exception as e:
            print(f"Hypertable already exists or error: {e}")
        
        # Enable compression for old data
        try:
            conn.execute(text("""
                ALTER TABLE market_candles 
                SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol',
                    timescaledb.compress_orderby = 'time DESC'
                )
            """))
            
            conn.execute(text("""
                SELECT add_compression_policy(
                    'market_candles',
                    INTERVAL '14 days',
                    if_not_exists => TRUE
                )
            """))
            print("✓ Enabled compression policy for market_candles")
        except Exception as e:
            print(f"Compression setup error: {e}")
        
        conn.commit()
    
    print("✓ TimescaleDB initialization complete")


# Import auth models to ensure they're registered with Base
# This ensures auth tables are created when init_database() is called
try:
    from src.auth.models import User, Subscription, APIKey, SubscriptionLimit
    # Models are now registered with Base
except ImportError:
    # Auth module not yet created, skip
    pass

# Import payment models to ensure they're registered
try:
    from src.payments.models import Payment, Invoice
    # Models are now registered with Base
except ImportError:
    # Payments module not yet created, skip
    pass

# Import admin models to ensure they're registered
try:
    from src.admin.models import AuditLog, SystemSettings, AdminActivity
    # Models are now registered with Base
except ImportError:
    # Admin module not yet created, skip
    pass


if __name__ == "__main__":
    # Run directly to initialize database
    init_database()
    init_timescaledb()
