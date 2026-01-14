"""
Data ingestion Celery tasks.

Tasks for fetching market data and news:
- fetch_prices: Fetch OHLCV for a single symbol
- fetch_all_prices: Fetch for all active assets
- fetch_news: Fetch news for a symbol
- fetch_all_news: Fetch news for all assets

All tasks are idempotent and safe to retry.
"""

from celery import shared_task
from datetime import datetime, timedelta
from typing import Optional, List

from src.tasks.celery_app import app
from src.data.ingestion import MarketDataClient, NewsDataClient
from src.data.persistence import get_database
from src.data.cache import get_cache
from src.logging_config import get_logger

logger = get_logger(__name__)


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=3600,
    max_retries=5,
    name='src.tasks.ingestion_tasks.fetch_prices'
)
def fetch_prices(self, symbol: str, days: int = 30) -> dict:
    """
    Fetch OHLCV data for a single symbol and save to database.
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        days: Number of trading days to fetch
    
    Returns:
        Result dictionary with status and count
    """
    logger.info("Fetching prices", symbol=symbol, days=days, task_id=self.request.id)
    
    try:
        # Fetch data
        client = MarketDataClient()
        df = client.fetch_daily_candles(symbol, days=days)
        
        if df is None or df.empty:
            logger.warning("No data returned", symbol=symbol)
            return {"status": "no_data", "symbol": symbol, "count": 0}
        
        # Save to database
        db = get_database()
        count = db.save_candles(df)
        
        # Invalidate cache
        cache = get_cache()
        cache.invalidate("candles", symbol)
        cache.invalidate("indicators", symbol)
        
        logger.info("Saved prices", symbol=symbol, count=count)
        
        return {
            "status": "success",
            "symbol": symbol,
            "count": count,
            "latest_date": df['time'].max().isoformat() if not df.empty else None
        }
        
    except Exception as e:
        logger.error(
            "Failed to fetch prices",
            symbol=symbol,
            error=str(e),
            attempt=self.request.retries
        )
        raise


@app.task(
    bind=True,
    name='src.tasks.ingestion_tasks.fetch_all_prices'
)
def fetch_all_prices(self, symbols: List[str] = None) -> dict:
    """
    Fetch prices for all active assets.
    
    Args:
        symbols: Optional list of symbols (defaults to all active)
    
    Returns:
        Summary of fetch operation
    """
    logger.info("Starting batch price fetch", task_id=self.request.id)
    
    # Get symbols to fetch
    if not symbols:
        db = get_database()
        assets = db.get_active_assets()
        symbols = [a.symbol for a in assets]
        
        # Default symbols if no assets configured
        if not symbols:
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
                'META', 'TSLA', 'BRK-B', 'JPM', 'V',
                'UNH', 'MA', 'PG', 'HD', 'DIS'
            ]
    
    logger.info("Fetching prices", symbol_count=len(symbols))
    
    # Queue individual tasks
    results = []
    for symbol in symbols:
        task = fetch_prices.delay(symbol, days=30)
        results.append({"symbol": symbol, "task_id": task.id})
    
    return {
        "status": "queued",
        "symbols": len(symbols),
        "tasks": results
    }


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
    name='src.tasks.ingestion_tasks.fetch_news'
)
def fetch_news(self, symbol: str) -> dict:
    """
    Fetch news for a single symbol.
    
    Args:
        symbol: Stock ticker
    
    Returns:
        Result dictionary with status and count
    """
    logger.info("Fetching news", symbol=symbol, task_id=self.request.id)
    
    try:
        client = NewsDataClient()
        news_items = client.fetch_news(symbol, days=7)
        
        if not news_items:
            logger.info("No news found", symbol=symbol)
            return {"status": "no_news", "symbol": symbol, "count": 0}
        
        # Save to database
        db = get_database()
        saved_count = 0
        
        for item in news_items:
            try:
                news_data = {
                    "symbol": symbol,
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "published_at": datetime.fromisoformat(
                        item.get("published_at", datetime.utcnow().isoformat())
                        .replace("Z", "+00:00")[:19]
                    ) if item.get("published_at") else datetime.utcnow(),
                }
                
                # Add sentiment if available from API
                if item.get("overall_sentiment_score"):
                    news_data["finbert_score"] = item["overall_sentiment_score"]
                    news_data["finbert_label"] = item.get("overall_sentiment_label", "neutral")
                
                db.save_news(news_data)
                saved_count += 1
                
            except Exception as e:
                logger.warning(
                    "Failed to save news item",
                    symbol=symbol,
                    headline=item.get("headline", "")[:50],
                    error=str(e)
                )
        
        # Invalidate sentiment cache
        cache = get_cache()
        cache.invalidate("sentiment", symbol)
        
        logger.info("Saved news", symbol=symbol, count=saved_count)
        
        return {
            "status": "success",
            "symbol": symbol,
            "fetched": len(news_items),
            "saved": saved_count
        }
        
    except Exception as e:
        logger.error(
            "Failed to fetch news",
            symbol=symbol,
            error=str(e),
            attempt=self.request.retries
        )
        raise


@app.task(
    bind=True,
    name='src.tasks.ingestion_tasks.fetch_all_news'
)
def fetch_all_news(self, symbols: List[str] = None) -> dict:
    """
    Fetch news for all active assets.
    
    Args:
        symbols: Optional list of symbols
    
    Returns:
        Summary of fetch operation
    """
    logger.info("Starting batch news fetch", task_id=self.request.id)
    
    # Get symbols
    if not symbols:
        db = get_database()
        assets = db.get_active_assets()
        symbols = [a.symbol for a in assets]
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    logger.info("Fetching news", symbol_count=len(symbols))
    
    # Queue individual tasks
    results = []
    for symbol in symbols:
        task = fetch_news.delay(symbol)
        results.append({"symbol": symbol, "task_id": task.id})
    
    return {
        "status": "queued",
        "symbols": len(symbols),
        "tasks": results
    }


@app.task(
    bind=True,
    name='src.tasks.ingestion_tasks.backfill_historical'
)
def backfill_historical(self, symbol: str, days: int = 252) -> dict:
    """
    Backfill historical data for a symbol.
    
    Used for initial data population or gap filling.
    
    Args:
        symbol: Stock ticker
        days: Number of trading days (252 = 1 year)
    
    Returns:
        Result dictionary
    """
    logger.info("Backfilling historical data", symbol=symbol, days=days)
    
    try:
        client = MarketDataClient()
        df = client.fetch_daily_candles(symbol, days=days)
        
        if df is None or df.empty:
            return {"status": "no_data", "symbol": symbol, "count": 0}
        
        db = get_database()
        count = db.save_candles(df)
        
        logger.info("Backfilled historical data", symbol=symbol, count=count)
        
        return {
            "status": "success",
            "symbol": symbol,
            "count": count,
            "date_range": {
                "start": df['time'].min().isoformat(),
                "end": df['time'].max().isoformat()
            }
        }
        
    except Exception as e:
        logger.error("Backfill failed", symbol=symbol, error=str(e))
        raise
