"""
Market data ingestion with intelligent fallback.

Supports multiple data sources:
1. Alpha Vantage (primary - reliable, rate-limited)
2. Yahoo Finance (fallback - free, good coverage)

Usage:
    client = MarketDataClient()
    df = client.fetch_daily_candles("AAPL", days=252)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import time

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except (ImportError, TypeError):
    # Fallback if yfinance not available or version incompatible (Python 3.9 issue)
    YFINANCE_AVAILABLE = False
    yf = None

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class MarketDataClient:
    """
    Fetch OHLCV data with intelligent fallback mechanism.
    
    Priority order:
    1. Alpha Vantage (reliable, rate-limited at 5/min free tier)
    2. Yahoo Finance (free, good backup)
    
    Features:
    - Automatic fallback on failure
    - Rate limiting respect
    - Data validation
    - Retry with exponential backoff
    """
    
    def __init__(self):
        self.alpha_vantage_key = settings.ALPHA_VANTAGE_KEY
        self.timeout = 30
        self.max_retries = 3
        self._alpha_vantage_last_call = 0
        self._alpha_vantage_rate_limit = 12  # seconds between calls (5/min)
    
    def fetch_daily_candles(
        self, 
        symbol: str, 
        days: int = 252
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with priority-based fallback.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            days: Lookback period in trading days (252 = 1 year)
        
        Returns:
            DataFrame with columns: [time, symbol, open, high, low, close, volume, adj_close]
            Returns None if all sources fail.
        """
        
        # Try Alpha Vantage first (if key is valid)
        if self.alpha_vantage_key and self.alpha_vantage_key != "demo":
            try:
                df = self._fetch_alpha_vantage(symbol)
                if df is not None and len(df) > 0:
                    logger.info(
                        "Fetched from Alpha Vantage",
                        symbol=symbol,
                        rows=len(df)
                    )
                    return self._standardize_dataframe(df, symbol, days)
            except Exception as e:
                logger.warning(
                    "Alpha Vantage failed, trying fallback",
                    symbol=symbol,
                    error=str(e)[:100]
                )
        
        # Fallback to Yahoo Finance (if available)
        if YFINANCE_AVAILABLE:
            try:
                df = self._fetch_yfinance(symbol, days)
                if df is not None and len(df) > 0:
                    logger.info(
                        "Fetched from Yahoo Finance",
                        symbol=symbol,
                        rows=len(df)
                    )
                    return self._standardize_dataframe(df, symbol, days)
            except Exception as e:
                logger.error(
                    "Yahoo Finance failed",
                    symbol=symbol,
                    error=str(e)[:100]
                )
        else:
            logger.warning("yfinance not available (version incompatible with Python 3.9)")
        
        # All sources failed
        logger.error("All data sources exhausted", symbol=symbol)
        return None
    
    def _fetch_alpha_vantage(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch from Alpha Vantage API.
        
        Rate limiting: Max 5 calls per minute on free tier.
        """
        # Rate limiting
        elapsed = time.time() - self._alpha_vantage_last_call
        if elapsed < self._alpha_vantage_rate_limit:
            time.sleep(self._alpha_vantage_rate_limit - elapsed)
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": self.alpha_vantage_key,
            "outputsize": "full",
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                self._alpha_vantage_last_call = time.time()
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}")
                
                data = response.json()
                
                # Check for API errors
                if "Error Message" in data:
                    raise Exception(data["Error Message"])
                if "Note" in data:
                    raise Exception("Rate limit exceeded")
                
                # Parse time series data
                time_series = data.get("Time Series (Daily)", {})
                if not time_series:
                    raise Exception("No data returned")
                
                # Convert to DataFrame
                records = []
                for date_str, values in time_series.items():
                    records.append({
                        "time": pd.to_datetime(date_str),
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "adj_close": float(values["5. adjusted close"]),
                        "volume": int(values["6. volume"]),
                    })
                
                df = pd.DataFrame(records)
                return df.sort_values("time").reset_index(drop=True)
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.debug(
                        "Retrying Alpha Vantage",
                        attempt=attempt + 1,
                        wait=wait_time
                    )
                    time.sleep(wait_time)
                else:
                    raise
    
    def _fetch_yfinance(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """
        Fetch from Yahoo Finance using yfinance library.
        
        More reliable for most tickers, no API key required.
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))  # Extra buffer
        
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=False
                )
                
                if df.empty:
                    raise Exception("No data returned")
                
                # Rename columns to match our schema
                df = df.reset_index()
                df.columns = df.columns.str.lower()
                df = df.rename(columns={
                    "date": "time",
                    "adj close": "adj_close"
                })
                
                # Select and reorder columns
                columns = ["time", "open", "high", "low", "close", "volume"]
                if "adj_close" in df.columns:
                    columns.append("adj_close")
                
                df = df[columns]
                return df
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.debug(
                        "Retrying Yahoo Finance",
                        attempt=attempt + 1,
                        wait=wait_time
                    )
                    time.sleep(wait_time)
                else:
                    raise
    
    def _standardize_dataframe(
        self, 
        df: pd.DataFrame, 
        symbol: str,
        days: int
    ) -> pd.DataFrame:
        """
        Standardize DataFrame format and filter to requested days.
        """
        df = df.copy()
        
        # Add symbol column
        df["symbol"] = symbol
        
        # Ensure time is datetime
        df["time"] = pd.to_datetime(df["time"])
        
        # Remove timezone info if present
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_localize(None)
        
        # Sort by time
        df = df.sort_values("time", ascending=True)
        
        # Keep only requested number of days
        if len(df) > days:
            df = df.tail(days)
        
        # Reorder columns
        columns = ["time", "symbol", "open", "high", "low", "close", "volume"]
        if "adj_close" in df.columns:
            columns.append("adj_close")
        
        df = df[columns].reset_index(drop=True)
        
        # Validate data
        self._validate_data(df)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate OHLCV data integrity.
        
        Raises ValueError if data is invalid.
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for required columns
        required = ["time", "symbol", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Check for nulls in critical columns
        null_counts = df[["open", "high", "low", "close"]].isnull().sum()
        if null_counts.any():
            logger.warning(
                "Found null values in OHLC data",
                null_counts=null_counts.to_dict()
            )
        
        # Check OHLC relationships (high >= low)
        invalid_hl = df[df["high"] < df["low"]]
        if not invalid_hl.empty:
            logger.warning(
                "Found invalid high/low relationships",
                count=len(invalid_hl)
            )
    
    def fetch_multiple(
        self, 
        symbols: List[str], 
        days: int = 252,
        delay: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock tickers
            days: Lookback period
            delay: Seconds between requests
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(
                "Fetching",
                symbol=symbol,
                progress=f"{i+1}/{len(symbols)}"
            )
            
            df = self.fetch_daily_candles(symbol, days)
            if df is not None:
                results[symbol] = df
            else:
                logger.warning("Failed to fetch", symbol=symbol)
            
            if i < len(symbols) - 1:
                time.sleep(delay)
        
        logger.info(
            "Batch fetch complete",
            success=len(results),
            failed=len(symbols) - len(results)
        )
        
        return results


class NewsDataClient:
    """
    Fetch financial news for sentiment analysis.
    
    Sources (in priority order):
    1. Alpha Vantage News (if available)
    2. NewsAPI (fallback)
    """
    
    def __init__(self):
        self.alpha_vantage_key = settings.ALPHA_VANTAGE_KEY
        self.timeout = 30
    
    def fetch_news(
        self, 
        symbol: str,
        days: int = 7
    ) -> List[Dict]:
        """
        Fetch recent news articles for a symbol.
        
        Args:
            symbol: Stock ticker
            days: Number of days to look back
        
        Returns:
            List of news article dictionaries
        """
        articles = []
        
        # Try Alpha Vantage News
        if self.alpha_vantage_key and self.alpha_vantage_key != "demo":
            try:
                articles = self._fetch_alpha_vantage_news(symbol)
                if articles:
                    logger.info(
                        "Fetched news from Alpha Vantage",
                        symbol=symbol,
                        count=len(articles)
                    )
                    return articles
            except Exception as e:
                logger.warning(
                    "Alpha Vantage news failed",
                    symbol=symbol,
                    error=str(e)[:100]
                )
        
        logger.warning("No news sources available", symbol=symbol)
        return []
    
    def _fetch_alpha_vantage_news(self, symbol: str) -> List[Dict]:
        """Fetch from Alpha Vantage News API."""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self.alpha_vantage_key,
            "limit": 50,
        }
        
        response = requests.get(url, params=params, timeout=self.timeout)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        data = response.json()
        
        if "feed" not in data:
            return []
        
        articles = []
        for item in data["feed"]:
            articles.append({
                "headline": item.get("title", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "published_at": item.get("time_published", ""),
                "overall_sentiment_score": item.get("overall_sentiment_score"),
                "overall_sentiment_label": item.get("overall_sentiment_label"),
            })
        
        return articles


# Convenience function
def get_market_data_client() -> MarketDataClient:
    """Get a configured MarketDataClient instance."""
    return MarketDataClient()


def get_news_data_client() -> NewsDataClient:
    """Get a configured NewsDataClient instance."""
    return NewsDataClient()


if __name__ == "__main__":
    # Test the client
    client = MarketDataClient()
    df = client.fetch_daily_candles("AAPL", days=30)
    if df is not None:
        print(f"Fetched {len(df)} rows for AAPL")
        print(df.head())
    else:
        print("Failed to fetch data")
