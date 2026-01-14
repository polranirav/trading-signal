"""
Redis caching layer for fast data retrieval.

Caches:
- Computed indicators (TTL: 4 hours)
- Market data (TTL: 5 minutes during trading hours)
- Aggregated sentiment (TTL: 1 hour)

Usage:
    from src.data.cache import CacheManager
    cache = CacheManager()
    cache.set_indicators("AAPL", indicators_dict)
    cached = cache.get_indicators("AAPL")
"""

import redis
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pickle

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Redis cache manager for trading data.
    
    Key prefixes:
    - indicators:{symbol} - Technical indicators
    - candles:{symbol} - Recent price data
    - sentiment:{symbol} - Aggregated sentiment
    - signal:{symbol} - Latest signal
    """
    
    # TTL constants (in seconds)
    TTL_INDICATORS = 4 * 60 * 60      # 4 hours
    TTL_CANDLES = 5 * 60               # 5 minutes
    TTL_SENTIMENT = 60 * 60            # 1 hour
    TTL_SIGNAL = 24 * 60 * 60          # 24 hours
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._client = None
    
    @property
    def client(self) -> redis.Redis:
        """Lazy-load Redis client."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=False  # We handle encoding ourselves
            )
        return self._client
    
    def _make_key(self, prefix: str, identifier: str) -> str:
        """Create a cache key."""
        return f"{prefix}:{identifier}"
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for Redis storage."""
        return pickle.dumps(data)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from Redis."""
        if data is None:
            return None
        return pickle.loads(data)
    
    # ============ HEALTH CHECK ============
    
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error("Redis connection failed", error=str(e))
            return False
    
    # ============ INDICATORS CACHE ============
    
    def set_indicators(
        self, 
        symbol: str, 
        indicators: Dict,
        ttl: int = None
    ) -> bool:
        """
        Cache computed indicators for a symbol.
        
        Args:
            symbol: Stock ticker
            indicators: Dictionary of indicator values
            ttl: Time-to-live in seconds (default: 4 hours)
        """
        try:
            key = self._make_key("indicators", symbol)
            data = self._serialize(indicators)
            self.client.setex(key, ttl or self.TTL_INDICATORS, data)
            logger.debug("Cached indicators", symbol=symbol)
            return True
        except Exception as e:
            logger.error("Failed to cache indicators", symbol=symbol, error=str(e))
            return False
    
    def get_indicators(self, symbol: str) -> Optional[Dict]:
        """Get cached indicators for a symbol."""
        try:
            key = self._make_key("indicators", symbol)
            data = self.client.get(key)
            return self._deserialize(data)
        except Exception as e:
            logger.error("Failed to get cached indicators", symbol=symbol, error=str(e))
            return None
    
    # ============ CANDLES CACHE ============
    
    def set_candles(
        self,
        symbol: str,
        candles: Dict,
        ttl: int = None
    ) -> bool:
        """Cache recent candle data."""
        try:
            key = self._make_key("candles", symbol)
            data = self._serialize(candles)
            self.client.setex(key, ttl or self.TTL_CANDLES, data)
            return True
        except Exception as e:
            logger.error("Failed to cache candles", symbol=symbol, error=str(e))
            return False
    
    def get_candles(self, symbol: str) -> Optional[Dict]:
        """Get cached candle data."""
        try:
            key = self._make_key("candles", symbol)
            data = self.client.get(key)
            return self._deserialize(data)
        except Exception as e:
            logger.error("Failed to get cached candles", symbol=symbol, error=str(e))
            return None
    
    # ============ SENTIMENT CACHE ============
    
    def set_sentiment(
        self,
        symbol: str,
        sentiment: Dict,
        ttl: int = None
    ) -> bool:
        """Cache aggregated sentiment data."""
        try:
            key = self._make_key("sentiment", symbol)
            data = self._serialize(sentiment)
            self.client.setex(key, ttl or self.TTL_SENTIMENT, data)
            return True
        except Exception as e:
            logger.error("Failed to cache sentiment", symbol=symbol, error=str(e))
            return False
    
    def get_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get cached sentiment data."""
        try:
            key = self._make_key("sentiment", symbol)
            data = self.client.get(key)
            return self._deserialize(data)
        except Exception as e:
            logger.error("Failed to get cached sentiment", symbol=symbol, error=str(e))
            return None
    
    # ============ SIGNAL CACHE ============
    
    def set_signal(
        self,
        symbol: str,
        signal: Dict,
        ttl: int = None
    ) -> bool:
        """Cache latest signal for a symbol."""
        try:
            key = self._make_key("signal", symbol)
            data = self._serialize(signal)
            self.client.setex(key, ttl or self.TTL_SIGNAL, data)
            return True
        except Exception as e:
            logger.error("Failed to cache signal", symbol=symbol, error=str(e))
            return False
    
    def get_signal(self, symbol: str) -> Optional[Dict]:
        """Get cached signal."""
        try:
            key = self._make_key("signal", symbol)
            data = self.client.get(key)
            return self._deserialize(data)
        except Exception as e:
            logger.error("Failed to get cached signal", symbol=symbol, error=str(e))
            return None
    
    # ============ BULK SIGNAL OPERATIONS ============
    
    def get_all_signals(self, limit: int = 50) -> Optional[List[Dict]]:
        """
        Get all cached signals from dashboard cache.
        
        Args:
            limit: Maximum number of signals to return
        
        Returns:
            List of signal dictionaries or None if not cached
        """
        try:
            key = "dashboard:latest_signals"
            data = self.client.get(key)
            if data:
                signals = self._deserialize(data)
                if isinstance(signals, list):
                    return signals[:limit]
            return None
        except Exception as e:
            logger.error("Failed to get all cached signals", error=str(e))
            return None
    
    def set_all_signals(self, signals: List[Dict], ttl: int = 900) -> bool:
        """
        Cache all signals for dashboard display.
        
        Args:
            signals: List of signal dictionaries
            ttl: Time-to-live in seconds (default: 15 minutes)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            key = "dashboard:latest_signals"
            data = self._serialize(signals)
            self.client.setex(key, ttl, data)
            logger.debug("Cached all signals", count=len(signals))
            return True
        except Exception as e:
            logger.error("Failed to cache all signals", error=str(e))
            return False
    
    # ============ UTILITY METHODS ============
    
    def invalidate(self, prefix: str, symbol: str) -> bool:
        """Invalidate a specific cache entry."""
        try:
            key = self._make_key(prefix, symbol)
            self.client.delete(key)
            logger.debug("Invalidated cache", key=key)
            return True
        except Exception as e:
            logger.error("Failed to invalidate", key=key, error=str(e))
            return False
    
    def invalidate_all(self, symbol: str) -> bool:
        """Invalidate all cache entries for a symbol."""
        prefixes = ["indicators", "candles", "sentiment", "signal"]
        success = True
        for prefix in prefixes:
            if not self.invalidate(prefix, symbol):
                success = False
        return success
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            info = self.client.info()
            return {
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
                )
            }
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {}


# Singleton instance
_cache_manager = None

def get_cache() -> CacheManager:
    """Get the cache manager singleton."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
