"""
Live Data Status Component.

Tracks and displays live data source status (API keys, data freshness).
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
from dataclasses import dataclass

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except (ImportError, TypeError):
    # Fallback if yfinance not available or version incompatible
    YFINANCE_AVAILABLE = False
    yf = None

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DataSourceStatus:
    """Status of a single data source."""
    name: str
    status: str  # 'live', 'slow', 'down', 'unknown'
    last_update: Optional[datetime]
    api_key: Optional[str]
    response_time_ms: Optional[float] = None
    
    @property
    def freshness(self) -> str:
        """Get data freshness description."""
        if self.last_update is None:
            return "Never"
        
        diff = datetime.utcnow() - self.last_update
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "real-time"
        elif seconds < 300:
            return f"{int(seconds / 60)}min ago"
        else:
            return f"{int(seconds / 60)}min ago (stale)"


class LiveDataStatus:
    """
    Track and display live data source status.
    
    Monitors:
    - yfinance (free, no API key needed)
    - Alpha Vantage (requires API key)
    - News API (if configured)
    - Sentiment API (if configured)
    """
    
    def __init__(self):
        self.data_sources = {
            'yfinance': {
                'name': 'Yahoo Finance',
                'api_key': None,
                'status': 'unknown',
                'last_update': None,
                'test_symbol': 'AAPL'
            },
            'alpha_vantage': {
                'name': 'Alpha Vantage',
                'api_key': settings.ALPHA_VANTAGE_KEY,
                'status': 'unknown',
                'last_update': None,
                'test_symbol': 'AAPL'
            },
            'news_api': {
                'name': 'News API',
                'api_key': None,  # Not currently configured
                'status': 'unknown',
                'last_update': None
            },
            'sentiment': {
                'name': 'Sentiment Analysis',
                'api_key': settings.OPENAI_API_KEY if settings.OPENAI_API_KEY else None,
                'status': 'unknown',
                'last_update': None
            }
        }
    
    def check_yfinance(self) -> Dict:
        """Check yfinance API status."""
        if not YFINANCE_AVAILABLE:
            return {
                'status': 'unknown',
                'last_update': None,
                'response_time_ms': None
            }
        
        try:
            start_time = datetime.utcnow()
            ticker = yf.Ticker('AAPL')
            info = ticker.info
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            if info and 'symbol' in info:
                return {
                    'status': 'live',
                    'last_update': datetime.utcnow(),
                    'response_time_ms': response_time
                }
        except Exception as e:
            logger.warning(f"yfinance check failed: {e}")
        
        return {
            'status': 'down',
            'last_update': None,
            'response_time_ms': None
        }
    
    def check_alpha_vantage(self) -> Dict:
        """Check Alpha Vantage API status."""
        api_key = self.data_sources['alpha_vantage']['api_key']
        
        # No API key configured
        if not api_key or api_key == 'demo':
            return {
                'status': 'unknown',
                'last_update': None,
                'response_time_ms': None
            }
        
        try:
            start_time = datetime.utcnow()
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': 'AAPL',
                'interval': '1min',
                'apikey': api_key
            }
            response = requests.get(url, params=params, timeout=5)
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                if 'Time Series (1min)' in data or 'Note' in data:
                    # Note usually means rate limit, but API is working
                    status = 'slow' if 'Note' in data else 'live'
                    return {
                        'status': status,
                        'last_update': datetime.utcnow(),
                        'response_time_ms': response_time
                    }
        except requests.exceptions.Timeout:
            return {
                'status': 'slow',
                'last_update': datetime.utcnow(),
                'response_time_ms': None
            }
        except Exception as e:
            logger.warning(f"Alpha Vantage check failed: {e}")
        
        return {
            'status': 'down',
            'last_update': None,
            'response_time_ms': None
        }
    
    def check_news_api(self) -> Dict:
        """Check News API status (if configured)."""
        # News API not currently configured
        return {
            'status': 'unknown',
            'last_update': None,
            'response_time_ms': None
        }
    
    def check_sentiment_api(self) -> Dict:
        """Check Sentiment API status (OpenAI if configured)."""
        api_key = self.data_sources['sentiment']['api_key']
        
        if not api_key:
            return {
                'status': 'unknown',
                'last_update': None,
                'response_time_ms': None
            }
        
        # Simple check - just verify key format
        if api_key.startswith('sk-') and len(api_key) > 40:
            return {
                'status': 'live',
                'last_update': datetime.utcnow(),
                'response_time_ms': None
            }
        
        return {
            'status': 'unknown',
            'last_update': None,
            'response_time_ms': None
        }
    
    def check_all_sources(self) -> Dict[str, DataSourceStatus]:
        """
        Check status of all data sources.
        
        Returns:
            Dictionary mapping source name to DataSourceStatus
        """
        results = {}
        
        # Check each source
        yf_result = self.check_yfinance()
        av_result = self.check_alpha_vantage()
        news_result = self.check_news_api()
        sentiment_result = self.check_sentiment_api()
        
        # Update internal state
        self.data_sources['yfinance'].update(yf_result)
        self.data_sources['alpha_vantage'].update(av_result)
        self.data_sources['news_api'].update(news_result)
        self.data_sources['sentiment'].update(sentiment_result)
        
        # Build results
        for key, config in self.data_sources.items():
            results[key] = DataSourceStatus(
                name=config['name'],
                status=config['status'],
                last_update=config.get('last_update'),
                api_key=config.get('api_key'),
                response_time_ms=config.get('response_time_ms')
            )
        
        return results
    
    def get_status_summary(self) -> Dict:
        """
        Get summary status for dashboard display.
        
        Returns:
            Dictionary with active_count, total_count, last_update
        """
        statuses = self.check_all_sources()
        
        active_count = sum(1 for s in statuses.values() if s.status == 'live')
        total_count = sum(1 for s in statuses.values() if s.status != 'unknown')
        
        # Get most recent update
        updates = [s.last_update for s in statuses.values() if s.last_update]
        last_update = max(updates) if updates else None
        
        return {
            'active_count': active_count,
            'total_count': total_count if total_count > 0 else len(statuses),
            'last_update': last_update,
            'statuses': statuses
        }


def get_live_status() -> Dict:
    """Get live data status (convenience function)."""
    monitor = LiveDataStatus()
    return monitor.get_status_summary()
