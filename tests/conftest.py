"""
Pytest configuration and fixtures.

Provides common fixtures for testing:
- Database session
- Sample price data
- Mock signals
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Generate realistic price movements
    returns = np.random.normal(0.001, 0.02, 100)
    close = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'time': dates,
        'open': close * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'high': close * (1 + np.random.uniform(0, 0.02, 100)),
        'low': close * (1 - np.random.uniform(0, 0.02, 100)),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    return df


@pytest.fixture
def sample_news_data():
    """Generate sample news data for testing."""
    return [
        {
            'headline': 'Company reports strong quarterly earnings',
            'published_at': (datetime.now() - timedelta(days=5)).isoformat(),
            'source': 'Reuters'
        },
        {
            'headline': 'Analyst upgrades rating to buy',
            'published_at': (datetime.now() - timedelta(days=10)).isoformat(),
            'source': 'Bloomberg'
        },
        {
            'headline': 'CEO announces expansion plans',
            'published_at': (datetime.now() - timedelta(days=15)).isoformat(),
            'source': 'CNBC'
        },
    ]


@pytest.fixture
def sample_signal():
    """Generate sample trading signal for testing."""
    return {
        'symbol': 'AAPL',
        'signal_type': 'BUY',
        'confluence_score': 0.72,
        'technical_score': 0.68,
        'sentiment_score': 0.75,
        'ml_score': 0.70,
        'var_95': 0.035,
        'suggested_position_size': 0.02,
        'price_at_signal': 175.50,
        'technical_rationale': 'RSI is bullish (45), MACD shows positive momentum.',
        'sentiment_rationale': 'News sentiment is positive based on 3 articles.',
        'created_at': datetime.now()
    }


@pytest.fixture
def mock_db(mocker):
    """Mock database for testing."""
    mock = mocker.MagicMock()
    mocker.patch('src.data.persistence.get_database', return_value=mock)
    return mock


@pytest.fixture
def mock_cache(mocker):
    """Mock cache for testing."""
    mock = mocker.MagicMock()
    mocker.patch('src.data.cache.get_cache', return_value=mock)
    return mock
