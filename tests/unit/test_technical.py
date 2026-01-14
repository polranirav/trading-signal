"""
Unit tests for technical analysis module.

Tests:
- Indicator calculations (RSI, MACD, BB, SMA)
- Technical score calculation
- Signal generation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Sample price data for testing
@pytest.fixture
def sample_price_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate 100 days of price data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Start at $100, random walk
    close_prices = [100.0]
    for i in range(99):
        change = np.random.normal(0, 2)
        close_prices.append(max(50, close_prices[-1] + change))
    
    close = np.array(close_prices)
    
    # Generate OHLCV
    data = {
        'time': dates,
        'symbol': 'TEST',
        'open': close * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'high': close * (1 + np.random.uniform(0.005, 0.02, 100)),
        'low': close * (1 - np.random.uniform(0.005, 0.02, 100)),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, 100),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def uptrend_data():
    """Create data with clear uptrend."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Strong uptrend: +1% daily on average
    close = 100 * (1.01 ** np.arange(100))
    
    data = {
        'time': dates,
        'symbol': 'UPTREND',
        'open': close * 0.99,
        'high': close * 1.02,
        'low': close * 0.98,
        'close': close,
        'volume': np.full(100, 5000000),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def downtrend_data():
    """Create data with clear downtrend."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Strong downtrend: -0.5% daily on average
    close = 100 * (0.995 ** np.arange(100))
    
    data = {
        'time': dates,
        'symbol': 'DOWNTREND',
        'open': close * 1.01,
        'high': close * 1.02,
        'low': close * 0.98,
        'close': close,
        'volume': np.full(100, 5000000),
    }
    
    return pd.DataFrame(data)


class TestTechnicalAnalyzer:
    """Test suite for TechnicalAnalyzer class."""
    
    def test_compute_all_indicators(self, sample_price_data):
        """Test that all indicators are computed."""
        from src.analytics.technical import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(sample_price_data)
        
        # Check that key indicators exist
        expected_columns = [
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'sma_20', 'sma_50', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'obv', 'mfi'
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_rsi_range(self, sample_price_data):
        """Test that RSI is within 0-100 range."""
        from src.analytics.technical import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(sample_price_data)
        
        rsi = result['rsi_14'].dropna()
        assert (rsi >= 0).all(), "RSI should be >= 0"
        assert (rsi <= 100).all(), "RSI should be <= 100"
    
    def test_bollinger_band_relationships(self, sample_price_data):
        """Test that BB upper > middle > lower."""
        from src.analytics.technical import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(sample_price_data)
        
        # After warmup period
        valid = result.dropna(subset=['bb_upper', 'bb_middle', 'bb_lower'])
        
        assert (valid['bb_upper'] >= valid['bb_middle']).all(), "BB upper should be >= middle"
        assert (valid['bb_middle'] >= valid['bb_lower']).all(), "BB middle should be >= lower"
    
    def test_sma_ordering_uptrend(self, uptrend_data):
        """In uptrend, SMA20 > SMA50 > SMA200."""
        from src.analytics.technical import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(uptrend_data)
        
        # Check last row (after SMA200 has values)
        last = result.iloc[-1]
        
        # In strong uptrend, shorter SMAs should be higher
        assert last['sma_20'] > last['sma_50'], "SMA20 should > SMA50 in uptrend"
    
    def test_technical_score_range(self, sample_price_data):
        """Test that technical score is between 0 and 1."""
        from src.analytics.technical import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(sample_price_data)
        score = analyzer.calculate_technical_score(result)
        
        assert 0 <= score['technical_score'] <= 1, "Score should be between 0 and 1"
    
    def test_signal_types(self, sample_price_data):
        """Test that signal type is valid."""
        from src.analytics.technical import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(sample_price_data)
        score = analyzer.calculate_technical_score(result)
        
        valid_signals = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
        assert score['signal_type'] in valid_signals
    
    def test_uptrend_bullish_score(self, uptrend_data):
        """Test that uptrend data produces bullish score."""
        from src.analytics.technical import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(uptrend_data)
        score = analyzer.calculate_technical_score(result)
        
        # Uptrend should have score > 0.5 (bullish)
        assert score['technical_score'] > 0.5, f"Uptrend should be bullish, got {score['technical_score']}"
    
    def test_downtrend_bearish_score(self, downtrend_data):
        """Test that downtrend data produces bearish score."""
        from src.analytics.technical import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(downtrend_data)
        score = analyzer.calculate_technical_score(result)
        
        # Downtrend should have score < 0.5 (bearish)
        assert score['technical_score'] < 0.5, f"Downtrend should be bearish, got {score['technical_score']}"
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        from src.analytics.technical import TechnicalAnalyzer
        
        # Only 10 rows
        small_df = pd.DataFrame({
            'time': pd.date_range(end=datetime.now(), periods=10, freq='D'),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000000] * 10,
        })
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(small_df)
        
        # Should return original df when insufficient data
        assert len(result) == 10
    
    def test_rationale_is_string(self, sample_price_data):
        """Test that rationale is a non-empty string."""
        from src.analytics.technical import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(sample_price_data)
        score = analyzer.calculate_technical_score(result)
        
        assert isinstance(score['rationale'], str)
        assert len(score['rationale']) > 0


class TestSignalGenerator:
    """Test suite for SignalGenerator class."""
    
    def test_signal_has_required_fields(self, sample_price_data, mocker):
        """Test that generated signal has all required fields."""
        from src.analytics.signals import SignalGenerator
        
        # Mock database
        mock_db = mocker.Mock()
        mock_db.get_candles.return_value = sample_price_data
        
        mock_cache = mocker.Mock()
        
        mocker.patch('src.analytics.signals.get_database', return_value=mock_db)
        mocker.patch('src.analytics.signals.get_cache', return_value=mock_cache)
        
        generator = SignalGenerator()
        generator.db = mock_db
        generator.cache = mock_cache
        
        signal = generator.generate_signal("TEST")
        
        required_fields = [
            'symbol', 'created_at', 'technical_score', 
            'signal_type', 'indicators', 'technical_rationale'
        ]
        
        for field in required_fields:
            assert field in signal, f"Missing field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
