"""
Realistic Market Data Generator for ML Training.

Generates synthetic but realistic OHLCV data that mimics real market behavior.
This allows training the ML model when real API access is limited.

The generated data includes:
- Realistic price movements (trending, mean-reverting, volatile periods)
- Volume patterns (higher on big moves, lower on consolidation)
- Seasonal patterns (gaps, different volatility periods)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.logging_config import get_logger

logger = get_logger(__name__)


class RealisticMarketDataGenerator:
    """
    Generate realistic OHLCV data for training ML models.
    
    Uses Geometric Brownian Motion with regime switching to create
    realistic price series that exhibit real market properties:
    - Trending periods
    - Mean-reversion
    - Volatility clustering
    - Volume-price correlation
    """
    
    # Base prices for common stocks
    BASE_PRICES = {
        "AAPL": 185.0,
        "MSFT": 420.0,
        "GOOGL": 175.0,
        "TSLA": 250.0,
        "NVDA": 140.0,
        "SPY": 500.0,
        "BTC-USD": 95000.0,
        "ETH-USD": 3500.0,
    }
    
    # Volatility profiles (annualized)
    VOLATILITY = {
        "AAPL": 0.25,
        "MSFT": 0.22,
        "GOOGL": 0.28,
        "TSLA": 0.55,
        "NVDA": 0.45,
        "SPY": 0.15,
        "BTC-USD": 0.60,
        "ETH-USD": 0.70,
    }
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
    
    def generate(
        self, 
        symbol: str = "AAPL", 
        days: int = 252,
        interval: str = "1D"
    ) -> pd.DataFrame:
        """
        Generate realistic OHLCV data.
        
        Args:
            symbol: Stock ticker (affects starting price and volatility)
            days: Number of trading days to generate
            interval: Time interval ("1D", "1H", "15M", "5M", "1M")
            
        Returns:
            DataFrame with columns: time, open, high, low, close, volume, symbol
        """
        # Determine number of candles based on interval
        candles_per_day = {
            "1D": 1,
            "4H": 6,  # 6 4-hour periods in trading day
            "1H": 6.5,  # 6.5 hours trading day
            "15M": 26,  # 26 15-min periods
            "5M": 78,   # 78 5-min periods
            "1M": 390,  # 390 1-min periods
        }
        
        num_candles = int(days * candles_per_day.get(interval, 1))
        
        # Get base parameters
        base_price = self.BASE_PRICES.get(symbol, 100.0)
        annual_vol = self.VOLATILITY.get(symbol, 0.25)
        
        # Scale volatility to interval
        intervals_per_year = 252 * candles_per_day.get(interval, 1)
        interval_vol = annual_vol / np.sqrt(intervals_per_year)
        
        logger.info(f"Generating {num_candles} candles for {symbol} ({interval})")
        
        # Generate price series with regime switching
        prices = self._generate_price_series(
            start_price=base_price,
            num_candles=num_candles,
            volatility=interval_vol
        )
        
        # Generate OHLC from close prices
        ohlc = self._generate_ohlc(prices, interval_vol)
        
        # Generate volume
        volumes = self._generate_volume(prices, ohlc, symbol)
        
        # Generate timestamps
        timestamps = self._generate_timestamps(num_candles, interval)
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': timestamps,
            'symbol': symbol,
            'open': ohlc['open'],
            'high': ohlc['high'],
            'low': ohlc['low'],
            'close': ohlc['close'],
            'volume': volumes
        })
        
        return df
    
    def _generate_price_series(
        self, 
        start_price: float,
        num_candles: int,
        volatility: float,
        drift: float = 0.0001
    ) -> np.ndarray:
        """
        Generate price series using Geometric Brownian Motion with regime switching.
        """
        prices = np.zeros(num_candles)
        prices[0] = start_price
        
        # Regime switching: trending, mean-reverting, volatile
        regime_probs = [0.4, 0.4, 0.2]  # 40% trending, 40% mean-reverting, 20% volatile
        regime_lengths = [20, 40]  # Min/max regime length
        
        current_regime = np.random.choice(3, p=regime_probs)
        regime_counter = 0
        regime_length = np.random.randint(*regime_lengths)
        trend_direction = np.random.choice([-1, 1])
        
        for i in range(1, num_candles):
            # Switch regime occasionally
            regime_counter += 1
            if regime_counter >= regime_length:
                current_regime = np.random.choice(3, p=regime_probs)
                regime_counter = 0
                regime_length = np.random.randint(*regime_lengths)
                trend_direction = np.random.choice([-1, 1])
            
            # Calculate return based on regime
            if current_regime == 0:  # Trending
                expected_return = drift * trend_direction * 2
                vol_mult = 0.8
            elif current_regime == 1:  # Mean-reverting
                mean_price = start_price
                expected_return = 0.01 * (mean_price - prices[i-1]) / prices[i-1]
                vol_mult = 0.6
            else:  # Volatile
                expected_return = 0
                vol_mult = 2.0
            
            # Generate return
            random_shock = np.random.normal(0, volatility * vol_mult)
            return_pct = expected_return + random_shock
            
            # Apply return
            prices[i] = prices[i-1] * (1 + return_pct)
            
            # Prevent negative prices
            prices[i] = max(prices[i], 1.0)
        
        return prices
    
    def _generate_ohlc(
        self, 
        closes: np.ndarray,
        volatility: float
    ) -> dict:
        """Generate open, high, low from close prices."""
        n = len(closes)
        
        opens = np.zeros(n)
        highs = np.zeros(n)
        lows = np.zeros(n)
        
        opens[0] = closes[0] * (1 + np.random.uniform(-0.002, 0.002))
        
        for i in range(1, n):
            # Open is typically near previous close (with gap sometimes)
            gap = np.random.normal(0, 0.001)  # Small gap
            if np.random.random() < 0.05:  # 5% chance of bigger gap
                gap = np.random.normal(0, 0.01)
            opens[i] = closes[i-1] * (1 + gap)
        
        for i in range(n):
            # High and low based on open/close range plus some wick
            base_range = abs(closes[i] - opens[i])
            wick_range = closes[i] * volatility * np.random.uniform(0.5, 1.5)
            
            if closes[i] > opens[i]:  # Bullish candle
                highs[i] = closes[i] + wick_range * np.random.uniform(0, 0.5)
                lows[i] = opens[i] - wick_range * np.random.uniform(0, 0.5)
            else:  # Bearish candle
                highs[i] = opens[i] + wick_range * np.random.uniform(0, 0.5)
                lows[i] = closes[i] - wick_range * np.random.uniform(0, 0.5)
            
            # Ensure high > low
            lows[i] = min(lows[i], min(opens[i], closes[i]))
            highs[i] = max(highs[i], max(opens[i], closes[i]))
        
        return {
            'open': np.round(opens, 2),
            'high': np.round(highs, 2),
            'low': np.round(lows, 2),
            'close': np.round(closes, 2)
        }
    
    def _generate_volume(
        self, 
        prices: np.ndarray,
        ohlc: dict,
        symbol: str
    ) -> np.ndarray:
        """Generate realistic volume correlated with price moves."""
        n = len(prices)
        
        # Base volume depends on stock
        base_volumes = {
            "AAPL": 50_000_000,
            "MSFT": 25_000_000,
            "GOOGL": 20_000_000,
            "TSLA": 100_000_000,
            "NVDA": 40_000_000,
            "SPY": 80_000_000,
            "BTC-USD": 30_000_000_000,  # Notional USD
            "ETH-USD": 15_000_000_000,
        }
        
        base_vol = base_volumes.get(symbol, 10_000_000)
        
        volumes = np.zeros(n)
        for i in range(n):
            # Higher volume on bigger moves
            price_change = abs(ohlc['close'][i] - ohlc['open'][i]) / ohlc['open'][i]
            volume_multiplier = 1 + price_change * 10  # More volume on big moves
            
            # Random variation
            random_mult = np.random.lognormal(0, 0.3)
            
            volumes[i] = base_vol * volume_multiplier * random_mult
        
        return volumes.astype(int)
    
    def _generate_timestamps(
        self, 
        num_candles: int,
        interval: str
    ) -> pd.DatetimeIndex:
        """Generate timestamps for candles."""
        interval_deltas = {
            "1D": timedelta(days=1),
            "4H": timedelta(hours=4),
            "1H": timedelta(hours=1),
            "15M": timedelta(minutes=15),
            "5M": timedelta(minutes=5),
            "1M": timedelta(minutes=1),
        }
        
        delta = interval_deltas.get(interval, timedelta(days=1))
        
        # Start from recent past
        end_time = datetime.now().replace(second=0, microsecond=0)
        start_time = end_time - delta * num_candles
        
        timestamps = pd.date_range(start=start_time, periods=num_candles, freq=delta)
        
        return timestamps


def generate_training_data(
    symbols: list = None,
    days: int = 200,
    interval: str = "1D"
) -> dict:
    """
    Generate realistic training data for multiple symbols.
    
    Returns:
        Dict mapping symbol to DataFrame
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    generator = RealisticMarketDataGenerator(seed=42)
    data = {}
    
    for symbol in symbols:
        df = generator.generate(symbol, days, interval)
        data[symbol] = df
        logger.info(f"Generated {len(df)} candles for {symbol}")
    
    return data


if __name__ == "__main__":
    # Test the generator
    generator = RealisticMarketDataGenerator(seed=42)
    df = generator.generate("AAPL", days=30, interval="1D")
    print(f"Generated {len(df)} candles")
    print(df.head(10))
    print(df.tail(5))
