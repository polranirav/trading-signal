"""
Feature Engineering Module for ML Predictions.

Calculates technical indicators as specified in research:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (20, 50, 200)
- ATR (Average True Range)
- OBV (On-Balance Volume)
- Price Momentum
- Volatility
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.logging_config import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Calculate technical indicators for ML model."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns: time, open, high, low, close, volume
        """
        self.df = df.copy()
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        """
        Relative Strength Index.
        > 70: Overbought (likely to fall)
        < 30: Oversold (likely to rise)
        """
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        return self.df
    
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence).
        Positive histogram: Uptrend momentum
        Negative histogram: Downtrend momentum
        """
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()
        
        self.df['macd'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']
        
        return self.df
    
    def add_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands for volatility.
        Near upper band: Price may fall
        Near lower band: Price may rise
        """
        sma = self.df['close'].rolling(window=period).mean()
        std = self.df['close'].rolling(window=period).std()
        
        self.df['bb_upper'] = sma + (std * std_dev)
        self.df['bb_middle'] = sma
        self.df['bb_lower'] = sma - (std * std_dev)
        self.df['bb_width'] = self.df['bb_upper'] - self.df['bb_lower']
        
        # Position within bands (0 = at lower, 1 = at upper)
        self.df['bb_position'] = (self.df['close'] - self.df['bb_lower']) / self.df['bb_width'].replace(0, np.nan)
        
        return self.df
    
    def add_moving_averages(self) -> pd.DataFrame:
        """
        Simple Moving Averages.
        Price > MA50: Uptrend
        Price < MA50: Downtrend
        """
        self.df['ma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['ma_50'] = self.df['close'].rolling(window=50).mean()
        self.df['ma_200'] = self.df['close'].rolling(window=200).mean()
        
        # Trend indicators
        self.df['price_above_ma20'] = (self.df['close'] > self.df['ma_20']).astype(int)
        self.df['price_above_ma50'] = (self.df['close'] > self.df['ma_50']).astype(int)
        self.df['ma20_above_ma50'] = (self.df['ma_20'] > self.df['ma_50']).astype(int)
        
        return self.df
    
    def add_atr(self, period: int = 14) -> pd.DataFrame:
        """
        Average True Range (volatility indicator).
        High ATR: High volatility
        Low ATR: Low volatility
        """
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift()).abs()
        low_close = (self.df['low'] - self.df['close'].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(window=period).mean()
        
        # Normalized ATR (as percentage of price)
        self.df['atr_pct'] = (self.df['atr'] / self.df['close']) * 100
        
        return self.df
    
    def add_obv(self) -> pd.DataFrame:
        """
        On-Balance Volume.
        Rising OBV: Buying pressure
        Falling OBV: Selling pressure
        """
        price_change = self.df['close'].diff()
        volume_direction = np.where(price_change > 0, self.df['volume'],
                                   np.where(price_change < 0, -self.df['volume'], 0))
        
        self.df['obv'] = pd.Series(volume_direction).cumsum()
        
        # OBV momentum (rate of change)
        self.df['obv_momentum'] = self.df['obv'].diff(5)
        
        return self.df
    
    def add_price_momentum(self) -> pd.DataFrame:
        """
        Price momentum indicators.
        Positive: Upward momentum
        Negative: Downward momentum
        """
        # Percentage change
        self.df['momentum'] = self.df['close'].pct_change() * 100
        
        # Lagged momentum
        self.df['momentum_lag1'] = self.df['momentum'].shift(1)
        self.df['momentum_lag3'] = self.df['momentum'].shift(3)
        self.df['momentum_lag5'] = self.df['momentum'].shift(5)
        
        # Rate of change (5-period)
        self.df['roc_5'] = ((self.df['close'] - self.df['close'].shift(5)) / 
                           self.df['close'].shift(5)) * 100
        
        return self.df
    
    def add_volatility(self, period: int = 20) -> pd.DataFrame:
        """
        Rolling volatility (standard deviation of returns).
        High volatility: Larger price swings expected
        Low volatility: Smaller price swings expected
        """
        returns = self.df['close'].pct_change()
        self.df['volatility'] = returns.rolling(window=period).std() * 100
        
        # Volatility ratio (current vs historical)
        self.df['volatility_ratio'] = self.df['volatility'] / self.df['volatility'].rolling(50).mean()
        
        return self.df
    
    def add_labels(self, future_periods: int = 1) -> pd.DataFrame:
        """
        Create target labels for ML training.
        1 = Price went UP (close[t+n] > close[t])
        0 = Price went DOWN (close[t+n] <= close[t])
        """
        self.df['target'] = (
            self.df['close'].shift(-future_periods) > self.df['close']
        ).astype(int)
        
        return self.df
    
    def get_all_features(self, add_labels: bool = True) -> pd.DataFrame:
        """
        Calculate all features.
        
        Returns:
            DataFrame with all technical indicators calculated.
        """
        logger.info("Calculating all technical features...")
        
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_moving_averages()
        self.add_atr()
        self.add_obv()
        self.add_price_momentum()
        self.add_volatility()
        
        if add_labels:
            self.add_labels()
        
        # Drop NaN rows (first ~50 rows will have NaN from MA200)
        original_len = len(self.df)
        self.df = self.df.dropna()
        logger.info(f"Calculated features. Rows: {len(self.df)} (dropped {original_len - len(self.df)} NaN rows)")
        
        return self.df
    
    @property
    def feature_columns(self) -> list:
        """
        Get list of feature columns for ML model.
        """
        return [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position',
            'ma_20', 'ma_50', 'ma_200',
            'price_above_ma20', 'price_above_ma50', 'ma20_above_ma50',
            'atr', 'atr_pct',
            'obv_momentum',
            'momentum', 'momentum_lag1', 'momentum_lag3', 'roc_5',
            'volatility', 'volatility_ratio'
        ]


def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to prepare features for prediction.
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        DataFrame with features calculated (no labels)
    """
    engineer = FeatureEngineer(df)
    return engineer.get_all_features(add_labels=False)
