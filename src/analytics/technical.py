"""
Technical Analysis Engine using TA-Lib.

Computes all technical indicators and generates signals:

Advanced Momentum Indicators:
- RSI (9, 14, 21) - Overbought/Oversold, multiple periods
- MACD (12, 26, 9) - Trend momentum
- Stochastic (14, 3, 3) - Momentum oscillator
- Rate of Change (ROC) - Multiple periods (12, 25, 50)
- Momentum Oscillator - Multiple periods (10, 14, 20)
- Williams %R - Overbought/oversold
- Commodity Channel Index (CCI) - Momentum divergence
- Stochastic RSI - More sensitive momentum
- True Strength Index (TSI) - Double-smoothed momentum
- Awesome Oscillator - Momentum acceleration
- Percentage Price Oscillator (PPO) - Momentum relative to price

Trend:
- SMA (20, 50, 200) - Moving averages
- ADX - Trend strength

Volatility:
- Bollinger Bands (20, 2σ)
- ATR (14) - Average True Range

Volume:
- OBV - On-Balance Volume
- MFI - Money Flow Index

Usage:
    analyzer = TechnicalAnalyzer()
    df_with_indicators = analyzer.compute_all(price_df)
    score = analyzer.calculate_technical_score(df_with_indicators)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from src.logging_config import get_logger

logger = get_logger(__name__)

if not TALIB_AVAILABLE:
    logger.warning("TA-Lib not installed. Using pandas-based fallbacks for indicators.")


class TechnicalAnalyzer:
    """
    Technical analysis engine for computing indicators and signals.
    
    All indicators are computed using TA-Lib for speed and accuracy.
    Falls back to pandas implementations if TA-Lib is not available.
    """
    
    # Signal thresholds based on research
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    ADX_STRONG_TREND = 25
    
    def __init__(self):
        self.use_talib = TALIB_AVAILABLE
        if not self.use_talib:
            logger.warning("TA-Lib not available, using pandas fallbacks")
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators.
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
        
        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()
        
        # Ensure we have enough data
        if len(df) < 50:
            logger.warning("Insufficient data for indicators", rows=len(df))
            return df
        
        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)
        
        # Compute all indicator groups
        df = self._compute_momentum_indicators(df)
        df = self._compute_trend_indicators(df)
        df = self._compute_volatility_indicators(df)
        df = self._compute_volume_indicators(df)
        
        logger.info("Computed all indicators", rows=len(df))
        return df
    
    # ============ MOMENTUM INDICATORS ============
    
    def _compute_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute advanced momentum indicators.
        
        Includes:
        - RSI (multiple periods: 9, 14, 21)
        - MACD (12, 26, 9)
        - Stochastic (14, 3, 3)
        - Rate of Change (ROC) - multiple periods (12, 25, 50)
        - Momentum Oscillator - multiple periods
        - Williams %R - overbought/oversold
        - Commodity Channel Index (CCI) - momentum divergence
        - Stochastic RSI - more sensitive momentum
        - True Strength Index (TSI) - double-smoothed momentum
        - Awesome Oscillator - momentum acceleration
        - Percentage Price Oscillator (PPO) - momentum relative to price
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values.astype(float) if 'volume' in df.columns else None
        
        # RSI - multiple periods (9, 14, 21)
        for period in [9, 14, 21]:
            if self.use_talib:
                df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
            else:
                df[f'rsi_{period}'] = self._rsi_pandas(df['close'], period)
        
        # RSI Signal (using 14-period as primary)
        df['rsi_signal'] = df['rsi_14'].apply(self._rsi_signal_label)
        
        # MACD (12, 26, 9)
        if self.use_talib:
            df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
        else:
            df['macd'], df['macd_signal'], df['macd_histogram'] = self._macd_pandas(
                df['close'], 12, 26, 9
            )
        
        # MACD Crossover
        df['macd_crossover'] = self._detect_macd_crossover(
            df['macd'], df['macd_signal']
        )
        
        # Stochastic (14, 3, 3)
        if self.use_talib:
            df['stoch_k'], df['stoch_d'] = talib.STOCH(
                high, low, close,
                fastk_period=14, slowk_period=3, slowd_period=3
            )
        else:
            df['stoch_k'], df['stoch_d'] = self._stochastic_pandas(
                df['high'], df['low'], df['close'], 14, 3, 3
            )
        
        # Rate of Change (ROC) - multiple periods (12, 25, 50)
        for period in [12, 25, 50]:
            if self.use_talib:
                df[f'roc_{period}'] = talib.ROC(close, timeperiod=period)
            else:
                df[f'roc_{period}'] = df['close'].pct_change(period) * 100
        
        # Momentum Oscillator - multiple periods (10, 14, 20)
        for period in [10, 14, 20]:
            if self.use_talib:
                df[f'momentum_{period}'] = talib.MOM(close, timeperiod=period)
            else:
                df[f'momentum_{period}'] = df['close'].diff(period)
        
        # Williams %R - overbought/oversold
        if self.use_talib:
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        else:
            df['williams_r'] = self._williams_r_pandas(df['high'], df['low'], df['close'], 14)
        
        # Commodity Channel Index (CCI) - momentum divergence
        if self.use_talib:
            df['cci'] = talib.CCI(high, low, close, timeperiod=20)
        else:
            df['cci'] = self._cci_pandas(df['high'], df['low'], df['close'], 20)
        
        # Stochastic RSI - more sensitive momentum
        df['stoch_rsi'] = self._stochastic_rsi_pandas(df['close'], 14, 3, 3)
        
        # True Strength Index (TSI) - double-smoothed momentum
        df['tsi'] = self._tsi_pandas(df['close'], 25, 13)
        
        # Awesome Oscillator - momentum acceleration
        df['awesome_oscillator'] = self._awesome_oscillator_pandas(df['high'], df['low'], 5, 34)
        
        # Percentage Price Oscillator (PPO) - momentum relative to price
        df['ppo'], df['ppo_signal'], df['ppo_histogram'] = self._ppo_pandas(df['close'], 12, 26, 9)
        
        return df
    
    def _rsi_signal_label(self, rsi: float) -> str:
        """Convert RSI value to signal label."""
        if pd.isna(rsi):
            return 'NEUTRAL'
        if rsi >= self.RSI_OVERBOUGHT:
            return 'OVERBOUGHT'
        elif rsi <= self.RSI_OVERSOLD:
            return 'OVERSOLD'
        return 'NEUTRAL'
    
    def _detect_macd_crossover(
        self, 
        macd: pd.Series, 
        signal: pd.Series
    ) -> pd.Series:
        """Detect MACD crossovers."""
        result = pd.Series(index=macd.index, dtype='object')
        result[:] = 'NONE'
        
        # Get previous values
        macd_prev = macd.shift(1)
        signal_prev = signal.shift(1)
        
        # Bullish crossover: MACD crosses above signal
        bullish = (macd > signal) & (macd_prev <= signal_prev)
        result[bullish] = 'BULLISH'
        
        # Bearish crossover: MACD crosses below signal
        bearish = (macd < signal) & (macd_prev >= signal_prev)
        result[bearish] = 'BEARISH'
        
        return result
    
    # ============ TREND INDICATORS ============
    
    def _compute_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute advanced trend indicators.
        
        Includes:
        - SMA (20, 50, 200)
        - EMA - multiple periods (12, 26, 50, 200)
        - DEMA (Double EMA) - faster response
        - TEMA (Triple EMA) - even faster
        - KAMA (Kaufman Adaptive MA) - volatility-adjusted
        - Parabolic SAR - trend reversal signals
        - Ichimoku Cloud - complete trend system (5 components)
        - ADX (enhanced) - with DI+ and DI- analysis
        - Aroon Indicator - trend strength and direction
        - Directional Movement Index (DMI) - trend confirmation
        - Trend Strength Score - composite metric
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages - multiple periods (12, 26, 50, 200)
        for period in [12, 26, 50, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # DEMA (Double EMA) - faster response
        df['dema'] = self._dema_pandas(df['close'], 14)
        
        # TEMA (Triple EMA) - even faster
        df['tema'] = self._tema_pandas(df['close'], 14)
        
        # KAMA (Kaufman Adaptive MA) - volatility-adjusted
        df['kama'] = self._kama_pandas(df['close'], 10, 2, 30)
        
        # Parabolic SAR - trend reversal signals
        if self.use_talib:
            df['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        else:
            df['sar'] = self._parabolic_sar_pandas(df['high'], df['low'], df['close'], 0.02, 0.2)
        
        # Ichimoku Cloud - complete trend system (5 components)
        ichimoku = self._ichimoku_cloud_pandas(df['high'], df['low'], df['close'], 9, 26, 52)
        df['ichimoku_tenkan'] = ichimoku['tenkan']
        df['ichimoku_kijun'] = ichimoku['kijun']
        df['ichimoku_senkou_a'] = ichimoku['senkou_a']
        df['ichimoku_senkou_b'] = ichimoku['senkou_b']
        df['ichimoku_chikou'] = ichimoku['chikou']
        
        # ADX (14) - Trend strength
        if self.use_talib:
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        else:
            df['adx'], df['plus_di'], df['minus_di'] = self._adx_pandas(
                df['high'], df['low'], df['close'], 14
            )
        
        # Aroon Indicator - trend strength and direction
        if self.use_talib:
            df['aroon_down'], df['aroon_up'] = talib.AROON(high, low, timeperiod=14)
            df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']
        else:
            aroon = self._aroon_pandas(df['high'], df['low'], 14)
            df['aroon_down'] = aroon['aroon_down']
            df['aroon_up'] = aroon['aroon_up']
            df['aroon_oscillator'] = aroon['aroon_oscillator']
        
        # Trend signal based on moving averages
        df['trend_signal'] = self._detect_trend_signal(df)
        
        # Golden/Death cross detection
        df['ma_crossover'] = self._detect_ma_crossover(df['sma_50'], df['sma_200'])
        
        # Trend Strength Score - composite metric
        df['trend_strength_score'] = self._calculate_trend_strength_score(df)
        
        return df
    
    def _detect_trend_signal(self, df: pd.DataFrame) -> pd.Series:
        """Determine overall trend based on moving averages."""
        result = pd.Series(index=df.index, dtype='object')
        result[:] = 'SIDEWAYS'
        
        # Strong uptrend: Close > SMA50 > SMA200
        uptrend = (
            (df['close'] > df['sma_50']) & 
            (df['sma_50'] > df['sma_200'])
        )
        result[uptrend] = 'UPTREND'
        
        # Strong downtrend: Close < SMA50 < SMA200
        downtrend = (
            (df['close'] < df['sma_50']) & 
            (df['sma_50'] < df['sma_200'])
        )
        result[downtrend] = 'DOWNTREND'
        
        return result
    
    def _detect_ma_crossover(
        self, 
        fast_ma: pd.Series, 
        slow_ma: pd.Series
    ) -> pd.Series:
        """Detect moving average crossovers (Golden Cross / Death Cross)."""
        result = pd.Series(index=fast_ma.index, dtype='object')
        result[:] = 'NONE'
        
        fast_prev = fast_ma.shift(1)
        slow_prev = slow_ma.shift(1)
        
        # Golden cross: Fast MA crosses above slow MA
        golden = (fast_ma > slow_ma) & (fast_prev <= slow_prev)
        result[golden] = 'GOLDEN_CROSS'
        
        # Death cross: Fast MA crosses below slow MA
        death = (fast_ma < slow_ma) & (fast_prev >= slow_prev)
        result[death] = 'DEATH_CROSS'
        
        return result
    
    # ============ VOLATILITY INDICATORS ============
    
    def _compute_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute advanced volatility indicators.
        
        Includes:
        - Bollinger Bands - multiple periods (20, 50)
        - Keltner Channels - volatility bands with ATR
        - Donchian Channels - price range channels
        - ATR - multiple periods (14, 20, 50)
        - Historical Volatility - multiple windows
        - Garman-Klass Volatility - uses OHLC (more accurate)
        - Parkinson Volatility - uses high/low
        - Rogers-Satchell Volatility - handles drift
        - Yang-Zhang Volatility - combines methods
        - Volatility Ratio - current vs. historical
        - Volatility of Volatility - meta-volatility
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values if 'open' in df.columns else close
        
        # Bollinger Bands - multiple periods (20, 50)
        for period in [20, 50]:
            if self.use_talib:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close, timeperiod=period, nbdevup=2, nbdevdn=2
                )
                df[f'bb_upper_{period}'] = bb_upper
                df[f'bb_middle_{period}'] = bb_middle
                df[f'bb_lower_{period}'] = bb_lower
            else:
                df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (2 * std)
                df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (2 * std)
        
        # Primary Bollinger Bands (20, 2σ) - for backward compatibility
        if self.use_talib:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2
            )
        else:
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (2 * std)
            df['bb_lower'] = df['bb_middle'] - (2 * std)
        
        # Bollinger Band Width and %B
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR - multiple periods (14, 20, 50)
        for period in [14, 20, 50]:
            if self.use_talib:
                df[f'atr_{period}'] = talib.ATR(high, low, close, timeperiod=period)
            else:
                df[f'atr_{period}'] = self._atr_pandas(df['high'], df['low'], df['close'], period)
        
        # Primary ATR (14) - for backward compatibility
        if self.use_talib:
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        else:
            df['atr'] = self._atr_pandas(df['high'], df['low'], df['close'], 14)
        
        # ATR as percentage of price
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # Keltner Channels - volatility bands with ATR
        keltner = self._keltner_channels_pandas(df['high'], df['low'], df['close'], 20, 2.0)
        df['keltner_upper'] = keltner['upper']
        df['keltner_middle'] = keltner['middle']
        df['keltner_lower'] = keltner['lower']
        
        # Donchian Channels - price range channels
        donchian = self._donchian_channels_pandas(df['high'], df['low'], df['close'], 20)
        df['donchian_upper'] = donchian['upper']
        df['donchian_middle'] = donchian['middle']
        df['donchian_lower'] = donchian['lower']
        
        # Historical Volatility - multiple windows (20, 60, 252 days)
        for window in [20, 60, 252]:
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std() * np.sqrt(252) * 100
        
        # Garman-Klass Volatility - uses OHLC (more accurate)
        df['gk_volatility'] = self._garman_klass_volatility_pandas(
            df['open'], df['high'], df['low'], df['close'], 20
        )
        
        # Parkinson Volatility - uses high/low
        df['parkinson_volatility'] = self._parkinson_volatility_pandas(df['high'], df['low'], 20)
        
        # Rogers-Satchell Volatility - handles drift
        df['rs_volatility'] = self._rogers_satchell_volatility_pandas(
            df['open'], df['high'], df['low'], df['close'], 20
        )
        
        # Yang-Zhang Volatility - combines methods
        df['yz_volatility'] = self._yang_zhang_volatility_pandas(
            df['open'], df['high'], df['low'], df['close'], 20
        )
        
        # Volatility Ratio - current vs. historical
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_60']
        
        # Volatility of Volatility - meta-volatility
        df['volatility_of_volatility'] = df['volatility_20'].rolling(20).std()
        
        return df
    
    # ============ VOLUME INDICATORS ============
    
    def _compute_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute OBV, MFI, Volume SMA."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values.astype(float)
        
        # On-Balance Volume
        if self.use_talib:
            df['obv'] = talib.OBV(close, volume)
        else:
            df['obv'] = self._obv_pandas(df['close'], df['volume'])
        
        # Money Flow Index (14)
        if self.use_talib:
            df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
        else:
            df['mfi'] = self._mfi_pandas(
                df['high'], df['low'], df['close'], df['volume'], 14
            )
        
        # Volume SMA
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volume trend
        df['volume_trend'] = df['volume'].pct_change(5)
        
        return df
    
    # ============ TECHNICAL SCORE CALCULATION ============
    
    def calculate_technical_score(self, df: pd.DataFrame) -> Dict:
        """
        Calculate overall technical score (0-1 scale).
        
        Based on research from Project Research documents:
        - RSI: 25% weight
        - MACD: 25% weight
        - Moving Averages: 30% weight
        - Bollinger Bands: 20% weight
        
        Args:
            df: DataFrame with all indicators computed
        
        Returns:
            Dictionary with scores and signal
        """
        if df.empty:
            return {"technical_score": 0.5, "signal_type": "HOLD", "rationale": "No data"}
        
        # Get latest row
        latest = df.iloc[-1]
        
        # Component scores (0-1 scale, 0.5 = neutral)
        rsi_score = self._score_rsi(latest.get('rsi_14'))
        macd_score = self._score_macd(latest)
        ma_score = self._score_moving_averages(latest)
        bb_score = self._score_bollinger(latest)
        
        # Weighted average
        weights = {
            'rsi': 0.25,
            'macd': 0.25,
            'ma': 0.30,
            'bb': 0.20
        }
        
        technical_score = (
            rsi_score * weights['rsi'] +
            macd_score * weights['macd'] +
            ma_score * weights['ma'] +
            bb_score * weights['bb']
        )
        
        # Determine signal type
        signal_type = self._score_to_signal(technical_score)
        
        # Build rationale
        rationale = self._build_rationale(latest, rsi_score, macd_score, ma_score, bb_score)
        
        return {
            "technical_score": round(technical_score, 3),
            "signal_type": signal_type,
            "component_scores": {
                "rsi": round(rsi_score, 3),
                "macd": round(macd_score, 3),
                "moving_averages": round(ma_score, 3),
                "bollinger": round(bb_score, 3)
            },
            "rationale": rationale,
            "as_of": latest.get('time', datetime.utcnow())
        }
    
    def _score_rsi(self, rsi: float) -> float:
        """
        Convert RSI to 0-1 score.
        - RSI < 30: Score near 0.8 (oversold = bullish)
        - RSI > 70: Score near 0.2 (overbought = bearish)
        - RSI = 50: Score = 0.5 (neutral)
        """
        if pd.isna(rsi):
            return 0.5
        
        # Invert RSI: Low RSI = bullish potential
        # Scale to 0-1 where 0.5 = neutral
        # RSI 30 → 0.7, RSI 50 → 0.5, RSI 70 → 0.3
        score = 1 - (rsi / 100)
        
        # Clamp to reasonable range
        return max(0.1, min(0.9, score))
    
    def _score_macd(self, row: pd.Series) -> float:
        """
        Score MACD signal.
        - MACD > Signal: Bullish (0.6-0.8)
        - MACD < Signal: Bearish (0.2-0.4)
        - Crossover adds strength
        """
        macd = row.get('macd')
        signal = row.get('macd_signal')
        histogram = row.get('macd_histogram')
        crossover = row.get('macd_crossover', 'NONE')
        
        if pd.isna(macd) or pd.isna(signal):
            return 0.5
        
        # Base score from MACD vs Signal
        if macd > signal:
            base_score = 0.6
        elif macd < signal:
            base_score = 0.4
        else:
            base_score = 0.5
        
        # Adjust for crossover
        if crossover == 'BULLISH':
            base_score = min(0.85, base_score + 0.15)
        elif crossover == 'BEARISH':
            base_score = max(0.15, base_score - 0.15)
        
        # Adjust for histogram strength
        if not pd.isna(histogram):
            strength = min(abs(histogram) / 2, 0.1)  # Cap adjustment at 0.1
            if histogram > 0:
                base_score = min(0.9, base_score + strength)
            else:
                base_score = max(0.1, base_score - strength)
        
        return base_score
    
    def _score_moving_averages(self, row: pd.Series) -> float:
        """
        Score based on moving average positioning.
        - Price > SMA50 > SMA200: Strong uptrend (0.8)
        - Price < SMA50 < SMA200: Strong downtrend (0.2)
        """
        close = row.get('close')
        sma_50 = row.get('sma_50')
        sma_200 = row.get('sma_200')
        trend = row.get('trend_signal', 'SIDEWAYS')
        ma_cross = row.get('ma_crossover', 'NONE')
        
        if pd.isna(close) or pd.isna(sma_50):
            return 0.5
        
        # Base score from trend
        if trend == 'UPTREND':
            score = 0.7
        elif trend == 'DOWNTREND':
            score = 0.3
        else:
            score = 0.5
        
        # Adjust for golden/death cross
        if ma_cross == 'GOLDEN_CROSS':
            score = min(0.9, score + 0.15)
        elif ma_cross == 'DEATH_CROSS':
            score = max(0.1, score - 0.15)
        
        # Adjust for price distance from SMA50
        if not pd.isna(sma_50) and sma_50 > 0:
            distance_pct = (close - sma_50) / sma_50
            adjustment = min(max(distance_pct * 2, -0.1), 0.1)
            score = max(0.1, min(0.9, score + adjustment))
        
        return score
    
    def _score_bollinger(self, row: pd.Series) -> float:
        """
        Score based on Bollinger Band position.
        - Price near lower band: Oversold (0.7)
        - Price near upper band: Overbought (0.3)
        """
        percent_b = row.get('bb_percent_b')
        
        if pd.isna(percent_b):
            return 0.5
        
        # %B < 0 means below lower band (potential bounce = bullish)
        # %B > 1 means above upper band (potential reversal = bearish)
        # %B = 0.5 means at middle band (neutral)
        
        # Invert for mean reversion logic
        score = 1 - percent_b
        
        # Clamp to reasonable range
        return max(0.15, min(0.85, score))
    
    def _score_to_signal(self, score: float) -> str:
        """
        Convert technical score to signal type.
        
        Based on research thresholds:
        - < 0.30: STRONG_SELL
        - 0.30-0.40: SELL
        - 0.40-0.60: HOLD
        - 0.60-0.70: BUY
        - > 0.70: STRONG_BUY
        """
        if score < 0.30:
            return "STRONG_SELL"
        elif score < 0.40:
            return "SELL"
        elif score < 0.60:
            return "HOLD"
        elif score < 0.70:
            return "BUY"
        else:
            return "STRONG_BUY"
    
    def _build_rationale(
        self,
        row: pd.Series,
        rsi_score: float,
        macd_score: float,
        ma_score: float,
        bb_score: float
    ) -> str:
        """Build human-readable rationale for the signal."""
        parts = []
        
        # RSI
        rsi = row.get('rsi_14')
        if not pd.isna(rsi):
            if rsi >= 70:
                parts.append(f"RSI overbought at {rsi:.1f}")
            elif rsi <= 30:
                parts.append(f"RSI oversold at {rsi:.1f}")
            else:
                parts.append(f"RSI neutral at {rsi:.1f}")
        
        # MACD
        crossover = row.get('macd_crossover', 'NONE')
        if crossover == 'BULLISH':
            parts.append("MACD bullish crossover")
        elif crossover == 'BEARISH':
            parts.append("MACD bearish crossover")
        elif row.get('macd', 0) > row.get('macd_signal', 0):
            parts.append("MACD above signal line")
        else:
            parts.append("MACD below signal line")
        
        # Trend
        trend = row.get('trend_signal', 'SIDEWAYS')
        if trend == 'UPTREND':
            parts.append("Price in uptrend (above SMA50/200)")
        elif trend == 'DOWNTREND':
            parts.append("Price in downtrend (below SMA50/200)")
        
        # Bollinger
        percent_b = row.get('bb_percent_b')
        if not pd.isna(percent_b):
            if percent_b < 0.2:
                parts.append("Near lower Bollinger Band (oversold)")
            elif percent_b > 0.8:
                parts.append("Near upper Bollinger Band (overbought)")
        
        return ". ".join(parts) if parts else "Insufficient data for analysis"
    
    # ============ PANDAS FALLBACK IMPLEMENTATIONS ============
    
    def _rsi_pandas(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using pandas."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _macd_pandas(
        self, 
        series: pd.Series, 
        fast: int, 
        slow: int, 
        signal: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD using pandas."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _stochastic_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int,
        k_smooth: int,
        d_smooth: int
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic using pandas."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        stoch_k = stoch_k.rolling(window=k_smooth).mean()
        stoch_d = stoch_k.rolling(window=d_smooth).mean()
        return stoch_k, stoch_d
    
    def _adx_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX using pandas."""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def _atr_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate ATR using pandas."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _keltner_channels_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        multiplier: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Calculate Keltner Channels using pandas."""
        # Middle line: EMA
        middle = close.ewm(span=period, adjust=False).mean()
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Upper and lower bands
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def _donchian_channels_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> Dict[str, pd.Series]:
        """Calculate Donchian Channels using pandas."""
        # Upper band: highest high
        upper = high.rolling(window=period).max()
        
        # Lower band: lowest low
        lower = low.rolling(window=period).min()
        
        # Middle line: average of upper and lower
        middle = (upper + lower) / 2
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def _garman_klass_volatility_pandas(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Calculate Garman-Klass Volatility using pandas."""
        # Garman-Klass volatility estimator (uses OHLC)
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_price) ** 2
        
        # Garman-Klass estimator
        gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        
        # Annualized volatility
        gk_vol = np.sqrt(gk_var.rolling(window=period).mean()) * np.sqrt(252) * 100
        
        return gk_vol
    
    def _parkinson_volatility_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Calculate Parkinson Volatility using pandas."""
        # Parkinson volatility estimator (uses high/low only)
        log_hl = np.log(high / low) ** 2
        
        # Parkinson estimator
        parkinson_var = (1 / (4 * np.log(2))) * log_hl
        
        # Annualized volatility
        parkinson_vol = np.sqrt(parkinson_var.rolling(window=period).mean()) * np.sqrt(252) * 100
        
        return parkinson_vol
    
    def _rogers_satchell_volatility_pandas(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Calculate Rogers-Satchell Volatility using pandas."""
        # Rogers-Satchell volatility estimator (handles drift)
        log_ho = np.log(high / open_price) * np.log(high / close)
        log_lo = np.log(low / open_price) * np.log(low / close)
        
        # Rogers-Satchell estimator
        rs_var = log_ho.rolling(window=period).sum() + log_lo.rolling(window=period).sum()
        rs_var = rs_var / period
        
        # Annualized volatility
        rs_vol = np.sqrt(rs_var) * np.sqrt(252) * 100
        
        return rs_vol
    
    def _yang_zhang_volatility_pandas(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Calculate Yang-Zhang Volatility using pandas (combines methods)."""
        # Calculate overnight volatility (close to open)
        overnight_var = np.log(close / open_price.shift(1)) ** 2
        
        # Calculate open-to-close volatility
        open_close_var = np.log(open_price / close.shift(1)) ** 2
        
        # Garman-Klass component
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_price) ** 2
        gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        
        # Rogers-Satchell component
        log_ho = np.log(high / open_price) * np.log(high / close)
        log_lo = np.log(low / open_price) * np.log(low / close)
        rs_var = log_ho + log_lo
        
        # Yang-Zhang estimator (combines overnight, open-to-close, RS, GK)
        k = 0.34 / (1.34 + (period + 1) / (period - 1))
        
        yz_var = (
            overnight_var.rolling(window=period).mean() +
            k * open_close_var.rolling(window=period).mean() +
            (1 - k) * rs_var.rolling(window=period).mean()
        )
        
        # Annualized volatility
        yz_vol = np.sqrt(yz_var) * np.sqrt(252) * 100
        
        return yz_vol
    
    def _obv_pandas(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV using pandas."""
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        return (volume * direction).cumsum()
    
    def _mfi_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate MFI using pandas."""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        diff = typical_price.diff()
        positive_flow = raw_money_flow.where(diff > 0, 0).rolling(window=period).sum()
        negative_flow = raw_money_flow.where(diff < 0, 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def _detect_obv_divergence(
        self,
        close: pd.Series,
        obv: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Detect OBV divergence from price."""
        result = pd.Series(index=close.index, dtype='object')
        result[:] = 'NONE'
        
        # Calculate price and OBV momentum
        price_momentum = close.pct_change(period)
        obv_momentum = obv.pct_change(period)
        
        # Bullish divergence: Price down, OBV up
        bullish_div = (price_momentum < 0) & (obv_momentum > 0)
        result[bullish_div] = 'BULLISH'
        
        # Bearish divergence: Price up, OBV down
        bearish_div = (price_momentum > 0) & (obv_momentum < 0)
        result[bearish_div] = 'BEARISH'
        
        return result
    
    def _vpt_pandas(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Price Trend using pandas."""
        # VPT = Previous VPT + Volume * (Current Close - Previous Close) / Previous Close
        price_change_pct = close.pct_change()
        vpt = (volume * price_change_pct).cumsum()
        return vpt
    
    def _ad_line_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Calculate Accumulation/Distribution Line using pandas."""
        # Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Money Flow Volume
        mfv = clv * volume
        
        # Accumulation/Distribution Line
        ad_line = mfv.cumsum()
        return ad_line
    
    def _chaikin_money_flow_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Calculate Chaikin Money Flow using pandas."""
        # Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Money Flow Volume
        mfv = clv * volume
        
        # Chaikin Money Flow = Sum of MFV / Sum of Volume
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf
    
    def _vwap_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Calculate Volume Weighted Average Price using pandas (rolling window)."""
        # Typical price
        typical_price = (high + low + close) / 3
        
        # VWAP = Sum(Price * Volume) / Sum(Volume) over rolling window
        pv_sum = (typical_price * volume).rolling(window=period).sum()
        v_sum = volume.rolling(window=period).sum()
        vwap = pv_sum / v_sum
        
        return vwap
    
    def _pvi_nvi_pandas(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> Dict[str, pd.Series]:
        """Calculate Positive/Negative Volume Index using pandas."""
        pvi = pd.Series(index=close.index, dtype=float)
        nvi = pd.Series(index=close.index, dtype=float)
        
        # Initialize
        pvi.iloc[0] = 1000.0
        nvi.iloc[0] = 1000.0
        
        # Calculate returns
        returns = close.pct_change()
        
        for i in range(1, len(close)):
            if pd.isna(volume.iloc[i]) or pd.isna(volume.iloc[i-1]):
                pvi.iloc[i] = pvi.iloc[i-1]
                nvi.iloc[i] = nvi.iloc[i-1]
                continue
            
            # Positive Volume Index: Only update when volume increases
            if volume.iloc[i] > volume.iloc[i-1]:
                if not pd.isna(returns.iloc[i]):
                    pvi.iloc[i] = pvi.iloc[i-1] * (1 + returns.iloc[i])
                else:
                    pvi.iloc[i] = pvi.iloc[i-1]
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
            
            # Negative Volume Index: Only update when volume decreases
            if volume.iloc[i] < volume.iloc[i-1]:
                if not pd.isna(returns.iloc[i]):
                    nvi.iloc[i] = nvi.iloc[i-1] * (1 + returns.iloc[i])
                else:
                    nvi.iloc[i] = nvi.iloc[i-1]
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        return {
            'pvi': pvi,
            'nvi': nvi
        }
    
    def _ease_of_movement_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Ease of Movement using pandas."""
        # Distance moved
        distance = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        
        # Box ratio (high - low)
        box_ratio = volume / (high - low)
        box_ratio = box_ratio.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Ease of Movement
        eom = distance / box_ratio
        
        # Smooth with moving average
        eom_smoothed = eom.rolling(window=period).mean()
        
        return eom_smoothed
    
    def _force_index_pandas(
        self,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Force Index using pandas."""
        # Force Index = Price Change * Volume
        price_change = close.diff()
        force_index = price_change * volume
        
        # Smooth with EMA
        force_index_smoothed = force_index.ewm(span=period, adjust=False).mean()
        
        return force_index_smoothed
    
    def _volume_oscillator_pandas(
        self,
        volume: pd.Series,
        fast_period: int = 5,
        slow_period: int = 10
    ) -> pd.Series:
        """Calculate Volume Oscillator using pandas."""
        # Volume moving averages
        fast_ma = volume.rolling(window=fast_period).mean()
        slow_ma = volume.rolling(window=slow_period).mean()
        
        # Volume Oscillator = ((Fast MA - Slow MA) / Slow MA) * 100
        volume_osc = ((fast_ma - slow_ma) / slow_ma) * 100
        
        return volume_osc
    
    def _williams_r_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Williams %R using pandas."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def _cci_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Commodity Channel Index using pandas."""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def _stochastic_rsi_pandas(
        self,
        close: pd.Series,
        rsi_period: int = 14,
        stoch_period: int = 3,
        smooth_period: int = 3
    ) -> pd.Series:
        """Calculate Stochastic RSI using pandas."""
        # Calculate RSI first
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic of RSI
        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()
        stoch_rsi = 100 * ((rsi - rsi_min) / (rsi_max - rsi_min))
        
        # Smooth
        stoch_rsi = stoch_rsi.rolling(window=smooth_period).mean()
        return stoch_rsi
    
    def _tsi_pandas(
        self,
        close: pd.Series,
        long_period: int = 25,
        short_period: int = 13
    ) -> pd.Series:
        """Calculate True Strength Index using pandas."""
        # Price change
        pc = close.diff()
        
        # First smoothing (EMA)
        pc_smooth1 = pc.ewm(span=long_period, adjust=False).mean()
        pc_abs_smooth1 = pc.abs().ewm(span=long_period, adjust=False).mean()
        
        # Second smoothing (EMA)
        pc_smooth2 = pc_smooth1.ewm(span=short_period, adjust=False).mean()
        pc_abs_smooth2 = pc_abs_smooth1.ewm(span=short_period, adjust=False).mean()
        
        # TSI
        tsi = 100 * (pc_smooth2 / pc_abs_smooth2)
        return tsi
    
    def _awesome_oscillator_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        fast_period: int = 5,
        slow_period: int = 34
    ) -> pd.Series:
        """Calculate Awesome Oscillator using pandas."""
        typical_price = (high + low) / 2
        fast_sma = typical_price.rolling(window=fast_period).mean()
        slow_sma = typical_price.rolling(window=slow_period).mean()
        awesome_osc = fast_sma - slow_sma
        return awesome_osc
    
    def _ppo_pandas(
        self,
        close: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Percentage Price Oscillator using pandas."""
        # EMAs
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()
        
        # PPO (percentage difference)
        ppo = 100 * ((ema_fast - ema_slow) / ema_slow)
        
        # Signal line (EMA of PPO)
        ppo_signal = ppo.ewm(span=signal_period, adjust=False).mean()
        
        # Histogram
        ppo_histogram = ppo - ppo_signal
        
        return ppo, ppo_signal, ppo_histogram
    
    def _dema_pandas(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Double Exponential Moving Average using pandas."""
        ema1 = close.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        dema = 2 * ema1 - ema2
        return dema
    
    def _tema_pandas(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Triple Exponential Moving Average using pandas."""
        ema1 = close.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        tema = 3 * ema1 - 3 * ema2 + ema3
        return tema
    
    def _kama_pandas(
        self,
        close: pd.Series,
        er_period: int = 10,
        fast_period: int = 2,
        slow_period: int = 30
    ) -> pd.Series:
        """Calculate Kaufman Adaptive Moving Average using pandas."""
        # Efficiency Ratio (ER)
        change = abs(close - close.shift(er_period))
        volatility = close.diff().abs().rolling(window=er_period).sum()
        er = change / volatility
        er = er.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Smoothing Constant (SC)
        sc = (er * (2 / (fast_period + 1) - 2 / (slow_period + 1)) + 2 / (slow_period + 1)) ** 2
        
        # KAMA
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[0] = close.iloc[0]
        
        for i in range(1, len(close)):
            if pd.isna(sc.iloc[i]) or pd.isna(kama.iloc[i-1]) or pd.isna(close.iloc[i]):
                kama.iloc[i] = kama.iloc[i-1]
            else:
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        
        return kama
    
    def _parabolic_sar_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        acceleration: float = 0.02,
        maximum: float = 0.2
    ) -> pd.Series:
        """Calculate Parabolic SAR using pandas."""
        sar = pd.Series(index=close.index, dtype=float)
        ep = pd.Series(index=close.index, dtype=float)  # Extreme Point
        af = pd.Series(index=close.index, dtype=float)  # Acceleration Factor
        trend = pd.Series(index=close.index, dtype=int)  # 1 for uptrend, -1 for downtrend
        
        # Initialize
        sar.iloc[0] = low.iloc[0]
        ep.iloc[0] = high.iloc[0]
        af.iloc[0] = acceleration
        trend.iloc[0] = 1
        
        for i in range(1, len(close)):
            prev_sar = sar.iloc[i-1]
            prev_ep = ep.iloc[i-1]
            prev_af = af.iloc[i-1]
            prev_trend = trend.iloc[i-1]
            
            # Calculate SAR
            sar.iloc[i] = prev_sar + prev_af * (prev_ep - prev_sar)
            
            # Adjust SAR for current period
            if prev_trend == 1:  # Uptrend
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i])
            else:  # Downtrend
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i])
            
            # Check for trend reversal
            if prev_trend == 1:  # Uptrend
                if low.iloc[i] < sar.iloc[i]:  # Reversal
                    trend.iloc[i] = -1
                    sar.iloc[i] = prev_ep
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = acceleration
                else:  # Continue uptrend
                    trend.iloc[i] = 1
                    if high.iloc[i] > prev_ep:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(prev_af + acceleration, maximum)
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af
            else:  # Downtrend
                if high.iloc[i] > sar.iloc[i]:  # Reversal
                    trend.iloc[i] = 1
                    sar.iloc[i] = prev_ep
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = acceleration
                else:  # Continue downtrend
                    trend.iloc[i] = -1
                    if low.iloc[i] < prev_ep:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(prev_af + acceleration, maximum)
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af
        
        return sar
    
    def _ichimoku_cloud_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_period: int = 52
    ) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud using pandas."""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        tenkan = (high.rolling(window=tenkan_period).max() + 
                  low.rolling(window=tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        kijun = (high.rolling(window=kijun_period).max() + 
                 low.rolling(window=kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods forward
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods forward
        senkou_b = ((high.rolling(window=senkou_period).max() + 
                     low.rolling(window=senkou_period).min()) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span): Close price, shifted 26 periods backward
        chikou = close.shift(-kijun_period)
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }
    
    def _aroon_pandas(
        self,
        high: pd.Series,
        low: pd.Series,
        period: int = 14
    ) -> Dict[str, pd.Series]:
        """Calculate Aroon Indicator using pandas."""
        aroon_up = pd.Series(index=high.index, dtype=float)
        aroon_down = pd.Series(index=low.index, dtype=float)
        
        for i in range(period, len(high)):
            # Highest high position in last period periods
            high_window = high.iloc[i-period+1:i+1]
            high_max_val = high_window.max()
            # Find index of max value within window
            high_max_idx = high_window[high_window == high_max_val].index[-1]
            # Calculate periods since high (0 = most recent, period-1 = oldest)
            window_indices = high_window.index
            periods_since_high = len(window_indices) - 1 - list(window_indices).index(high_max_idx)
            aroon_up.iloc[i] = 100 * (period - periods_since_high) / period
            
            # Lowest low position in last period periods
            low_window = low.iloc[i-period+1:i+1]
            low_min_val = low_window.min()
            # Find index of min value within window
            low_min_idx = low_window[low_window == low_min_val].index[-1]
            # Calculate periods since low
            window_indices = low_window.index
            periods_since_low = len(window_indices) - 1 - list(window_indices).index(low_min_idx)
            aroon_down.iloc[i] = 100 * (period - periods_since_low) / period
        
        # Fill NaN values
        aroon_up = aroon_up.fillna(50.0)  # Neutral value
        aroon_down = aroon_down.fillna(50.0)
        
        # Aroon Oscillator: Aroon Up - Aroon Down
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
    
    def _calculate_trend_strength_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite trend strength score."""
        # Combine multiple trend indicators into composite score
        score = pd.Series(index=df.index, dtype=float)
        score[:] = 0.5  # Neutral
        
        # ADX contribution (0-25 points)
        if 'adx' in df.columns:
            adx_score = df['adx'].fillna(0) / 100  # Normalize to 0-1
            score += adx_score * 0.25
        
        # Aroon Oscillator contribution (0-25 points)
        if 'aroon_oscillator' in df.columns:
            aroon_score = df['aroon_oscillator'].fillna(0) / 200 + 0.5  # Normalize to 0-1
            score += (aroon_score - 0.5) * 0.25
        
        # Trend signal contribution (0-25 points)
        if 'trend_signal' in df.columns:
            trend_map = {'UPTREND': 0.75, 'DOWNTREND': 0.25, 'SIDEWAYS': 0.5}
            trend_score = df['trend_signal'].map(trend_map).fillna(0.5)
            score += (trend_score - 0.5) * 0.25
        
        # Price vs SMA contribution (0-25 points)
        if 'close' in df.columns and 'sma_50' in df.columns:
            price_sma_diff = (df['close'] - df['sma_50']) / df['sma_50']
            price_score = (price_sma_diff * 10).clip(-0.5, 0.5) + 0.5  # Normalize to 0-1
            score += (price_score - 0.5) * 0.25
        
        # Normalize to 0-1 range
        score = score.clip(0.0, 1.0)
        
        return score


# Convenience function
def get_technical_analyzer() -> TechnicalAnalyzer:
    """Get a TechnicalAnalyzer instance."""
    return TechnicalAnalyzer()


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1y")
    df = df.reset_index()
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"date": "time"})
    
    analyzer = TechnicalAnalyzer()
    df_with_indicators = analyzer.compute_all(df)
    
    print("\nLast 5 rows with indicators:")
    print(df_with_indicators[['time', 'close', 'rsi_14', 'macd', 'sma_50', 'bb_upper']].tail())
    
    score = analyzer.calculate_technical_score(df_with_indicators)
    print(f"\nTechnical Score: {score}")
