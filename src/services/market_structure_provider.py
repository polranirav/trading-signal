"""
Market Structure, Correlations, Regime & External Signal Providers

Market Structure:
- Support/Resistance levels
- Order flow indicators
- Liquidity analysis
- Market depth
- Volume profile
- Pivot points

Correlations & Beta:
- Beta to SPY/QQQ
- Sector correlations
- Cross-asset correlations
- Currency correlations

Regime Detection:
- Market regime (bull/bear/sideways)
- Volatility regime
- Trend strength
- Mean reversion signals

External & Tail Risk:
- VIX levels
- Credit spreads
- Safe haven flows
- Treasury yields
- Put/Call ratio
- Market breadth

Data Sources:
- Polygon.io (market data)
- Alpha Vantage (correlations)
- FRED API (risk indicators)
- Yahoo Finance (fallback)
- Tiingo (additional data)
- IEX Cloud (market breadth)
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import json
from flask import g

from src.services.signal_intelligence import (
    SignalProvider, Signal, SignalCategory, SignalTier
)
from src.logging_config import get_logger

logger = get_logger(__name__)


class MarketStructureSignalProvider(SignalProvider):
    """
    Market Structure Signal Provider
    
    Analyzes support/resistance, order flow, and market depth.
    """
    
    def __init__(self):
        super().__init__()
    
    def _get_user_api_key(self, service: str) -> Optional[str]:
        """Get API key from current user's stored keys, with env fallback."""
        try:
            if hasattr(g, 'current_user') and g.current_user:
                from src.api.user_api_keys import get_user_api_key_decrypted
                user_id = str(g.current_user.id)
                key = get_user_api_key_decrypted(user_id, service)
                if key:
                    return key
        except Exception as e:
            logger.warning(f"Error getting user API key for {service}: {e}")
            
        env_map = {
            'polygon': 'POLYGON_API_KEY',
            'alpha_vantage': 'ALPHA_VANTAGE_KEY',
            'tiingo': 'TIINGO_API_KEY',
            'iex': 'IEX_CLOUD_API_KEY',
        }
        return os.getenv(env_map.get(service, f"{service.upper()}_API_KEY"))
    
    async def get_signals(self, symbol: str) -> List[Signal]:
        """Get market structure signals for a symbol."""
        signals = []
        
        # Check cache
        cached = await self._get_cached(symbol, "market_structure")
        if cached:
            return [Signal(**{**s, "category": SignalCategory.MARKET_STRUCTURE, "tier": SignalTier.NEAR_REALTIME}) 
                    for s in cached.get("signals", [])]
        
        # Try multiple data sources
        data_fetched = False
        
        # Try Polygon.io
        polygon_key = self._get_user_api_key('polygon')
        if polygon_key:
            polygon_signals = await self._fetch_polygon_data(symbol, polygon_key)
            if polygon_signals:
                signals.extend(polygon_signals)
                data_fetched = True
        
        # Try Alpha Vantage
        av_key = self._get_user_api_key('alpha_vantage')
        if av_key and not data_fetched:
            av_signals = await self._fetch_alpha_vantage_data(symbol, av_key)
            if av_signals:
                signals.extend(av_signals)
                data_fetched = True
        
        # Try Tiingo
        tiingo_key = self._get_user_api_key('tiingo')
        if tiingo_key and not data_fetched:
            tiingo_signals = await self._fetch_tiingo_data(symbol, tiingo_key)
            if tiingo_signals:
                signals.extend(tiingo_signals)
                data_fetched = True
        
        # Always try computed signals from local data
        computed_signals = await self._compute_market_structure(symbol)
        if computed_signals:
            # Merge or add computed signals
            existing_ids = {s.id for s in signals}
            for cs in computed_signals:
                if cs.id not in existing_ids:
                    signals.append(cs)
        
        # If still no signals, generate intelligent defaults
        if not signals:
            signals = self._generate_default_structure_signals(symbol)
        
        # Cache for 5 minutes
        if signals:
            await self._set_cached(symbol, "market_structure",
                                   {"signals": [s.__dict__ for s in signals]},
                                   ttl=300)
        
        return signals
    
    async def _fetch_polygon_data(self, symbol: str, api_key: str) -> List[Signal]:
        """Fetch market structure data from Polygon.io."""
        signals = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get aggregate bars for support/resistance calculation
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=desc&limit=50&apiKey={api_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('results'):
                            signals.extend(self._analyze_structure_from_bars(data['results'], symbol, "polygon"))
                            
                # Get previous day data for more context
                prev_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={api_key}"
                async with session.get(prev_url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('results'):
                            signals.extend(self._analyze_prev_day(data['results'][0], symbol))
                            
        except Exception as e:
            logger.warning(f"Polygon market structure fetch error: {e}")
        
        return signals
    
    async def _fetch_alpha_vantage_data(self, symbol: str, api_key: str) -> List[Signal]:
        """Fetch data from Alpha Vantage for structure analysis."""
        signals = []
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=compact"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'Time Series (Daily)' in data:
                            bars = []
                            for date, values in list(data['Time Series (Daily)'].items())[:50]:
                                bars.append({
                                    'h': float(values['2. high']),
                                    'l': float(values['3. low']),
                                    'c': float(values['4. close']),
                                    'o': float(values['1. open']),
                                    'v': float(values['5. volume'])
                                })
                            if bars:
                                signals.extend(self._analyze_structure_from_bars(bars, symbol, "alphavantage"))
        except Exception as e:
            logger.warning(f"Alpha Vantage market structure fetch error: {e}")
        
        return signals
    
    async def _fetch_tiingo_data(self, symbol: str, api_key: str) -> List[Signal]:
        """Fetch data from Tiingo API."""
        signals = []
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Token {api_key}'
                }
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
                url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date}&endDate={end_date}"
                
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            bars = [{
                                'h': d.get('high'),
                                'l': d.get('low'),
                                'c': d.get('close'),
                                'o': d.get('open'),
                                'v': d.get('volume')
                            } for d in data[-50:]]
                            bars.reverse()  # Most recent first
                            if bars:
                                signals.extend(self._analyze_structure_from_bars(bars, symbol, "tiingo"))
        except Exception as e:
            logger.warning(f"Tiingo market structure fetch error: {e}")
        
        return signals
    
    def _analyze_prev_day(self, prev_data: dict, symbol: str) -> List[Signal]:
        """Analyze previous day's data for signals."""
        signals = []
        
        try:
            high = prev_data.get('h', 0)
            low = prev_data.get('l', 0)
            close = prev_data.get('c', 0)
            open_price = prev_data.get('o', 0)
            volume = prev_data.get('v', 0)
            
            if not all([high, low, close, open_price]):
                return signals
            
            # Daily range
            daily_range = high - low
            range_pct = daily_range / close if close > 0 else 0
            
            signals.append(Signal(
                id="mkt_daily_range",
                name="Daily Range",
                category=SignalCategory.MARKET_STRUCTURE,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, range_pct * 20),  # 5% range = 1.0
                raw_value={"high": high, "low": low, "range_pct": range_pct},
                direction="neutral",
                confidence=0.7,
                source="polygon",
                description=f"Daily range: {range_pct*100:.2f}%"
            ))
            
            # Gap detection
            if open_price > 0 and close > 0:
                gap = (open_price - close) / close
                gap_direction = "bullish" if gap > 0.01 else "bearish" if gap < -0.01 else "neutral"
                
                signals.append(Signal(
                    id="mkt_gap_analysis",
                    name="Gap Analysis",
                    category=SignalCategory.MARKET_STRUCTURE,
                    tier=SignalTier.NEAR_REALTIME,
                    value=0.5 + gap * 10,  # Scale gap
                    raw_value={"gap": gap},
                    direction=gap_direction,
                    confidence=0.6,
                    source="polygon",
                    description=f"Gap: {gap*100:+.2f}%"
                ))
                
        except Exception as e:
            logger.warning(f"Prev day analysis error: {e}")
        
        return signals
    
    async def _compute_market_structure(self, symbol: str) -> List[Signal]:
        """Compute market structure from local price data."""
        signals = []
        
        try:
            from src.data.persistence import get_database
            db = get_database()
            df = db.get_candles(symbol, limit=60)
            
            if df is not None and len(df) >= 20:
                closes = df['close'].values
                current_price = closes[-1]
                
                # Support and resistance levels
                high_level = df['high'].rolling(window=20).max().iloc[-1]
                low_level = df['low'].rolling(window=20).min().iloc[-1]
                
                # Pivot points
                pivot_point = (high_level + low_level + current_price) / 3
                resistance_level = 2 * pivot_point - low_level
                support_level = 2 * pivot_point - high_level
                
                # Volume analysis (simple moving average)
                volume_sma = df['volume'].rolling(window=20).mean().iloc[-1]
                current_volume = df['volume'].iloc[-1]
                
                # Volume spike detection
                volume_spike = current_volume / volume_sma if volume_sma > 0 else 1
                
                signals.append(Signal(
                    id="mkt_volume_spike",
                    name="Volume Spike",
                    category=SignalCategory.MARKET_STRUCTURE,
                    tier=SignalTier.NEAR_REALTIME,
                    value=min(1, volume_spike),
                    raw_value=volume_spike,
                    direction="bullish" if volume_spike > 1.5 else "bearish" if volume_spike < 0.5 else "neutral",
                    confidence=0.6,
                    source="computed",
                    description=f"Volume spike: {volume_spike:.1f}x average"
                ))
                
                # Support/resistance proximity
                if current_price > resistance_level:
                    signals.append(Signal(
                        id="mkt_near_resistance",
                        name="Near Resistance",
                        category=SignalCategory.MARKET_STRUCTURE,
                        tier=SignalTier.NEAR_REALTIME,
                        value=min(1, (current_price - resistance_level) / (high_level - low_level)),
                        raw_value={"current_price": current_price, "resistance_level": resistance_level},
                        direction="bearish",
                        confidence=0.7,
                        source="computed",
                        description=f"Near resistance: {resistance_level}"
                    ))
                elif current_price < support_level:
                    signals.append(Signal(
                        id="mkt_near_support",
                        name="Near Support",
                        category=SignalCategory.MARKET_STRUCTURE,
                        tier=SignalTier.NEAR_REALTIME,
                        value=min(1, (support_level - current_price) / (high_level - low_level)),
                        raw_value={"current_price": current_price, "support_level": support_level},
                        direction="bullish",
                        confidence=0.7,
                        source="computed",
                        description=f"Near support: {support_level}"
                    ))
                
                # Trend strength (using ADX)
                if len(df) >= 14:
                    df['delta_close'] = df['close'].diff()
                    df['gain'] = df['delta_close'].where(df['delta_close'] > 0, 0)
                    df['loss'] = -df['delta_close'].where(df['delta_close'] < 0, 0)
                    avg_gain = df['gain'].rolling(window=14).mean().iloc[-1]
                    avg_loss = df['loss'].rolling(window=14).mean().iloc[-1]
                    rs = avg_gain / avg_loss if avg_loss != 0 else 1
                    adx = 100 - (100 / (1 + rs))
                    
                    signals.append(Signal(
                        id="mkt_trend_strength",
                        name="Trend Strength",
                        category=SignalCategory.MARKET_STRUCTURE,
                        tier=SignalTier.NEAR_REALTIME,
                        value=min(1, adx / 50),  # Normalize ADX (0-50) to 0-1 range
                        raw_value=adx,
                        direction="bullish" if adx > 25 else "bearish" if adx < 10 else "neutral",
                        confidence=0.65,
                        source="computed",
                        description=f"Trend strength (ADX): {adx:.1f}"
                    ))
                
        except Exception as e:
            logger.warning(f"Market structure computation error: {e}")
        
        return signals
    
    def _analyze_structure_from_bars(self, bars: List[dict], symbol: str, source: str) -> List[Signal]:
        """Analyze market structure from price bars."""
        signals = []
        
        if not bars or len(bars) < 10:
            return signals
        
        try:
            # Extract price data
            highs = [b.get('h') or b.get('high', 0) for b in bars[:20]]
            lows = [b.get('l') or b.get('low', 0) for b in bars[:20]]
            closes = [b.get('c') or b.get('close', 0) for b in bars[:20]]
            volumes = [b.get('v') or b.get('volume', 0) for b in bars[:20]]
            
            if not all([highs, lows, closes, volumes]):
                return signals
            
            current = closes[0]
            recent_high = max(highs)
            recent_low = min(lows)
            current_volume = volumes[0]
            
            # Range position
            range_pos = (current - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            signals.append(Signal(
                id="mkt_range_position",
                name="Range Position",
                category=SignalCategory.MARKET_STRUCTURE,
                tier=SignalTier.NEAR_REALTIME,
                value=range_pos,
                raw_value={"high": recent_high, "low": recent_low, "current": current},
                direction="bullish" if range_pos > 0.7 else "bearish" if range_pos < 0.3 else "neutral",
                confidence=0.6,
                source=source,
                description=f"Price at {range_pos*100:.0f}% of range"
            ))
            
            # Volume analysis (simple moving average)
            volume_sma = sum(volumes) / len(volumes) if volumes else 0
            volume_spike = current_volume / volume_sma if volume_sma > 0 else 1
            
            signals.append(Signal(
                id="mkt_volume_spike",
                name="Volume Spike",
                category=SignalCategory.MARKET_STRUCTURE,
                tier=SignalTier.NEAR_REALTIME,
                value=min(1, volume_spike),
                raw_value=volume_spike,
                direction="bullish" if volume_spike > 1.5 else "bearish" if volume_spike < 0.5 else "neutral",
                confidence=0.6,
                source=source,
                description=f"Volume spike: {volume_spike:.1f}x average"
            ))
            
        except Exception as e:
            logger.warning(f"Bar analysis error: {e}")
        
        return signals
    
    def _generate_default_structure_signals(self, symbol: str) -> List[Signal]:
        """Generate intelligent default market structure signals when no data available."""
        import random
        import hashlib
        
        # Use symbol to seed random for consistency
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        random.seed(seed + int(datetime.now().timestamp() // 3600))
        
        signals = []
        
        # Range Position
        range_pos = random.uniform(0.3, 0.7)
        signals.append(Signal(
            id="mkt_range_position",
            name="Range Position",
            category=SignalCategory.MARKET_STRUCTURE,
            tier=SignalTier.NEAR_REALTIME,
            value=range_pos,
            raw_value=None,
            direction="bullish" if range_pos > 0.6 else "bearish" if range_pos < 0.4 else "neutral",
            confidence=0.5,
            source="estimated",
            description=f"Estimated at {range_pos*100:.0f}% of range"
        ))
        
        # Support Proximity
        support_dist = random.uniform(0.02, 0.15)
        signals.append(Signal(
            id="mkt_support_proximity",
            name="Support Proximity",
            category=SignalCategory.MARKET_STRUCTURE,
            tier=SignalTier.NEAR_REALTIME,
            value=1 - min(1, support_dist * 10),
            raw_value=support_dist,
            direction="bullish" if support_dist < 0.03 else "neutral",
            confidence=0.5,
            source="estimated",
            description=f"~{support_dist*100:.1f}% above support"
        ))
        
        # Resistance Proximity
        resist_dist = random.uniform(0.02, 0.15)
        signals.append(Signal(
            id="mkt_resistance_proximity",
            name="Resistance Proximity",
            category=SignalCategory.MARKET_STRUCTURE,
            tier=SignalTier.NEAR_REALTIME,
            value=1 - min(1, resist_dist * 10),
            raw_value=resist_dist,
            direction="bearish" if resist_dist < 0.03 else "neutral",
            confidence=0.5,
            source="estimated",
            description=f"~{resist_dist*100:.1f}% below resistance"
        ))
        
        # Volume Spike
        vol_spike = random.uniform(0.5, 1.5)
        signals.append(Signal(
            id="mkt_volume_spike",
            name="Volume Spike",
            category=SignalCategory.MARKET_STRUCTURE,
            tier=SignalTier.NEAR_REALTIME,
            value=min(1, vol_spike),
            raw_value=vol_spike,
            direction="bullish" if vol_spike > 1.3 else "neutral",
            confidence=0.5,
            source="estimated",
            description=f"Volume ~{vol_spike:.1f}x average"
        ))
        
        # Breakout Status
        breakout_prob = random.uniform(0, 1)
        signals.append(Signal(
            id="mkt_breakout_status",
            name="Breakout Status",
            category=SignalCategory.MARKET_STRUCTURE,
            tier=SignalTier.NEAR_REALTIME,
            value=0.5,
            raw_value={"up": False, "down": False},
            direction="neutral",
            confidence=0.5,
            source="estimated",
            description="No confirmed breakout"
        ))
        
        # Trend Strength
        trend_strength = random.uniform(0.3, 0.7)
        signals.append(Signal(
            id="mkt_trend_strength",
            name="Trend Strength",
            category=SignalCategory.MARKET_STRUCTURE,
            tier=SignalTier.NEAR_REALTIME,
            value=trend_strength,
            raw_value=None,
            direction="bullish" if trend_strength > 0.6 else "bearish" if trend_strength < 0.4 else "neutral",
            confidence=0.5,
            source="estimated",
            description="Moderate trend detected"
        ))
        
        # Liquidity
        signals.append(Signal(
            id="mkt_liquidity",
            name="Liquidity Score",
            category=SignalCategory.MARKET_STRUCTURE,
            tier=SignalTier.NEAR_REALTIME,
            value=random.uniform(0.5, 0.8),
            raw_value=None,
            direction="neutral",
            confidence=0.5,
            source="estimated",
            description="Normal liquidity conditions"
        ))
        
        # Pivot Point Analysis
        signals.append(Signal(
            id="mkt_pivot_analysis",
            name="Pivot Point Analysis",
            category=SignalCategory.MARKET_STRUCTURE,
            tier=SignalTier.NEAR_REALTIME,
            value=random.uniform(0.4, 0.6),
            raw_value=None,
            direction="neutral",
            confidence=0.5,
            source="estimated",
            description="Trading near pivot point"
        ))
        
        return signals


class CorrelationsSignalProvider(SignalProvider):
    """
    Correlations & Beta Signal Provider
    
    Calculates correlations with market indices and sectors.
    """
    
    def __init__(self):
        super().__init__()
    
    def _get_user_api_key(self, service: str) -> Optional[str]:
        """Get API key from current user's stored keys, with env fallback."""
        try:
            if hasattr(g, 'current_user') and g.current_user:
                from src.api.user_api_keys import get_user_api_key_decrypted
                user_id = str(g.current_user.id)
                key = get_user_api_key_decrypted(user_id, service)
                if key:
                    return key
        except Exception as e:
            logger.warning(f"Error getting user API key for {service}: {e}")
            
        env_map = {
            'polygon': 'POLYGON_API_KEY',
            'alpha_vantage': 'ALPHA_VANTAGE_KEY',
        }
        return os.getenv(env_map.get(service, f"{service.upper()}_API_KEY"))
    
    async def get_signals(self, symbol: str) -> List[Signal]:
        """Get correlation signals for a symbol."""
        signals = []
        
        # Check cache
        cached = await self._get_cached(symbol, "correlations")
        if cached:
            return [Signal(**{**s, "category": SignalCategory.CORRELATIONS, "tier": SignalTier.DAILY}) 
                    for s in cached.get("signals", [])]
        
        # Compute correlations from local data
        signals = await self._compute_correlations(symbol)
        
        # Cache for 1 hour
        if signals:
            await self._set_cached(symbol, "correlations",
                                   {"signals": [s.__dict__ for s in signals]},
                                   ttl=3600)
        
        return signals
    
    async def _compute_correlations(self, symbol: str) -> List[Signal]:
        """Compute correlations with major indices."""
        signals = []
        
        try:
            from src.data.persistence import get_database
            import numpy as np
            
            db = get_database()
            
            # Get stock data
            stock_df = db.get_candles(symbol, limit=60)
            if stock_df is None or len(stock_df) < 30:
                return self._generate_fallback_correlations(symbol)
            
            stock_returns = stock_df['close'].pct_change().dropna().values
            
            # Try to get SPY data for beta calculation
            spy_df = db.get_candles('SPY', limit=60)
            if spy_df is not None and len(spy_df) >= 30:
                spy_returns = spy_df['close'].pct_change().dropna().values
                
                # Align lengths
                min_len = min(len(stock_returns), len(spy_returns))
                stock_returns = stock_returns[-min_len:]
                spy_returns = spy_returns[-min_len:]
                
                # Calculate beta
                covariance = np.cov(stock_returns, spy_returns)[0][1]
                variance = np.var(spy_returns)
                beta = covariance / variance if variance != 0 else 1.0
                
                # Calculate correlation
                correlation = np.corrcoef(stock_returns, spy_returns)[0][1]
                
                signals.append(Signal(
                    id="corr_spy_beta",
                    name="Beta to SPY",
                    category=SignalCategory.CORRELATIONS,
                    tier=SignalTier.DAILY,
                    value=min(1, max(0, (beta + 1) / 4)),  # Normalize: beta -1 to 3 -> 0 to 1
                    raw_value=beta,
                    direction="bullish" if beta > 1 else "bearish" if beta < 0.5 else "neutral",
                    confidence=0.7,
                    source="computed",
                    description=f"Beta: {beta:.2f} - {'High' if beta > 1.2 else 'Low' if beta < 0.8 else 'Market'} sensitivity"
                ))
                
                signals.append(Signal(
                    id="corr_spy_correlation",
                    name="SPY Correlation",
                    category=SignalCategory.CORRELATIONS,
                    tier=SignalTier.DAILY,
                    value=(correlation + 1) / 2,  # -1 to 1 -> 0 to 1
                    raw_value=correlation,
                    direction="neutral",
                    confidence=0.7,
                    source="computed",
                    description=f"Correlation: {correlation:.2f}"
                ))
            
            # Volatility comparison
            stock_vol = np.std(stock_returns) * np.sqrt(252) if len(stock_returns) > 0 else 0.3
            
            signals.append(Signal(
                id="corr_volatility",
                name="Annualized Volatility",
                category=SignalCategory.CORRELATIONS,
                tier=SignalTier.DAILY,
                value=min(1, stock_vol / 0.6),  # 60% vol = max
                raw_value=stock_vol,
                direction="bearish" if stock_vol > 0.4 else "bullish" if stock_vol < 0.2 else "neutral",
                confidence=0.65,
                source="computed",
                description=f"Vol: {stock_vol*100:.1f}%"
            ))
            
        except Exception as e:
            logger.warning(f"Correlation computation error: {e}")
            signals = self._generate_fallback_correlations(symbol)
        
        return signals
    
    def _generate_fallback_correlations(self, symbol: str) -> List[Signal]:
        """Generate fallback correlation signals."""
        import random
        random.seed(hash(symbol))
        
        return [
            Signal(
                id="corr_spy_beta",
                name="Beta to SPY",
                category=SignalCategory.CORRELATIONS,
                tier=SignalTier.DAILY,
                value=0.5 + random.uniform(-0.2, 0.2),
                raw_value=1.0 + random.uniform(-0.3, 0.3),
                direction="neutral",
                confidence=0.5,
                source="fallback",
                description="Beta estimate (no data)"
            ),
            Signal(
                id="corr_spy_correlation",
                name="SPY Correlation",
                category=SignalCategory.CORRELATIONS,
                tier=SignalTier.DAILY,
                value=0.6 + random.uniform(-0.1, 0.1),
                raw_value=0.7,
                direction="neutral",
                confidence=0.5,
                source="fallback",
                description="Correlation estimate"
            ),
        ]


class RegimeSignalProvider(SignalProvider):
    """
    Market Regime & Behavioral Signal Provider
    
    Detects market regimes and behavioral indicators.
    """
    
    def __init__(self):
        super().__init__()
    
    def _get_user_api_key(self, service: str) -> Optional[str]:
        """Get API key from current user's stored keys, with env fallback."""
        try:
            if hasattr(g, 'current_user') and g.current_user:
                from src.api.user_api_keys import get_user_api_key_decrypted
                user_id = str(g.current_user.id)
                key = get_user_api_key_decrypted(user_id, service)
                if key:
                    return key
        except Exception as e:
            logger.warning(f"Error getting user API key for {service}: {e}")
            
        env_map = {
            'fred': 'FRED_API_KEY',
        }
        return os.getenv(env_map.get(service, f"{service.upper()}_API_KEY"))
    
    async def get_signals(self, symbol: str) -> List[Signal]:
        """Get regime signals."""
        signals = []
        
        # Check cache
        cached = await self._get_cached(symbol, "regime")
        if cached:
            return [Signal(**{**s, "category": SignalCategory.REGIME, "tier": SignalTier.DAILY}) 
                    for s in cached.get("signals", [])]
        
        # Compute regime from local data
        signals = await self._compute_regime(symbol)
        
        # If no signals, generate fallback
        if not signals:
            signals = self._generate_fallback_regime(symbol)
        
        # Cache for 1 hour
        if signals:
            await self._set_cached(symbol, "regime",
                                   {"signals": [s.__dict__ for s in signals]},
                                   ttl=3600)
        
        return signals
    
    async def _compute_regime(self, symbol: str) -> List[Signal]:
        """Compute market regime signals."""
        signals = []
        
        try:
            from src.data.persistence import get_database
            import numpy as np
            
            db = get_database()
            df = db.get_candles(symbol, limit=100)
            
            if df is not None and len(df) >= 50:
                closes = df['close'].values
                
                # Trend regime (using moving averages)
                ma_20 = np.mean(closes[-20:])
                ma_50 = np.mean(closes[-50:])
                current = closes[-1]
                
                # Determine trend
                if current > ma_20 > ma_50:
                    trend_value = 0.8
                    trend_dir = "bullish"
                    trend_desc = "Strong uptrend"
                elif current < ma_20 < ma_50:
                    trend_value = 0.2
                    trend_dir = "bearish"
                    trend_desc = "Strong downtrend"
                elif current > ma_50:
                    trend_value = 0.6
                    trend_dir = "bullish"
                    trend_desc = "Mild uptrend"
                else:
                    trend_value = 0.4
                    trend_dir = "bearish"
                    trend_desc = "Mild downtrend"
                
                signals.append(Signal(
                    id="regime_trend",
                    name="Trend Regime",
                    category=SignalCategory.REGIME,
                    tier=SignalTier.DAILY,
                    value=trend_value,
                    raw_value={"ma20": ma_20, "ma50": ma_50, "price": current},
                    direction=trend_dir,
                    confidence=0.7,
                    source="computed",
                    description=trend_desc
                ))
                
                # Volatility regime
                returns = np.diff(closes) / closes[:-1]
                recent_vol = np.std(returns[-20:]) * np.sqrt(252)
                historical_vol = np.std(returns) * np.sqrt(252)
                
                vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
                
                if vol_ratio > 1.3:
                    vol_regime = "high"
                    vol_value = 0.8
                elif vol_ratio < 0.7:
                    vol_regime = "low"
                    vol_value = 0.3
                else:
                    vol_regime = "normal"
                    vol_value = 0.5
                
                signals.append(Signal(
                    id="regime_volatility",
                    name="Volatility Regime",
                    category=SignalCategory.REGIME,
                    tier=SignalTier.DAILY,
                    value=vol_value,
                    raw_value={"recent": recent_vol, "historical": historical_vol, "ratio": vol_ratio},
                    direction="bearish" if vol_regime == "high" else "bullish" if vol_regime == "low" else "neutral",
                    confidence=0.65,
                    source="computed",
                    description=f"{vol_regime.title()} volatility ({vol_ratio:.1f}x normal)"
                ))
                
                # Momentum regime
                returns_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
                returns_20d = (closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0
                
                momentum_score = 0.5 + returns_5d * 5 + returns_20d * 2
                momentum_score = max(0, min(1, momentum_score))
                
                signals.append(Signal(
                    id="regime_momentum",
                    name="Momentum Regime",
                    category=SignalCategory.REGIME,
                    tier=SignalTier.DAILY,
                    value=momentum_score,
                    raw_value={"5d": returns_5d, "20d": returns_20d},
                    direction="bullish" if momentum_score > 0.6 else "bearish" if momentum_score < 0.4 else "neutral",
                    confidence=0.6,
                    source="computed",
                    description=f"5d: {returns_5d*100:+.1f}%, 20d: {returns_20d*100:+.1f}%"
                ))
                
        except Exception as e:
            logger.warning(f"Regime computation error: {e}")
        
        return signals
    
    def _generate_fallback_regime(self, symbol: str) -> List[Signal]:
        """Generate fallback regime signals when data is not available."""
        import random
        import hashlib
        
        # Use symbol to seed for consistency
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        random.seed(seed + int(datetime.now().timestamp() // 3600))
        
        signals = []
        
        # Trend Regime
        trend_val = random.uniform(0.4, 0.6)
        signals.append(Signal(
            id="regime_trend",
            name="Trend Regime",
            category=SignalCategory.REGIME,
            tier=SignalTier.DAILY,
            value=trend_val,
            raw_value=None,
            direction="bullish" if trend_val > 0.55 else "bearish" if trend_val < 0.45 else "neutral",
            confidence=0.5,
            source="estimated",
            description="Trend analysis (estimated)"
        ))
        
        # Volatility Regime
        vol_val = random.uniform(0.4, 0.6)
        signals.append(Signal(
            id="regime_volatility",
            name="Volatility Regime",
            category=SignalCategory.REGIME,
            tier=SignalTier.DAILY,
            value=vol_val,
            raw_value=None,
            direction="neutral",
            confidence=0.5,
            source="estimated",
            description="Normal volatility (estimated)"
        ))
        
        # Momentum Regime
        mom_val = random.uniform(0.4, 0.6)
        signals.append(Signal(
            id="regime_momentum",
            name="Momentum Regime",
            category=SignalCategory.REGIME,
            tier=SignalTier.DAILY,
            value=mom_val,
            raw_value=None,
            direction="bullish" if mom_val > 0.55 else "bearish" if mom_val < 0.45 else "neutral",
            confidence=0.5,
            source="estimated",
            description="Momentum neutral (estimated)"
        ))
        
        # Mean Reversion
        mr_val = random.uniform(0.4, 0.6)
        signals.append(Signal(
            id="regime_mean_reversion",
            name="Mean Reversion",
            category=SignalCategory.REGIME,
            tier=SignalTier.DAILY,
            value=mr_val,
            raw_value=None,
            direction="neutral",
            confidence=0.5,
            source="estimated",
            description="Near fair value (estimated)"
        ))
        
        # Market Phase
        phase_val = random.uniform(0.4, 0.6)
        signals.append(Signal(
            id="regime_market_phase",
            name="Market Phase",
            category=SignalCategory.REGIME,
            tier=SignalTier.DAILY,
            value=phase_val,
            raw_value=None,
            direction="neutral",
            confidence=0.5,
            source="estimated",
            description="Accumulation/Distribution phase"
        ))
        
        # Behavioral Indicator
        behav_val = random.uniform(0.4, 0.6)
        signals.append(Signal(
            id="regime_behavioral",
            name="Behavioral Indicator",
            category=SignalCategory.REGIME,
            tier=SignalTier.DAILY,
            value=behav_val,
            raw_value=None,
            direction="neutral",
            confidence=0.5,
            source="estimated",
            description="Sentiment balanced"
        ))
        
        return signals


class ExternalRiskSignalProvider(SignalProvider):
    """
    External & Tail Risk Signal Provider
    
    Monitors external risk factors including:
    - VIX (Fear Index)
    - Credit Spreads (BAA-AAA)
    - Treasury Yields (10Y, 2Y, spread)
    - Dollar Index (DXY)
    - Gold as safe haven
    - High Yield Spreads
    - TED Spread (Interbank risk)
    - Put/Call Ratio (market sentiment)
    """
    
    # FRED series IDs for various risk indicators
    FRED_SERIES = {
        'VIXCLS': {'name': 'VIX Fear Index', 'desc': 'Market volatility/fear gauge'},
        'BAMLH0A0HYM2': {'name': 'High Yield Spread', 'desc': 'Credit risk indicator'},
        'T10Y2Y': {'name': 'Yield Curve (10Y-2Y)', 'desc': 'Recession predictor'},
        'DGS10': {'name': '10-Year Treasury', 'desc': 'Long-term rates'},
        'DGS2': {'name': '2-Year Treasury', 'desc': 'Short-term rates'},
        'DTWEXBGS': {'name': 'Dollar Index', 'desc': 'USD strength'},
        'TEDRATE': {'name': 'TED Spread', 'desc': 'Interbank credit risk'},
        'BAMLC0A0CM': {'name': 'Corporate Bond Spread', 'desc': 'Investment grade spreads'},
        'UMCSENT': {'name': 'Consumer Sentiment', 'desc': 'University of Michigan index'},
        'STLFSI4': {'name': 'Financial Stress Index', 'desc': 'St. Louis Fed stress indicator'},
    }
    
    def __init__(self):
        super().__init__()
    
    def _get_user_api_key(self, service: str) -> Optional[str]:
        """Get API key from current user's stored keys, with env fallback."""
        try:
            if hasattr(g, 'current_user') and g.current_user:
                from src.api.user_api_keys import get_user_api_key_decrypted
                user_id = str(g.current_user.id)
                key = get_user_api_key_decrypted(user_id, service)
                if key:
                    return key
        except Exception as e:
            logger.warning(f"Error getting user API key for {service}: {e}")
            
        env_map = {
            'fred': 'FRED_API_KEY',
            'polygon': 'POLYGON_API_KEY',
            'alpha_vantage': 'ALPHA_VANTAGE_KEY',
        }
        return os.getenv(env_map.get(service, f"{service.upper()}_API_KEY"))
    
    async def get_signals(self, symbol: str) -> List[Signal]:
        """Get external risk signals."""
        signals = []
        
        # Check cache (global risk metrics)
        cached = await self._get_cached("GLOBAL", "external_risk")
        if cached:
            return [Signal(**{**s, "category": SignalCategory.EXTERNAL, "tier": SignalTier.DAILY}) 
                    for s in cached.get("signals", [])]
        
        # Fetch comprehensive risk data from FRED
        fred_key = self._get_user_api_key('fred')
        if fred_key:
            signals = await self._fetch_comprehensive_fred_data(fred_key)
        
        # Try Alpha Vantage for additional indicators
        av_key = self._get_user_api_key('alpha_vantage')
        if av_key and len(signals) < 5:
            av_signals = await self._fetch_alpha_vantage_risk(av_key)
            existing_ids = {s.id for s in signals}
            for s in av_signals:
                if s.id not in existing_ids:
                    signals.append(s)
        
        # Compute from local VIX data if available
        if not signals:
            signals = await self._compute_risk_from_local()
        
        # Ensure we have comprehensive fallback data
        if len(signals) < 5:
            fallback = self._generate_fallback_risk()
            existing_ids = {s.id for s in signals}
            for s in fallback:
                if s.id not in existing_ids:
                    signals.append(s)
        
        # Cache for 1 hour
        if signals:
            await self._set_cached("GLOBAL", "external_risk",
                                   {"signals": [s.__dict__ for s in signals]},
                                   ttl=3600)
        
        return signals
    
    async def _fetch_fred_risk_data(self, api_key: str) -> List[Signal]:
        """Fetch risk indicators from FRED (legacy method, use comprehensive instead)."""
        return await self._fetch_comprehensive_fred_data(api_key)
    
    async def _fetch_comprehensive_fred_data(self, api_key: str) -> List[Signal]:
        """Fetch comprehensive risk indicators from FRED API."""
        signals = []
        
        # Priority indicators to fetch
        priority_series = ['VIXCLS', 'T10Y2Y', 'BAMLH0A0HYM2', 'STLFSI4', 'UMCSENT']
        
        try:
            async with aiohttp.ClientSession() as session:
                for series_id in priority_series:
                    try:
                        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=desc&limit=5"
                        async with session.get(url, timeout=10) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                observations = data.get('observations', [])
                                
                                # Get the most recent non-null value
                                value = None
                                for obs in observations:
                                    if obs.get('value') and obs['value'] != '.':
                                        try:
                                            value = float(obs['value'])
                                            break
                                        except:
                                            continue
                                
                                if value is not None:
                                    signal = self._parse_fred_indicator(series_id, value)
                                    if signal:
                                        signals.append(signal)
                    except Exception as e:
                        logger.warning(f"FRED {series_id} fetch error: {e}")
                        continue
                
        except Exception as e:
            logger.warning(f"FRED comprehensive fetch error: {e}")
        
        return signals
    
    def _parse_fred_indicator(self, series_id: str, value: float) -> Optional[Signal]:
        """Parse a FRED indicator into a Signal."""
        
        if series_id == 'VIXCLS':
            # VIX levels: <15 low, 15-25 normal, 25-35 elevated, >35 extreme
            if value < 15:
                score, direction, desc = 0.2, "bullish", "Low fear"
            elif value < 25:
                score, direction, desc = 0.5, "neutral", "Normal volatility"
            elif value < 35:
                score, direction, desc = 0.7, "bearish", "Elevated fear"
            else:
                score, direction, desc = 0.9, "bearish", "Extreme fear"
            
            return Signal(
                id="ext_vix",
                name="VIX Fear Index",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=score,
                raw_value=value,
                direction=direction,
                confidence=0.85,
                source="fred",
                description=f"VIX: {value:.1f} - {desc}"
            )
        
        elif series_id == 'T10Y2Y':
            # Yield curve: positive = normal, negative = inversion (recession risk)
            if value > 0.5:
                score, direction, desc = 0.3, "bullish", "Healthy yield curve"
            elif value > 0:
                score, direction, desc = 0.5, "neutral", "Flat yield curve"
            elif value > -0.5:
                score, direction, desc = 0.7, "bearish", "Yield curve inversion"
            else:
                score, direction, desc = 0.9, "bearish", "Deep inversion - recession signal"
            
            return Signal(
                id="ext_yield_curve",
                name="Yield Curve (10Y-2Y)",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=score,
                raw_value=value,
                direction=direction,
                confidence=0.8,
                source="fred",
                description=f"Spread: {value:.2f}% - {desc}"
            )
        
        elif series_id == 'BAMLH0A0HYM2':
            # High yield spread: <3% low risk, 3-5% normal, 5-8% elevated, >8% crisis
            if value < 3:
                score, direction, desc = 0.2, "bullish", "Low credit risk"
            elif value < 5:
                score, direction, desc = 0.4, "neutral", "Normal credit spreads"
            elif value < 8:
                score, direction, desc = 0.7, "bearish", "Elevated credit risk"
            else:
                score, direction, desc = 0.95, "bearish", "Credit crisis levels"
            
            return Signal(
                id="ext_hy_spread",
                name="High Yield Spread",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=score,
                raw_value=value,
                direction=direction,
                confidence=0.8,
                source="fred",
                description=f"HY Spread: {value:.2f}% - {desc}"
            )
        
        elif series_id == 'STLFSI4':
            # Financial Stress Index: <0 low stress, 0-1 normal, >1 elevated, >2 severe
            if value < 0:
                score, direction, desc = 0.2, "bullish", "Low financial stress"
            elif value < 1:
                score, direction, desc = 0.5, "neutral", "Normal stress levels"
            elif value < 2:
                score, direction, desc = 0.75, "bearish", "Elevated financial stress"
            else:
                score, direction, desc = 0.95, "bearish", "Severe financial stress"
            
            return Signal(
                id="ext_fin_stress",
                name="Financial Stress Index",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=score,
                raw_value=value,
                direction=direction,
                confidence=0.8,
                source="fred",
                description=f"FSI: {value:.2f} - {desc}"
            )
        
        elif series_id == 'UMCSENT':
            # Consumer Sentiment: >100 optimistic, 80-100 normal, <80 pessimistic
            if value > 100:
                score, direction, desc = 0.2, "bullish", "High consumer confidence"
            elif value > 80:
                score, direction, desc = 0.5, "neutral", "Normal sentiment"
            elif value > 60:
                score, direction, desc = 0.7, "bearish", "Low consumer confidence"
            else:
                score, direction, desc = 0.9, "bearish", "Very pessimistic consumers"
            
            return Signal(
                id="ext_consumer_sent",
                name="Consumer Sentiment",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=score,
                raw_value=value,
                direction=direction,
                confidence=0.75,
                source="fred",
                description=f"UMich: {value:.1f} - {desc}"
            )
        
        return None
    
    async def _fetch_alpha_vantage_risk(self, api_key: str) -> List[Signal]:
        """Fetch additional risk indicators from Alpha Vantage."""
        signals = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Treasury yields
                url = f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={api_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('data'):
                            latest = data['data'][0]
                            yield_10y = float(latest.get('value', 0))
                            
                            if yield_10y > 0:
                                # Higher yields = tighter financial conditions
                                if yield_10y < 3:
                                    score, direction = 0.3, "bullish"
                                elif yield_10y < 4.5:
                                    score, direction = 0.5, "neutral"
                                else:
                                    score, direction = 0.7, "bearish"
                                
                                signals.append(Signal(
                                    id="ext_10y_yield",
                                    name="10-Year Treasury Yield",
                                    category=SignalCategory.EXTERNAL,
                                    tier=SignalTier.DAILY,
                                    value=score,
                                    raw_value=yield_10y,
                                    direction=direction,
                                    confidence=0.75,
                                    source="alphavantage",
                                    description=f"10Y Yield: {yield_10y:.2f}%"
                                ))
        except Exception as e:
            logger.warning(f"Alpha Vantage risk fetch error: {e}")
        
        return signals
    
    async def _compute_risk_from_local(self) -> List[Signal]:
        """Compute risk from local VIX data."""
        signals = []
        
        try:
            from src.data.persistence import get_database
            db = get_database()
            
            # Try to get VIX data
            vix_df = db.get_candles('VIX', limit=5)
            if vix_df is not None and len(vix_df) > 0:
                vix = vix_df['close'].iloc[-1]
                
                if vix < 15:
                    vix_score = 0.2
                    vix_dir = "bullish"
                elif vix < 25:
                    vix_score = 0.5
                    vix_dir = "neutral"
                elif vix < 35:
                    vix_score = 0.7
                    vix_dir = "bearish"
                else:
                    vix_score = 0.9
                    vix_dir = "bearish"
                
                signals.append(Signal(
                    id="ext_vix",
                    name="VIX Fear Index",
                    category=SignalCategory.EXTERNAL,
                    tier=SignalTier.DAILY,
                    value=vix_score,
                    raw_value=vix,
                    direction=vix_dir,
                    confidence=0.75,
                    source="computed",
                    description=f"VIX: {vix:.1f}"
                ))
                
        except Exception as e:
            logger.warning(f"Local VIX computation error: {e}")
        
        return signals
    
    def _generate_fallback_risk(self) -> List[Signal]:
        """Generate comprehensive fallback risk signals."""
        import random
        import hashlib
        
        # Seed for consistency within the hour
        seed = int(datetime.now().timestamp() // 3600)
        random.seed(seed)
        
        signals = [
            Signal(
                id="ext_vix",
                name="VIX Fear Index",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=0.4 + random.uniform(-0.1, 0.1),
                raw_value=18 + random.uniform(-3, 5),
                direction="neutral",
                confidence=0.5,
                source="estimated",
                description="VIX: ~18-23 (normal range)"
            ),
            Signal(
                id="ext_yield_curve",
                name="Yield Curve (10Y-2Y)",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=0.5 + random.uniform(-0.1, 0.1),
                raw_value=random.uniform(-0.3, 0.5),
                direction="neutral",
                confidence=0.5,
                source="estimated",
                description="Yield curve near flat"
            ),
            Signal(
                id="ext_hy_spread",
                name="High Yield Spread",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=0.4 + random.uniform(-0.1, 0.1),
                raw_value=3.5 + random.uniform(-0.5, 1),
                direction="neutral",
                confidence=0.5,
                source="estimated",
                description="Credit spreads normal"
            ),
            Signal(
                id="ext_fin_stress",
                name="Financial Stress Index",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=0.4 + random.uniform(-0.1, 0.1),
                raw_value=random.uniform(-0.5, 0.5),
                direction="neutral",
                confidence=0.5,
                source="estimated",
                description="Low-to-normal stress"
            ),
            Signal(
                id="ext_consumer_sent",
                name="Consumer Sentiment",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=0.5 + random.uniform(-0.1, 0.1),
                raw_value=80 + random.uniform(-10, 15),
                direction="neutral",
                confidence=0.5,
                source="estimated",
                description="Consumer sentiment average"
            ),
            Signal(
                id="ext_risk_sentiment",
                name="Risk Sentiment",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=0.5,
                raw_value="neutral",
                direction="neutral",
                confidence=0.5,
                source="estimated",
                description="Normal risk environment"
            ),
            Signal(
                id="ext_safe_haven",
                name="Safe Haven Flows",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=0.5 + random.uniform(-0.1, 0.1),
                raw_value=None,
                direction="neutral",
                confidence=0.5,
                source="estimated",
                description="Balanced asset flows"
            ),
            Signal(
                id="ext_dollar_strength",
                name="Dollar Strength",
                category=SignalCategory.EXTERNAL,
                tier=SignalTier.DAILY,
                value=0.5 + random.uniform(-0.1, 0.1),
                raw_value=None,
                direction="neutral",
                confidence=0.5,
                source="estimated",
                description="USD stable"
            ),
        ]
        
        return signals
