"""
Fundamentals & Macro Signal Providers

Fundamentals Provider:
- P/E Ratio, EPS, Revenue Growth
- Free Cash Flow, ROE, Debt/Equity
- Book Value, Dividend Yield
- Earnings Surprise, Guidance

Macro Provider:
- Fed Rate Expectations
- Inflation (CPI, PPI)
- GDP Growth
- Unemployment
- Consumer Confidence
- Dollar Strength
- Oil Prices
- Treasury Yields

Data Sources:
- Financial Modeling Prep (fundamentals)
- Alpha Vantage (fundamentals)
- FRED API (macroeconomic data)
- Yahoo Finance (basic fundamentals)
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import json

from src.services.signal_intelligence import (
    SignalProvider, Signal, SignalCategory, SignalTier
)
from src.logging_config import get_logger

logger = get_logger(__name__)


class FundamentalsSignalProvider(SignalProvider):
    """
    Fundamentals Signal Provider
    
    Fetches company fundamentals from financial APIs.
    """
    
    def __init__(self):
        super().__init__()
        self.fmp_key = os.getenv('FMP_API_KEY', '')
        self.alphavantage_key = os.getenv('ALPHA_VANTAGE_KEY', '')
    
    async def get_signals(self, symbol: str) -> List[Signal]:
        """Get fundamental signals for a symbol."""
        signals = []
        
        # Check cache (fundamentals change daily, so longer TTL)
        cached = await self._get_cached(symbol, "fundamentals")
        if cached:
            return [Signal(**{**s, "category": SignalCategory.FUNDAMENTALS, "tier": SignalTier.DAILY}) 
                    for s in cached.get("signals", [])]
        
        # Try Financial Modeling Prep first
        if self.fmp_key:
            signals = await self._fetch_fmp_fundamentals(symbol)
        
        # Fallback to Alpha Vantage
        if not signals and self.alphavantage_key:
            signals = await self._fetch_alphavantage_fundamentals(symbol)
        
        # Fallback to mock data
        if not signals:
            signals = self._generate_mock_fundamentals(symbol)
        
        # Cache for 24 hours
        await self._set_cached(symbol, "fundamentals",
                               {"signals": [s.__dict__ for s in signals]},
                               ttl=86400)
        
        return signals
    
    async def _fetch_fmp_fundamentals(self, symbol: str) -> List[Signal]:
        """Fetch from Financial Modeling Prep API."""
        signals = []
        
        try:
            session = await self.get_session()
            
            # Key Metrics
            url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}?apikey={self.fmp_key}"
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        metrics = data[0]
                        signals.extend(self._parse_fmp_metrics(metrics, symbol))
            
            # Ratios
            url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={self.fmp_key}"
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        ratios = data[0]
                        signals.extend(self._parse_fmp_ratios(ratios, symbol))
            
            # Rating
            url = f"https://financialmodelingprep.com/api/v3/rating/{symbol}?apikey={self.fmp_key}"
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        rating = data[0]
                        signals.extend(self._parse_fmp_rating(rating, symbol))
                        
        except Exception as e:
            logger.warning(f"FMP API error: {e}")
        
        return signals
    
    def _parse_fmp_metrics(self, metrics: Dict, symbol: str) -> List[Signal]:
        """Parse FMP key metrics."""
        signals = []
        
        # P/E Ratio
        pe = metrics.get('peRatioTTM')
        if pe:
            # Normalize: P/E 15-25 is normal, <15 undervalued, >35 overvalued
            pe_score = max(0, min(1, 1 - (pe - 20) / 30))
            signals.append(Signal(
                id="fund_pe_ratio",
                name="P/E Ratio",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=pe_score,
                raw_value=pe,
                direction="bullish" if pe < 18 else "bearish" if pe > 35 else "neutral",
                confidence=0.7,
                source="fmp",
                description=f"P/E: {pe:.1f} ({'undervalued' if pe < 18 else 'overvalued' if pe > 35 else 'fair'})"
            ))
        
        # ROE
        roe = metrics.get('roeTTM')
        if roe:
            roe_score = max(0, min(1, roe / 0.3))  # 30% ROE = 1.0
            signals.append(Signal(
                id="fund_roe",
                name="Return on Equity (ROE)",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=roe_score,
                raw_value=roe * 100,
                direction="bullish" if roe > 0.15 else "bearish" if roe < 0.05 else "neutral",
                confidence=0.75,
                source="fmp",
                description=f"ROE: {roe*100:.1f}%"
            ))
        
        # FCF Yield
        fcf_yield = metrics.get('freeCashFlowYieldTTM')
        if fcf_yield:
            fcf_score = max(0, min(1, fcf_yield / 0.1))  # 10% FCF yield = 1.0
            signals.append(Signal(
                id="fund_fcf_yield",
                name="Free Cash Flow Yield",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=fcf_score,
                raw_value=fcf_yield * 100,
                direction="bullish" if fcf_yield > 0.05 else "bearish" if fcf_yield < 0 else "neutral",
                confidence=0.7,
                source="fmp",
                description=f"FCF Yield: {fcf_yield*100:.1f}%"
            ))
        
        # Revenue Per Share Growth
        rev_growth = metrics.get('revenuePerShareTTM')
        if rev_growth:
            signals.append(Signal(
                id="fund_rev_per_share",
                name="Revenue Per Share",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=0.5,  # Need comparison to historical for signal
                raw_value=rev_growth,
                direction="neutral",
                confidence=0.5,
                source="fmp",
                description=f"Revenue/Share: ${rev_growth:.2f}"
            ))
        
        return signals
    
    def _parse_fmp_ratios(self, ratios: Dict, symbol: str) -> List[Signal]:
        """Parse FMP financial ratios."""
        signals = []
        
        # Debt to Equity
        de = ratios.get('debtEquityRatioTTM')
        if de:
            de_score = max(0, min(1, 1 - de / 2))  # Lower is better, 2.0 = 0.0
            signals.append(Signal(
                id="fund_debt_equity",
                name="Debt/Equity Ratio",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=de_score,
                raw_value=de,
                direction="bullish" if de < 0.5 else "bearish" if de > 1.5 else "neutral",
                confidence=0.7,
                source="fmp",
                description=f"D/E: {de:.2f} ({'low' if de < 0.5 else 'high' if de > 1.5 else 'moderate'} leverage)"
            ))
        
        # Current Ratio (Liquidity)
        current = ratios.get('currentRatioTTM')
        if current:
            current_score = min(1, current / 2)  # 2.0 current ratio = 1.0
            signals.append(Signal(
                id="fund_current_ratio",
                name="Current Ratio",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=current_score,
                raw_value=current,
                direction="bullish" if current > 1.5 else "bearish" if current < 1 else "neutral",
                confidence=0.65,
                source="fmp",
                description=f"Current Ratio: {current:.2f}"
            ))
        
        # Gross Margin
        gm = ratios.get('grossProfitMarginTTM')
        if gm:
            gm_score = min(1, gm)  # 100% margin = 1.0
            signals.append(Signal(
                id="fund_gross_margin",
                name="Gross Margin",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=gm_score,
                raw_value=gm * 100,
                direction="bullish" if gm > 0.4 else "bearish" if gm < 0.2 else "neutral",
                confidence=0.7,
                source="fmp",
                description=f"Gross Margin: {gm*100:.1f}%"
            ))
        
        # Operating Margin
        om = ratios.get('operatingProfitMarginTTM')
        if om:
            om_score = max(0, min(1, (om + 0.1) / 0.4))  # -10% to 30% range
            signals.append(Signal(
                id="fund_operating_margin",
                name="Operating Margin",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=om_score,
                raw_value=om * 100,
                direction="bullish" if om > 0.15 else "bearish" if om < 0.05 else "neutral",
                confidence=0.7,
                source="fmp",
                description=f"Operating Margin: {om*100:.1f}%"
            ))
        
        # Dividend Yield
        div = ratios.get('dividendYieldTTM')
        if div:
            div_score = min(1, div / 0.05)  # 5% yield = 1.0
            signals.append(Signal(
                id="fund_dividend_yield",
                name="Dividend Yield",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=div_score,
                raw_value=div * 100,
                direction="bullish" if div > 0.02 else "neutral",
                confidence=0.6,
                source="fmp",
                description=f"Dividend Yield: {div*100:.2f}%"
            ))
        
        return signals
    
    def _parse_fmp_rating(self, rating: Dict, symbol: str) -> List[Signal]:
        """Parse FMP stock rating."""
        signals = []
        
        overall = rating.get('ratingScore')
        if overall:
            signals.append(Signal(
                id="fund_overall_rating",
                name="FMP Rating Score",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=overall / 5,  # 5 = best
                raw_value={"score": overall, "recommendation": rating.get('ratingRecommendation')},
                direction="bullish" if overall >= 4 else "bearish" if overall <= 2 else "neutral",
                confidence=0.75,
                source="fmp",
                description=f"Rating: {rating.get('ratingRecommendation', 'N/A')} ({overall}/5)"
            ))
        
        # DCF Value vs Price
        dcf = rating.get('ratingDetailsDCFScore')
        if dcf:
            signals.append(Signal(
                id="fund_dcf_score",
                name="DCF Valuation Score",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=dcf / 5,
                raw_value=dcf,
                direction="bullish" if dcf >= 4 else "bearish" if dcf <= 2 else "neutral",
                confidence=0.7,
                source="fmp",
                description=f"DCF Score: {dcf}/5"
            ))
        
        return signals
    
    async def _fetch_alphavantage_fundamentals(self, symbol: str) -> List[Signal]:
        """Fetch from Alpha Vantage."""
        signals = []
        
        try:
            session = await self.get_session()
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={self.alphavantage_key}"
            
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and 'Symbol' in data:
                        signals = self._parse_alphavantage_overview(data)
                        
        except Exception as e:
            logger.warning(f"Alpha Vantage fundamentals error: {e}")
        
        return signals
    
    def _parse_alphavantage_overview(self, data: Dict) -> List[Signal]:
        """Parse Alpha Vantage company overview."""
        signals = []
        
        # P/E Ratio
        pe = data.get('PERatio')
        if pe and pe != 'None':
            pe = float(pe)
            pe_score = max(0, min(1, 1 - (pe - 20) / 30))
            signals.append(Signal(
                id="fund_pe_ratio",
                name="P/E Ratio",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=pe_score,
                raw_value=pe,
                direction="bullish" if pe < 18 else "bearish" if pe > 35 else "neutral",
                confidence=0.7,
                source="alphavantage",
                description=f"P/E: {pe:.1f}"
            ))
        
        # EPS
        eps = data.get('EPS')
        if eps and eps != 'None':
            eps = float(eps)
            signals.append(Signal(
                id="fund_eps",
                name="Earnings Per Share",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=0.5 + (0.5 if eps > 0 else -0.5) * min(1, abs(eps) / 10),
                raw_value=eps,
                direction="bullish" if eps > 2 else "bearish" if eps < 0 else "neutral",
                confidence=0.7,
                source="alphavantage",
                description=f"EPS: ${eps:.2f}"
            ))
        
        # Profit Margin
        margin = data.get('ProfitMargin')
        if margin and margin != 'None':
            margin = float(margin)
            signals.append(Signal(
                id="fund_profit_margin",
                name="Profit Margin",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=max(0, min(1, (margin + 0.1) / 0.4)),
                raw_value=margin * 100,
                direction="bullish" if margin > 0.1 else "bearish" if margin < 0 else "neutral",
                confidence=0.7,
                source="alphavantage",
                description=f"Profit Margin: {margin*100:.1f}%"
            ))
        
        # ROE
        roe = data.get('ReturnOnEquityTTM')
        if roe and roe != 'None':
            roe = float(roe)
            roe_score = max(0, min(1, roe / 0.3))
            signals.append(Signal(
                id="fund_roe",
                name="Return on Equity",
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=roe_score,
                raw_value=roe * 100,
                direction="bullish" if roe > 0.15 else "bearish" if roe < 0.05 else "neutral",
                confidence=0.75,
                source="alphavantage",
                description=f"ROE: {roe*100:.1f}%"
            ))
        
        return signals
    
    def _generate_mock_fundamentals(self, symbol: str) -> List[Signal]:
        """Generate mock fundamental signals."""
        import random
        seed = sum(ord(c) for c in symbol)
        random.seed(seed)
        
        # Stock-specific biases
        strong_fundamentals = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'V', 'MA', 'JPM', 'UNH']
        weak_fundamentals = ['TSLA', 'NFLX', 'SNAP', 'RIVN']
        
        base = 0.55 if symbol in strong_fundamentals else 0.4 if symbol in weak_fundamentals else 0.5
        
        mock_signals = [
            ("P/E Ratio", base + random.random() * 0.3 - 0.1),
            ("EPS Growth", base + random.random() * 0.35 - 0.1),
            ("Revenue Growth", base + random.random() * 0.3),
            ("Gross Margin", base + random.random() * 0.25 + 0.1),
            ("Free Cash Flow", base + random.random() * 0.3),
            ("Debt/Equity", base + random.random() * 0.25),
            ("ROE", base + random.random() * 0.3),
            ("Dividend Yield", 0.3 + random.random() * 0.4),
            ("Book Value Growth", base + random.random() * 0.25),
            ("Working Capital", base + random.random() * 0.2 + 0.1),
        ]
        
        return [
            Signal(
                id=f"fund_{name.lower().replace(' ', '_').replace('/', '_')}",
                name=name,
                category=SignalCategory.FUNDAMENTALS,
                tier=SignalTier.DAILY,
                value=min(1, max(0, value)),
                direction="bullish" if value > 0.6 else "bearish" if value < 0.4 else "neutral",
                confidence=0.5 + random.random() * 0.2,
                source="mock",
                description=f"{name} analysis"
            )
            for name, value in mock_signals
        ]


class MacroSignalProvider(SignalProvider):
    """
    Macroeconomic Signal Provider
    
    Fetches macro data from FRED and other sources.
    """
    
    def __init__(self):
        super().__init__()
        self.fred_key = os.getenv('FRED_API_KEY', '')
    
    async def get_signals(self, symbol: str) -> List[Signal]:
        """Get macro signals (same for all symbols, but with sector adjustments)."""
        signals = []
        
        # Check cache (macro data, longer TTL)
        cached = await self._get_cached("MACRO", "macro_global")
        if cached:
            base_signals = [Signal(**{**s, "category": SignalCategory.MACROECONOMICS, "tier": SignalTier.DAILY}) 
                           for s in cached.get("signals", [])]
            return self._adjust_for_sector(base_signals, symbol)
        
        # Fetch from FRED API
        if self.fred_key:
            signals = await self._fetch_fred_data()
        
        # Fallback to mock
        if not signals:
            signals = self._generate_mock_macro()
        
        # Cache for 6 hours
        await self._set_cached("MACRO", "macro_global",
                               {"signals": [s.__dict__ for s in signals]},
                               ttl=21600)
        
        return self._adjust_for_sector(signals, symbol)
    
    async def _fetch_fred_data(self) -> List[Signal]:
        """Fetch data from FRED API."""
        signals = []
        
        # Key FRED series
        series_map = {
            'FEDFUNDS': ('Fed Funds Rate', 'Interest rate environment'),
            'CPIAUCSL': ('CPI Inflation', 'Consumer price inflation'),
            'GDP': ('GDP Growth', 'Economic growth indicator'),
            'UNRATE': ('Unemployment Rate', 'Labor market health'),
            'UMCSENT': ('Consumer Sentiment', 'Consumer confidence'),
            'DFF': ('Effective Fed Rate', 'Current fed rate'),
            'T10Y2Y': ('Yield Curve (10Y-2Y)', 'Recession indicator'),
            'VIXCLS': ('VIX Index', 'Market fear gauge'),
        }
        
        try:
            session = await self.get_session()
            
            for series_id, (name, desc) in series_map.items():
                try:
                    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.fred_key}&file_type=json&sort_order=desc&limit=5"
                    
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            observations = data.get('observations', [])
                            
                            if observations:
                                latest = float(observations[0]['value']) if observations[0]['value'] != '.' else None
                                if latest is not None:
                                    signal = self._create_macro_signal(series_id, name, latest, desc)
                                    if signal:
                                        signals.append(signal)
                                        
                except Exception as e:
                    logger.warning(f"FRED series {series_id} error: {e}")
                    
        except Exception as e:
            logger.warning(f"FRED API error: {e}")
        
        return signals
    
    def _create_macro_signal(self, series_id: str, name: str, value: float, desc: str) -> Optional[Signal]:
        """Create a signal from FRED data."""
        
        # Normalize based on series type
        if series_id == 'FEDFUNDS' or series_id == 'DFF':
            # Fed rate: lower is more accommodative (bullish for stocks)
            normalized = max(0, min(1, 1 - value / 8))  # 0-8% range
            direction = "bullish" if value < 3 else "bearish" if value > 5 else "neutral"
            
        elif series_id == 'CPIAUCSL':
            # CPI: lower is better, high inflation is bearish
            yoy_change = 3.5  # Approximate, would need to calculate
            normalized = max(0, min(1, 1 - yoy_change / 10))
            direction = "bullish" if yoy_change < 2.5 else "bearish" if yoy_change > 4 else "neutral"
            
        elif series_id == 'UNRATE':
            # Unemployment: Goldilocks zone 3.5-5%
            normalized = max(0, min(1, 1 - abs(value - 4.2) / 4))
            direction = "bullish" if 3.5 < value < 5 else "bearish" if value > 6 else "neutral"
            
        elif series_id == 'UMCSENT':
            # Consumer sentiment: higher is better, 60-100 range
            normalized = max(0, min(1, (value - 50) / 50))
            direction = "bullish" if value > 80 else "bearish" if value < 60 else "neutral"
            
        elif series_id == 'T10Y2Y':
            # Yield curve: positive is normal, negative = recession risk
            normalized = max(0, min(1, (value + 1) / 2))  # -1 to +1 range
            direction = "bullish" if value > 0.5 else "bearish" if value < 0 else "neutral"
            
        elif series_id == 'VIXCLS':
            # VIX: lower is calmer markets
            normalized = max(0, min(1, 1 - (value - 12) / 40))  # 12-52 range
            direction = "bullish" if value < 18 else "bearish" if value > 30 else "neutral"
            
        else:
            normalized = 0.5
            direction = "neutral"
        
        return Signal(
            id=f"macro_{series_id.lower()}",
            name=name,
            category=SignalCategory.MACROECONOMICS,
            tier=SignalTier.DAILY,
            value=normalized,
            raw_value=value,
            direction=direction,
            confidence=0.7,
            source="fred",
            description=f"{desc}: {value:.2f}"
        )
    
    def _adjust_for_sector(self, signals: List[Signal], symbol: str) -> List[Signal]:
        """Adjust macro signals based on stock's sector sensitivity."""
        # Sector mappings
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'META', 'AMZN', 'CRM']
        financial_stocks = ['JPM', 'BAC', 'GS', 'V', 'MA', 'BRK.B', 'SCHW']
        consumer_stocks = ['CMG', 'SBUX', 'MCD', 'NKE', 'DIS', 'COST', 'TGT', 'WMT']
        energy_stocks = ['XOM', 'CVX', 'OXY', 'COP']
        
        # Apply sector-specific adjustments
        adjusted = []
        for signal in signals:
            adj_signal = Signal(**signal.__dict__)
            
            if 'FEDFUNDS' in signal.id or 'DFF' in signal.id:
                # Tech and growth stocks more sensitive to rates
                if symbol in tech_stocks:
                    adj_signal.confidence = min(0.9, signal.confidence + 0.1)
                    adj_signal.description += " (high tech sensitivity)"
                elif symbol in financial_stocks:
                    # Banks benefit from higher rates
                    adj_signal.value = 1 - signal.value  # Invert
                    adj_signal.direction = "bearish" if signal.direction == "bullish" else "bullish" if signal.direction == "bearish" else "neutral"
                    adj_signal.description += " (banks benefit from higher rates)"
            
            elif 'UMCSENT' in signal.id:
                # Consumer stocks very sensitive to consumer sentiment
                if symbol in consumer_stocks:
                    adj_signal.confidence = min(0.9, signal.confidence + 0.15)
                    adj_signal.description += " (high consumer sensitivity)"
            
            adjusted.append(adj_signal)
        
        return adjusted
    
    def _generate_mock_macro(self) -> List[Signal]:
        """Generate mock macro signals."""
        import random
        
        # Use current hour as seed for some variation
        seed = int(datetime.now().timestamp() // 3600)
        random.seed(seed)
        
        mock_signals = [
            ("Fed Rate Expectations", 0.45 + random.random() * 0.3, "Interest rate trajectory"),
            ("Inflation (CPI)", 0.4 + random.random() * 0.35, "Consumer price effects"),
            ("GDP Growth Outlook", 0.5 + random.random() * 0.3, "Economic expansion"),
            ("Unemployment Trends", 0.55 + random.random() * 0.25, "Labor market health"),
            ("Dollar Strength (DXY)", 0.45 + random.random() * 0.3, "Currency impact"),
            ("Treasury Yields", 0.5 + random.random() * 0.3, "Bond market signals"),
            ("Oil/Energy Prices", 0.4 + random.random() * 0.35, "Commodity impact"),
            ("Consumer Confidence", 0.5 + random.random() * 0.3, "Spending outlook"),
            ("VIX Volatility", 0.55 + random.random() * 0.25, "Market fear gauge"),
        ]
        
        return [
            Signal(
                id=f"macro_{name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}",
                name=name,
                category=SignalCategory.MACROECONOMICS,
                tier=SignalTier.DAILY,
                value=value,
                direction="bullish" if value > 0.6 else "bearish" if value < 0.4 else "neutral",
                confidence=0.6,
                source="mock",
                description=desc
            )
            for name, value, desc in mock_signals
        ]
