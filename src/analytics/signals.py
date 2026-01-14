"""
Signal generation based on technical analysis.

Generates trading signals with:
- Technical score (0-1)
- Sentiment score (0-1) via FinBERT
- Risk-adjusted position sizing via Monte Carlo VaR
- Signal type (BUY/SELL/HOLD)
- Rationale explanation

Usage:
    generator = SignalGenerator()
    signal = generator.generate_signal("AAPL")
"""

from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd

from src.analytics.technical import TechnicalAnalyzer
from src.data.persistence import get_database
from src.data.cache import get_cache
from src.logging_config import get_logger

logger = get_logger(__name__)


class SignalGenerator:
    """
    Generate trading signals based on technical analysis, sentiment, and risk.
    
    Combines:
    - Technical analysis (RSI, MACD, BB, trend)
    - Sentiment analysis (FinBERT - time-weighted)
    - Risk metrics (Monte Carlo VaR, position sizing)
    """
    
    # Weights for confluence score
    TECHNICAL_WEIGHT = 0.50  # Technical indicators
    SENTIMENT_WEIGHT = 0.30  # FinBERT sentiment (Days 6-30 peak)
    ML_WEIGHT = 0.20         # ML/ensemble predictions (future)
    
    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        self.db = get_database()
        self.cache = get_cache()
        self._risk_engine = None
        self._sentiment_analyzer = None
    
    @property
    def risk_engine(self):
        """Lazy load risk engine."""
        if self._risk_engine is None:
            try:
                from src.analytics.risk import RiskEngine
                self._risk_engine = RiskEngine()
            except Exception as e:
                logger.warning("Risk engine unavailable", error=str(e))
        return self._risk_engine
    
    @property
    def sentiment_analyzer(self):
        """Lazy load sentiment analyzer."""
        if self._sentiment_analyzer is None:
            try:
                from src.analytics.sentiment import FinBERTAnalyzer
                self._sentiment_analyzer = FinBERTAnalyzer()
            except Exception as e:
                logger.warning("Sentiment analyzer unavailable", error=str(e))
        return self._sentiment_analyzer
    
    def generate_signal(self, symbol: str, min_confidence: float = 0.0) -> Optional[Dict]:
        """
        Generate a trading signal for a symbol.
        
        Args:
            symbol: Stock ticker
            min_confidence: Minimum confidence threshold (0-1)
        
        Returns:
            Signal dictionary or None if insufficient data
        """
        logger.info("Generating signal", symbol=symbol)
        
        # Get price data
        df = self.db.get_candles(symbol, limit=250)
        
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data", symbol=symbol, rows=len(df))
            return None
        
        # Compute indicators
        df_with_indicators = self.analyzer.compute_all(df)
        
        # Calculate technical score
        technical_result = self.analyzer.calculate_technical_score(df_with_indicators)
        
        # Get sentiment score (from cache or compute)
        sentiment_data = self._get_sentiment_score(symbol)
        
        # Get risk metrics
        risk_data = self._get_risk_metrics(symbol, df)
        
        # Calculate confluence score (weighted combination)
        confluence_score = self._calculate_confluence(
            technical_result['technical_score'],
            sentiment_data['normalized_score'],
            0.5  # ML placeholder
        )
        
        # Get latest price
        latest = df_with_indicators.iloc[-1]
        
        # Determine signal type from confluence
        signal_type = self._determine_signal_type(confluence_score)
        
        # Build signal
        signal = {
            "symbol": symbol,
            "created_at": datetime.utcnow(),
            "price_at_signal": float(latest['close']),
            
            # Scores
            "technical_score": technical_result['technical_score'],
            "sentiment_score": sentiment_data['normalized_score'],
            "ml_score": 0.5,  # Placeholder - will be from ML models
            "confluence_score": confluence_score,
            
            # Signal type (from confluence, not just technical)
            "signal_type": signal_type,
            
            # Component scores
            "component_scores": technical_result['component_scores'],
            
            # Rationale
            "technical_rationale": technical_result['rationale'],
            "sentiment_rationale": self._build_sentiment_rationale(sentiment_data),
            
            # Risk metrics
            "risk": {
                "var_95": risk_data.get('var_95'),
                "recommended_position_pct": risk_data.get('recommended_position_pct'),
                "max_position_pct": risk_data.get('max_position_pct'),
            },
            
            # Key indicators
            "indicators": {
                "rsi_14": float(latest.get('rsi_14', 0)) if not pd.isna(latest.get('rsi_14')) else None,
                "macd": float(latest.get('macd', 0)) if not pd.isna(latest.get('macd')) else None,
                "macd_signal": float(latest.get('macd_signal', 0)) if not pd.isna(latest.get('macd_signal')) else None,
                "sma_50": float(latest.get('sma_50', 0)) if not pd.isna(latest.get('sma_50')) else None,
                "sma_200": float(latest.get('sma_200', 0)) if not pd.isna(latest.get('sma_200')) else None,
                "bb_upper": float(latest.get('bb_upper', 0)) if not pd.isna(latest.get('bb_upper')) else None,
                "bb_lower": float(latest.get('bb_lower', 0)) if not pd.isna(latest.get('bb_lower')) else None,
                "atr": float(latest.get('atr', 0)) if not pd.isna(latest.get('atr')) else None,
                "adx": float(latest.get('adx', 0)) if not pd.isna(latest.get('adx')) else None,
            },
            
            # Sentiment details
            "sentiment": {
                "weighted_score": sentiment_data.get('weighted_score', 0),
                "label": sentiment_data.get('overall_label', 'neutral'),
                "article_count": sentiment_data.get('article_count', 0),
                "signal_quality": sentiment_data.get('signal_quality', 'NO_DATA'),
            },
            
            # Trend info
            "trend_signal": latest.get('trend_signal', 'SIDEWAYS'),
            "rsi_signal": latest.get('rsi_signal', 'NEUTRAL'),
            "macd_crossover": latest.get('macd_crossover', 'NONE'),
        }
        
        logger.info(
            "Signal generated",
            symbol=symbol,
            type=signal['signal_type'],
            technical=signal['technical_score'],
            sentiment=signal['sentiment_score'],
            confluence=signal['confluence_score']
        )
        
        return signal
    
    def _get_sentiment_score(self, symbol: str) -> Dict:
        """Get sentiment score from cache or compute."""
        # Try cache first
        cached = self.cache.get_sentiment(symbol)
        if cached:
            return cached
        
        # Return neutral if analyzer unavailable
        if self.sentiment_analyzer is None:
            return {'normalized_score': 0.5, 'weighted_score': 0, 'overall_label': 'neutral', 
                    'article_count': 0, 'signal_quality': 'NO_DATA'}
        
        # Get news and analyze
        try:
            news_items = self.db.get_recent_news(symbol, days=90)
            if not news_items:
                return {'normalized_score': 0.5, 'weighted_score': 0, 'overall_label': 'neutral',
                        'article_count': 0, 'signal_quality': 'NO_DATA'}
            
            news_dicts = [
                {'headline': n.headline, 'published_at': n.published_at.isoformat() if n.published_at else None}
                for n in news_items
            ]
            
            result = self.sentiment_analyzer.aggregate_sentiment(symbol, news_dicts)
            self.cache.set_sentiment(symbol, result)
            return result
        except Exception as e:
            logger.warning("Sentiment analysis failed", symbol=symbol, error=str(e))
            return {'normalized_score': 0.5, 'weighted_score': 0, 'overall_label': 'neutral',
                    'article_count': 0, 'signal_quality': 'NO_DATA'}
    
    def _get_risk_metrics(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Get risk metrics from risk engine."""
        if self.risk_engine is None:
            return {'var_95': 0.05, 'recommended_position_pct': 0.02, 'max_position_pct': 0.05}
        
        try:
            returns = df['close'].pct_change().dropna()
            metrics = self.risk_engine.calculate_portfolio_risk(returns)
            return {
                'var_95': metrics.var_95,
                'var_99': metrics.var_99,
                'recommended_position_pct': metrics.recommended_position_pct,
                'max_position_pct': metrics.max_position_pct,
                'sharpe_ratio': metrics.sharpe_ratio,
            }
        except Exception as e:
            logger.warning("Risk calculation failed", symbol=symbol, error=str(e))
            return {'var_95': 0.05, 'recommended_position_pct': 0.02, 'max_position_pct': 0.05}
    
    def _calculate_confluence(
        self, 
        technical: float, 
        sentiment: float, 
        ml: float
    ) -> float:
        """
        Calculate weighted confluence score.
        
        Combines technical (50%), sentiment (30%), ML (20%).
        """
        confluence = (
            technical * self.TECHNICAL_WEIGHT +
            sentiment * self.SENTIMENT_WEIGHT +
            ml * self.ML_WEIGHT
        )
        return round(confluence, 4)
    
    def _determine_signal_type(self, confluence_score: float) -> str:
        """Determine signal type from confluence score."""
        if confluence_score >= 0.75:
            return "STRONG_BUY"
        elif confluence_score >= 0.65:
            return "BUY"
        elif confluence_score <= 0.25:
            return "STRONG_SELL"
        elif confluence_score <= 0.35:
            return "SELL"
        else:
            return "HOLD"
    
    def _build_sentiment_rationale(self, sentiment_data: Dict) -> str:
        """Build sentiment rationale string."""
        label = sentiment_data.get('overall_label', 'neutral')
        count = sentiment_data.get('article_count', 0)
        quality = sentiment_data.get('signal_quality', 'NO_DATA')
        
        if count == 0:
            return "No recent news available."
        
        return f"Sentiment is {label} based on {count} articles. Signal quality: {quality}."
    
    def generate_signals_batch(
        self, 
        symbols: List[str],
        min_confidence: float = 0.65
    ) -> List[Dict]:
        """
        Generate signals for multiple symbols.
        
        Args:
            symbols: List of stock tickers
            min_confidence: Minimum confidence for actionable signals
        
        Returns:
            List of signals, filtered by minimum confidence
        """
        signals = []
        
        for symbol in symbols:
            try:
                signal = self.generate_signal(symbol, min_confidence)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(
                    "Failed to generate signal",
                    symbol=symbol,
                    error=str(e)
                )
        
        # Sort by confluence score (highest first)
        signals.sort(key=lambda x: x['confluence_score'], reverse=True)
        
        # Filter actionable signals
        actionable = [s for s in signals if s['signal_type'] in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']]
        
        logger.info(
            "Batch signal generation complete",
            total=len(signals),
            actionable=len(actionable)
        )
        
        return signals
    
    def save_signal(self, signal: Dict) -> str:
        """
        Save a signal to the database.
        
        Args:
            signal: Signal dictionary
        
        Returns:
            Signal ID
        """
        # Prepare for database
        db_signal = {
            "symbol": signal['symbol'],
            "created_at": signal['created_at'],
            "signal_type": signal['signal_type'],
            "technical_score": signal['technical_score'],
            "sentiment_score": signal.get('sentiment_score', 0.5),
            "ml_score": signal.get('ml_score', 0.5),
            "confluence_score": signal['confluence_score'],
            "technical_rationale": signal.get('technical_rationale', ''),
            "sentiment_rationale": signal.get('sentiment_rationale', ''),
            "price_at_signal": signal.get('price_at_signal'),
        }
        
        signal_id = self.db.save_signal(db_signal)
        
        # Cache the signal
        self.cache.set_signal(signal['symbol'], signal)
        
        return signal_id
    
    def update_and_save_indicators(self, symbol: str) -> Optional[Dict]:
        """
        Update indicators and save to database/cache.
        
        Args:
            symbol: Stock ticker
        
        Returns:
            Indicator dictionary or None
        """
        # Get price data
        df = self.db.get_candles(symbol, limit=250)
        
        if df.empty or len(df) < 50:
            return None
        
        # Compute indicators
        df_with_indicators = self.analyzer.compute_all(df)
        latest = df_with_indicators.iloc[-1]
        
        # Calculate score
        score_result = self.analyzer.calculate_technical_score(df_with_indicators)
        
        # Prepare indicator cache entry
        indicators = {
            "symbol": symbol,
            "as_of_date": latest['time'],
            "computed_at": datetime.utcnow(),
            
            # Momentum
            "rsi_14": float(latest.get('rsi_14', 0)) if not pd.isna(latest.get('rsi_14')) else None,
            "rsi_signal": latest.get('rsi_signal', 'NEUTRAL'),
            "macd": float(latest.get('macd', 0)) if not pd.isna(latest.get('macd')) else None,
            "macd_signal": float(latest.get('macd_signal', 0)) if not pd.isna(latest.get('macd_signal')) else None,
            "macd_histogram": float(latest.get('macd_histogram', 0)) if not pd.isna(latest.get('macd_histogram')) else None,
            "macd_crossover": latest.get('macd_crossover', 'NONE'),
            
            # Trend
            "sma_20": float(latest.get('sma_20', 0)) if not pd.isna(latest.get('sma_20')) else None,
            "sma_50": float(latest.get('sma_50', 0)) if not pd.isna(latest.get('sma_50')) else None,
            "sma_200": float(latest.get('sma_200', 0)) if not pd.isna(latest.get('sma_200')) else None,
            "trend_signal": latest.get('trend_signal', 'SIDEWAYS'),
            "adx": float(latest.get('adx', 0)) if not pd.isna(latest.get('adx')) else None,
            
            # Volatility
            "bb_upper": float(latest.get('bb_upper', 0)) if not pd.isna(latest.get('bb_upper')) else None,
            "bb_middle": float(latest.get('bb_middle', 0)) if not pd.isna(latest.get('bb_middle')) else None,
            "bb_lower": float(latest.get('bb_lower', 0)) if not pd.isna(latest.get('bb_lower')) else None,
            "bb_bandwidth": float(latest.get('bb_bandwidth', 0)) if not pd.isna(latest.get('bb_bandwidth')) else None,
            "atr": float(latest.get('atr', 0)) if not pd.isna(latest.get('atr')) else None,
            
            # Score
            "technical_score": score_result['technical_score'],
        }
        
        # Save to database
        self.db.save_indicators(indicators)
        
        # Cache
        self.cache.set_indicators(symbol, indicators)
        
        logger.info(
            "Updated indicators",
            symbol=symbol,
            score=indicators['technical_score']
        )
        
        return indicators


# Convenience function
def get_signal_generator() -> SignalGenerator:
    """Get a SignalGenerator instance."""
    return SignalGenerator()
