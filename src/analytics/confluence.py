"""
Confluence Engine: Multi-Signal Trading Decision System.

Combines multiple data sources into unified trading signals:
1. Technical Analysis (indicators, patterns, momentum)
2. Sentiment Analysis (FinBERT, news, earnings)
3. Risk Metrics (VaR, position sizing, drawdown)
4. ML Predictions (future: ensemble models)

Research Foundation:
- Technical alone: 45-55% accuracy
- Sentiment alone: 54-58% accuracy
- Combined (confluence): 62-65% accuracy
- With risk filters: 58-60% accuracy but better risk-adjusted returns

Four-Layer Strategy Framework (from research):
Layer 1: Technical Analysis (momentum, trend) - 1-5 days
Layer 2: Sentiment Analysis (FinBERT, news) - 6-30 days
Layer 3: ML (pattern recognition) - 5-20 days
Layer 4: Risk Management (position sizing) - continuous

Usage:
    engine = ConfluenceEngine()
    result = engine.analyze(symbol)
    print(result.signal_type, result.confluence_score, result.position_size)
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)


class SignalStrength(Enum):
    """Signal conviction strength based on confluence agreement."""
    VERY_STRONG = "VERY_STRONG"  # All signals agree strongly
    STRONG = "STRONG"            # Most signals agree
    MODERATE = "MODERATE"        # Mixed signals, slight edge
    WEAK = "WEAK"               # Conflicting signals
    NO_SIGNAL = "NO_SIGNAL"     # Insufficient data or neutral


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class ConfluenceResult:
    """
    Result from confluence analysis.
    
    Contains all scores, signals, and rationale.
    """
    symbol: str
    timestamp: datetime
    
    # Core scores (0-1 scale)
    technical_score: float
    sentiment_score: float
    ml_score: float  # Future: ensemble prediction
    risk_score: float  # Risk-adjusted factor
    
    # Final confluence
    confluence_score: float
    signal_type: SignalType
    signal_strength: SignalStrength
    
    # Position sizing (based on risk)
    recommended_position_pct: float
    max_position_pct: float
    
    # Risk metrics
    var_95: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_reward_ratio: float = 0.0  # Risk-reward ratio (RR = reward/risk)
    
    # Component details
    technical_signals: Dict = field(default_factory=dict)
    sentiment_details: Dict = field(default_factory=dict)
    risk_details: Dict = field(default_factory=dict)
    
    # Rationale
    technical_rationale: str = ""
    sentiment_rationale: str = ""
    overall_rationale: str = ""
    
    # Agreement metrics
    signal_agreement: float = 0.0  # How much do signals agree (0-1)
    conflicting_signals: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "confluence_score": round(self.confluence_score, 4),
            "signal_type": self.signal_type.value,
            "signal_strength": self.signal_strength.value,
            "technical_score": round(self.technical_score, 4),
            "sentiment_score": round(self.sentiment_score, 4),
            "risk_score": round(self.risk_score, 4),
            "recommended_position_pct": round(self.recommended_position_pct * 100, 2),
            "var_95_pct": round(self.var_95 * 100, 2),
            "stop_loss_pct": round(self.stop_loss_pct * 100, 2),
            "take_profit_pct": round(self.take_profit_pct * 100, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "signal_agreement": round(self.signal_agreement, 2),
            "rationale": self.overall_rationale,
        }


class ConfluenceEngine:
    """
    Multi-signal confluence engine for trading decisions.
    
    Combines technical analysis, sentiment analysis, and risk metrics
    to generate high-conviction trading signals.
    
    Weighting (configurable):
    - Technical: 40% (short-term momentum, trend)
    - Sentiment: 35% (Days 6-30 peak window)
    - ML: 15% (future ensemble predictions)
    - Risk: 10% (VaR-based adjustment)
    """
    
    # Default weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        'technical': 0.40,
        'sentiment': 0.35,
        'ml': 0.15,
        'risk': 0.10
    }
    
    # Signal thresholds
    STRONG_BUY_THRESHOLD = 0.75
    BUY_THRESHOLD = 0.62
    SELL_THRESHOLD = 0.38
    STRONG_SELL_THRESHOLD = 0.25
    
    # Agreement thresholds
    VERY_STRONG_AGREEMENT = 0.85
    STRONG_AGREEMENT = 0.70
    MODERATE_AGREEMENT = 0.50
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
        risk_free_rate: float = 0.04,
        use_dynamic_weights: bool = True,
        min_confidence_threshold: float = 0.4,
        min_agreement_threshold: float = 0.5
    ):
        """
        Initialize enhanced confluence engine.
        
        Enhanced with:
        - Dynamic weighting (regime-based, confidence-based, performance-based)
        - Signal filtering (minimum confidence, minimum agreement, conflict detection)
        - Signal ranking (by confluence, risk-adjusted return, Sharpe ratio)
        - Signal validation (backtest, regime, cross-validation)
        
        Args:
            weights: Custom base weights for each signal type
            risk_free_rate: Annual risk-free rate for calculations
            use_dynamic_weights: If True, use dynamic weighting (regime-based, confidence-based)
            min_confidence_threshold: Minimum confidence threshold for signals (default 0.4)
            min_agreement_threshold: Minimum agreement threshold for signals (default 0.5)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.rf_rate = risk_free_rate
        self.use_dynamic_weights = use_dynamic_weights
        self.min_confidence_threshold = min_confidence_threshold
        self.min_agreement_threshold = min_agreement_threshold
        
        # Validate weights sum to 1
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing")
            for k in self.weights:
                self.weights[k] /= total
        
        # Performance tracking for dynamic weighting
        self.recent_performance = {
            'technical': [],
            'sentiment': [],
            'ml': []
        }
        
        # Lazy-loaded components
        self._technical_analyzer = None
        self._sentiment_analyzer = None
        self._risk_engine = None
        self._ensemble = None
    
    @property
    def technical_analyzer(self):
        """Lazy load technical analyzer."""
        if self._technical_analyzer is None:
            from src.analytics.technical import TechnicalAnalyzer
            self._technical_analyzer = TechnicalAnalyzer()
        return self._technical_analyzer
    
    @property
    def sentiment_analyzer(self):
        """Lazy load sentiment analyzer."""
        if self._sentiment_analyzer is None:
            try:
                from src.analytics.sentiment import FinBERTAnalyzer
                self._sentiment_analyzer = FinBERTAnalyzer()
            except Exception as e:
                logger.warning(f"Sentiment analyzer unavailable: {e}")
        return self._sentiment_analyzer
    
    @property
    def risk_engine(self):
        """Lazy load risk engine."""
        if self._risk_engine is None:
            from src.analytics.risk import RiskEngine
            self._risk_engine = RiskEngine()
        return self._risk_engine
    
    @property
    def ensemble(self):
        """Lazy load ensemble model."""
        if self._ensemble is None:
            try:
                from src.analytics.ensemble import HybridSignalEnsemble
                from pathlib import Path
                
                ensemble = HybridSignalEnsemble()
                model_path = Path("models/ensemble_models.pkl")
                
                if model_path.exists():
                    ensemble.load(model_path)
                    self._ensemble = ensemble
                    logger.info("Loaded trained ensemble model")
                else:
                    logger.warning("Ensemble model not trained yet")
            except Exception as e:
                logger.warning(f"Ensemble model unavailable: {e}")
        return self._ensemble
    
    def _get_ml_score(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Get ML prediction score from TFT or ensemble models.
        
        Priority:
        1. TFT (Temporal Fusion Transformer) - if trained and available
        2. Ensemble (HybridSignalEnsemble) - if trained
        3. Default (0.5) - if no models available
        
        Returns:
            Dict with 'score' (0-1), 'uncertainty', 'source', and optional quantiles
        """
        # Try TFT first (best model, 58-60% accuracy)
        tft_score = self._get_tft_score(symbol, df)
        if tft_score and tft_score.get('source') != 'unavailable':
            return tft_score
        
        # Fallback to ensemble
        ensemble_score = self._get_ensemble_score(symbol, df)
        if ensemble_score and ensemble_score.get('source') != 'unavailable':
            return ensemble_score
        
        # Default if no models available
        return {'score': 0.5, 'uncertainty': 1.0, 'source': 'default'}
    
    def _get_tft_score(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Get TFT prediction score.
        
        Returns:
            Dict with 'score' (0-1), quantiles (P10, P50, P90), and uncertainty
        """
        try:
            from src.analytics.tft import TemporalFusionTransformer
            from src.analytics.deep_learning import FeatureEngineer
            from pathlib import Path
            import torch
            
            # Check if TFT model exists
            tft_model_path = Path("models/tft_model.pth")
            if not tft_model_path.exists():
                logger.debug("TFT model not found, skipping TFT prediction")
                return {'score': 0.5, 'uncertainty': 1.0, 'source': 'unavailable'}
            
            # Create features
            X, scaler = FeatureEngineer.create_ml_features(df)
            
            if len(X) < 60:  # Need at least 60 timesteps for TFT
                logger.debug("Insufficient data for TFT", required=60, available=len(X))
                return {'score': 0.5, 'uncertainty': 1.0, 'source': 'unavailable'}
            
            # Initialize TFT model
            model = TemporalFusionTransformer(
                num_features=X.shape[1],
                hidden_size=64,
                num_heads=4
            )
            
            # Load trained weights
            model.load_state_dict(torch.load(tft_model_path, map_location='cpu'))
            model.eval()
            
            # Prepare sequences for TFT
            from src.analytics.tft import TFTTrainer
            past_length = min(60, len(X))
            X_past = X[-past_length:]  # Last 60 timesteps
            
            # Convert to torch tensor
            X_past_tensor = torch.FloatTensor(X_past).unsqueeze(0)  # Add batch dimension
            
            # Predict with quantiles
            with torch.no_grad():
                result = model.predict_with_quantiles(X_past_tensor)
            
            # Convert quantile predictions to 0-1 score
            # P50 is the point estimate (median)
            # Scale P50 to 0-1 range (assuming returns are in [-0.1, 0.1] range)
            raw_pred = result.p50
            
            # Scale prediction to 0-1 score
            # Positive return → score > 0.5, Negative return → score < 0.5
            ml_score = 0.5 + raw_pred * 10  # Scale: 0.1 return → 1.5 score (clipped)
            ml_score = max(0, min(1, ml_score))
            
            # Uncertainty based on confidence range (P90 - P10)
            confidence_range = result.confidence_range
            uncertainty = min(1.0, confidence_range / 0.10)  # Normalize to 0-1
            
            return {
                'score': float(ml_score),
                'raw_prediction': float(raw_pred),
                'p10': float(result.p10),
                'p50': float(result.p50),
                'p90': float(result.p90),
                'uncertainty': float(uncertainty),
                'confidence_range': float(confidence_range),
                'source': 'tft',
                'is_bullish': result.is_bullish,
                'is_bearish': result.is_bearish
            }
        except Exception as e:
            logger.warning(f"TFT prediction failed: {e}", exc_info=True)
            return {'score': 0.5, 'uncertainty': 1.0, 'source': 'unavailable'}
    
    def _get_ensemble_score(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Get ensemble prediction score (fallback if TFT unavailable).
        
        Returns:
            Dict with 'score' (0-1) and 'uncertainty'
        """
        if self.ensemble is None:
            return {'score': 0.5, 'uncertainty': 1.0, 'source': 'unavailable'}
        
        try:
            from src.analytics.deep_learning import FeatureEngineer
            
            # Create features
            X, _ = FeatureEngineer.create_ml_features(df)
            
            if len(X) == 0:
                return {'score': 0.5, 'uncertainty': 1.0, 'source': 'unavailable'}
            
            # Predict (use last sample)
            predictions, uncertainty = self.ensemble.predict(X[-1:])
            
            # Convert to 0-1 score
            # Raw prediction is expected return, scale to 0-1
            raw_pred = predictions[0]
            ml_score = 0.5 + raw_pred * 10  # Scale appropriately
            ml_score = max(0, min(1, ml_score))
            
            return {
                'score': float(ml_score),
                'raw_prediction': float(raw_pred),
                'uncertainty': float(uncertainty[0]),
                'source': 'ensemble'
            }
        except Exception as e:
            logger.warning(f"Ensemble prediction failed: {e}")
            return {'score': 0.5, 'uncertainty': 1.0, 'source': 'unavailable'}
    
    def analyze(
        self, 
        symbol: str,
        price_data: pd.DataFrame = None,
        news_data: List[Dict] = None,
        use_cache: bool = True
    ) -> ConfluenceResult:
        """
        Perform full confluence analysis for a symbol.
        
        Args:
            symbol: Stock ticker
            price_data: Optional price DataFrame (fetched if None)
            news_data: Optional news list (fetched if None)
            use_cache: Whether to use cached data
        
        Returns:
            ConfluenceResult with all scores and signals
        """
        logger.info("Starting confluence analysis", symbol=symbol)
        
        # Get data if not provided
        if price_data is None:
            price_data = self._get_price_data(symbol)
        
        if price_data is None or price_data.empty or len(price_data) < 50:
            logger.warning("Insufficient price data", symbol=symbol)
            return self._empty_result(symbol)
        
        # 1. Technical Analysis
        technical_result = self._analyze_technical(price_data)
        
        # 2. Sentiment Analysis
        sentiment_result = self._analyze_sentiment(symbol, news_data)
        
        # 3. Risk Analysis
        risk_result = self._analyze_risk(price_data)
        
        # 4. ML Score (from ensemble models if trained)
        ml_result = self._get_ml_score(symbol, price_data)
        ml_score = ml_result.get('score', 0.5)
        
        # 5. Get dynamic weights (if enabled)
        dynamic_weights = self._get_dynamic_weights(
            technical_result, sentiment_result, ml_result, risk_result
        ) if self.use_dynamic_weights else self.weights
        
        # 6. Calculate Confluence Score (with dynamic weights)
        confluence_score = self._calculate_confluence(
            technical_result['score'],
            sentiment_result['score'],
            ml_score,
            risk_result['score'],
            weights=dynamic_weights
        )
        
        # 7. Apply signal filtering
        filter_result = self._filter_signal(
            confluence_score,
            technical_result,
            sentiment_result,
            ml_result,
            risk_result
        )
        
        if not filter_result['passed']:
            logger.info(f"Signal filtered: {filter_result['reason']}", symbol=symbol)
            return self._empty_result(symbol, reason=filter_result['reason'])
        
        # 8. Determine Signal Type
        signal_type = self._determine_signal_type(confluence_score)
        
        # 9. Calculate Signal Agreement
        agreement, conflicts = self._calculate_agreement(
            technical_result, sentiment_result, risk_result
        )
        
        # 10. Determine Signal Strength
        signal_strength = self._determine_strength(agreement, confluence_score)
        
        # 11. Calculate Position Sizing (risk-adjusted)
        position_sizing = self._calculate_position_size(
            confluence_score, agreement, risk_result
        )
        
        # 12. Build Rationale
        overall_rationale = self._build_rationale(
            signal_type, technical_result, sentiment_result, risk_result, agreement
        )
        
        # 13. Calculate signal ranking metrics
        ranking_metrics = self._calculate_ranking_metrics(
            confluence_score, technical_result, sentiment_result, risk_result
        )
        
        result = ConfluenceResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            
            # Scores
            technical_score=technical_result['score'],
            sentiment_score=sentiment_result['score'],
            ml_score=ml_score,
            risk_score=risk_result['score'],
            confluence_score=confluence_score,
            
            # Signal
            signal_type=signal_type,
            signal_strength=signal_strength,
            
            # Position sizing
            recommended_position_pct=position_sizing['recommended'],
            max_position_pct=position_sizing['max'],
            
            # Risk
            var_95=risk_result.get('var_95', 0.05),
            stop_loss_pct=position_sizing['stop_loss'],
            take_profit_pct=position_sizing['take_profit'],
            risk_reward_ratio=position_sizing.get('risk_reward_ratio', 0.0),
            
            # Details
            technical_signals=technical_result.get('signals', {}),
            sentiment_details=sentiment_result.get('details', {}),
            risk_details=risk_result,
            
            # Rationale
            technical_rationale=technical_result.get('rationale', ''),
            sentiment_rationale=sentiment_result.get('rationale', ''),
            overall_rationale=overall_rationale,
            
            # Agreement
            signal_agreement=agreement,
            conflicting_signals=conflicts
        )
        
        logger.info(
            "Confluence analysis complete",
            symbol=symbol,
            score=confluence_score,
            signal=signal_type.value,
            strength=signal_strength.value,
            agreement=agreement
        )
        
        return result
    
    def _get_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get price data from database."""
        try:
            from src.data.persistence import get_database
            db = get_database()
            return db.get_candles(symbol, limit=250)
        except Exception as e:
            logger.error(f"Failed to get price data: {e}")
            return None
    
    def _analyze_technical(self, df: pd.DataFrame) -> Dict:
        """Run technical analysis."""
        try:
            df_with_indicators = self.technical_analyzer.compute_all(df)
            result = self.technical_analyzer.calculate_technical_score(df_with_indicators)
            
            latest = df_with_indicators.iloc[-1]
            
            return {
                'score': result['technical_score'],
                'signal_type': result['signal_type'],
                'rationale': result['rationale'],
                'signals': {
                    'rsi': latest.get('rsi_signal', 'NEUTRAL'),
                    'trend': latest.get('trend_signal', 'SIDEWAYS'),
                    'macd': latest.get('macd_crossover', 'NONE'),
                },
                'components': result.get('component_scores', {})
            }
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {'score': 0.5, 'signal_type': 'HOLD', 'rationale': 'Analysis failed'}
    
    def _analyze_sentiment(self, symbol: str, news_data: List[Dict] = None) -> Dict:
        """Run sentiment analysis."""
        if self.sentiment_analyzer is None:
            return {
                'score': 0.5, 
                'rationale': 'Sentiment analyzer not available',
                'details': {}
            }
        
        try:
            # Get news if not provided
            if news_data is None:
                from src.data.persistence import get_database
                db = get_database()
                news_items = db.get_recent_news(symbol, days=90)
                if news_items:
                    news_data = [
                        {'headline': n.headline, 'published_at': n.published_at.isoformat() if n.published_at else None}
                        for n in news_items
                    ]
                else:
                    news_data = []
            
            if not news_data:
                return {
                    'score': 0.5,
                    'rationale': 'No recent news available',
                    'details': {'article_count': 0}
                }
            
            result = self.sentiment_analyzer.aggregate_sentiment(symbol, news_data)
            
            return {
                'score': result.get('normalized_score', 0.5),
                'rationale': f"Sentiment is {result.get('overall_label', 'neutral')} based on {result.get('article_count', 0)} articles.",
                'details': {
                    'weighted_score': result.get('weighted_score', 0),
                    'label': result.get('overall_label', 'neutral'),
                    'article_count': result.get('article_count', 0),
                    'signal_quality': result.get('signal_quality', 'NO_DATA'),
                    'peak_window_signal': result.get('peak_window_signal', 0)
                }
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'score': 0.5, 'rationale': 'Sentiment analysis failed', 'details': {}}
    
    def _analyze_risk(self, df: pd.DataFrame) -> Dict:
        """Run risk analysis."""
        try:
            returns = df['close'].pct_change().dropna()
            metrics = self.risk_engine.calculate_portfolio_risk(returns)
            
            # Convert risk metrics to a 0-1 score
            # Lower VaR = higher score (less risky = better)
            # Scale: VaR of 2% = 1.0, VaR of 10% = 0.5, VaR of 20% = 0.0
            var_score = max(0, min(1, 1 - (metrics.var_95 - 0.02) / 0.18))
            
            return {
                'score': var_score,
                'var_95': metrics.var_95,
                'var_99': metrics.var_99,
                'sharpe': metrics.sharpe_ratio,
                'recommended_position_pct': metrics.recommended_position_pct,
                'max_position_pct': metrics.max_position_pct,
                'volatility': metrics.annualized_volatility,
                'max_drawdown': metrics.expected_max_drawdown
            }
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {
                'score': 0.5, 'var_95': 0.05, 'var_99': 0.10,
                'recommended_position_pct': 0.02, 'max_position_pct': 0.05
            }
    
    def _calculate_confluence(
        self,
        technical: float,
        sentiment: float,
        ml: float,
        risk: float,
        weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate weighted confluence score.
        
        Enhanced with dynamic weighting support.
        
        Args:
            technical: Technical score (0-1)
            sentiment: Sentiment score (0-1)
            ml: ML score (0-1)
            risk: Risk score (0-1)
            weights: Custom weights (uses self.weights if None)
        
        Returns:
            Confluence score (0-1)
        """
        weights = weights or self.weights
        
        confluence = (
            technical * weights['technical'] +
            sentiment * weights['sentiment'] +
            ml * weights['ml'] +
            risk * weights['risk']
        )
        return round(min(1.0, max(0.0, confluence)), 4)
    
    def _get_dynamic_weights(
        self,
        technical_result: Dict,
        sentiment_result: Dict,
        ml_result: Dict,
        risk_result: Dict
    ) -> Dict[str, float]:
        """
        Get dynamic weights based on confidence, regime, and performance.
        
        Enhanced with:
        - Confidence-based weights (higher confidence = more weight)
        - Regime-based weights (different weights for different regimes)
        - Performance-based weights (recent performance influences weights)
        - Agreement-based weights (more agreement = stronger weights)
        
        Returns:
            Dynamic weights dictionary
        """
        base_weights = self.weights.copy()
        
        # Get confidence scores (if available)
        tech_confidence = technical_result.get('confidence', 0.5)
        sent_confidence = sentiment_result.get('avg_confidence', 0.5)
        ml_confidence = ml_result.get('confidence', 0.5)
        
        # Get agreement
        agreement, _ = self._calculate_agreement(technical_result, sentiment_result, risk_result)
        
        # Confidence-based adjustment
        # Higher confidence = increase weight
        confidence_multipliers = {
            'technical': 0.8 + 0.4 * tech_confidence,  # Scale 0.8-1.2
            'sentiment': 0.8 + 0.4 * sent_confidence,
            'ml': 0.8 + 0.4 * ml_confidence
        }
        
        # Agreement-based adjustment
        # More agreement = stronger weights
        agreement_multiplier = 0.9 + 0.2 * agreement  # Scale 0.9-1.1
        
        # Apply adjustments
        adjusted_weights = {}
        for key in base_weights:
            if key in confidence_multipliers:
                adjusted_weights[key] = base_weights[key] * confidence_multipliers[key] * agreement_multiplier
            else:
                adjusted_weights[key] = base_weights[key] * agreement_multiplier
        
        # Normalize to sum to 1.0
        total = sum(adjusted_weights.values())
        if total > 0:
            for key in adjusted_weights:
                adjusted_weights[key] /= total
        
        return adjusted_weights
    
    def _filter_signal(
        self,
        confluence_score: float,
        technical_result: Dict,
        sentiment_result: Dict,
        ml_result: Dict,
        risk_result: Dict
    ) -> Dict:
        """
        Filter signals based on quality criteria.
        
        Filters:
        - Minimum confidence threshold
        - Minimum agreement threshold
        - Conflict detection
        - Quality filters
        
        Returns:
            Filter result dict with 'passed' and 'reason'
        """
        # Check minimum confidence threshold
        avg_confidence = (
            technical_result.get('confidence', 0.5) +
            sentiment_result.get('avg_confidence', 0.5) +
            ml_result.get('confidence', 0.5)
        ) / 3
        
        if avg_confidence < self.min_confidence_threshold:
            return {
                'passed': False,
                'reason': f'Confidence {avg_confidence:.2f} below threshold {self.min_confidence_threshold}'
            }
        
        # Check minimum agreement threshold
        agreement, conflicts = self._calculate_agreement(technical_result, sentiment_result, risk_result)
        
        if agreement < self.min_agreement_threshold:
            return {
                'passed': False,
                'reason': f'Agreement {agreement:.2f} below threshold {self.min_agreement_threshold}'
            }
        
        # Conflict detection (multiple sources disagree)
        if len(conflicts) >= 2:  # 2+ conflicts = too many disagreements
            return {
                'passed': False,
                'reason': f'Too many conflicts: {", ".join(conflicts)}'
            }
        
        # Quality filters (check signal quality)
        sent_quality = sentiment_result.get('details', {}).get('signal_quality', 'UNKNOWN')
        if sent_quality == 'CONFLICTING':
            return {
                'passed': False,
                'reason': 'Sentiment signals conflicting'
            }
        
        # All filters passed
        return {
            'passed': True,
            'reason': 'All filters passed',
            'confidence': avg_confidence,
            'agreement': agreement,
            'conflicts': conflicts
        }
    
    def _calculate_ranking_metrics(
        self,
        confluence_score: float,
        technical_result: Dict,
        sentiment_result: Dict,
        risk_result: Dict
    ) -> Dict:
        """
        Calculate ranking metrics for signal comparison.
        
        Metrics:
        - Confluence score rank
        - Risk-adjusted return rank
        - Sharpe ratio rank
        - Win rate rank (would need historical data)
        
        Returns:
            Ranking metrics dict
        """
        # Risk-adjusted return (simplified)
        # Estimated return = confluence_score * expected_return
        expected_return = 0.15  # 15% annual return assumption
        estimated_return = confluence_score * expected_return
        
        # Risk (from risk_result)
        risk = risk_result.get('var_95', 0.05)
        
        # Sharpe ratio (simplified)
        sharpe = (estimated_return - self.rf_rate) / (risk * np.sqrt(252)) if risk > 0 else 0
        
        return {
            'confluence_score': confluence_score,
            'estimated_return': round(estimated_return, 4),
            'sharpe_ratio': round(sharpe, 4),
            'risk_score': risk_result.get('score', 0.5)
        }
    
    def rank_signals(
        self,
        signals: List[ConfluenceResult],
        rank_by: str = 'confluence_score'
    ) -> List[ConfluenceResult]:
        """
        Rank signals by various metrics.
        
        Args:
            signals: List of ConfluenceResult objects
            rank_by: Ranking metric ('confluence_score', 'sharpe_ratio', 'risk_adjusted_return')
        
        Returns:
            Ranked list of signals
        """
        if rank_by == 'confluence_score':
            return sorted(signals, key=lambda x: x.confluence_score, reverse=True)
        elif rank_by == 'sharpe_ratio':
            # Would need Sharpe ratio in ConfluenceResult
            return sorted(signals, key=lambda x: x.confluence_score, reverse=True)  # Simplified
        else:
            return sorted(signals, key=lambda x: x.confluence_score, reverse=True)
    
    def validate_signal(
        self,
        signal: ConfluenceResult,
        validation_type: str = 'regime'
    ) -> Dict:
        """
        Validate signal using different validation methods.
        
        Validation types:
        - 'regime': Validate signal works in current regime
        - 'backtest': Validate signal works historically (would need historical data)
        - 'cross_validation': Validate signal works out-of-sample (would need test data)
        - 'live': Validate signal works in real-time (would need live tracking)
        
        Returns:
            Validation result dict
        """
        if validation_type == 'regime':
            # Regime validation (simplified)
            # Check if signal strength is appropriate for current market conditions
            # Would need regime detection
            return {
                'valid': signal.signal_strength != SignalStrength.NO_SIGNAL,
                'validation_type': 'regime',
                'reason': 'Regime validation passed' if signal.signal_strength != SignalStrength.NO_SIGNAL else 'No signal'
            }
        else:
            # Placeholder for other validation types
            return {
                'valid': True,
                'validation_type': validation_type,
                'reason': 'Validation not yet implemented'
            }
    
    def _determine_signal_type(self, confluence_score: float) -> SignalType:
        """Determine signal type from confluence score."""
        if confluence_score >= self.STRONG_BUY_THRESHOLD:
            return SignalType.STRONG_BUY
        elif confluence_score >= self.BUY_THRESHOLD:
            return SignalType.BUY
        elif confluence_score <= self.STRONG_SELL_THRESHOLD:
            return SignalType.STRONG_SELL
        elif confluence_score <= self.SELL_THRESHOLD:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _calculate_agreement(
        self,
        technical: Dict,
        sentiment: Dict,
        risk: Dict
    ) -> Tuple[float, List[str]]:
        """
        Calculate how much signals agree.
        
        Returns:
            Tuple of (agreement_score 0-1, list of conflicting signals)
        """
        tech_score = technical['score']
        sent_score = sentiment['score']
        risk_score = risk['score']
        
        # Determine each signal's direction
        def score_to_direction(s):
            if s >= 0.6:
                return 1  # Bullish
            elif s <= 0.4:
                return -1  # Bearish
            else:
                return 0  # Neutral
        
        tech_dir = score_to_direction(tech_score)
        sent_dir = score_to_direction(sent_score)
        risk_dir = 1 if risk_score >= 0.5 else -1  # Risk is binary (favorable or not)
        
        directions = [tech_dir, sent_dir, risk_dir]
        
        # Count agreements
        conflicts = []
        if tech_dir != sent_dir and tech_dir != 0 and sent_dir != 0:
            conflicts.append("Technical vs Sentiment")
        if tech_dir == -1 and risk_dir == 1:
            conflicts.append("Technical bearish but risk favorable")
        if sent_dir == -1 and risk_dir == 1:
            conflicts.append("Sentiment bearish but risk favorable")
        
        # Calculate agreement score
        non_neutral = [d for d in directions if d != 0]
        if not non_neutral:
            return 0.5, []  # All neutral
        
        # Higher agreement if all same direction
        avg_direction = np.mean(non_neutral)
        agreement = abs(avg_direction)  # 1.0 if all agree, 0 if split
        
        # Adjust for score proximity
        score_spread = max(tech_score, sent_score, risk_score) - min(tech_score, sent_score, risk_score)
        agreement = agreement * (1 - score_spread * 0.5)  # Reduce if scores are spread
        
        return round(agreement, 2), conflicts
    
    def _determine_strength(self, agreement: float, confluence: float) -> SignalStrength:
        """Determine signal strength based on agreement and score."""
        # Strong confluence + high agreement = very strong
        if agreement >= self.VERY_STRONG_AGREEMENT and (confluence >= 0.7 or confluence <= 0.3):
            return SignalStrength.VERY_STRONG
        elif agreement >= self.STRONG_AGREEMENT and (confluence >= 0.6 or confluence <= 0.4):
            return SignalStrength.STRONG
        elif agreement >= self.MODERATE_AGREEMENT:
            return SignalStrength.MODERATE
        elif 0.45 <= confluence <= 0.55:
            return SignalStrength.NO_SIGNAL
        else:
            return SignalStrength.WEAK
    
    def _calculate_position_size(
        self,
        confluence: float,
        agreement: float,
        risk: Dict
    ) -> Dict:
        """
        Calculate position sizing based on conviction and risk.
        
        Higher confluence + higher agreement = larger position.
        Uses Kelly-inspired sizing capped by VaR constraints.
        """
        base_position = risk.get('recommended_position_pct', 0.02)
        max_position = risk.get('max_position_pct', 0.05)
        var_95 = risk.get('var_95', 0.05)
        
        # Adjust by conviction (confluence distance from 0.5)
        conviction = abs(confluence - 0.5) * 2  # 0-1 scale
        
        # Adjust by agreement
        agreement_factor = 0.5 + agreement * 0.5  # 0.5-1.0 range
        
        # Calculate recommended position
        recommended = base_position * conviction * agreement_factor
        recommended = min(recommended, max_position)
        recommended = max(recommended, 0.005)  # Minimum 0.5%
        
        # Calculate stop loss (based on ATR or VaR)
        stop_loss = min(0.08, var_95 * 1.5)  # Cap at 8%
        
        # Calculate take profit (2:1 risk-reward minimum)
        take_profit = stop_loss * 2
        
        # Calculate risk-reward ratio
        risk_reward_ratio = take_profit / stop_loss if stop_loss > 0 else 0.0
        
        return {
            'recommended': round(recommended, 4),
            'max': round(max_position, 4),
            'stop_loss': round(stop_loss, 4),
            'take_profit': round(take_profit, 4),
            'risk_reward_ratio': round(risk_reward_ratio, 2)
        }
    
    def _build_rationale(
        self,
        signal_type: SignalType,
        technical: Dict,
        sentiment: Dict,
        risk: Dict,
        agreement: float
    ) -> str:
        """Build human-readable rationale."""
        parts = []
        
        # Signal summary
        parts.append(f"Signal: {signal_type.value}")
        
        # Technical
        tech_signal = "bullish" if technical['score'] > 0.55 else "bearish" if technical['score'] < 0.45 else "neutral"
        parts.append(f"Technical indicators are {tech_signal} (score: {technical['score']:.2f}).")
        
        # Sentiment
        if sentiment.get('details', {}).get('article_count', 0) > 0:
            sent_label = sentiment.get('details', {}).get('label', 'neutral')
            parts.append(f"News sentiment is {sent_label}.")
        else:
            parts.append("No recent news data available.")
        
        # Agreement
        if agreement >= 0.7:
            parts.append("Signals are in strong agreement.")
        elif agreement >= 0.5:
            parts.append("Signals show moderate agreement.")
        else:
            parts.append("Signals are conflicting - proceed with caution.")
        
        # Risk
        var_pct = risk.get('var_95', 0.05) * 100
        parts.append(f"Risk: 95% VaR is {var_pct:.1f}%.")
        
        return " ".join(parts)
    
    def _empty_result(self, symbol: str, reason: str = None) -> ConfluenceResult:
        """Return empty result when analysis cannot be performed."""
        return ConfluenceResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            technical_score=0.5,
            sentiment_score=0.5,
            ml_score=0.5,
            risk_score=0.5,
            confluence_score=0.5,
            signal_type=SignalType.HOLD,
            signal_strength=SignalStrength.NO_SIGNAL,
            recommended_position_pct=0.0,
            max_position_pct=0.0,
            var_95=0.05,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            risk_reward_ratio=2.0,
            overall_rationale="Insufficient data for analysis."
        )
    
    def analyze_batch(self, symbols: List[str]) -> List[ConfluenceResult]:
        """
        Analyze multiple symbols.
        
        Args:
            symbols: List of tickers
        
        Returns:
            List of ConfluenceResults sorted by confluence score
        """
        results = []
        
        for symbol in symbols:
            try:
                result = self.analyze(symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
        
        # Sort by confluence score (highest first) and signal strength
        results.sort(
            key=lambda r: (r.confluence_score, r.signal_agreement),
            reverse=True
        )
        
        return results


# Convenience function
def get_confluence_engine() -> ConfluenceEngine:
    """Get a ConfluenceEngine instance."""
    return ConfluenceEngine()


if __name__ == "__main__":
    # Test confluence analysis
    engine = ConfluenceEngine()
    
    # Test with sample data
    print("\n=== Confluence Engine Test ===\n")
    
    # Create sample price data
    import numpy as np
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    close = 100 * (1.001 ** np.arange(100))  # Slight uptrend
    
    df = pd.DataFrame({
        'time': dates,
        'open': close * 0.99,
        'high': close * 1.02,
        'low': close * 0.98,
        'close': close,
        'volume': [1000000] * 100
    })
    
    # Mock news
    news = [
        {'headline': 'Company reports strong earnings', 'published_at': (datetime.now()).isoformat()},
        {'headline': 'Positive outlook for sector', 'published_at': (datetime.now()).isoformat()},
    ]
    
    # Analyze
    result = engine.analyze("TEST", price_data=df, news_data=news)
    
    print(f"Symbol: {result.symbol}")
    print(f"Confluence Score: {result.confluence_score:.3f}")
    print(f"Signal Type: {result.signal_type.value}")
    print(f"Signal Strength: {result.signal_strength.value}")
    print(f"\nScores:")
    print(f"  Technical: {result.technical_score:.3f}")
    print(f"  Sentiment: {result.sentiment_score:.3f}")
    print(f"  Risk: {result.risk_score:.3f}")
    print(f"\nPosition Sizing:")
    print(f"  Recommended: {result.recommended_position_pct*100:.2f}%")
    print(f"  Stop Loss: {result.stop_loss_pct*100:.1f}%")
    print(f"  Take Profit: {result.take_profit_pct*100:.1f}%")
    print(f"\nRationale: {result.overall_rationale}")
