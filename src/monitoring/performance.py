"""
Performance Monitoring and Adaptation Module.

Provides comprehensive performance tracking:
1. Signal-level metrics (win rate, returns, Sharpe by signal type)
2. Model-level metrics (accuracy, drift detection, performance by regime)
3. Feature-level metrics (importance, correlation, stability)
4. Adaptation triggers (retrain, adjust weights, remove features)

Usage:
    monitor = PerformanceMonitor()
    metrics = monitor.track_signal_result(signal_id, actual_return)
    adaptation = monitor.check_adaptation_triggers()
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict

from src.logging_config import get_logger

logger = get_logger(__name__)


class AdaptationTrigger(Enum):
    """Adaptation trigger types."""
    RETRAIN_MODEL = "RETRAIN_MODEL"
    ADJUST_WEIGHTS = "ADJUST_WEIGHTS"
    REMOVE_FEATURE = "REMOVE_FEATURE"
    ADD_FEATURE = "ADD_FEATURE"
    NO_ACTION = "NO_ACTION"


@dataclass
class SignalMetrics:
    """Metrics for a single signal."""
    signal_id: str
    symbol: str
    signal_type: str
    confluence_score: float
    timestamp: datetime
    actual_return: Optional[float] = None
    realized_return: Optional[float] = None
    win: Optional[bool] = None
    holding_period: Optional[int] = None  # Days


@dataclass
class ModelMetrics:
    """Metrics for a model."""
    model_name: str
    accuracy: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    timestamp: datetime
    regime: str = "UNKNOWN"
    drift_score: float = 0.0  # Model drift detection score


@dataclass
class FeatureMetrics:
    """Metrics for a feature."""
    feature_name: str
    importance: float
    correlation: float
    stability: float
    performance_by_regime: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring and adaptation system.
    
    Tracks:
    - Signal-level metrics (win rate, returns, Sharpe by signal type)
    - Model-level metrics (accuracy, drift detection, performance by regime)
    - Feature-level metrics (importance, correlation, stability)
    - Adaptation triggers (retrain, adjust weights, remove features)
    """
    
    def __init__(self, lookback_periods: int = 252):
        """
        Initialize performance monitor.
        
        Args:
            lookback_periods: Number of periods to look back for metrics
        """
        self.lookback_periods = lookback_periods
        self.signal_history: List[SignalMetrics] = []
        self.model_metrics_history: List[ModelMetrics] = []
        self.feature_metrics_history: List[FeatureMetrics] = []
        
        # Adaptation thresholds
        self.performance_degradation_threshold = 0.05  # 5% drop
        self.drift_threshold = 0.15  # 15% drift
        self.feature_importance_threshold = 0.001  # Minimum importance
    
    def track_signal_result(
        self,
        signal_id: str,
        symbol: str,
        signal_type: str,
        confluence_score: float,
        actual_return: float,
        holding_period: int = None
    ) -> Dict:
        """
        Track signal result.
        
        Args:
            signal_id: Unique signal identifier
            symbol: Stock ticker
            signal_type: Signal type (STRONG_BUY, BUY, etc.)
            confluence_score: Confluence score (0-1)
            actual_return: Actual return realized
            holding_period: Holding period in days
        
        Returns:
            Signal metrics dict
        """
        win = actual_return > 0
        
        signal_metric = SignalMetrics(
            signal_id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            confluence_score=confluence_score,
            timestamp=datetime.utcnow(),
            actual_return=actual_return,
            realized_return=actual_return,
            win=win,
            holding_period=holding_period
        )
        
        self.signal_history.append(signal_metric)
        
        # Keep only recent history
        if len(self.signal_history) > self.lookback_periods * 2:
            self.signal_history = self.signal_history[-self.lookback_periods:]
        
        logger.info(f"Tracked signal result: {signal_id}, return: {actual_return:.2%}")
        
        return signal_metric.__dict__
    
    def get_signal_level_metrics(self, signal_type: str = None) -> Dict:
        """
        Calculate signal-level metrics.
        
        Metrics:
        - Win rate by signal type
        - Average return by signal type
        - Sharpe ratio by signal type
        - Maximum drawdown by signal type
        
        Args:
            signal_type: Filter by signal type (None = all types)
        
        Returns:
            Metrics dict
        """
        if not self.signal_history:
            return {
                'win_rate': 0.5,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'signal_count': 0
            }
        
        # Filter by signal type if specified
        signals = [s for s in self.signal_history if s.actual_return is not None]
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        
        if not signals:
            return {
                'win_rate': 0.5,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'signal_count': 0
            }
        
        # Calculate metrics
        returns = [s.actual_return for s in signals]
        wins = [s.win for s in signals if s.win is not None]
        
        win_rate = sum(wins) / len(wins) if wins else 0.5
        avg_return = np.mean(returns) if returns else 0.0
        std_return = np.std(returns) if len(returns) > 1 else 0.01
        
        # Sharpe ratio (annualized)
        sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        return {
            'win_rate': round(win_rate, 4),
            'avg_return': round(avg_return, 4),
            'sharpe_ratio': round(sharpe, 4),
            'max_drawdown': round(max_drawdown, 4),
            'signal_count': len(signals)
        }
    
    def track_model_metrics(
        self,
        model_name: str,
        accuracy: float,
        sharpe_ratio: float,
        win_rate: float,
        max_drawdown: float,
        regime: str = "UNKNOWN",
        drift_score: float = 0.0
    ) -> Dict:
        """
        Track model-level metrics.
        
        Args:
            model_name: Model name
            accuracy: Model accuracy
            sharpe_ratio: Sharpe ratio
            win_rate: Win rate
            max_drawdown: Maximum drawdown
            regime: Market regime
            drift_score: Model drift score
        
        Returns:
            Model metrics dict
        """
        model_metric = ModelMetrics(
            model_name=model_name,
            accuracy=accuracy,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            timestamp=datetime.utcnow(),
            regime=regime,
            drift_score=drift_score
        )
        
        self.model_metrics_history.append(model_metric)
        
        # Keep only recent history
        if len(self.model_metrics_history) > self.lookback_periods:
            self.model_metrics_history = self.model_metrics_history[-self.lookback_periods:]
        
        return model_metric.__dict__
    
    def detect_model_drift(self, model_name: str) -> Dict:
        """
        Detect model drift.
        
        Compares recent performance to historical performance.
        
        Args:
            model_name: Model name
        
        Returns:
            Drift detection result
        """
        model_history = [m for m in self.model_metrics_history if m.model_name == model_name]
        
        if len(model_history) < 30:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'reason': 'Insufficient data'
            }
        
        # Compare recent vs. historical performance
        recent_periods = 30
        recent = model_history[-recent_periods:]
        historical = model_history[:-recent_periods]
        
        if not historical:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'reason': 'No historical data'
            }
        
        recent_accuracy = np.mean([m.accuracy for m in recent])
        historical_accuracy = np.mean([m.accuracy for m in historical])
        
        accuracy_drop = historical_accuracy - recent_accuracy
        drift_score = accuracy_drop / historical_accuracy if historical_accuracy > 0 else 0.0
        
        drift_detected = drift_score > self.drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'drift_score': round(drift_score, 4),
            'recent_accuracy': round(recent_accuracy, 4),
            'historical_accuracy': round(historical_accuracy, 4),
            'reason': f'Accuracy drop: {accuracy_drop:.2%}' if drift_detected else 'No drift detected'
        }
    
    def track_feature_metrics(
        self,
        feature_name: str,
        importance: float,
        correlation: float,
        stability: float,
        performance_by_regime: Dict[str, float] = None
    ) -> Dict:
        """
        Track feature-level metrics.
        
        Args:
            feature_name: Feature name
            importance: Feature importance score
            correlation: Feature correlation with target
            stability: Feature stability score (0-1)
            performance_by_regime: Performance by regime
        
        Returns:
            Feature metrics dict
        """
        feature_metric = FeatureMetrics(
            feature_name=feature_name,
            importance=importance,
            correlation=correlation,
            stability=stability,
            performance_by_regime=performance_by_regime or {},
            timestamp=datetime.utcnow()
        )
        
        self.feature_metrics_history.append(feature_metric)
        
        # Keep only recent history
        if len(self.feature_metrics_history) > self.lookback_periods:
            self.feature_metrics_history = self.feature_metrics_history[-self.lookback_periods:]
        
        return feature_metric.__dict__
    
    def check_adaptation_triggers(self) -> Dict:
        """
        Check if adaptation is needed.
        
        Triggers:
        - Retrain if performance degrades
        - Adjust weights if regime changes
        - Remove features if they stop working
        - Add features if they improve performance
        
        Returns:
            Adaptation trigger dict
        """
        triggers = []
        actions = []
        
        # Check model performance degradation
        if self.model_metrics_history:
            recent_metrics = self.model_metrics_history[-30:]
            if len(recent_metrics) >= 30:
                recent_accuracy = np.mean([m.accuracy for m in recent_metrics])
                historical_accuracy = np.mean([m.accuracy for m in self.model_metrics_history[:-30]])
                
                if historical_accuracy - recent_accuracy > self.performance_degradation_threshold:
                    triggers.append(AdaptationTrigger.RETRAIN_MODEL)
                    actions.append(f'Model accuracy dropped {historical_accuracy - recent_accuracy:.2%}')
        
        # Check model drift
        unique_models = set(m.model_name for m in self.model_metrics_history)
        for model_name in unique_models:
            drift_result = self.detect_model_drift(model_name)
            if drift_result['drift_detected']:
                triggers.append(AdaptationTrigger.RETRAIN_MODEL)
                actions.append(f'Model {model_name} drift detected: {drift_result["drift_score"]:.2%}')
        
        # Check feature importance
        if self.feature_metrics_history:
            recent_features = [f for f in self.feature_metrics_history 
                             if (datetime.utcnow() - f.timestamp).days < 30]
            
            low_importance_features = [
                f for f in recent_features
                if f.importance < self.feature_importance_threshold
            ]
            
            if low_importance_features:
                triggers.append(AdaptationTrigger.REMOVE_FEATURE)
                actions.append(f'Features with low importance: {[f.feature_name for f in low_importance_features[:5]]}')
        
        # Default: no action
        if not triggers:
            triggers.append(AdaptationTrigger.NO_ACTION)
            actions.append('No adaptation needed')
        
        return {
            'triggers': [t.value for t in triggers],
            'primary_trigger': triggers[0].value if triggers else AdaptationTrigger.NO_ACTION.value,
            'actions': actions,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_comprehensive_metrics(self) -> Dict:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Complete metrics dict
        """
        # Signal-level metrics
        signal_metrics = self.get_signal_level_metrics()
        
        # Model-level metrics
        model_metrics = {}
        if self.model_metrics_history:
            recent_models = self.model_metrics_history[-1]
            model_metrics = {
                'accuracy': recent_models.accuracy,
                'sharpe_ratio': recent_models.sharpe_ratio,
                'win_rate': recent_models.win_rate,
                'drift_score': recent_models.drift_score
            }
        
        # Feature-level metrics
        feature_metrics = {}
        if self.feature_metrics_history:
            recent_features = [f for f in self.feature_metrics_history 
                             if (datetime.utcnow() - f.timestamp).days < 30]
            if recent_features:
                avg_importance = np.mean([f.importance for f in recent_features])
                avg_stability = np.mean([f.stability for f in recent_features])
                feature_metrics = {
                    'avg_importance': round(avg_importance, 4),
                    'avg_stability': round(avg_stability, 4),
                    'num_features': len(recent_features)
                }
        
        # Adaptation triggers
        adaptation = self.check_adaptation_triggers()
        
        return {
            'signal_metrics': signal_metrics,
            'model_metrics': model_metrics,
            'feature_metrics': feature_metrics,
            'adaptation_triggers': adaptation,
            'timestamp': datetime.utcnow().isoformat()
        }


def get_performance_monitor() -> PerformanceMonitor:
    """Get a PerformanceMonitor instance."""
    return PerformanceMonitor()
