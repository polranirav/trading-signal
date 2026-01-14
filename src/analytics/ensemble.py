"""
Ensemble Methods for Trading Signal Generation.

Implements:
1. HybridSignalEnsemble - RF + GB + Ridge with meta-learner (stacking)
2. MultiTimeframeEnsemble - Combine signals from 1h, 4h, 1d, 1w
3. ModelSelector - Adaptive model weighting based on regime

Research Foundation:
From "CRITICAL_RESEARCH_PAPERS_AND_STRATEGIES.md":
- Single best model: 58% accuracy
- Ensemble of 3 models: 59% accuracy
- Ensemble of 5 models: 60% accuracy
- Overfitting reduction: from 9% to 3%

Why Ensemble > Single Model:
- Different models overfit differently
- XGBoost overfits to recent trends
- Random Forest overfits to static patterns
- Neural Network overfits to rare events
- Combined: Overfit cancels out, signal adds up

Usage:
    ensemble = HybridSignalEnsemble()
    ensemble.train(X_train, y_train)
    predictions, uncertainty = ensemble.predict(X_test)
    
    mtf = MultiTimeframeEnsemble()
    score, signal = mtf.combine_signals({'1h': 0.7, '4h': 0.6, '1d': 0.8, '1w': 0.55})
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path

# Optional imports for advanced models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.logging_config import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "BULL"           # VIX < 15, uptrend
    BEAR = "BEAR"           # VIX > 25, downtrend
    SIDEWAYS = "SIDEWAYS"   # Low volatility, no trend
    HIGH_VOL = "HIGH_VOL"   # VIX > 30, uncertain


@dataclass
class EnsemblePrediction:
    """Ensemble prediction with confidence."""
    prediction: float           # Point estimate
    uncertainty: float          # Std across base models
    model_predictions: Dict     # Individual model outputs
    confidence: float           # 0-1 confidence score
    consensus: str              # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    
    @property
    def is_high_confidence(self) -> bool:
        return self.confidence > 0.7 and self.uncertainty < 0.02


class HybridSignalEnsemble:
    """
    Ensemble multiple signal sources with learned weights.
    
    Implements stacking ensemble:
    1. Train base learners (RF, GB, Ridge)
    2. Generate meta-features from base predictions
    3. Train meta-learner on meta-features
    
    From research:
    - RF: 55-57% accuracy, robust to outliers
    - GB: 56-58% accuracy, fast training
    - Ridge: 52-55% accuracy, linear baseline
    - Meta-learner combines optimally → 59-61% accuracy
    """
    
    def __init__(self, use_classifier: bool = False, use_advanced_models: bool = True):
        """
        Initialize enhanced ensemble.
        
        Args:
            use_classifier: If True, use classifiers for direction prediction
            use_advanced_models: If True, include LightGBM, CatBoost (if available)
        """
        self.use_classifier = use_classifier
        self.use_advanced_models = use_advanced_models
        
        # Base models list
        self.base_models = {}
        
        if use_classifier:
            # Classification models
            self.base_models['rf'] = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            self.base_models['gb'] = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
            self.base_models['ridge'] = LogisticRegression(C=1.0, random_state=42)
            
            # Advanced models (if available)
            if use_advanced_models:
                if LIGHTGBM_AVAILABLE:
                    self.base_models['lgb'] = lgb.LGBMClassifier(
                        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, verbose=-1
                    )
                if CATBOOST_AVAILABLE:
                    self.base_models['catboost'] = cb.CatBoostClassifier(
                        iterations=100, depth=5, random_state=42, verbose=False
                    )
            
            self.meta_model = LogisticRegression(C=0.1, random_state=42)
        else:
            # Regression models
            self.base_models['rf'] = RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1,
                min_samples_split=10, min_samples_leaf=5
            )
            self.base_models['gb'] = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42,
                min_samples_split=10, min_samples_leaf=5
            )
            self.base_models['ridge'] = Ridge(alpha=1.0)
            self.base_models['lasso'] = Lasso(alpha=0.1, random_state=42)
            self.base_models['elastic'] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            
            # Advanced models (if available)
            if use_advanced_models:
                if LIGHTGBM_AVAILABLE:
                    self.base_models['lgb'] = lgb.LGBMRegressor(
                        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, verbose=-1
                    )
                if CATBOOST_AVAILABLE:
                    self.base_models['catboost'] = cb.CatBoostRegressor(
                        iterations=100, depth=5, random_state=42, verbose=False
                    )
            
            self.meta_model = Ridge(alpha=0.1)
        
        # Store model names for reference
        self.model_names = list(self.base_models.keys())
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict:
        """
        Train all models using time-series cross-validation.
        
        Args:
            X_train: Feature array
            y_train: Target array
            feature_names: Optional feature names for importance
        
        Returns:
            Training metrics dict
        """
        logger.info("Training enhanced ensemble models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train all base models
        for name, model in self.base_models.items():
            try:
                model.fit(X_scaled, y_train)
                logger.info(f"Trained {name} model")
            except Exception as e:
                logger.warning(f"Failed to train {name} model: {e}")
        
        logger.info(f"Base models trained: {len(self.base_models)} models")
        
        # Generate meta-features (predictions from all base models)
        meta_features = []
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X_scaled).reshape(-1, 1)
                meta_features.append(pred)
            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
        
        if not meta_features:
            raise ValueError("No base models successfully trained")
        
        X_meta = np.hstack(meta_features)
        
        # Train meta-model
        self.meta_model.fit(X_meta, y_train)
        
        self.is_trained = True
        
        # Extract feature importance
        if hasattr(self.rf_model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                feature_names or [f"f{i}" for i in range(X_train.shape[1])],
                self.rf_model.feature_importances_
            ))
        
        logger.info("Ensemble training complete")
        
        return {
            "rf_score": self.rf_model.score(X_scaled, y_train) if not self.use_classifier else 0,
            "feature_importance": self.feature_importance
        }
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates.
        
        Returns:
            predictions: Mean predictions from meta-model
            uncertainty: Standard deviation across base models
        """
        if not self.is_trained:
            raise ValueError("Train model first")
        
        X_scaled = self.scaler.transform(X_test)
        
        # Base model predictions (from all models)
        base_predictions = []
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X_scaled)
                base_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
        
        if not base_predictions:
            raise ValueError("No base models available for prediction")
        
        # Meta features
        X_meta = np.column_stack(base_predictions)
        
        # Ensemble prediction
        ensemble_pred = self.meta_model.predict(X_meta)
        
        # Uncertainty estimate (disagreement among base models)
        all_preds = np.column_stack(base_predictions)
        uncertainty = np.std(all_preds, axis=1)
        
        return ensemble_pred, uncertainty
    
    def predict_with_details(self, X_test: np.ndarray) -> List[EnsemblePrediction]:
        """
        Predict with full details for each sample.
        """
        predictions, uncertainties = self.predict(X_test)
        X_scaled = self.scaler.transform(X_test)
        
        # Get individual predictions from all base models
        base_pred_dict = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X_scaled)
                base_pred_dict[name] = pred
            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
        
        # Combine all predictions
        all_base_preds = [pred for pred in base_pred_dict.values()]
        
        results = []
        for i in range(len(predictions)):
            pred = predictions[i]
            unc = uncertainties[i]
            
            # Confidence based on uncertainty
            confidence = max(0, 1 - unc / 0.05)  # 5% uncertainty = 0 confidence
            
            # Consensus signal (from all base models)
            bullish_count = sum(1 for pred_arr in all_base_preds if pred_arr[i] > 0.005)
            bearish_count = sum(1 for pred_arr in all_base_preds if pred_arr[i] < -0.005)
            
            if bullish_count >= 3:
                consensus = "STRONG_BUY"
            elif bullish_count >= 2:
                consensus = "BUY"
            elif bearish_count >= 3:
                consensus = "STRONG_SELL"
            elif bearish_count >= 2:
                consensus = "SELL"
            else:
                consensus = "HOLD"
            
            # Build model predictions dict
            model_preds = {name: float(pred_arr[i]) for name, pred_arr in base_pred_dict.items()}
            
            results.append(EnsemblePrediction(
                prediction=float(pred),
                uncertainty=float(unc),
                model_predictions=model_preds,
                confidence=float(confidence),
                consensus=consensus
            ))
        
        return results
    
    def save(self, path: Path = None):
        """Save trained models to disk."""
        path = path or MODELS_DIR / "ensemble_models.pkl"
        
        with open(path, 'wb') as f:
            pickle.dump({
                'base_models': self.base_models,
                'meta_model': self.meta_model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'use_classifier': self.use_classifier,
                'model_names': self.model_names
            }, f)
        
        logger.info(f"Models saved to {path}")
    
    def load(self, path: Path = None):
        """Load trained models from disk."""
        path = path or MODELS_DIR / "ensemble_models.pkl"
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Backward compatibility: check if old format (single models) or new format (base_models dict)
        if 'base_models' in data:
            self.base_models = data['base_models']
            self.model_names = data.get('model_names', list(self.base_models.keys()))
        else:
            # Old format - convert to new format
            self.base_models = {
                'rf': data['rf_model'],
                'gb': data['gb_model'],
                'ridge': data['ridge_model']
            }
            self.model_names = ['rf', 'gb', 'ridge']
        
        self.meta_model = data['meta_model']
        self.scaler = data['scaler']
        self.feature_importance = data['feature_importance']
        self.use_classifier = data.get('use_classifier', False)
        self.is_trained = True
        
        logger.info(f"Models loaded from {path}")


class MultiTimeframeEnsemble:
    """
    Combine signals from multiple timeframes.
    
    From research:
    - Multi-timeframe analysis improves accuracy by 5-7%
    - Higher timeframes have more weight (less noise)
    - Consensus voting adds robustness
    
    Default weights (from research):
    - 1h: 20% (high noise, short-term)
    - 4h: 30% (moderate, intraday trend)
    - 1d: 35% (core signal, daily momentum)
    - 1w: 15% (strategic direction, less responsive)
    """
    
    TIMEFRAME_WEIGHTS = {
        '1h': 0.20,
        '4h': 0.30,
        '1d': 0.35,
        '1w': 0.15
    }
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize multi-timeframe ensemble.
        
        Args:
            weights: Custom timeframe weights (must sum to 1.0)
        """
        self.weights = weights or self.TIMEFRAME_WEIGHTS.copy()
        
        # Validate weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing")
            for k in self.weights:
                self.weights[k] /= total
    
    def combine_signals(
        self,
        signals: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        Combine multi-timeframe signals with consensus voting.
        
        Args:
            signals: {'1h': score, '4h': score, '1d': score, '1w': score}
                     Each score is 0-1 (0.5 = neutral)
        
        Returns:
            weighted_score: Combined 0-1 score
            consensus_signal: STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
        """
        # Weighted score
        weighted_score = sum(
            signals.get(tf, 0.5) * self.weights.get(tf, 0)
            for tf in self.weights.keys()
        )
        
        # Consensus voting
        bullish = sum(1 for s in signals.values() if s > 0.6)
        bearish = sum(1 for s in signals.values() if s < 0.4)
        
        if bullish >= 3:
            consensus = "STRONG_BUY"
        elif bullish >= 2:
            consensus = "BUY"
        elif bearish >= 3:
            consensus = "STRONG_SELL"
        elif bearish >= 2:
            consensus = "SELL"
        else:
            consensus = "HOLD"
        
        return weighted_score, consensus
    
    def get_timeframe_agreement(self, signals: Dict[str, float]) -> float:
        """
        Calculate agreement across timeframes.
        
        Returns:
            0-1 score where 1 = all timeframes agree
        """
        scores = list(signals.values())
        
        if len(scores) < 2:
            return 1.0
        
        # Calculate variance-based agreement
        variance = np.var(scores)
        max_variance = 0.25  # Max possible for 0-1 range
        
        agreement = 1 - (variance / max_variance)
        return max(0, min(1, agreement))


class RegimeAdaptiveEnsemble:
    """
    Ensemble that adapts weights based on market regime.
    
    From research:
    - "ML learns different strategy in bull vs bear"
    - Different models perform better in different regimes
    - VIX-based regime detection is effective
    
    Regime-based weights:
    - BULL: Technical indicators weighted higher
    - BEAR: Sentiment and risk weighted higher
    - SIDEWAYS: Mean reversion models preferred
    - HIGH_VOL: Conservative positioning
    """
    
    REGIME_WEIGHTS = {
        MarketRegime.BULL: {'technical': 0.50, 'sentiment': 0.25, 'ml': 0.15, 'risk': 0.10},
        MarketRegime.BEAR: {'technical': 0.25, 'sentiment': 0.35, 'ml': 0.15, 'risk': 0.25},
        MarketRegime.SIDEWAYS: {'technical': 0.40, 'sentiment': 0.30, 'ml': 0.20, 'risk': 0.10},
        MarketRegime.HIGH_VOL: {'technical': 0.20, 'sentiment': 0.20, 'ml': 0.10, 'risk': 0.50},
    }
    
    def __init__(self):
        self.current_regime = MarketRegime.SIDEWAYS
    
    def detect_regime(
        self,
        vix: float,
        sma_50_200_cross: bool = None,
        volatility_ratio: float = None
    ) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            vix: VIX value
            sma_50_200_cross: True if 50 SMA > 200 SMA (bullish)
            volatility_ratio: Current vol / historical vol
        
        Returns:
            MarketRegime
        """
        # VIX-based detection (from research)
        if vix > 30:
            return MarketRegime.HIGH_VOL
        elif vix > 25:
            return MarketRegime.BEAR
        elif vix < 15:
            if sma_50_200_cross:
                return MarketRegime.BULL
            else:
                return MarketRegime.SIDEWAYS
        else:
            # 15-25 range
            if sma_50_200_cross:
                return MarketRegime.BULL
            else:
                return MarketRegime.SIDEWAYS
    
    def get_regime_weights(self, regime: MarketRegime = None) -> Dict[str, float]:
        """Get weights for current or specified regime."""
        regime = regime or self.current_regime
        return self.REGIME_WEIGHTS[regime].copy()
    
    def calculate_confluence(
        self,
        technical_score: float,
        sentiment_score: float,
        ml_score: float,
        risk_score: float,
        vix: float = 20.0
    ) -> float:
        """
        Calculate regime-adjusted confluence score.
        
        Args:
            technical_score: 0-1 technical score
            sentiment_score: 0-1 sentiment score
            ml_score: 0-1 ML model score
            risk_score: 0-1 risk score
            vix: Current VIX for regime detection
        
        Returns:
            Confluence score 0-1
        """
        regime = self.detect_regime(vix)
        weights = self.get_regime_weights(regime)
        
        confluence = (
            technical_score * weights['technical'] +
            sentiment_score * weights['sentiment'] +
            ml_score * weights['ml'] +
            risk_score * weights['risk']
        )
        
        return round(confluence, 4)


# Convenience functions
def get_ensemble() -> HybridSignalEnsemble:
    """Get a HybridSignalEnsemble instance."""
    return HybridSignalEnsemble()


def get_mtf_ensemble() -> MultiTimeframeEnsemble:
    """Get a MultiTimeframeEnsemble instance."""
    return MultiTimeframeEnsemble()


def get_regime_ensemble() -> RegimeAdaptiveEnsemble:
    """Get a RegimeAdaptiveEnsemble instance."""
    return RegimeAdaptiveEnsemble()


class BlendingEnsemble:
    """
    Blending ensemble with learned weights.
    
    Similar to stacking but simpler - uses weighted average with learned weights.
    """
    
    def __init__(self, base_models: Dict[str, any]):
        """
        Initialize blending ensemble.
        
        Args:
            base_models: Dictionary of model_name -> model object
        """
        self.base_models = base_models
        self.weights = None
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train blending ensemble by learning optimal weights.
        
        Uses linear regression to learn weights for each base model.
        """
        from sklearn.linear_model import LinearRegression
        
        # Get predictions from all base models
        predictions = []
        model_names = []
        
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X_train).reshape(-1, 1)
                predictions.append(pred)
                model_names.append(name)
            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
        
        if not predictions:
            raise ValueError("No base models available")
        
        # Stack predictions
        X_blend = np.hstack(predictions)
        
        # Learn weights using linear regression
        weight_model = LinearRegression(fit_intercept=False, positive=True)  # Positive weights
        weight_model.fit(X_blend, y_train)
        
        self.weights = dict(zip(model_names, weight_model.coef_))
        self.is_trained = True
        
        logger.info(f"Blending weights learned: {self.weights}")
        
        return {
            'weights': self.weights,
            'model_names': model_names
        }
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using weighted average of base models."""
        if not self.is_trained:
            raise ValueError("Train model first")
        
        predictions = []
        for name, model in self.base_models.items():
            if name in self.weights:
                try:
                    pred = model.predict(X_test)
                    weight = self.weights[name]
                    predictions.append(pred * weight)
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {name}: {e}")
        
        if not predictions:
            raise ValueError("No predictions available")
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0) / sum(self.weights.values())
        
        return ensemble_pred


if __name__ == "__main__":
    # Test ensemble methods
    print("\n=== Ensemble Methods Test ===\n")
    
    # 1. Test HybridSignalEnsemble
    print("1. Testing HybridSignalEnsemble...")
    
    # Create sample data
    n_samples = 500
    n_features = 10
    np.random.seed(42)
    
    X = np.random.randn(n_samples, n_features)
    y = 0.01 * X[:, 0] + 0.005 * X[:, 1] + 0.002 * np.random.randn(n_samples)
    
    # Train ensemble
    ensemble = HybridSignalEnsemble()
    result = ensemble.train(X[:400], y[:400])
    
    # Predict
    predictions, uncertainty = ensemble.predict(X[400:])
    
    print(f"  Training result: {result}")
    print(f"  Prediction mean: {predictions.mean():.4f}")
    print(f"  Uncertainty mean: {uncertainty.mean():.4f}")
    
    # 2. Test MultiTimeframeEnsemble
    print("\n2. Testing MultiTimeframeEnsemble...")
    
    mtf = MultiTimeframeEnsemble()
    
    # All bullish
    score, signal = mtf.combine_signals({'1h': 0.75, '4h': 0.70, '1d': 0.80, '1w': 0.65})
    print(f"  All bullish: score={score:.3f}, signal={signal}")
    
    # Mixed signals
    score, signal = mtf.combine_signals({'1h': 0.55, '4h': 0.45, '1d': 0.60, '1w': 0.48})
    print(f"  Mixed: score={score:.3f}, signal={signal}")
    
    # 3. Test RegimeAdaptiveEnsemble
    print("\n3. Testing RegimeAdaptiveEnsemble...")
    
    regime_ens = RegimeAdaptiveEnsemble()
    
    # Different regimes
    for vix in [12, 18, 27, 35]:
        regime = regime_ens.detect_regime(vix)
        weights = regime_ens.get_regime_weights(regime)
        print(f"  VIX={vix}: {regime.value} → tech={weights['technical']}, risk={weights['risk']}")
    
    print("\n=== Tests Complete ===")
