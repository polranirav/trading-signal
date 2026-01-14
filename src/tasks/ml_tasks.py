"""
ML Model Celery Tasks.

Tasks for:
- train_models: Train deep learning and ensemble models
- predict_ml: Get ML predictions for symbols
- update_ml_scores: Update ML scores in confluence analysis
- retrain_ensemble: Periodic model retraining
"""

from celery import shared_task
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np

from src.tasks.celery_app import app
from src.logging_config import get_logger

logger = get_logger(__name__)


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=2,
    name='src.tasks.ml_tasks.train_ensemble'
)
def train_ensemble(self, symbols: List[str] = None) -> Dict:
    """
    Train ensemble models on historical data.
    
    Args:
        symbols: List of symbols to train on
    
    Returns:
        Training result dict
    """
    logger.info("Starting ensemble training", task_id=self.request.id)
    
    try:
        from src.analytics.ensemble import HybridSignalEnsemble
        from src.analytics.deep_learning import FeatureEngineer
        from src.data.persistence import get_database
        import pandas as pd
        
        db = get_database()
        
        if not symbols:
            assets = db.get_active_assets()
            symbols = [a.symbol for a in assets][:10] if assets else ['AAPL', 'MSFT', 'GOOGL']
        
        # Collect training data
        all_X = []
        all_y = []
        
        for symbol in symbols:
            df = db.get_candles(symbol, limit=500)
            if df.empty or len(df) < 100:
                continue
            
            try:
                X, _ = FeatureEngineer.create_ml_features(df)
                y = FeatureEngineer.create_target(df, horizon=5)
                
                # Remove NaN
                valid_idx = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
                all_X.append(X[valid_idx])
                all_y.append(y[valid_idx])
            except Exception as e:
                logger.warning(f"Feature creation failed for {symbol}: {e}")
        
        if not all_X:
            return {"status": "error", "message": "No training data available"}
        
        X_train = np.vstack(all_X)
        y_train = np.hstack(all_y)
        
        logger.info(f"Training on {len(X_train)} samples from {len(symbols)} symbols")
        
        # Train ensemble
        ensemble = HybridSignalEnsemble()
        result = ensemble.train(X_train, y_train)
        
        # Save model
        ensemble.save()
        
        return {
            "status": "success",
            "samples_trained": len(X_train),
            "symbols": len(symbols),
            "feature_importance": result.get('feature_importance', {})
        }
        
    except Exception as e:
        logger.error(f"Ensemble training failed: {e}")
        raise


@app.task(
    bind=True,
    name='src.tasks.ml_tasks.predict_ml'
)
def predict_ml(self, symbol: str) -> Dict:
    """
    Get ML prediction for a symbol.
    
    Args:
        symbol: Stock ticker
    
    Returns:
        Prediction dict with score and uncertainty
    """
    logger.info(f"Getting ML prediction for {symbol}", task_id=self.request.id)
    
    try:
        from src.analytics.ensemble import HybridSignalEnsemble
        from src.analytics.deep_learning import FeatureEngineer
        from src.data.persistence import get_database
        from src.data.cache import get_cache
        from pathlib import Path
        
        db = get_database()
        cache = get_cache()
        
        # Get price data
        df = db.get_candles(symbol, limit=100)
        
        if df.empty or len(df) < 50:
            return {"status": "error", "message": "Insufficient data"}
        
        # Create features
        X, _ = FeatureEngineer.create_ml_features(df)
        
        # Load ensemble
        ensemble = HybridSignalEnsemble()
        model_path = Path("models/ensemble_models.pkl")
        
        if model_path.exists():
            ensemble.load(model_path)
        else:
            return {"status": "error", "message": "Model not trained yet"}
        
        # Predict (use last sample)
        predictions, uncertainty = ensemble.predict(X[-1:])
        
        # Convert to 0-1 score
        # Prediction is expected return, normalize
        raw_pred = predictions[0]
        ml_score = 0.5 + raw_pred * 10  # Scale to 0-1
        ml_score = max(0, min(1, ml_score))
        
        result = {
            "status": "success",
            "symbol": symbol,
            "ml_score": float(ml_score),
            "raw_prediction": float(raw_pred),
            "uncertainty": float(uncertainty[0]),
            "confidence": float(1 - min(1, uncertainty[0] / 0.05))
        }
        
        # Cache result
        cache_key = f"ml_prediction:{symbol}"
        cache.client.setex(cache_key, 3600, str(result))
        
        return result
        
    except Exception as e:
        logger.error(f"ML prediction failed for {symbol}: {e}")
        return {"status": "error", "message": str(e)}


@app.task(
    bind=True,
    name='src.tasks.ml_tasks.predict_all_ml'
)
def predict_all_ml(self, symbols: List[str] = None) -> Dict:
    """
    Get ML predictions for all active symbols.
    
    Args:
        symbols: Optional list of symbols
    
    Returns:
        Summary dict
    """
    logger.info("Starting batch ML predictions", task_id=self.request.id)
    
    if not symbols:
        from src.data.persistence import get_database
        db = get_database()
        assets = db.get_active_assets()
        symbols = [a.symbol for a in assets] if assets else ['AAPL', 'MSFT', 'GOOGL']
    
    # Queue individual predictions
    results = []
    for symbol in symbols:
        task = predict_ml.delay(symbol)
        results.append({"symbol": symbol, "task_id": task.id})
    
    return {
        "status": "queued",
        "symbols_count": len(symbols),
        "tasks": results
    }


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=2,
    name='src.tasks.ml_tasks.train_tft'
)
def train_tft(self, symbol: str, epochs: int = 100, hidden_size: int = 64) -> Dict:
    """
    Train Temporal Fusion Transformer (TFT) for a symbol.
    
    Args:
        symbol: Stock ticker
        epochs: Maximum training epochs
        hidden_size: TFT hidden size
    
    Returns:
        Training result dict
    """
    logger.info("Starting TFT training", symbol=symbol, task_id=self.request.id)
    
    try:
        from src.analytics.tft import TemporalFusionTransformer, TFTTrainer
        from src.analytics.deep_learning import FeatureEngineer
        from src.data.persistence import get_database
        from pathlib import Path
        import torch
        
        db = get_database()
        
        # Get historical price data (need at least 252 days = 1 year)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=504)  # 2 years of data
        
        candles_df = db.get_candles(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=504
        )
        
        if candles_df.empty or len(candles_df) < 252:
            logger.warning(f"Insufficient data for TFT training: {symbol}", available=len(candles_df))
            return {"status": "insufficient_data", "symbol": symbol, "available_days": len(candles_df)}
        
        logger.info(f"Retrieved {len(candles_df)} days of data for {symbol}")
        
        # Prepare features
        X, scaler = FeatureEngineer.create_ml_features(candles_df)
        y = FeatureEngineer.create_target(candles_df, horizon=1, target_type='returns')
        
        # Ensure same length
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        if len(X) < 120:  # Need at least 120 samples for TFT (60 past + 60 for sequences)
            logger.warning(f"Insufficient features for TFT: {symbol}", available=len(X))
            return {"status": "insufficient_features", "symbol": symbol, "available_samples": len(X)}
        
        # Prepare TFT sequences
        X_past, X_future, y_target = TFTTrainer.prepare_tft_sequences(
            X, y,
            past_length=60,
            future_length=1
        )
        
        if len(X_past) < 50:  # Need at least 50 sequences
            logger.warning(f"Insufficient sequences for TFT: {symbol}", available=len(X_past))
            return {"status": "insufficient_sequences", "symbol": symbol, "available_sequences": len(X_past)}
        
        logger.info(f"Prepared {len(X_past)} TFT sequences for {symbol}")
        
        # Create model
        num_features = X_past.shape[2]
        model = TemporalFusionTransformer(
            num_features=num_features,
            hidden_size=hidden_size,
            num_heads=4,
            dropout=0.1
        )
        
        # Create data loaders
        train_loader, val_loader = TFTTrainer.create_dataloaders(
            X_past, X_future, y_target,
            train_ratio=0.8,
            batch_size=32
        )
        
        # Save path
        models_dir = Path("models/tft")
        models_dir.mkdir(parents=True, exist_ok=True)
        save_path = models_dir / f"tft_{symbol.lower()}.pth"
        
        # Train model
        history = TFTTrainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=1e-3,
            patience=10,
            device='cpu',  # Use CPU by default (can be changed to 'cuda' if available)
            save_path=save_path
        )
        
        logger.info(
            f"TFT training complete for {symbol}: "
            f"epochs={history['epochs_trained']}, "
            f"best_val_loss={history['best_val_loss']:.6f}"
        )
        
        return {
            "status": "success",
            "symbol": symbol,
            "epochs_trained": history['epochs_trained'],
            "best_val_loss": float(history['best_val_loss']),
            "final_train_loss": float(history['train_loss'][-1]),
            "final_val_loss": float(history['val_loss'][-1]),
            "sequences_trained": len(X_past),
            "model_path": str(save_path)
        }
        
    except Exception as e:
        logger.error(f"TFT training failed for {symbol}: {e}", exc_info=True)
        raise


@app.task(
    bind=True,
    name='src.tasks.ml_tasks.train_tft_all'
)
def train_tft_all(self, symbols: List[str] = None, epochs: int = 100) -> Dict:
    """
    Train TFT models for all active symbols (or provided symbols).
    
    Args:
        symbols: Optional list of symbols (uses all active if None)
        epochs: Maximum training epochs
    
    Returns:
        Summary dict with queued tasks
    """
    logger.info("Starting batch TFT training", task_id=self.request.id)
    
    if not symbols:
        from src.data.persistence import get_database
        db = get_database()
        assets = db.get_active_assets()
        symbols = [a.symbol for a in assets] if assets else ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Limit to top 10 symbols (TFT training is expensive)
    symbols = symbols[:10]
    
    # Queue individual training tasks
    results = []
    for symbol in symbols:
        task = train_tft.delay(symbol, epochs=epochs)
        results.append({"symbol": symbol, "task_id": task.id})
    
    return {
        "status": "queued",
        "symbols_count": len(symbols),
        "symbols": symbols,
        "tasks": results,
        "note": "TFT training is expensive. Tasks will run asynchronously."
    }
