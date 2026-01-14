#!/usr/bin/env python3
"""
TFT Training Script.

Train Temporal Fusion Transformer (TFT) on historical price data.

Usage:
    python scripts/train_tft.py --symbol AAPL --epochs 50
    python scripts/train_tft.py --symbols AAPL MSFT GOOGL --epochs 100
    python scripts/train_tft.py --all-symbols --epochs 50 --batch-size 64

Example:
    # Train on single symbol
    python scripts/train_tft.py --symbol AAPL --epochs 100 --hidden-size 64
    
    # Train on multiple symbols (ensemble)
    python scripts/train_tft.py --symbols AAPL MSFT GOOGL NVDA --epochs 150
    
    # Train on all active symbols
    python scripts/train_tft.py --all-symbols --epochs 50 --batch-size 32
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from src.logging_config import get_logger
from src.data.persistence import get_database
from src.analytics.tft import (
    TemporalFusionTransformer,
    TFTTrainer,
    create_tft_features
)
from src.analytics.deep_learning import FeatureEngineer

logger = get_logger(__name__)


def prepare_training_data(
    symbol: str,
    min_days: int = 252,  # At least 1 year of data
    past_length: int = 60,
    future_length: int = 1
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Prepare training data for TFT from database.
    
    Args:
        symbol: Stock ticker
        min_days: Minimum days of data required
        past_length: Lookback window
        future_length: Future horizon
    
    Returns:
        (X_past, X_future, y_target) or None if insufficient data
    """
    logger.info(f"Preparing training data for {symbol}")
    
    try:
        db = get_database()
        
        # Get historical price data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=min_days * 2)  # Get extra for safety
        
        candles_df = db.get_candles(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=min_days * 2
        )
        
        if candles_df.empty or len(candles_df) < min_days:
            logger.warning(f"Insufficient data for {symbol}: {len(candles_df)} days (need {min_days})")
            return None
        
        logger.info(f"Retrieved {len(candles_df)} days of data for {symbol}")
        
        # Prepare features
        X, scaler = FeatureEngineer.create_ml_features(candles_df)
        y = FeatureEngineer.create_target(candles_df, horizon=1, target_type='returns')
        
        # Ensure same length
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        if len(X) < past_length + future_length + 10:
            logger.warning(f"Insufficient features after processing for {symbol}: {len(X)} samples")
            return None
        
        logger.info(f"Created {X.shape[1]} features from {len(X)} samples")
        
        # Prepare TFT sequences
        X_past, X_future, y_target = TFTTrainer.prepare_tft_sequences(
            X, y,
            past_length=past_length,
            future_length=future_length,
            step_size=1
        )
        
        if len(X_past) < 50:  # Need at least 50 sequences
            logger.warning(f"Insufficient sequences for {symbol}: {len(X_past)} (need 50)")
            return None
        
        logger.info(f"Prepared {len(X_past)} TFT sequences for {symbol}")
        
        return X_past, X_future, y_target
        
    except Exception as e:
        logger.error(f"Failed to prepare training data for {symbol}: {e}", exc_info=True)
        return None


def train_tft_model(
    symbol: str,
    epochs: int = 100,
    hidden_size: int = 64,
    num_heads: int = 4,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    patience: int = 10,
    device: str = 'cpu',
    save_path: Path = None
) -> Optional[TemporalFusionTransformer]:
    """
    Train TFT model for a symbol.
    
    Args:
        symbol: Stock ticker
        epochs: Maximum training epochs
        hidden_size: TFT hidden size
        num_heads: Number of attention heads
        learning_rate: Initial learning rate
        batch_size: Training batch size
        patience: Early stopping patience
        device: 'cpu' or 'cuda'
        save_path: Path to save trained model
    
    Returns:
        Trained TFT model or None if training failed
    """
    logger.info(f"Starting TFT training for {symbol}")
    
    # Prepare data
    training_data = prepare_training_data(symbol)
    
    if training_data is None:
        logger.error(f"Failed to prepare training data for {symbol}")
        return None
    
    X_past, X_future, y_target = training_data
    
    # Create model
    num_features = X_past.shape[2]
    
    model = TemporalFusionTransformer(
        num_features=num_features,
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=0.1
    )
    
    logger.info(
        f"Created TFT model: features={num_features}, "
        f"hidden_size={hidden_size}, num_heads={num_heads}"
    )
    
    # Create data loaders
    train_loader, val_loader = TFTTrainer.create_dataloaders(
        X_past, X_future, y_target,
        train_ratio=0.8,
        batch_size=batch_size
    )
    
    logger.info(
        f"Created data loaders: train={len(train_loader.dataset)} samples, "
        f"val={len(val_loader.dataset)} samples"
    )
    
    # Set save path
    if save_path is None:
        models_dir = Path("models/tft")
        models_dir.mkdir(parents=True, exist_ok=True)
        save_path = models_dir / f"tft_{symbol.lower()}.pth"
    
    # Train model
    try:
        history = TFTTrainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
            device=device,
            save_path=save_path
        )
        
        logger.info(
            f"Training complete for {symbol}: "
            f"epochs={history['epochs_trained']}, "
            f"best_val_loss={history['best_val_loss']:.6f}"
        )
        
        # Log training history
        logger.info(
            f"Training history: "
            f"final_train_loss={history['train_loss'][-1]:.6f}, "
            f"final_val_loss={history['val_loss'][-1]:.6f}"
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}", exc_info=True)
        return None


def train_multiple_symbols(
    symbols: List[str],
    epochs: int = 100,
    hidden_size: int = 64,
    batch_size: int = 32,
    device: str = 'cpu'
) -> Dict[str, Optional[TemporalFusionTransformer]]:
    """
    Train TFT models for multiple symbols.
    
    Args:
        symbols: List of stock tickers
        epochs: Maximum training epochs
        hidden_size: TFT hidden size
        batch_size: Training batch size
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary mapping symbol to trained model (or None if failed)
    """
    logger.info(f"Training TFT models for {len(symbols)} symbols")
    
    results = {}
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"Training {i}/{len(symbols)}: {symbol}")
        
        model = train_tft_model(
            symbol=symbol,
            epochs=epochs,
            hidden_size=hidden_size,
            batch_size=batch_size,
            device=device
        )
        
        results[symbol] = model
        
        if model is None:
            logger.warning(f"Failed to train model for {symbol}")
        else:
            logger.info(f"Successfully trained model for {symbol}")
    
    # Summary
    successful = sum(1 for m in results.values() if m is not None)
    logger.info(
        f"Training complete: {successful}/{len(symbols)} models trained successfully"
    )
    
    return results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Temporal Fusion Transformer (TFT)')
    
    # Data selection
    parser.add_argument('--symbol', type=str, help='Single symbol to train on')
    parser.add_argument('--symbols', nargs='+', help='Multiple symbols to train on')
    parser.add_argument('--all-symbols', action='store_true', help='Train on all active symbols')
    
    # Model hyperparameters
    parser.add_argument('--hidden-size', type=int, default=64, help='TFT hidden size (default: 64)')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Maximum epochs (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')
    
    # Training configuration
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device (default: cpu)')
    parser.add_argument('--past-length', type=int, default=60, help='Lookback window (default: 60)')
    parser.add_argument('--future-length', type=int, default=1, help='Future horizon (default: 1)')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='models/tft', help='Directory to save models (default: models/tft)')
    
    args = parser.parse_args()
    
    # Determine symbols to train
    symbols = []
    
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = args.symbols
    elif args.all_symbols:
        db = get_database()
        assets = db.get_active_assets()
        symbols = [a.symbol for a in assets]
        
        if not symbols:
            logger.warning("No active assets found, using default symbols")
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    else:
        parser.error("Must specify --symbol, --symbols, or --all-symbols")
    
    logger.info(f"Training TFT models for {len(symbols)} symbols")
    logger.info(f"Hyperparameters: hidden_size={args.hidden_size}, num_heads={args.num_heads}, "
                f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    if len(symbols) == 1:
        # Single symbol
        symbol = symbols[0]
        save_path = save_dir / f"tft_{symbol.lower()}.pth"
        
        model = train_tft_model(
            symbol=symbol,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device,
            save_path=save_path
        )
        
        if model is not None:
            logger.info(f"✅ Successfully trained TFT model for {symbol}")
            logger.info(f"Model saved to: {save_path}")
        else:
            logger.error(f"❌ Failed to train TFT model for {symbol}")
            sys.exit(1)
    else:
        # Multiple symbols
        results = train_multiple_symbols(
            symbols=symbols,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            device=args.device
        )
        
        successful = [s for s, m in results.items() if m is not None]
        failed = [s for s, m in results.items() if m is None]
        
        logger.info(f"✅ Successfully trained: {len(successful)} models")
        if successful:
            logger.info(f"Successful symbols: {', '.join(successful)}")
        
        if failed:
            logger.warning(f"❌ Failed to train: {len(failed)} models")
            logger.warning(f"Failed symbols: {', '.join(failed)}")
        
        if not successful:
            logger.error("All training failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
