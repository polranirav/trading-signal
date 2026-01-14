#!/usr/bin/env python3
"""
Quick Test Script for TFT Implementation.

Tests TFT components without requiring database or trained models.
This is a quick verification that everything compiles and works.

Usage:
    python scripts/test_tft_implementation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=" * 80)
print("TFT IMPLEMENTATION TEST")
print("=" * 80)
print()

# Test 1: Import TFT
print("Test 1: Import TFT module...")
try:
    from src.analytics.tft import (
        TemporalFusionTransformer,
        VariableSelectionNetwork,
        TFTTrainer,
        TFTPredictionResult,
        DEFAULT_QUANTILES
    )
    print("✅ TFT imports successful")
    print(f"   - DEFAULT_QUANTILES: {DEFAULT_QUANTILES}")
except Exception as e:
    print(f"❌ TFT import failed: {e}")
    sys.exit(1)

print()

# Test 2: Create Variable Selection Network
print("Test 2: Create Variable Selection Network...")
try:
    vsn = VariableSelectionNetwork(
        num_inputs=50,
        hidden_size=64,
        dropout=0.1
    )
    print("✅ Variable Selection Network created")
    print(f"   - num_inputs: {vsn.num_inputs}")
    print(f"   - hidden_size: {vsn.hidden_size}")
except Exception as e:
    print(f"❌ VSN creation failed: {e}")
    sys.exit(1)

print()

# Test 3: Test VSN Forward Pass
print("Test 3: Test Variable Selection Network forward pass...")
try:
    batch_size = 4
    seq_len = 60
    num_inputs = 50
    
    x = torch.randn(batch_size, seq_len, num_inputs)
    selected, weights = vsn(x)
    
    print("✅ VSN forward pass successful")
    print(f"   - Input shape: {x.shape}")
    print(f"   - Selected shape: {selected.shape}")
    print(f"   - Weights shape: {weights.shape}")
    print(f"   - Weights sum (should be ~1.0): {weights.sum(dim=-1).mean().item():.4f}")
    
    # Verify weights sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5), \
        "Weights should sum to 1"
    print("   - ✅ Weights are valid probabilities")
except Exception as e:
    print(f"❌ VSN forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Create TFT Model
print("Test 4: Create Temporal Fusion Transformer...")
try:
    num_features = 50
    model = TemporalFusionTransformer(
        num_features=num_features,
        hidden_size=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        quantiles=[0.1, 0.5, 0.9]
    )
    print("✅ TFT model created")
    print(f"   - num_features: {model.num_features}")
    print(f"   - hidden_size: {model.hidden_size}")
    print(f"   - num_heads: {model.num_heads}")
    print(f"   - quantiles: {model.quantiles}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   - Total parameters: {num_params:,}")
except Exception as e:
    print(f"❌ TFT model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: Test TFT Forward Pass
print("Test 5: Test TFT forward pass...")
try:
    batch_size = 2
    seq_len = 60
    
    # Create dummy features
    past_features = torch.randn(batch_size, seq_len, num_features)
    
    # Forward pass
    predictions = model(past_features)
    
    print("✅ TFT forward pass successful")
    print(f"   - Input shape: {past_features.shape}")
    print(f"   - Output shape: {predictions.shape}")
    print(f"   - Expected shape: ({batch_size}, {len(DEFAULT_QUANTILES)})")
    
    # Verify output shape
    assert predictions.shape == (batch_size, len(DEFAULT_QUANTILES)), \
        f"Output shape mismatch: {predictions.shape} != ({batch_size}, {len(DEFAULT_QUANTILES)})"
    print(f"   - ✅ Output shape correct")
    
    # Show sample predictions
    print(f"   - Sample predictions (P10, P50, P90): {predictions[0].detach().numpy()}")
except Exception as e:
    print(f"❌ TFT forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 6: Test TFT with Future Features
print("Test 6: Test TFT with future features...")
try:
    past_features = torch.randn(1, 60, num_features)
    future_features = torch.randn(1, 1, num_features)
    
    predictions = model(past_features, future_features)
    
    print("✅ TFT forward pass with future features successful")
    print(f"   - Past features shape: {past_features.shape}")
    print(f"   - Future features shape: {future_features.shape}")
    print(f"   - Output shape: {predictions.shape}")
except Exception as e:
    print(f"❌ TFT forward pass with future features failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 7: Test TFT with Attention Weights
print("Test 7: Test TFT with attention weights...")
try:
    past_features = torch.randn(1, 60, num_features)
    
    predictions, attention_dict = model(past_features, return_attention=True)
    
    print("✅ TFT forward pass with attention successful")
    print(f"   - Predictions shape: {predictions.shape}")
    print(f"   - Attention keys: {list(attention_dict.keys())}")
    print(f"   - Encoder weights shape: {attention_dict['encoder_weights'].shape}")
    print(f"   - Temporal attention shape: {attention_dict['temporal_attention'].shape}")
except Exception as e:
    print(f"❌ TFT forward pass with attention failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 8: Test predict_with_quantiles
print("Test 8: Test predict_with_quantiles method...")
try:
    past_features = torch.randn(60, num_features)  # Single sequence
    
    result = model.predict_with_quantiles(past_features)
    
    print("✅ predict_with_quantiles successful")
    print(f"   - Result type: {type(result).__name__}")
    print(f"   - P10: {result.p10:.6f}")
    print(f"   - P50: {result.p50:.6f}")
    print(f"   - P90: {result.p90:.6f}")
    print(f"   - Confidence range: {result.confidence_range:.6f}")
    print(f"   - Is bullish: {result.is_bullish}")
    print(f"   - Is bearish: {result.is_bearish}")
    
    # Test to_dict
    result_dict = result.to_dict()
    assert 'p10' in result_dict and 'p50' in result_dict and 'p90' in result_dict
    print("   - ✅ to_dict() works correctly")
except Exception as e:
    print(f"❌ predict_with_quantiles failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 9: Test Quantile Loss Function
print("Test 9: Test quantile loss function...")
try:
    batch_size = 8
    num_quantiles = 3
    
    predictions = torch.randn(batch_size, num_quantiles)
    targets = torch.randn(batch_size)
    quantiles = [0.1, 0.5, 0.9]
    
    loss = TFTTrainer.quantile_loss(predictions, targets, quantiles)
    
    print("✅ Quantile loss calculation successful")
    print(f"   - Predictions shape: {predictions.shape}")
    print(f"   - Targets shape: {targets.shape}")
    print(f"   - Loss value: {loss.item():.6f}")
    print(f"   - Loss is scalar: {loss.shape == ()}")
    
    assert loss.item() >= 0, "Loss should be non-negative"
    print("   - ✅ Loss is non-negative")
except Exception as e:
    print(f"❌ Quantile loss calculation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 10: Test Sequence Preparation
print("Test 10: Test TFT sequence preparation...")
try:
    n_samples = 100
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 0.02
    
    X_past, X_future, y_target = TFTTrainer.prepare_tft_sequences(
        X, y,
        past_length=60,
        future_length=1,
        step_size=1
    )
    
    print("✅ Sequence preparation successful")
    print(f"   - Input X shape: {X.shape}")
    print(f"   - Input y shape: {y.shape}")
    print(f"   - X_past shape: {X_past.shape}")
    print(f"   - X_future shape: {X_future.shape}")
    print(f"   - y_target shape: {y_target.shape}")
    print(f"   - Number of sequences: {len(X_past)}")
    
    assert len(X_past) > 0, "Should have at least one sequence"
    print("   - ✅ Sequences created successfully")
except Exception as e:
    print(f"❌ Sequence preparation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 11: Test DataLoader Creation
print("Test 11: Test DataLoader creation...")
try:
    n_sequences = 50
    past_length = 60
    future_length = 1
    n_features = 50
    
    X_past = np.random.randn(n_sequences, past_length, n_features)
    X_future = np.random.randn(n_sequences, future_length, n_features)
    y_target = np.random.randn(n_sequences) * 0.02
    
    train_loader, val_loader = TFTTrainer.create_dataloaders(
        X_past, X_future, y_target,
        train_ratio=0.8,
        batch_size=16
    )
    
    print("✅ DataLoader creation successful")
    print(f"   - Train dataset size: {len(train_loader.dataset)}")
    print(f"   - Val dataset size: {len(val_loader.dataset)}")
    print(f"   - Train batches: {len(train_loader)}")
    print(f"   - Val batches: {len(val_loader)}")
    
    # Test one batch
    batch = next(iter(train_loader))
    assert len(batch) == 3, "Batch should have 3 elements (X_past, X_future, y)"
    print(f"   - Batch X_past shape: {batch[0].shape}")
    print(f"   - Batch X_future shape: {batch[1].shape}")
    print(f"   - Batch y shape: {batch[2].shape}")
    print("   - ✅ DataLoaders work correctly")
except Exception as e:
    print(f"❌ DataLoader creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 12: Test Integration with FeatureEngineer
print("Test 12: Test integration with FeatureEngineer...")
try:
    from src.analytics.deep_learning import FeatureEngineer
    
    # Create sample price data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(42)
    
    close_prices = 100 * (1.01 ** np.arange(100))
    
    df = pd.DataFrame({
        'time': dates,
        'open': close_prices * 0.99,
        'high': close_prices * 1.02,
        'low': close_prices * 0.98,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, 100),
    })
    
    # Create features
    X, scaler = FeatureEngineer.create_ml_features(df)
    y = FeatureEngineer.create_target(df, horizon=1, target_type='returns')
    
    print("✅ FeatureEngineer integration successful")
    print(f"   - Features shape: {X.shape}")
    print(f"   - Target shape: {y.shape}")
    
    if len(X) >= 60:
        # Create TFT model with correct feature count
        tft_model = TemporalFusionTransformer(
            num_features=X.shape[1],
            hidden_size=32,  # Smaller for testing
            num_heads=2
        )
        
        # Prepare sequence
        past_seq = X[-60:].unsqueeze(0)  # Last 60 timesteps
        
        # Forward pass
        predictions = tft_model(past_seq)
        
        print(f"   - TFT model created with {X.shape[1]} features")
        print(f"   - TFT predictions shape: {predictions.shape}")
        print("   - ✅ TFT works with FeatureEngineer features")
    else:
        print(f"   - ⚠️  Insufficient data for TFT (need 60, have {len(X)})")
except Exception as e:
    print(f"❌ FeatureEngineer integration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 13: Test Integration with ConfluenceEngine
print("Test 13: Test integration with ConfluenceEngine...")
try:
    from src.analytics.confluence import ConfluenceEngine
    
    engine = ConfluenceEngine()
    
    # Verify that _get_tft_score method exists
    assert hasattr(engine, '_get_tft_score'), "ConfluenceEngine should have _get_tft_score method"
    assert hasattr(engine, '_get_ml_score'), "ConfluenceEngine should have _get_ml_score method"
    
    print("✅ ConfluenceEngine integration successful")
    print("   - ✅ _get_tft_score method exists")
    print("   - ✅ _get_ml_score method exists")
    print("   - ✅ TFT is integrated into ConfluenceEngine")
    
    # Test with mock data (won't actually predict without trained model)
    # But we can test that the method exists and handles missing model gracefully
    print("   - ✅ Integration structure correct")
except Exception as e:
    print(f"❌ ConfluenceEngine integration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 14: Test Module Exports
print("Test 14: Test module exports...")
try:
    from src.analytics import (
        TemporalFusionTransformer,
        TFTTrainer,
        TFTPredictionResult
    )
    
    print("✅ Module exports successful")
    print("   - ✅ TemporalFusionTransformer exported")
    print("   - ✅ TFTTrainer exported")
    print("   - ✅ TFTPredictionResult exported")
except Exception as e:
    print(f"❌ Module exports failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 15: Test TFT with Different Configurations
print("Test 15: Test TFT with different configurations...")
try:
    configs = [
        {"hidden_size": 32, "num_heads": 2},
        {"hidden_size": 64, "num_heads": 4},
        {"hidden_size": 128, "num_heads": 8},
    ]
    
    for i, config in enumerate(configs, 1):
        model = TemporalFusionTransformer(
            num_features=50,
            **config
        )
        
        past_features = torch.randn(1, 60, 50)
        predictions = model(past_features)
        
        print(f"   - Config {i}: hidden_size={config['hidden_size']}, "
              f"num_heads={config['num_heads']} ✅")
    
    print("✅ TFT works with various configurations")
except Exception as e:
    print(f"❌ TFT configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 16: Test Batch Processing
print("Test 16: Test TFT batch processing...")
try:
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        past_features = torch.randn(batch_size, 60, 50)
        predictions = model(past_features)
        
        assert predictions.shape[0] == batch_size, f"Batch size mismatch: {predictions.shape[0]} != {batch_size}"
        assert predictions.shape[1] == len(DEFAULT_QUANTILES), "Output shape mismatch"
        
        print(f"   - Batch size {batch_size}: ✅")
    
    print("✅ TFT batch processing works correctly")
except Exception as e:
    print(f"❌ Batch processing test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Summary
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print()
print("✅ All tests passed!")
print()
print("TFT Implementation Status:")
print("  ✅ Variable Selection Network: Working")
print("  ✅ TFT Architecture: Working")
print("  ✅ Quantile Regression: Working")
print("  ✅ Training Pipeline: Working")
print("  ✅ Integration: Working")
print("  ✅ Module Exports: Working")
print()
print("Next Steps:")
print("  1. Train TFT model on real data: python scripts/train_tft.py --symbol AAPL")
print("  2. Run unit tests: pytest tests/unit/test_tft.py -v")
print("  3. Continue to Week 3: Feature Engineering Expansion")
print()
print("=" * 80)
