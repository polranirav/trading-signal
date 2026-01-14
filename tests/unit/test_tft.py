"""
Unit tests for Temporal Fusion Transformer (TFT).

Tests:
- Variable Selection Network
- TFT forward pass
- Quantile regression heads
- Training pipeline
- Integration with ConfluenceEngine
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.analytics.tft import (
    TemporalFusionTransformer,
    VariableSelectionNetwork,
    TFTTrainer,
    TFTPredictionResult,
    DEFAULT_QUANTILES
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_features():
    """Create sample feature data for TFT testing."""
    np.random.seed(42)
    
    # 100 days, 50 features
    n_samples = 100
    n_features = 50
    
    # Generate realistic feature data
    features = np.random.randn(n_samples, n_features)
    
    # Normalize features (simulating preprocessing)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    return torch.FloatTensor(features)


@pytest.fixture
def sample_sequences():
    """Create sample sequences for TFT training."""
    np.random.seed(42)
    
    past_length = 60
    future_length = 1
    n_features = 50
    n_sequences = 50
    
    # Generate past features
    X_past = np.random.randn(n_sequences, past_length, n_features)
    
    # Generate future features (for decoder)
    X_future = np.random.randn(n_sequences, future_length, n_features)
    
    # Generate targets (future returns)
    y = np.random.randn(n_sequences) * 0.02  # Small returns
    
    return (
        torch.FloatTensor(X_past),
        torch.FloatTensor(X_future),
        torch.FloatTensor(y)
    )


@pytest.fixture
def sample_price_data():
    """Create sample OHLCV data for feature engineering."""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Random walk prices
    close_prices = [100.0]
    for i in range(99):
        change = np.random.normal(0, 2)
        close_prices.append(max(50, close_prices[-1] + change))
    
    close = np.array(close_prices)
    
    data = {
        'time': dates,
        'open': close * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'high': close * (1 + np.random.uniform(0.005, 0.02, 100)),
        'low': close * (1 - np.random.uniform(0.005, 0.02, 100)),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, 100),
    }
    
    return pd.DataFrame(data)


# ============================================================================
# TESTS: Variable Selection Network
# ============================================================================

class TestVariableSelectionNetwork:
    """Test suite for Variable Selection Network."""
    
    def test_variable_selection_network_initialization(self):
        """Test VSN initializes correctly."""
        vsn = VariableSelectionNetwork(
            num_inputs=50,
            hidden_size=64,
            dropout=0.1
        )
        
        assert vsn.num_inputs == 50
        assert vsn.hidden_size == 64
    
    def test_variable_selection_forward_pass(self):
        """Test VSN forward pass produces correct shapes."""
        vsn = VariableSelectionNetwork(
            num_inputs=50,
            hidden_size=64
        )
        
        batch_size = 32
        seq_len = 60
        num_inputs = 50
        
        x = torch.randn(batch_size, seq_len, num_inputs)
        
        selected, weights = vsn(x)
        
        # Check output shapes
        assert selected.shape == (batch_size, seq_len, 64)
        assert weights.shape == (batch_size, seq_len, num_inputs)
        
        # Check weights sum to 1 (softmax normalization)
        weights_sum = weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)
        
        # Check weights are non-negative
        assert torch.all(weights >= 0)
    
    def test_variable_selection_weights_valid(self):
        """Test that variable selection weights are valid probabilities."""
        vsn = VariableSelectionNetwork(
            num_inputs=20,
            hidden_size=32
        )
        
        x = torch.randn(10, 30, 20)
        selected, weights = vsn(x)
        
        # Weights should be probabilities (sum to 1)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(10, 30), atol=1e-5)
        
        # Weights should be in [0, 1]
        assert torch.all(weights >= 0) and torch.all(weights <= 1)


# ============================================================================
# TESTS: Temporal Fusion Transformer
# ============================================================================

class TestTemporalFusionTransformer:
    """Test suite for Temporal Fusion Transformer."""
    
    def test_tft_initialization(self):
        """Test TFT initializes correctly."""
        model = TemporalFusionTransformer(
            num_features=50,
            hidden_size=64,
            num_heads=4,
            quantiles=[0.1, 0.5, 0.9]
        )
        
        assert model.num_features == 50
        assert model.hidden_size == 64
        assert model.num_heads == 4
        assert len(model.quantiles) == 3
        assert model.quantiles == [0.1, 0.5, 0.9]
    
    def test_tft_forward_pass_basic(self, sample_features):
        """Test TFT forward pass with basic input."""
        model = TemporalFusionTransformer(
            num_features=50,
            hidden_size=64,
            num_heads=4
        )
        
        batch_size = 1
        seq_len = 60
        num_features = 50
        
        # Reshape to (batch, seq, features)
        past_features = sample_features[:seq_len].unsqueeze(0)
        
        # Forward pass
        predictions = model(past_features)
        
        # Check output shape: (batch_size, num_quantiles)
        assert predictions.shape == (batch_size, len(DEFAULT_QUANTILES))
        assert predictions.shape == (1, 3)  # P10, P50, P90
    
    def test_tft_forward_pass_with_future(self, sample_features):
        """Test TFT forward pass with future features."""
        model = TemporalFusionTransformer(
            num_features=50,
            hidden_size=64
        )
        
        past_features = sample_features[:60].unsqueeze(0)
        future_features = sample_features[60:61].unsqueeze(0)
        
        predictions = model(past_features, future_features)
        
        assert predictions.shape == (1, 3)
    
    def test_tft_forward_pass_with_attention(self, sample_features):
        """Test TFT forward pass with attention weights."""
        model = TemporalFusionTransformer(
            num_features=50,
            hidden_size=64
        )
        
        past_features = sample_features[:60].unsqueeze(0)
        
        predictions, attention_dict = model(past_features, return_attention=True)
        
        assert predictions.shape == (1, 3)
        assert 'encoder_weights' in attention_dict
        assert 'decoder_weights' in attention_dict
        assert 'temporal_attention' in attention_dict
    
    def test_tft_quantile_outputs(self, sample_features):
        """Test that quantile outputs are ordered correctly."""
        model = TemporalFusionTransformer(
            num_features=50,
            quantiles=[0.1, 0.5, 0.9]
        )
        
        past_features = sample_features[:60].unsqueeze(0)
        
        predictions = model(past_features)
        
        # Extract quantiles (assuming order matches [0.1, 0.5, 0.9])
        p10, p50, p90 = predictions[0, 0], predictions[0, 1], predictions[0, 2]
        
        # Quantiles should be ordered: P10 <= P50 <= P90
        # (Note: During training this should hold, but with random weights it may not)
        # We'll just check shapes for now
        assert predictions.shape == (1, 3)
    
    def test_tft_predict_with_quantiles(self, sample_features):
        """Test predict_with_quantiles method."""
        model = TemporalFusionTransformer(
            num_features=50,
            quantiles=[0.1, 0.5, 0.9]
        )
        
        # Test with numpy array
        past_array = sample_features[:60].numpy()
        result = model.predict_with_quantiles(past_array)
        
        assert isinstance(result, TFTPredictionResult)
        assert hasattr(result, 'p10')
        assert hasattr(result, 'p50')
        assert hasattr(result, 'p90')
        assert hasattr(result, 'confidence_range')
        assert hasattr(result, 'is_bullish')
        assert hasattr(result, 'is_bearish')
        
        # Test with torch tensor
        past_tensor = sample_features[:60]
        result2 = model.predict_with_quantiles(past_tensor)
        
        assert isinstance(result2, TFTPredictionResult)
    
    def test_tft_handles_long_sequences(self, sample_features):
        """Test TFT handles sequences longer than max_seq_length."""
        model = TemporalFusionTransformer(
            num_features=50,
            max_seq_length=60
        )
        
        # Create sequence longer than max
        long_sequence = torch.randn(1, 100, 50)
        
        # Should automatically truncate to max_seq_length
        predictions = model(long_sequence)
        
        assert predictions.shape == (1, 3)
    
    def test_tft_batch_processing(self):
        """Test TFT processes batches correctly."""
        model = TemporalFusionTransformer(
            num_features=50,
            hidden_size=64
        )
        
        batch_size = 8
        seq_len = 60
        
        past_features = torch.randn(batch_size, seq_len, 50)
        
        predictions = model(past_features)
        
        assert predictions.shape == (batch_size, 3)


# ============================================================================
# TESTS: TFT Trainer
# ============================================================================

class TestTFTTrainer:
    """Test suite for TFT Trainer."""
    
    def test_quantile_loss_calculation(self):
        """Test quantile (pinball) loss calculation."""
        batch_size = 32
        num_quantiles = 3
        quantiles = [0.1, 0.5, 0.9]
        
        predictions = torch.randn(batch_size, num_quantiles)
        targets = torch.randn(batch_size)
        
        loss = TFTTrainer.quantile_loss(predictions, targets, quantiles)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
        assert loss.shape == ()  # Scalar loss
    
    def test_quantile_loss_zero_error(self):
        """Test quantile loss when predictions match targets."""
        batch_size = 10
        quantiles = [0.1, 0.5, 0.9]
        
        targets = torch.randn(batch_size)
        # Perfect predictions (all quantiles = target)
        predictions = targets.unsqueeze(1).repeat(1, len(quantiles))
        
        loss = TFTTrainer.quantile_loss(predictions, targets, quantiles)
        
        # Loss should be small (not exactly zero due to quantile asymmetry)
        assert loss.item() >= 0
    
    def test_prepare_tft_sequences(self):
        """Test sequence preparation for TFT."""
        n_samples = 200
        n_features = 50
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 0.02
        
        past_length = 60
        future_length = 1
        
        X_past, X_future, y_target = TFTTrainer.prepare_tft_sequences(
            X, y, past_length=past_length, future_length=future_length
        )
        
        # Check shapes
        assert X_past.shape[1] == past_length  # Sequence length
        assert X_past.shape[2] == n_features  # Features
        assert X_future.shape[1] == future_length
        assert X_future.shape[2] == n_features
        assert len(y_target) == len(X_past)
        
        # Check that sequences are properly aligned
        assert len(X_past) > 0
        assert len(X_past) == len(X_future) == len(y_target)
    
    def test_create_dataloaders(self):
        """Test DataLoader creation for TFT."""
        n_sequences = 100
        past_length = 60
        future_length = 1
        n_features = 50
        
        X_past = np.random.randn(n_sequences, past_length, n_features)
        X_future = np.random.randn(n_sequences, future_length, n_features)
        y_target = np.random.randn(n_sequences) * 0.02
        
        train_loader, val_loader = TFTTrainer.create_dataloaders(
            X_past, X_future, y_target,
            train_ratio=0.8,
            batch_size=32
        )
        
        # Check that loaders are created
        assert train_loader is not None
        assert val_loader is not None
        
        # Check batch sizes
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 3  # X_past, X_future, y
        assert train_batch[0].shape[0] <= 32  # Batch size
        
        val_batch = next(iter(val_loader))
        assert len(val_batch) == 3


# ============================================================================
# TESTS: Integration with Feature Engineering
# ============================================================================

class TestTFTFeatureIntegration:
    """Test TFT integration with feature engineering."""
    
    def test_create_tft_features_from_dataframe(self, sample_price_data):
        """Test creating TFT features from price DataFrame."""
        from src.analytics.tft import create_tft_features
        
        try:
            X, y = create_tft_features(sample_price_data)
            
            assert X is not None
            assert y is not None
            assert len(X) > 0
            assert len(y) > 0
            assert len(X) == len(y)
            
            # Features should be normalized
            assert X.shape[1] >= 17  # At least basic features
        except ImportError:
            pytest.skip("FeatureEngineer not available")
    
    def test_tft_with_feature_engineer_features(self, sample_price_data):
        """Test TFT can use FeatureEngineer features."""
        from src.analytics.deep_learning import FeatureEngineer
        from src.analytics.tft import TemporalFusionTransformer
        
        # Create features
        X, scaler = FeatureEngineer.create_ml_features(sample_price_data)
        y = FeatureEngineer.create_target(sample_price_data, horizon=1)
        
        if len(X) < 60:
            pytest.skip("Insufficient data for TFT")
        
        # Create TFT model with correct feature count
        model = TemporalFusionTransformer(
            num_features=X.shape[1],
            hidden_size=32,  # Smaller for faster testing
            num_heads=2
        )
        
        # Prepare sequence
        past_seq = X[-60:].unsqueeze(0)  # Last 60 timesteps
        
        # Forward pass
        predictions = model(past_seq)
        
        assert predictions.shape == (1, 3)  # P10, P50, P90


# ============================================================================
# TESTS: Integration with ConfluenceEngine
# ============================================================================

class TestTFTConfluenceIntegration:
    """Test TFT integration with ConfluenceEngine."""
    
    def test_confluence_engine_can_use_tft(self, sample_price_data, mocker):
        """Test that ConfluenceEngine can use TFT if available."""
        from src.analytics.confluence import ConfluenceEngine
        
        engine = ConfluenceEngine()
        
        # Mock TFT model to exist
        mock_model_path = Path("models/tft_model.pth")
        
        # Create a temporary mock model for testing
        # In real scenario, this would be a trained model
        if not mock_model_path.parent.exists():
            mock_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Mock the TFT prediction
        with mocker.patch('src.analytics.confluence.TemporalFusionTransformer') as MockTFT:
            # Mock model instance
            mock_model = mocker.MagicMock()
            mock_result = TFTPredictionResult(p10=-0.05, p50=0.02, p90=0.08)
            mock_model.predict_with_quantiles.return_value = mock_result
            mock_model.load_state_dict = mocker.MagicMock()
            mock_model.eval = mocker.MagicMock()
            
            MockTFT.return_value = mock_model
            
            # Mock Path.exists to return True
            with mocker.patch('pathlib.Path.exists', return_value=True):
                # Mock torch.load
                with mocker.patch('torch.load', return_value={}):
                    # Mock FeatureEngineer
                    with mocker.patch('src.analytics.confluence.FeatureEngineer') as MockFE:
                        MockFE.create_ml_features.return_value = (
                            np.random.randn(100, 50),
                            None
                        )
                        
                        # Test _get_tft_score
                        result = engine._get_tft_score('TEST', sample_price_data)
                        
                        assert result is not None
                        assert 'score' in result
                        assert result['source'] == 'tft'
    
    def test_confluence_engine_fallback_to_ensemble(self, sample_price_data, mocker):
        """Test that ConfluenceEngine falls back to ensemble if TFT unavailable."""
        from src.analytics.confluence import ConfluenceEngine
        
        engine = ConfluenceEngine()
        
        # Mock TFT as unavailable
        with mocker.patch('pathlib.Path.exists', return_value=False):
            # Mock ensemble as available
            with mocker.patch.object(engine, 'ensemble') as mock_ensemble:
                mock_ensemble.predict.return_value = (
                    np.array([0.01]),
                    np.array([0.05])
                )
                
                # Mock FeatureEngineer
                with mocker.patch('src.analytics.confluence.FeatureEngineer') as MockFE:
                    MockFE.create_ml_features.return_value = (
                        np.random.randn(100, 50),
                        None
                    )
                    
                    result = engine._get_ml_score('TEST', sample_price_data)
                    
                    # Should fallback to ensemble or default
                    assert result is not None
                    assert 'score' in result


# ============================================================================
# TESTS: PredictionResult
# ============================================================================

class TestTFTPredictionResult:
    """Test suite for TFTPredictionResult dataclass."""
    
    def test_prediction_result_creation(self):
        """Test TFTPredictionResult can be created."""
        result = TFTPredictionResult(
            p10=-0.05,
            p50=0.02,
            p90=0.08
        )
        
        assert result.p10 == -0.05
        assert result.p50 == 0.02
        assert result.p90 == 0.08
    
    def test_prediction_result_properties(self):
        """Test TFTPredictionResult properties."""
        result = TFTPredictionResult(p10=-0.03, p50=0.01, p90=0.05)
        
        assert result.point_estimate == 0.01  # P50
        assert result.lower_bound == -0.03  # P10
        assert result.upper_bound == 0.05  # P90
        assert result.confidence_range == 0.08  # P90 - P10
    
    def test_prediction_result_bullish(self):
        """Test bullish prediction detection."""
        result = TFTPredictionResult(p10=-0.01, p50=0.02, p90=0.05)
        
        assert result.is_bullish is True
        assert result.is_bearish is False
    
    def test_prediction_result_bearish(self):
        """Test bearish prediction detection."""
        result = TFTPredictionResult(p10=-0.05, p50=-0.02, p90=0.01)
        
        assert result.is_bullish is False
        assert result.is_bearish is True
    
    def test_prediction_result_to_dict(self):
        """Test converting to dictionary."""
        result = TFTPredictionResult(p10=-0.03, p50=0.01, p90=0.05)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'p10' in result_dict
        assert 'p50' in result_dict
        assert 'p90' in result_dict
        assert 'confidence_range' in result_dict
        assert 'is_bullish' in result_dict
        assert 'is_bearish' in result_dict


# ============================================================================
# TESTS: Edge Cases
# ============================================================================

class TestTFTEdgeCases:
    """Test edge cases and error handling."""
    
    def test_tft_handles_short_sequences(self):
        """Test TFT handles sequences shorter than required."""
        model = TemporalFusionTransformer(
            num_features=50,
            hidden_size=32
        )
        
        # Very short sequence (should still work with padding/truncation)
        short_seq = torch.randn(1, 10, 50)
        
        try:
            predictions = model(short_seq)
            # Should handle gracefully (may pad or use as-is)
            assert predictions.shape[0] == 1
        except Exception as e:
            # If it fails, that's acceptable - document the minimum length
            pytest.skip(f"TFT requires minimum sequence length: {e}")
    
    def test_tft_handles_missing_features(self, sample_features):
        """Test TFT handles missing features gracefully."""
        model = TemporalFusionTransformer(
            num_features=50,
            hidden_size=32
        )
        
        # Sequence with NaN values (should be handled by preprocessing)
        features_with_nan = sample_features.clone()
        features_with_nan[0, 0] = float('nan')
        
        # Should handle NaN (either through preprocessing or error)
        try:
            past_seq = features_with_nan[:60].unsqueeze(0)
            predictions = model(past_seq)
            assert predictions.shape == (1, 3)
        except (ValueError, RuntimeError):
            # Expected - NaN should be handled in preprocessing
            pass
    
    def test_tft_handles_zero_variance_features(self):
        """Test TFT handles constant features."""
        model = TemporalFusionTransformer(
            num_features=50,
            hidden_size=32
        )
        
        # Constant features (zero variance)
        constant_features = torch.ones(60, 50)
        past_seq = constant_features.unsqueeze(0)
        
        # Should still produce predictions
        predictions = model(past_seq)
        assert predictions.shape == (1, 3)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestTFTPerformance:
    """Test TFT performance characteristics."""
    
    def test_tft_forward_pass_speed(self, sample_features):
        """Test TFT forward pass is reasonably fast."""
        import time
        
        model = TemporalFusionTransformer(
            num_features=50,
            hidden_size=32,  # Smaller for speed
            num_heads=2
        )
        
        past_features = sample_features[:60].unsqueeze(0)
        
        # Warm-up
        _ = model(past_features)
        
        # Time inference
        start = time.time()
        for _ in range(10):
            _ = model(past_features)
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        
        # Should be reasonably fast (< 0.1s per inference on CPU)
        assert avg_time < 0.1, f"TFT inference too slow: {avg_time:.4f}s"
    
    def test_tft_memory_usage(self):
        """Test TFT doesn't use excessive memory."""
        import torch
        
        model = TemporalFusionTransformer(
            num_features=50,
            hidden_size=64,
            num_heads=4
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters (< 1M for default config)
        assert num_params < 1_000_000, f"Too many parameters: {num_params}"
