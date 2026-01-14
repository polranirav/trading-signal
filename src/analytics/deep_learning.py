"""
Deep Learning Models for Price Prediction.

Implements:
1. AttentionLSTM - LSTM with Multi-Head Attention
2. FeatureEngineer - 17+ ML features from research
3. ModelTrainer - Training with early stopping

Research Foundation:
From "Complete_Enterprise_Trading_Stack.md":
- AttentionLSTM: input_size=14, hidden_size=64, num_layers=2, num_heads=4
- Features: momentum, volatility, volume patterns, OHLC relationships
- Training: Adam optimizer, MSE loss, early stopping (patience=10)

From Research Papers:
- LSTM captures sequential patterns: 54-56% accuracy
- Attention focuses on relevant time steps (earnings, Fed meetings)
- Combined LSTM+Attention: 58-60% accuracy
- Quantile output for risk management (P10, P50, P90)

Usage:
    from src.analytics.deep_learning import AttentionLSTM, FeatureEngineer, ModelTrainer
    
    # Prepare features
    X, scaler = FeatureEngineer.create_ml_features(df)
    
    # Create sequences
    X_seq, y_seq = ModelTrainer.prepare_sequences(X, returns, seq_length=20)
    
    # Train model
    model = AttentionLSTM(input_size=X.shape[1])
    ModelTrainer.train(model, train_loader, val_loader)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path
import pickle

from src.logging_config import get_logger

logger = get_logger(__name__)

# Create models directory if not exists
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class PredictionResult:
    """Result from model prediction."""
    point_estimate: float  # P50 (median prediction)
    lower_bound: float     # P10 (10th percentile)
    upper_bound: float     # P90 (90th percentile)
    confidence: float      # How narrow is the range (0-1)
    
    @property
    def is_bullish(self) -> bool:
        return self.point_estimate > 0 and self.lower_bound > -0.02
    
    @property
    def is_bearish(self) -> bool:
        return self.point_estimate < 0 and self.upper_bound < 0.02


class AttentionLSTM(nn.Module):
    """
    LSTM with Multi-Head Attention for price prediction.
    
    Architecture (from research):
    - LSTM: Captures sequential patterns in time series
    - Multi-Head Attention: Focus on relevant time steps
    - MLP Head: Final prediction with dropout regularization
    
    From research:
    - input_size: 14-20 features
    - hidden_size: 64 (good balance of capacity vs overfitting)
    - num_layers: 2 (deeper = more overfitting risk)
    - num_heads: 4 (multi-scale attention)
    - dropout: 0.2 (regularization)
    """
    
    def __init__(
        self,
        input_size: int = 17,  # From FeatureEngineer
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        quantile_output: bool = True  # Output P10, P50, P90
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.quantile_output = quantile_output
        
        # Input normalization
        self.layer_norm = nn.LayerNorm(input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Causal: can't use future data
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP head
        output_dim = 3 if quantile_output else 1  # P10, P50, P90 or just P50
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            predictions: (batch_size, num_outputs)
        """
        # Normalize input
        x = self.layer_norm(x)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_size)
        
        # Multi-head attention (self-attention on LSTM outputs)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # attn_out: (batch_size, seq_len, hidden_size)
        
        # Global average pooling (capture all time steps)
        pooled = torch.mean(attn_out, dim=1)
        # pooled: (batch_size, hidden_size)
        
        # MLP head
        predictions = self.fc(pooled)
        
        return predictions
    
    def predict_with_confidence(self, x: torch.Tensor) -> PredictionResult:
        """
        Predict with confidence intervals.
        
        Returns:
            PredictionResult with P10, P50, P90
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            
            if self.quantile_output:
                # Output is [P10, P50, P90]
                p10, p50, p90 = output[0].numpy()
            else:
                p50 = output[0, 0].numpy()
                # Estimate range based on training volatility
                p10, p90 = p50 - 0.02, p50 + 0.02
            
            # Confidence based on range narrowness
            confidence = 1 - min(1, (p90 - p10) / 0.10)  # 10% range = 0 confidence
            
            return PredictionResult(
                point_estimate=float(p50),
                lower_bound=float(p10),
                upper_bound=float(p90),
                confidence=float(confidence)
            )


class FeatureEngineer:
    """
    Advanced feature engineering for ML models.
    
    Creates 17+ features from OHLCV data:
    - Momentum features (3): momentum_5, momentum_10, momentum_20
    - ROC features (3): roc_5, roc_10, roc_20
    - Volatility features (2): volatility_20, volatility_ratio
    - Volume features (3): volume_ma_ratio, volume_trend, volume_momentum
    - OHLC features (3): hl_range, co_position, ho_position
    - Statistical features (2): skewness, kurtosis
    - Trend features (1): price_sma_20_ratio
    
    From research:
    - These features capture non-linear patterns humans miss
    - ML considers 50+ variables simultaneously
    - Regime adaptation through rolling statistics
    """
    
    FEATURE_COLS = [
        'momentum_5', 'momentum_10', 'momentum_20',
        'roc_5', 'roc_10', 'roc_20',
        'volatility_20', 'volatility_ratio',
        'volume_ma_ratio', 'volume_trend', 'volume_momentum',
        'hl_range', 'co_position', 'ho_position',
        'skewness', 'kurtosis',
        'price_sma_20_ratio'
    ]
    
    @classmethod
    def _add_relative_strength_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add relative strength features (stock vs sector/market).
        
        Note: Requires sector/market data. Returns 0 if not available.
        
        Features:
        - Stock vs. Sector momentum (20d, 60d, 120d)
        - Stock vs. Market momentum (SPY relative strength)
        - Relative volatility (stock vol / sector vol)
        - Correlation with sector (rolling 60d)
        - Correlation with market (rolling 60d)
        - Sector rank percentile (within sector)
        - Market rank percentile (within market)
        - Relative strength ratio (stock / sector)
        - Momentum divergence (price vs. relative strength)
        - Sector leadership score
        """
        # Stock returns
        stock_returns = df['close'].pct_change()
        
        # Check if sector/market data available
        has_sector_returns = 'sector_returns' in df.columns
        has_market_returns = 'market_returns' in df.columns
        has_sector_prices = 'sector_prices' in df.columns
        has_market_prices = 'market_prices' in df.columns
        has_sector_vol = 'sector_volatility' in df.columns
        
        # Stock vs. Sector momentum (20d, 60d, 120d)
        for period in [20, 60, 120]:
            stock_momentum = df['close'].pct_change(period)
            
            if has_sector_prices:
                sector_momentum = df['sector_prices'].pct_change(period)
                df[f'rs_vs_sector_{period}d'] = stock_momentum - sector_momentum
            else:
                df[f'rs_vs_sector_{period}d'] = 0.0
        
        # Stock vs. Market momentum (SPY)
        for period in [20, 60, 120]:
            stock_momentum = df['close'].pct_change(period)
            
            if has_market_prices:
                market_momentum = df['market_prices'].pct_change(period)
                df[f'rs_vs_market_{period}d'] = stock_momentum - market_momentum
            else:
                df[f'rs_vs_market_{period}d'] = 0.0
        
        # Relative volatility (stock vol / sector vol)
        stock_vol = df['close'].pct_change().rolling(20).std()
        if has_sector_vol:
            df['rel_vol_sector'] = stock_vol / (df['sector_volatility'] + 1e-8)
        else:
            df['rel_vol_sector'] = 1.0  # Default to 1.0 if no sector data
        
        # Correlation with sector (rolling 60d)
        if has_sector_returns:
            df['corr_sector_60d'] = stock_returns.rolling(60).corr(df['sector_returns'])
        else:
            df['corr_sector_60d'] = 0.0
        
        # Correlation with market (rolling 60d)
        if has_market_returns:
            df['corr_market_60d'] = stock_returns.rolling(60).corr(df['market_returns'])
        else:
            df['corr_market_60d'] = 0.0
        
        # Sector rank percentile (within sector) - requires universe data
        # For now, calculate percentile rank within rolling window
        df['sector_rank_percentile'] = df['close'].rolling(252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if len(x) > 0 and x.max() != x.min() else 0.5,
            raw=False
        )
        
        # Market rank percentile (within market) - requires universe data
        # For now, calculate percentile rank within rolling window
        df['market_rank_percentile'] = df['close'].rolling(252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if len(x) > 0 and x.max() != x.min() else 0.5,
            raw=False
        )
        
        # Relative strength ratio (stock / sector)
        if has_sector_prices:
            df['relative_strength_ratio'] = df['close'] / (df['sector_prices'] + 1e-8)
        else:
            df['relative_strength_ratio'] = 1.0
        
        # Momentum divergence (price vs. relative strength)
        price_momentum = df['close'].pct_change(20)
        if 'rs_vs_sector_20d' in df.columns:
            rs_momentum = df['rs_vs_sector_20d'].rolling(20).mean()
            df['momentum_divergence'] = price_momentum - rs_momentum
        else:
            df['momentum_divergence'] = 0.0
        
        # Sector leadership score (simplified - would need sector data for full implementation)
        # Score = 1 if stock outperforming sector, 0 if underperforming
        if 'rs_vs_sector_60d' in df.columns:
            df['sector_leadership_score'] = (df['rs_vs_sector_60d'] > 0).astype(float)
        else:
            df['sector_leadership_score'] = 0.5  # Neutral if no data
        
        return df
    
    @classmethod
    def _add_macro_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add macroeconomic features.
        
        Note: Requires macro data (VIX, rates, etc.). Returns defaults if not available.
        
        Features:
        - VIX level and percentile
        - VIX term structure (VIX / VIX9D)
        - Interest rate sensitivity (beta to rates)
        - Inflation sensitivity (beta to inflation)
        - Credit spread impact (corp bond spreads)
        - Yield curve slope (10Y - 2Y)
        - Dollar strength (DXY impact)
        - Commodity correlation (oil, gold)
        - Sector rotation indicators
        - Economic surprise index
        """
        stock_returns = df['close'].pct_change()
        
        # Check if macro data available
        has_vix = 'vix' in df.columns
        has_vix9d = 'vix9d' in df.columns
        has_rates = 'interest_rates' in df.columns
        has_inflation = 'inflation' in df.columns
        has_credit_spread = 'credit_spread' in df.columns
        has_yield_10y = 'yield_10y' in df.columns
        has_yield_2y = 'yield_2y' in df.columns
        has_dxy = 'dxy' in df.columns
        has_oil = 'oil_price' in df.columns
        has_gold = 'gold_price' in df.columns
        
        # VIX level and percentile
        if has_vix:
            df['vix_level'] = df['vix']
            df['vix_percentile'] = df['vix'].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if len(x) > 0 and x.max() != x.min() else 0.5,
                raw=False
            )
        else:
            df['vix_level'] = 20.0  # Default VIX
            df['vix_percentile'] = 0.5
        
        # VIX term structure (VIX / VIX9D)
        if has_vix and has_vix9d:
            df['vix_term_structure'] = df['vix'] / (df['vix9d'] + 1e-8)
        else:
            df['vix_term_structure'] = 1.0  # Normal structure
        
        # Interest rate sensitivity (beta to rates)
        if has_rates:
            rate_returns = df['interest_rates'].pct_change()
            df['rate_sensitivity'] = stock_returns.rolling(60).cov(rate_returns) / (rate_returns.rolling(60).var() + 1e-8)
        else:
            df['rate_sensitivity'] = 0.0
        
        # Inflation sensitivity (beta to inflation)
        if has_inflation:
            inflation_returns = df['inflation'].pct_change()
            df['inflation_sensitivity'] = stock_returns.rolling(60).cov(inflation_returns) / (inflation_returns.rolling(60).var() + 1e-8)
        else:
            df['inflation_sensitivity'] = 0.0
        
        # Credit spread impact (corp bond spreads)
        if has_credit_spread:
            spread_returns = df['credit_spread'].pct_change()
            df['credit_spread_impact'] = stock_returns.rolling(60).cov(spread_returns) / (spread_returns.rolling(60).var() + 1e-8)
        else:
            df['credit_spread_impact'] = 0.0
        
        # Yield curve slope (10Y - 2Y)
        if has_yield_10y and has_yield_2y:
            df['yield_curve_slope'] = df['yield_10y'] - df['yield_2y']
        else:
            df['yield_curve_slope'] = 0.0
        
        # Dollar strength (DXY impact)
        if has_dxy:
            dxy_returns = df['dxy'].pct_change()
            df['dollar_strength'] = dxy_returns.rolling(20).mean()
            df['dollar_strength_correlation'] = stock_returns.rolling(60).corr(dxy_returns)
        else:
            df['dollar_strength'] = 0.0
            df['dollar_strength_correlation'] = 0.0
        
        # Commodity correlation (oil, gold)
        if has_oil:
            oil_returns = df['oil_price'].pct_change()
            df['oil_correlation'] = stock_returns.rolling(60).corr(oil_returns)
        else:
            df['oil_correlation'] = 0.0
        
        if has_gold:
            gold_returns = df['gold_price'].pct_change()
            df['gold_correlation'] = stock_returns.rolling(60).corr(gold_returns)
        else:
            df['gold_correlation'] = 0.0
        
        # Commodity correlation (combined)
        if has_oil and has_gold:
            df['commodity_correlation'] = (df['oil_correlation'] + df['gold_correlation']) / 2
        elif has_oil:
            df['commodity_correlation'] = df['oil_correlation']
        elif has_gold:
            df['commodity_correlation'] = df['gold_correlation']
        else:
            df['commodity_correlation'] = 0.0
        
        # Sector rotation indicator (simplified - would need sector data)
        # For now, use VIX as proxy (low VIX = risk-on, high VIX = risk-off)
        if has_vix:
            df['sector_rotation_indicator'] = (df['vix'] < 15).astype(float)  # Risk-on = 1, Risk-off = 0
        else:
            df['sector_rotation_indicator'] = 0.5
        
        # Economic surprise index (placeholder - would need actual data)
        df['economic_surprise_index'] = 0.0  # Neutral if no data
        
        return df
    
    @classmethod
    def _add_cross_sectional_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-sectional features (percentile ranks, z-scores, etc.).
        
        Note: Requires universe data. Uses rolling windows if not available.
        
        Features:
        - Percentile rank vs. universe (price, volume, momentum)
        - Z-score vs. universe (standardized)
        - Momentum rank (within sector)
        - Valuation rank (P/E, P/B vs. sector)
        - Quality rank (ROE, margins vs. sector)
        - Growth rank (revenue growth vs. sector)
        - Consensus score (analyst ratings)
        - Insider activity score
        - Institutional ownership changes
        - Short interest ratio and changes
        """
        stock_returns = df['close'].pct_change()
        
        # Check if universe/sector data available
        has_universe_prices = 'universe_prices' in df.columns
        has_universe_volume = 'universe_volume' in df.columns
        has_universe_momentum = 'universe_momentum' in df.columns
        has_pe = 'pe_ratio' in df.columns
        has_pb = 'pb_ratio' in df.columns
        has_roe = 'roe' in df.columns
        has_margins = 'margins' in df.columns
        has_revenue_growth = 'revenue_growth' in df.columns
        has_analyst_ratings = 'analyst_ratings' in df.columns
        has_insider_activity = 'insider_activity' in df.columns
        has_institutional_ownership = 'institutional_ownership' in df.columns
        has_short_interest = 'short_interest' in df.columns
        
        # Percentile rank vs. universe (price, volume, momentum)
        # For now, use rolling window percentile (would need universe data for true percentile)
        df['percentile_rank_price'] = df['close'].rolling(252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if len(x) > 0 and x.max() != x.min() else 0.5,
            raw=False
        )
        
        if 'volume' in df.columns:
            df['percentile_rank_volume'] = df['volume'].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if len(x) > 0 and x.max() != x.min() else 0.5,
                raw=False
            )
        else:
            df['percentile_rank_volume'] = 0.5
        
        momentum_20d = df['close'].pct_change(20)
        df['percentile_rank_momentum'] = momentum_20d.rolling(252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if len(x) > 0 and x.max() != x.min() else 0.5,
            raw=False
        )
        
        # Z-score vs. universe (standardized)
        # For now, use rolling window z-score
        df['z_score_price'] = (df['close'] - df['close'].rolling(252).mean()) / (df['close'].rolling(252).std() + 1e-8)
        
        if 'volume' in df.columns:
            df['z_score_volume'] = (df['volume'] - df['volume'].rolling(252).mean()) / (df['volume'].rolling(252).std() + 1e-8)
        else:
            df['z_score_volume'] = 0.0
        
        df['z_score_momentum'] = (momentum_20d - momentum_20d.rolling(252).mean()) / (momentum_20d.rolling(252).std() + 1e-8)
        
        # Momentum rank (within sector) - placeholder
        df['momentum_rank'] = df['percentile_rank_momentum']  # Use percentile rank as proxy
        
        # Valuation rank (P/E, P/B vs. sector) - placeholder
        if has_pe:
            df['valuation_rank'] = df['pe_ratio'].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if len(x) > 0 and x.max() != x.min() else 0.5,
                raw=False
            )
        else:
            df['valuation_rank'] = 0.5
        
        # Quality rank (ROE, margins vs. sector) - placeholder
        if has_roe:
            df['quality_rank'] = df['roe'].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if len(x) > 0 and x.max() != x.min() else 0.5,
                raw=False
            )
        else:
            df['quality_rank'] = 0.5
        
        # Growth rank (revenue growth vs. sector) - placeholder
        if has_revenue_growth:
            df['growth_rank'] = df['revenue_growth'].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if len(x) > 0 and x.max() != x.min() else 0.5,
                raw=False
            )
        else:
            df['growth_rank'] = 0.5
        
        # Consensus score (analyst ratings) - placeholder
        if has_analyst_ratings:
            # Analyst ratings typically 1-5 (buy-hold-sell), normalize to 0-1
            df['consensus_score'] = (df['analyst_ratings'] - 1) / 4  # Map 1-5 to 0-1
        else:
            df['consensus_score'] = 0.5
        
        # Insider activity score - placeholder
        if has_insider_activity:
            # Insider activity: positive = buying, negative = selling
            df['insider_activity_score'] = df['insider_activity'].rolling(60).mean()
            df['insider_activity_score'] = (df['insider_activity_score'] - df['insider_activity_score'].rolling(252).min()) / (
                df['insider_activity_score'].rolling(252).max() - df['insider_activity_score'].rolling(252).min() + 1e-8
            )
        else:
            df['insider_activity_score'] = 0.5
        
        # Institutional ownership changes
        if has_institutional_ownership:
            df['institutional_ownership_change'] = df['institutional_ownership'].pct_change(90)  # 3-month change
        else:
            df['institutional_ownership_change'] = 0.0
        
        # Short interest ratio and changes
        if has_short_interest:
            df['short_interest_ratio'] = df['short_interest']
            df['short_interest_change'] = df['short_interest'].pct_change(30)  # 1-month change
        else:
            df['short_interest_ratio'] = 0.0
            df['short_interest_change'] = 0.0
        
        return df
    
    @classmethod
    def _add_regime_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime features (market regime, volatility regime, etc.).
        
        Features:
        - Market regime (Bull/Bear/Sideways/High Vol) - encoded
        - Volatility regime (Low/Medium/High/Extreme)
        - Trend regime (Uptrend/Downtrend/Range)
        - Sector regime (rotation indicators)
        - Economic regime (expansion/contraction)
        - Risk-on/risk-off indicator
        - Regime duration (how long in current regime)
        - Regime transition probability
        """
        stock_returns = df['close'].pct_change()
        
        # Check if VIX data available
        has_vix = 'vix' in df.columns
        has_vix_level = 'vix_level' in df.columns
        vix_data = df['vix_level'] if has_vix_level else (df['vix'] if has_vix else None)
        
        # Market regime (Bull/Bear/Sideways/High Vol) - encoded
        # Use trend signal and VIX to determine regime
        if 'trend_signal' in df.columns and vix_data is not None:
            df['market_regime'] = df.apply(
                lambda row: cls._classify_market_regime(
                    row.get('trend_signal', 'SIDEWAYS'),
                    row.get('vix_level') if has_vix_level else (row.get('vix') if has_vix else 20.0),
                    row.get('close', 0),
                    row.get('sma_50', 0),
                    row.get('sma_200', 0)
                ),
                axis=1
            )
            # Encode as numeric (0=High Vol, 1=Bear, 2=Sideways, 3=Bull)
            regime_map = {'HIGH_VOL': 0, 'BEAR': 1, 'SIDEWAYS': 2, 'BULL': 3}
            df['market_regime_encoded'] = df['market_regime'].map(regime_map).fillna(2)
        else:
            df['market_regime'] = 'SIDEWAYS'
            df['market_regime_encoded'] = 2
        
        # Volatility regime (Low/Medium/High/Extreme)
        if vix_data is not None:
            df['volatility_regime'] = df.apply(
                lambda row: cls._classify_volatility_regime(
                    row.get('vix_level') if has_vix_level else (row.get('vix') if has_vix else 20.0)
                ),
                axis=1
            )
            # Encode as numeric (0=Low, 1=Medium, 2=High, 3=Extreme)
            vol_regime_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'EXTREME': 3}
            df['volatility_regime_encoded'] = df['volatility_regime'].map(vol_regime_map).fillna(1)
        else:
            df['volatility_regime'] = 'MEDIUM'
            df['volatility_regime_encoded'] = 1
        
        # Trend regime (Uptrend/Downtrend/Range)
        if 'trend_signal' in df.columns:
            trend_map = {'UPTREND': 2, 'DOWNTREND': 0, 'SIDEWAYS': 1}
            df['trend_regime'] = df['trend_signal']
            df['trend_regime_encoded'] = df['trend_signal'].map(trend_map).fillna(1)
        else:
            df['trend_regime'] = 'SIDEWAYS'
            df['trend_regime_encoded'] = 1
        
        # Sector regime (rotation indicators) - placeholder
        df['sector_regime'] = 'STABLE'
        df['sector_regime_encoded'] = 1
        
        # Economic regime (expansion/contraction) - placeholder
        # Would need GDP growth, unemployment data
        df['economic_regime'] = 'EXPANSION'
        df['economic_regime_encoded'] = 1
        
        # Risk-on/risk-off indicator (0=risk-off, 1=risk-on)
        if vix_data is not None:
            df['risk_on_off'] = (df['vix_level'] if has_vix_level else df['vix']) < 15  # Low VIX = risk-on
            df['risk_on_off'] = df['risk_on_off'].astype(float)
        else:
            df['risk_on_off'] = 0.5
        
        # Regime duration (how long in current regime) - simplified
        # Count consecutive periods with same regime
        if 'market_regime_encoded' in df.columns:
            regime_changes = (df['market_regime_encoded'] != df['market_regime_encoded'].shift(1)).astype(int)
            df['regime_duration'] = regime_changes.groupby((regime_changes != 0).cumsum()).cumsum()
        else:
            df['regime_duration'] = 0
        
        # Regime transition probability - placeholder (would need model)
        df['regime_transition_probability'] = 0.1  # 10% chance of transition (simplified)
        
        return df
    
    @staticmethod
    def _classify_market_regime(trend_signal: str, vix: float, close: float, sma_50: float, sma_200: float) -> str:
        """Classify market regime based on trend and VIX."""
        if vix > 30:
            return 'HIGH_VOL'
        elif vix > 25:
            return 'BEAR'
        elif trend_signal == 'UPTREND' and close > sma_50 > sma_200:
            return 'BULL'
        elif trend_signal == 'DOWNTREND' and close < sma_50 < sma_200:
            return 'BEAR'
        else:
            return 'SIDEWAYS'
    
    @staticmethod
    def _classify_volatility_regime(vix: float) -> str:
        """Classify volatility regime based on VIX."""
        if vix > 35:
            return 'EXTREME'
        elif vix > 25:
            return 'HIGH'
        elif vix > 15:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    @classmethod
    def _add_microstructure_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features (options flow, volume profile, etc.).
        
        Note: Requires high-frequency/options data. Returns defaults if not available.
        
        Features:
        - Options flow (put/call ratio)
        - Options skew (out-of-the-money puts vs. calls)
        - Implied volatility vs. realized volatility
        - Volume profile (support/resistance levels)
        - Large trade indicators (block trades)
        - Order imbalance (buy vs. sell volume)
        """
        # Check if options/microstructure data available
        has_put_call_ratio = 'options_put_call_ratio' in df.columns
        has_options_skew = 'options_skew' in df.columns
        has_iv = 'implied_volatility' in df.columns
        has_volume_profile = 'volume_profile' in df.columns
        has_large_trades = 'large_trade_volume' in df.columns
        has_order_imbalance = 'order_imbalance' in df.columns
        
        # Options put/call ratio
        if has_put_call_ratio:
            df['options_put_call_ratio'] = df['options_put_call_ratio']
        else:
            df['options_put_call_ratio'] = 1.0  # Neutral ratio
        
        # Options skew (out-of-the-money puts vs. calls)
        if has_options_skew:
            df['options_skew'] = df['options_skew']
        else:
            df['options_skew'] = 0.0  # No skew
        
        # Implied volatility vs. realized volatility
        if has_iv:
            rv = df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100  # Annualized RV
            df['iv_vs_rv'] = df['implied_volatility'] / (rv + 1e-8)
        else:
            df['iv_vs_rv'] = 1.0  # IV = RV (neutral)
        
        # Volume profile (support/resistance levels) - placeholder
        # Would need intraday volume data for true volume profile
        if has_volume_profile:
            df['volume_profile_support'] = df['volume_profile'].apply(lambda x: x.get('support', 0) if isinstance(x, dict) else 0)
            df['volume_profile_resistance'] = df['volume_profile'].apply(lambda x: x.get('resistance', 0) if isinstance(x, dict) else 0)
        else:
            # Use price levels as proxy
            df['volume_profile_support'] = df['close'].rolling(20).min()
            df['volume_profile_resistance'] = df['close'].rolling(20).max()
            # Normalize
            df['volume_profile_support'] = (df['volume_profile_support'] - df['close']) / (df['close'] + 1e-8)
            df['volume_profile_resistance'] = (df['volume_profile_resistance'] - df['close']) / (df['close'] + 1e-8)
        
        # Large trade indicators (block trades)
        if has_large_trades and 'volume' in df.columns:
            avg_volume = df['volume'].rolling(20).mean()
            df['large_trade_indicator'] = (df['large_trade_volume'] > avg_volume * 2).astype(float)
        else:
            df['large_trade_indicator'] = 0.0
        
        # Order imbalance (buy vs. sell volume)
        if has_order_imbalance:
            df['order_imbalance'] = df['order_imbalance']  # Already calculated
        else:
            # Use price action as proxy (simplified)
            price_change = df['close'].pct_change()
            df['order_imbalance'] = np.sign(price_change)  # Positive change = buying pressure
        
        return df
    
    @classmethod
    def create_ml_features(
        cls,
        df: pd.DataFrame,
        include_indicators: bool = True
    ) -> Tuple[np.ndarray, StandardScaler]:
        """
        Create comprehensive feature set.
        
        Args:
            df: DataFrame with OHLCV columns
            include_indicators: Whether to include technical indicators
        
        Returns:
            X: Feature array (n_samples, n_features)
            scaler: Fitted StandardScaler for inverse transform
        """
        df = df.copy()
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # === MOMENTUM FEATURES ===
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # === VOLATILITY FEATURES ===
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        df['volatility_60_ma'] = df['volatility_20'].rolling(60).mean()
        df['volatility_ratio'] = df['volatility_20'] / (df['volatility_60_ma'] + 1e-8)
        
        # === VOLUME FEATURES ===
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)
        df['volume_trend'] = df['volume'].pct_change(5)
        df['volume_momentum'] = df['volume'].pct_change(1).rolling(20).mean()
        
        # === OHLC RELATIONSHIP FEATURES ===
        df['hl_range'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
        df['co_position'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['ho_position'] = (df['high'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        
        # === STATISTICAL FEATURES ===
        returns = df['close'].pct_change()
        df['skewness'] = returns.rolling(20).skew()
        df['kurtosis'] = returns.rolling(20).apply(
            lambda x: x.kurtosis() if len(x) > 3 else 0,
            raw=False
        )
        
        # === TREND FEATURES ===
        df['sma_20'] = df['close'].rolling(20).mean()
        df['price_sma_20_ratio'] = df['close'] / (df['sma_20'] + 1e-8)
        
        # === RELATIVE STRENGTH FEATURES ===
        # Note: These require sector/market data - will be 0 if not available
        df = cls._add_relative_strength_features(df)
        
        # === MACRO FEATURES ===
        # Note: These require macro data - will be 0 if not available
        df = cls._add_macro_features(df)
        
        # === CROSS-SECTIONAL FEATURES ===
        # Note: These require universe data - will use rolling window if not available
        df = cls._add_cross_sectional_features(df)
        
        # === REGIME FEATURES ===
        df = cls._add_regime_features(df)
        
        # === MARKET MICROSTRUCTURE FEATURES ===
        # Note: These require high-frequency/options data - will be 0 if not available
        df = cls._add_microstructure_features(df)
        
        # === TECHNICAL INDICATORS (optional) ===
        feature_cols = list(cls.FEATURE_COLS)  # Start with base features
        
        if include_indicators and 'rsi_14' in df.columns:
            indicator_cols = ['rsi_14', 'macd', 'bb_bandwidth', 'adx']
            feature_cols.extend([c for c in indicator_cols if c in df.columns])
        
        # Add available relative strength features
        relative_strength_cols = [
            'rs_vs_sector_20d', 'rs_vs_sector_60d', 'rs_vs_sector_120d',
            'rs_vs_market_20d', 'rs_vs_market_60d', 'rs_vs_market_120d',
            'rel_vol_sector', 'corr_sector_60d', 'corr_market_60d',
            'sector_rank_percentile', 'market_rank_percentile',
            'relative_strength_ratio', 'momentum_divergence', 'sector_leadership_score'
        ]
        feature_cols.extend([c for c in relative_strength_cols if c in df.columns])
        
        # Add available macro features
        macro_cols = [
            'vix_level', 'vix_percentile', 'vix_term_structure',
            'rate_sensitivity', 'inflation_sensitivity', 'credit_spread_impact',
            'yield_curve_slope', 'dollar_strength', 'commodity_correlation',
            'sector_rotation_indicator', 'economic_surprise_index'
        ]
        feature_cols.extend([c for c in macro_cols if c in df.columns])
        
        # Add available cross-sectional features
        cross_sectional_cols = [
            'percentile_rank_price', 'percentile_rank_volume', 'percentile_rank_momentum',
            'z_score_price', 'z_score_volume', 'z_score_momentum',
            'momentum_rank', 'valuation_rank', 'quality_rank',
            'growth_rank', 'consensus_score', 'insider_activity_score',
            'institutional_ownership_change', 'short_interest_ratio'
        ]
        feature_cols.extend([c for c in cross_sectional_cols if c in df.columns])
        
        # Add available regime features
        regime_cols = [
            'market_regime_encoded', 'volatility_regime_encoded', 'trend_regime_encoded',
            'sector_regime_encoded', 'economic_regime_encoded', 'risk_on_off',
            'regime_duration', 'regime_transition_probability'
        ]
        feature_cols.extend([c for c in regime_cols if c in df.columns])
        
        # Add available microstructure features
        microstructure_cols = [
            'options_put_call_ratio', 'options_skew', 'iv_vs_rv',
            'volume_profile_support', 'volume_profile_resistance',
            'large_trade_indicator', 'order_imbalance'
        ]
        feature_cols.extend([c for c in microstructure_cols if c in df.columns])
        
        # Extract features
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        
        # Normalize using RobustScaler (handles outliers better)
        scaler = RobustScaler()
        X_normalized = scaler.fit_transform(X)
        
        # Clip extreme values
        X_normalized = np.clip(X_normalized, -5, 5)
        
        logger.info(f"Created {X.shape[1]} features from {len(df)} samples")
        
        return X_normalized, scaler
    
    @classmethod
    def create_target(
        cls,
        df: pd.DataFrame,
        horizon: int = 5,
        target_type: str = 'returns'
    ) -> np.ndarray:
        """
        Create prediction target.
        
        Args:
            df: DataFrame with 'close' column
            horizon: Forward-looking period (days)
            target_type: 'returns' or 'direction'
        
        Returns:
            y: Target array
        """
        if target_type == 'returns':
            # Future returns
            y = df['close'].pct_change(horizon).shift(-horizon)
        elif target_type == 'direction':
            # Binary direction (1 = up, 0 = down)
            future_returns = df['close'].pct_change(horizon).shift(-horizon)
            y = (future_returns > 0).astype(float)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        return y.fillna(0).values


class ModelTrainer:
    """
    Train and evaluate deep learning models.
    
    Features:
    - Sequence preparation for LSTM
    - Training with early stopping
    - Walk-forward validation
    - Model checkpointing
    
    From research:
    - Early stopping: patience=10 (prevents overfitting)
    - Adam optimizer with lr=1e-3
    - Gradient clipping: max_norm=1.0
    """
    
    @staticmethod
    def prepare_sequences(
        X: np.ndarray,
        y: np.ndarray,
        seq_length: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            seq_length: Lookback window
        
        Returns:
            X_seq: (n_sequences, seq_length, n_features)
            y_seq: (n_sequences,)
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    @staticmethod
    def create_dataloaders(
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        train_ratio: float = 0.8,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation DataLoaders.
        
        Uses chronological split (no shuffle for time series).
        """
        split_idx = int(len(X_seq) * train_ratio)
        
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    @staticmethod
    def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        patience: int = 10,
        device: str = 'cpu',
        save_path: Path = None
    ) -> Dict:
        """
        Train model with early stopping.
        
        Args:
            model: AttentionLSTM or similar
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Maximum epochs
            learning_rate: Adam learning rate
            patience: Early stopping patience
            device: 'cpu' or 'cuda'
            save_path: Path to save best model
        
        Returns:
            Training history dict
        """
        logger.info(f"Starting training: {epochs} epochs, patience={patience}")
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss function for quantile output
        if model.quantile_output:
            criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        else:
            criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # === TRAINING ===
            model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                if model.quantile_output:
                    # Expand y for quantile comparison
                    y_batch = y_batch.unsqueeze(1).expand(-1, 3)
                else:
                    y_batch = y_batch.unsqueeze(1)
                
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # === VALIDATION ===
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    if model.quantile_output:
                        y_batch = y_batch.unsqueeze(1).expand(-1, 3)
                    else:
                        y_batch = y_batch.unsqueeze(1)
                    
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}")
            
            # === EARLY STOPPING ===
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if save_path:
                    torch.save(model.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), MODELS_DIR / 'best_lstm.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"Training complete. Best val_loss: {best_val_loss:.6f}")
        
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'history': history
        }


class QuantileLoss(nn.Module):
    """
    Quantile loss for uncertainty estimation.
    
    From TFT research:
    - Output P10, P50, P90 instead of point estimate
    - Quantile loss is appropriate for this task
    - Allows risk management (know worst/best case)
    """
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.
        
        Args:
            predictions: (batch, num_quantiles)
            targets: (batch, num_quantiles) - same target repeated
        
        Returns:
            loss: Scalar
        """
        losses = []
        
        for i, q in enumerate(self.quantiles):
            errors = targets[:, i] - predictions[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors))
        
        return torch.mean(torch.stack(losses))


# Convenience functions
def get_model(input_size: int = 17, device: str = 'cpu') -> AttentionLSTM:
    """Get an AttentionLSTM model."""
    model = AttentionLSTM(input_size=input_size)
    return model.to(device)


def load_model(path: Path, input_size: int = 17, device: str = 'cpu') -> AttentionLSTM:
    """Load a trained model from disk."""
    model = AttentionLSTM(input_size=input_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    # Test with sample data
    print("\n=== Deep Learning Module Test ===\n")
    
    # Create sample data
    n_samples = 500
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'high': 0,
        'low': 0,
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'volume': np.random.randint(1000000, 10000000, n_samples)
    })
    df['high'] = df[['open', 'close']].max(axis=1) * 1.01
    df['low'] = df[['open', 'close']].min(axis=1) * 0.99
    
    # Create features
    X, scaler = FeatureEngineer.create_ml_features(df)
    y = FeatureEngineer.create_target(df, horizon=5)
    
    # Create sequences
    X_seq, y_seq = ModelTrainer.prepare_sequences(X, y, seq_length=20)
    
    print(f"Features shape: {X.shape}")
    print(f"Sequences shape: {X_seq.shape}")
    print(f"Targets shape: {y_seq.shape}")
    
    # Create model
    model = AttentionLSTM(input_size=X.shape[1])
    print(f"\nModel architecture:\n{model}")
    
    # Create dataloaders
    train_loader, val_loader = ModelTrainer.create_dataloaders(X_seq, y_seq, batch_size=32)
    
    # Train for a few epochs
    result = ModelTrainer.train(
        model, train_loader, val_loader,
        epochs=20, patience=5
    )
    
    print(f"\nTraining result:")
    print(f"  Best val_loss: {result['best_val_loss']:.6f}")
    print(f"  Epochs trained: {result['epochs_trained']}")
