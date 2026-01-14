"""
Risk Management Module.

Provides risk metrics and position sizing:
- Monte Carlo VaR (Value at Risk)
- Conditional VaR (CVaR / Expected Shortfall)
- Position sizing (Kelly Criterion-inspired)
- Risk-Reward Ratio calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from datetime import datetime, timedelta

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioRiskMetrics:
    """Container for portfolio risk metrics."""
    var_95: float
    var_99: float
    sharpe_ratio: float
    recommended_position_pct: float
    max_position_pct: float
    annualized_volatility: float
    expected_max_drawdown: float



class RiskEngine:
    """
    Risk management engine.
    
    Calculates risk metrics and provides position sizing recommendations.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize risk engine.
        
        Args:
            confidence_level: Confidence level for VaR (default 0.95 = 95%)
        """
        self.confidence_level = confidence_level
    
    def calculate_portfolio_risk(
        self,
        returns: pd.Series,
        portfolio_value: float = 10000.0
    ) -> PortfolioRiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Used by ConfluenceEngine for risk analysis.
        
        Args:
            returns: Series of historical returns
            portfolio_value: Portfolio value in dollars
        
        Returns:
            PortfolioRiskMetrics dataclass with all metrics
        """
        if len(returns) < 30:
            # Default values for insufficient data
            return PortfolioRiskMetrics(
                var_95=0.05,
                var_99=0.08,
                sharpe_ratio=0.0,
                recommended_position_pct=0.02,
                max_position_pct=0.05,
                annualized_volatility=0.20,
                expected_max_drawdown=0.10
            )
        
        # VaR calculation
        var_metrics = self.calculate_var(returns, portfolio_value)
        
        # Volatility (annualized)
        annualized_vol = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming risk-free rate of 4%)
        risk_free_rate = 0.04
        annualized_return = returns.mean() * 252
        sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Position sizing (based on volatility and VaR)
        # Higher volatility = smaller position
        vol_factor = max(0.1, 1 - annualized_vol)  # 0.1 to 1.0
        recommended_pct = min(0.05, vol_factor * 0.02)  # Cap at 5%
        max_pct = min(0.10, vol_factor * 0.05)  # Cap at 10%
        
        return PortfolioRiskMetrics(
            var_95=var_metrics['var_95'],
            var_99=var_metrics['var_99'],
            sharpe_ratio=float(sharpe),
            recommended_position_pct=recommended_pct,
            max_position_pct=max_pct,
            annualized_volatility=float(annualized_vol),
            expected_max_drawdown=float(max_drawdown)
        )
    
    def calculate_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 10000.0,
        method: str = 'monte_carlo'
    ) -> Dict:
        """
        Calculate Value at Risk (VaR) using multiple methods.
        
        Enhanced with:
        - Monte Carlo VaR (already implemented)
        - Parametric VaR
        - Historical VaR
        - Conditional VaR (CVaR / Expected Shortfall)
        
        VaR answers: "What's the worst loss I can expect with X% confidence?"
        
        Args:
            returns: Series of historical returns
            portfolio_value: Portfolio value in dollars
            method: VaR method ('monte_carlo', 'parametric', 'historical')
        
        Returns:
            Dictionary with VaR metrics
        """
        if len(returns) < 30:
            logger.warning("Insufficient data for VaR calculation")
            return {
                'var_95': 0.05,  # Default 5%
                'var_99': 0.08,
                'cvar_95': 0.07,
                'cvar_99': 0.10,
                'method': method
            }
        
        # Calculate statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        if method == 'parametric':
            # Parametric VaR (assumes normal distribution)
            # VaR = -mean + z_score * std
            z_95 = stats.norm.ppf(0.05)  # -1.645 for 95%
            z_99 = stats.norm.ppf(0.01)  # -2.326 for 99%
            
            var_95 = abs(-mean_return + z_95 * std_return)
            var_99 = abs(-mean_return + z_99 * std_return)
            
            # CVaR (analytical for normal distribution)
            cvar_95 = abs(-mean_return + (stats.norm.pdf(z_95) / 0.05) * std_return)
            cvar_99 = abs(-mean_return + (stats.norm.pdf(z_99) / 0.01) * std_return)
        
        elif method == 'historical':
            # Historical VaR (percentile of actual returns)
            var_95 = abs(np.percentile(returns, 5))
            var_99 = abs(np.percentile(returns, 1))
            
            # CVaR (average of tail returns)
            cvar_95 = abs(returns[returns <= np.percentile(returns, 5)].mean())
            cvar_99 = abs(returns[returns <= np.percentile(returns, 1)].mean())
        
        else:  # monte_carlo (default)
            # Monte Carlo simulation
            n_simulations = 10000
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # Calculate VaR (percentile)
            var_95 = abs(np.percentile(simulated_returns, 5))
            var_99 = abs(np.percentile(simulated_returns, 1))
            
            # Calculate CVaR (Conditional VaR / Expected Shortfall)
            cvar_95 = abs(simulated_returns[simulated_returns <= np.percentile(simulated_returns, 5)].mean())
            cvar_99 = abs(simulated_returns[simulated_returns <= np.percentile(simulated_returns, 1)].mean())
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'mean_return': mean_return,
            'std_return': std_return,
            'method': method
        }
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
        
        CVaR is the expected loss given that loss exceeds VaR.
        More conservative than VaR.
        
        Args:
            returns: Series of historical returns
            confidence_level: Confidence level (default 0.95)
        
        Returns:
            CVaR value
        """
        if len(returns) < 30:
            return 0.07  # Default 7%
        
        var_level = 1 - confidence_level
        var_threshold = np.percentile(returns, var_level * 100)
        
        # Average of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        cvar = tail_returns.mean() if len(tail_returns) > 0 else var_threshold
        
        return abs(cvar)
    
    def calculate_position_size(
        self,
        expected_return: float,
        win_rate: float,
        stop_loss_pct: float,
        portfolio_value: float = 10000.0,
        max_position_pct: float = 0.02
    ) -> Dict:
        """
        Calculate optimal position size using Kelly Criterion-inspired approach.
        
        Kelly Criterion: f* = (p * b - q) / b
        where:
        - f* = optimal fraction of capital
        - p = win probability
        - q = loss probability (1 - p)
        - b = win/loss ratio
        
        Args:
            expected_return: Expected return (0.1 = 10%)
            win_rate: Win probability (0.6 = 60%)
            stop_loss_pct: Stop loss percentage (0.05 = 5%)
            portfolio_value: Portfolio value
            max_position_pct: Maximum position size (0.02 = 2%)
        
        Returns:
            Dictionary with position sizing recommendations
        """
        if win_rate <= 0 or win_rate >= 1:
            win_rate = 0.5  # Default 50%
        
        if stop_loss_pct <= 0:
            stop_loss_pct = 0.05  # Default 5%
        
        # Calculate win/loss ratio
        # Assuming take-profit = 2 * stop-loss (2:1 risk-reward)
        take_profit_pct = stop_loss_pct * 2
        
        # Kelly Criterion
        loss_prob = 1 - win_rate
        win_loss_ratio = take_profit_pct / stop_loss_pct  # b in Kelly formula
        
        kelly_fraction = (win_rate * win_loss_ratio - loss_prob) / win_loss_ratio
        
        # Apply fractional Kelly (use 25% of Kelly for safety)
        kelly_fraction = max(0, kelly_fraction * 0.25)
        
        # Cap at max position size
        position_pct = min(kelly_fraction, max_position_pct)
        
        # Calculate position value
        position_value = portfolio_value * position_pct
        
        return {
            'position_pct': position_pct,
            'position_value': position_value,
            'kelly_fraction': kelly_fraction,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio
        }
    
    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> float:
        """
        Calculate Risk-Reward Ratio.
        
        RR = (Take Profit - Entry) / (Entry - Stop Loss)
        
        Example:
        Entry: $100, Stop Loss: $95, Take Profit: $110
        RR = (110 - 100) / (100 - 95) = 10 / 5 = 2.0
        
        Good RR: >= 2.0 (risk $1 to make $2)
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Risk-reward ratio (float)
        """
        if entry_price <= 0:
            return 0.0
        
        # Calculate risk and reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0.0
        
        rr_ratio = reward / risk
        
        return float(rr_ratio)
    
    def get_risk_metrics(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Calculate comprehensive risk metrics for a symbol.
        
        Args:
            symbol: Stock ticker
            price_data: DataFrame with OHLCV data
            confidence_level: Confidence level for VaR
        
        Returns:
            Dictionary with risk metrics
        """
        if price_data.empty or len(price_data) < 30:
            return {
                'var_95': 0.05,
                'cvar_95': 0.07,
                'max_drawdown': 0.10,
                'volatility': 0.20
            }
        
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        # VaR calculation
        var_metrics = self.calculate_var(returns, confidence_level=confidence_level)
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        return {
            'var_95': var_metrics.get('var_95', 0.05),
            'var_99': var_metrics.get('var_99', 0.08),
            'cvar_95': var_metrics.get('cvar_95', 0.07),
            'cvar_99': var_metrics.get('cvar_99', 0.10),
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'mean_return': returns.mean(),
            'std_return': returns.std()
        }


# Convenience functions
def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float
) -> float:
    """
    Calculate Risk-Reward Ratio (convenience function).
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
    
    Returns:
        Risk-reward ratio
    """
    engine = RiskEngine()
    return engine.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
