"""
Prediction Points Component.

Discrete prediction points (Perplexity style) with confidence and targets.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.logging_config import get_logger
from src.analytics.confluence import ConfluenceEngine
from src.analytics.ensemble import HybridSignalEnsemble
from src.data.ingestion import MarketDataClient

logger = get_logger(__name__)


@dataclass
class PredictionPoint:
    """Single prediction point (Perplexity style)."""
    target_price: float
    confidence: float  # 0-100%
    probability_change: float  # Change in probability (%)
    timeframe: str  # "1 day", "5 days", "1 month"
    timestamp: datetime
    direction: str = "↑"  # "↑" or "↓"
    
    @property
    def confidence_color(self) -> str:
        """Get color based on confidence."""
        if self.confidence >= 75:
            return 'green'
        elif self.confidence >= 50:
            return 'yellow'
        else:
            return 'red'


class PredictionPointsGenerator:
    """
    Generate discrete prediction points (like Perplexity).
    
    Creates 3-5 key price targets with confidence levels:
    - 1 day target (highest confidence)
    - 5 day target
    - 1 month target
    - Support/Resistance levels (optional)
    """
    
    def __init__(self):
        self.confluence_engine = ConfluenceEngine()
        self.market_data = MarketDataClient()
    
    def generate_prediction_points(
        self,
        symbol: str,
        current_price: float,
        timeframes: List[str] = None
    ) -> List[PredictionPoint]:
        """
        Generate discrete prediction points for a symbol.
        
        Args:
            symbol: Stock ticker
            current_price: Current stock price
            timeframes: List of timeframes (default: ["1 day", "5 days", "1 month"])
        
        Returns:
            List of PredictionPoint objects
        """
        if timeframes is None:
            timeframes = ["1 day", "5 days", "1 month"]
        
        try:
            # Get current analysis
            result = self.confluence_engine.analyze(symbol)
            
            # Get confluence score (0-1, maps to bullish/bearish)
            confluence_score = result.confluence_score
            
            # Calculate expected return based on confluence
            # Higher confluence = more bullish
            # Assumes 15% annual return for strong signals
            annual_return = 0.15
            expected_return_multiplier = (confluence_score - 0.5) * 2  # -1 to +1
            
            points = []
            
            for timeframe in timeframes:
                # Convert timeframe to days
                days = self._timeframe_to_days(timeframe)
                
                # Calculate expected return for this timeframe
                daily_return = expected_return_multiplier * annual_return / 252
                expected_return = daily_return * days
                
                # Calculate target price
                target_price = current_price * (1 + expected_return)
                
                # Calculate confidence based on confluence score and agreement
                agreement = result.signal_agreement
                base_confidence = confluence_score * 100
                agreement_boost = agreement * 20  # Up to 20% boost for high agreement
                confidence = min(95, base_confidence + agreement_boost)
                
                # Simulate probability change (would come from historical tracking)
                # For now, use confluence trend
                if confluence_score > 0.6:
                    prob_change = np.random.uniform(0.5, 2.0)  # Positive change
                elif confluence_score < 0.4:
                    prob_change = np.random.uniform(-2.0, -0.5)  # Negative change
                else:
                    prob_change = np.random.uniform(-0.5, 0.5)  # Neutral
                
                # Determine direction
                direction = "↑" if target_price > current_price else "↓"
                
                point = PredictionPoint(
                    target_price=target_price,
                    confidence=confidence,
                    probability_change=prob_change,
                    timeframe=timeframe,
                    timestamp=datetime.utcnow(),
                    direction=direction
                )
                points.append(point)
            
            return points
            
        except Exception as e:
            logger.error(f"Failed to generate prediction points for {symbol}: {e}")
            # Return default points
            return self._get_default_points(current_price, timeframes)
    
    def _timeframe_to_days(self, timeframe: str) -> int:
        """Convert timeframe string to days."""
        timeframe_lower = timeframe.lower()
        if 'day' in timeframe_lower or '1d' in timeframe_lower:
            parts = timeframe_lower.split()
            if len(parts) > 1 and parts[0].isdigit():
                return int(parts[0])
            return 1
        elif 'week' in timeframe_lower or '1w' in timeframe_lower:
            parts = timeframe_lower.split()
            if len(parts) > 1 and parts[0].isdigit():
                return int(parts[0]) * 7
            return 7
        elif 'month' in timeframe_lower or '1m' in timeframe_lower:
            parts = timeframe_lower.split()
            if len(parts) > 1 and parts[0].isdigit():
                return int(parts[0]) * 30
            return 30
        elif 'year' in timeframe_lower or '1y' in timeframe_lower:
            return 252
        return 1
    
    def _get_default_points(
        self,
        current_price: float,
        timeframes: List[str]
    ) -> List[PredictionPoint]:
        """Get default prediction points when analysis fails."""
        points = []
        for timeframe in timeframes:
            days = self._timeframe_to_days(timeframe)
            # Default: small positive return
            target_price = current_price * (1 + 0.001 * days)
            points.append(PredictionPoint(
                target_price=target_price,
                confidence=50.0,
                probability_change=0.0,
                timeframe=timeframe,
                timestamp=datetime.utcnow(),
                direction="↑"
            ))
        return points
    
    def get_prediction_volume(self, symbol: str) -> Dict:
        """
        Get prediction volume/metrics (for display).
        
        Args:
            symbol: Stock ticker
        
        Returns:
            Dictionary with volume, model info, etc.
        """
        # This would ideally come from a database tracking predictions
        # For now, return placeholder
        return {
            'volume': '$1.2M',  # Would be actual volume if tracked
            'model': 'HybridSignalEnsemble v2.1',
            'prediction_count': 150,  # Number of predictions made
            'last_update': datetime.utcnow()
        }


def get_prediction_points(symbol: str, current_price: float) -> List[PredictionPoint]:
    """
    Get prediction points for a symbol (convenience function).
    
    Args:
        symbol: Stock ticker
        current_price: Current stock price
    
    Returns:
        List of PredictionPoint objects
    """
    generator = PredictionPointsGenerator()
    return generator.generate_prediction_points(symbol, current_price)
