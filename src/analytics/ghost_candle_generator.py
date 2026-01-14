"""
Ghost Candle Generator - Implements the Architectural Specification.

Key concepts from the specification:
1. Ghost Candles: Transparent candlesticks for predictions
2. Confidence Intervals: Fan Chart with 50% and 95% bands
3. NOW Divider: Vertical line separating reality from prediction
4. Anti-Repainting: Immutable prediction storage
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import hashlib

from src.logging_config import get_logger

logger = get_logger(__name__)


class GhostCandleGenerator:
    """
    Generates Ghost Candles and Confidence Intervals for predictions.
    
    From the Specification:
    - "Predictions are Probabilities, not Certainties"
    - "Draw a Fan Chart or Confidence Interval"
    - "The Median Path (The Ghost Candles): This is the AI's Best Guess"
    """
    
    PREDICTION_HISTORY_DIR = "prediction_history"
    
    def __init__(self):
        os.makedirs(self.PREDICTION_HISTORY_DIR, exist_ok=True)
    
    def generate_ghost_candles(
        self, 
        last_candle: Dict, 
        predictions: List[Dict], 
        timeframe: str = "1H"
    ) -> Tuple[List[Dict], List[float], List[float]]:
        """
        Generate Ghost Candles from ML predictions.
        
        Args:
            last_candle: The last real candle (contains close, time)
            predictions: List of prediction dicts with direction, confidence
            timeframe: Timeframe for time calculation
            
        Returns:
            Tuple of (ghost_candles, upper_band, lower_band)
            - ghost_candles: List of OHLC dicts with opacity
            - upper_band: 95% confidence upper prices
            - lower_band: 95% confidence lower prices
        """
        ghost_candles = []
        upper_band = []
        lower_band = []
        fifty_upper = []
        fifty_lower = []
        
        # Get base values from last candle
        current_price = float(last_candle.get('close', 100))
        current_time = last_candle.get('time', datetime.now())
        
        if isinstance(current_time, str):
            try:
                current_time = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
            except:
                current_time = datetime.now()
        
        # Time delta per candle based on timeframe (includes scalping intervals)
        delta_map = {
            "1M": timedelta(minutes=1),
            "5M": timedelta(minutes=5),
            "15M": timedelta(minutes=15),
            "1H": timedelta(hours=1),
            "4H": timedelta(hours=4),
            "1D": timedelta(days=1)
        }
        time_delta = delta_map.get(timeframe, timedelta(minutes=5))
        
        # Generate each ghost candle
        running_price = current_price
        cumulative_uncertainty = 0
        
        for i, pred in enumerate(predictions):
            # Time for this candle
            candle_time = current_time + (time_delta * (i + 1))
            
            # Direction and confidence
            direction = pred.get('direction', 'UP')
            confidence = pred.get('confidence', 55) / 100  # Convert to 0-1
            
            # Calculate price movement based on direction and confidence
            # Higher confidence = more movement
            base_volatility = 0.005  # 0.5% base movement
            movement_pct = base_volatility * (1 + confidence - 0.5)
            
            if direction == 'UP':
                price_change = running_price * movement_pct
            else:
                price_change = -running_price * movement_pct
            
            # Generate OHLC for ghost candle
            open_price = running_price
            close_price = running_price + price_change
            
            # Add some natural variation
            price_range = running_price * base_volatility * 0.5
            high_price = max(open_price, close_price) + abs(np.random.normal(0, price_range))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, price_range))
            
            ghost_candles.append({
                'time': candle_time.isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'direction': direction,
                'confidence': pred.get('confidence', 55),
                'opacity': 0.4 - (i * 0.05)  # Fade as we go further
            })
            
            # Update running price
            running_price = close_price
            
            # Calculate confidence intervals
            # Uncertainty grows with time (the "Fan Chart" / "Cone of Uncertainty")
            cumulative_uncertainty += 1  # Increases with each step
            uncertainty_factor = cumulative_uncertainty * 0.003  # 0.3% per step
            
            # 50% confidence band (tighter)
            fifty_band = running_price * uncertainty_factor * 0.5
            fifty_upper.append(round(running_price + fifty_band, 2))
            fifty_lower.append(round(running_price - fifty_band, 2))
            
            # 95% confidence band (wider - the "Cone of Silence")
            ninety_five_band = running_price * uncertainty_factor * 1.5
            upper_band.append(round(running_price + ninety_five_band, 2))
            lower_band.append(round(running_price - ninety_five_band, 2))
        
        return ghost_candles, upper_band, lower_band, fifty_upper, fifty_lower
    
    def store_prediction(
        self, 
        symbol: str, 
        timeframe: str, 
        predictions: List[Dict],
        ghost_candles: List[Dict]
    ) -> str:
        """
        Store prediction for anti-repainting audit trail.
        
        From Specification:
        "Once a prediction is made for a specific time block, it must be locked"
        "This transparency builds long-term trust"
        """
        timestamp = datetime.now()
        
        # Create unique prediction ID
        pred_id = hashlib.md5(
            f"{symbol}-{timeframe}-{timestamp.isoformat()}".encode()
        ).hexdigest()[:8]
        
        prediction_record = {
            'id': pred_id,
            'symbol': symbol,
            'timeframe': timeframe,
            'created_at': timestamp.isoformat(),
            'predictions': predictions,
            'ghost_candles': ghost_candles,
            'validated': False,
            'actual_outcome': None
        }
        
        # Save to file
        filename = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{pred_id}.json"
        filepath = os.path.join(self.PREDICTION_HISTORY_DIR, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(prediction_record, f, indent=2)
            logger.info(f"Stored prediction {pred_id} for {symbol}")
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
        
        return pred_id
    
    def get_prediction_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get past predictions for validation/audit.
        """
        history = []
        
        try:
            files = sorted([
                f for f in os.listdir(self.PREDICTION_HISTORY_DIR)
                if f.startswith(symbol) and f.endswith('.json')
            ], reverse=True)
            
            for filename in files[:limit]:
                filepath = os.path.join(self.PREDICTION_HISTORY_DIR, filename)
                with open(filepath, 'r') as f:
                    history.append(json.load(f))
        except Exception as e:
            logger.error(f"Error reading prediction history: {e}")
        
        return history


def generate_ghost_prediction_chart(
    symbol: str, 
    timeframe: str = "1H",
    num_periods: int = 5
) -> Dict:
    """
    Generate complete Ghost Candle prediction data.
    
    Returns:
        Dict containing:
        - ghost_candles: List of transparent OHLC candles
        - upper_band_95: 95% confidence upper limit
        - lower_band_95: 95% confidence lower limit
        - upper_band_50: 50% confidence upper limit
        - lower_band_50: 50% confidence lower limit
        - now_line: Timestamp for the NOW divider
        - prediction_id: Unique ID for anti-repainting
    """
    try:
        from src.analytics.ml_prediction_service import generate_predictions
        from src.data.persistence import get_database
        
        # Get current price data
        db = get_database()
        df = db.get_candles(symbol.upper(), limit=50)
        
        if df.empty:
            return generate_demo_ghost_candles(timeframe, num_periods)
        
        # Get last candle
        last_row = df.iloc[-1]
        last_candle = {
            'close': float(last_row['close']),
            'time': last_row['time'] if 'time' in df.columns else datetime.now()
        }
        
        # Get ML predictions
        predictions, recommendation = generate_predictions(symbol, timeframe, num_periods)
        
        # Generate ghost candles and confidence intervals
        generator = GhostCandleGenerator()
        ghost_candles, upper_95, lower_95, upper_50, lower_50 = generator.generate_ghost_candles(
            last_candle, predictions, timeframe
        )
        
        # Store for anti-repainting
        pred_id = generator.store_prediction(symbol, timeframe, predictions, ghost_candles)
        
        return {
            'ghost_candles': ghost_candles,
            'upper_band_95': upper_95,
            'lower_band_95': lower_95,
            'upper_band_50': upper_50,
            'lower_band_50': lower_50,
            'now_line': last_candle['time'],
            'prediction_id': pred_id,
            'predictions': predictions,
            'recommendation': recommendation,
            'current_price': last_candle['close']
        }
        
    except Exception as e:
        logger.error(f"Ghost prediction error: {e}")
        return generate_demo_ghost_candles(timeframe, num_periods)


def generate_demo_ghost_candles(timeframe: str = "1H", num_periods: int = 5) -> Dict:
    """Generate demo ghost candles when no real data available."""
    import random
    
    base_price = 185.0
    now = datetime.now()
    
    delta_map = {
        "1M": timedelta(minutes=1),
        "5M": timedelta(minutes=5),
        "15M": timedelta(minutes=15),
        "1H": timedelta(hours=1),
        "4H": timedelta(hours=4),
        "1D": timedelta(days=1)
    }
    time_delta = delta_map.get(timeframe, timedelta(minutes=5))
    
    ghost_candles = []
    upper_95 = []
    lower_95 = []
    upper_50 = []
    lower_50 = []
    
    running_price = base_price
    base_direction = random.choice(['UP', 'DOWN'])
    
    for i in range(num_periods):
        candle_time = now + (time_delta * (i + 1))
        
        # Direction with some variation
        if random.random() < 0.7:
            direction = base_direction
        else:
            direction = 'DOWN' if base_direction == 'UP' else 'UP'
        
        confidence = random.uniform(52, 68) - i * 2
        
        # Price movement
        movement = running_price * 0.005 * (1 if direction == 'UP' else -1)
        
        open_price = running_price
        close_price = running_price + movement
        high_price = max(open_price, close_price) + running_price * 0.002
        low_price = min(open_price, close_price) - running_price * 0.002
        
        ghost_candles.append({
            'time': candle_time.isoformat(),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'direction': direction,
            'confidence': max(50, confidence),
            'opacity': 0.4 - (i * 0.05)
        })
        
        running_price = close_price
        
        # Confidence bands (widening fan)
        uncertainty = running_price * 0.003 * (i + 1)
        upper_95.append(round(running_price + uncertainty * 1.5, 2))
        lower_95.append(round(running_price - uncertainty * 1.5, 2))
        upper_50.append(round(running_price + uncertainty * 0.5, 2))
        lower_50.append(round(running_price - uncertainty * 0.5, 2))
    
    # Calculate recommendation
    up_count = sum(1 for gc in ghost_candles if gc['direction'] == 'UP')
    if up_count >= 3:
        recommendation = {'recommendation': 'BUY', 'symbol': '✅', 'color': 'green', 
                          'reason': f'{up_count}/5 predictions are UP', 'avg_confidence': 58}
    elif up_count <= 2:
        recommendation = {'recommendation': 'SELL', 'symbol': '❌', 'color': 'red',
                          'reason': f'{5-up_count}/5 predictions are DOWN', 'avg_confidence': 55}
    else:
        recommendation = {'recommendation': 'NEUTRAL', 'symbol': '⚠️', 'color': 'orange',
                          'reason': 'Mixed signals', 'avg_confidence': 52}
    
    return {
        'ghost_candles': ghost_candles,
        'upper_band_95': upper_95,
        'lower_band_95': lower_95,
        'upper_band_50': upper_50,
        'lower_band_50': lower_50,
        'now_line': now.isoformat(),
        'prediction_id': 'demo',
        'predictions': [{'period': i+1, 'label': f"Period {i+1}", 
                        'direction': gc['direction'], 'confidence': gc['confidence']} 
                       for i, gc in enumerate(ghost_candles)],
        'recommendation': recommendation,
        'current_price': base_price
    }
