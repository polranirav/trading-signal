"""
Backtesting Strategy Definitions.

Provides pre-built strategies and signal-based strategy wrapper
for the backtesting dashboard.
"""

from typing import Callable, List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.analytics.backtesting import Trade, SignalType
from src.analytics.technical import TechnicalAnalyzer
from src.logging_config import get_logger

logger = get_logger(__name__)


def create_signal_based_strategy(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    min_confidence: float = 0.5
) -> Callable:
    """
    Create a strategy based on actual signals from the database.
    
    Fetches signals from the History page and executes trades
    based on signal timing and confidence levels.
    
    Args:
        symbol: Stock symbol to backtest
        start_date: Start of backtest period
        end_date: End of backtest period
        min_confidence: Minimum confluence score to execute signal
    
    Returns:
        Strategy function compatible with WalkForwardBacktester
    """
    from src.data.persistence import get_database
    
    def strategy(train_data: pd.DataFrame, test_data: pd.DataFrame) -> List[Trade]:
        """Execute trades based on stored signals."""
        db = get_database()
        trades = []
        
        try:
            # Fetch signals for this symbol in the test period
            signals = db.get_latest_signals(limit=1000)
            relevant_signals = [
                s for s in signals
                if s.symbol == symbol
                and s.created_at
                and s.created_at >= test_data['time'].min()
                and s.created_at <= test_data['time'].max()
                and (s.confluence_score or 0) >= min_confidence
            ]
            
            # Sort by date
            relevant_signals.sort(key=lambda s: s.created_at)
            
            # Execute trades based on signals
            position = None
            for signal in relevant_signals:
                # Find closest price data point
                signal_time = signal.created_at
                price_row = test_data[test_data['time'] <= signal_time]
                
                if len(price_row) == 0:
                    continue
                
                price_row = price_row.iloc[-1]
                price = price_row['close']
                
                # Determine signal type
                if signal.signal_type in ["STRONG_BUY", "BUY"]:
                    if position is None:
                        # Enter long position
                        position = Trade(
                            symbol=symbol,
                            entry_date=signal_time,
                            entry_price=price,
                            signal_type=SignalType.BUY if signal.signal_type == "BUY" else SignalType.STRONG_BUY,
                            position_size=signal.suggested_position_size or 1.0
                        )
                elif signal.signal_type in ["STRONG_SELL", "SELL"]:
                    if position is not None and position.exit_price is None:
                        # Exit position
                        position.exit_date = signal_time
                        position.exit_price = price
                        trades.append(position)
                        position = None
                elif signal.signal_type == "HOLD":
                    # Hold - no action, but could exit if already in position
                    pass
            
            # Close any open position at end of test period
            if position is not None and position.exit_price is None:
                last_price = test_data.iloc[-1]['close']
                position.exit_date = test_data.iloc[-1]['time']
                position.exit_price = last_price
                trades.append(position)
                
        except Exception as e:
            logger.error(f"Error in signal-based strategy: {e}", exc_info=True)
        
        return trades
    
    return strategy


def create_rsi_strategy(
    oversold: int = 30,
    overbought: int = 70,
    stop_loss: float = 0.05,
    take_profit: float = 0.10,
    hold_days: int = 20
) -> Callable:
    """
    Create RSI Mean Reversion strategy.
    
    Entry: RSI < oversold (buy oversold)
    Exit: RSI > overbought OR stop-loss OR take-profit OR time-based
    
    Args:
        oversold: RSI level to trigger buy (default 30)
        overbought: RSI level to trigger sell (default 70)
        stop_loss: Stop loss percentage (default 5%)
        take_profit: Take profit percentage (default 10%)
        hold_days: Maximum days to hold (default 20)
    
    Returns:
        Strategy function
    """
    def strategy(train_data: pd.DataFrame, test_data: pd.DataFrame) -> List[Trade]:
        """RSI mean reversion strategy."""
        analyzer = TechnicalAnalyzer()
        trades = []
        
        # Compute indicators
        test_with_indicators = analyzer.compute_all(test_data)
        
        position = None
        for idx in range(14, len(test_with_indicators)):
            row = test_with_indicators.iloc[idx]
            current_price = row['close']
            current_time = row['time']
            
            # Get RSI
            rsi = row.get('rsi')
            if pd.isna(rsi) or rsi is None:
                continue
            
            # Entry logic: RSI oversold
            if position is None:
                if rsi < oversold:
                    position = Trade(
                        symbol=test_data.iloc[0]['symbol'] if 'symbol' in test_data.columns else "UNKNOWN",
                        entry_date=current_time,
                        entry_price=current_price,
                        signal_type=SignalType.BUY,
                        position_size=1.0
                    )
            
            # Exit logic
            elif position.exit_price is None:
                # Time-based exit
                days_held = (current_time - position.entry_date).days
                if days_held >= hold_days:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
                    continue
                
                # Stop loss
                price_change = (current_price - position.entry_price) / position.entry_price
                if price_change <= -stop_loss:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
                    continue
                
                # Take profit
                if price_change >= take_profit:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
                    continue
                
                # RSI overbought exit
                if rsi > overbought:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
        
        # Close any open position
        if position is not None and position.exit_price is None:
            last_row = test_with_indicators.iloc[-1]
            position.exit_date = last_row['time']
            position.exit_price = last_row['close']
            trades.append(position)
        
        return trades
    
    return strategy


def create_ma_crossover_strategy(
    fast_period: int = 50,
    slow_period: int = 200,
    stop_loss: float = 0.05,
    take_profit: float = 0.15,
    hold_days: int = 30
) -> Callable:
    """
    Create Moving Average Crossover strategy.
    
    Entry: Fast MA crosses above Slow MA (golden cross)
    Exit: Fast MA crosses below Slow MA (death cross) OR stop-loss/take-profit
    
    Args:
        fast_period: Fast MA period (default 50)
        slow_period: Slow MA period (default 200)
        stop_loss: Stop loss percentage (default 5%)
        take_profit: Take profit percentage (default 15%)
        hold_days: Maximum days to hold (default 30)
    
    Returns:
        Strategy function
    """
    def strategy(train_data: pd.DataFrame, test_data: pd.DataFrame) -> List[Trade]:
        """MA crossover strategy."""
        analyzer = TechnicalAnalyzer()
        trades = []
        
        # Compute indicators
        test_with_indicators = analyzer.compute_all(test_data)
        
        # Calculate MAs
        test_with_indicators['ma_fast'] = test_with_indicators['close'].rolling(fast_period).mean()
        test_with_indicators['ma_slow'] = test_with_indicators['close'].rolling(slow_period).mean()
        
        position = None
        prev_fast_above = False
        
        for idx in range(slow_period, len(test_with_indicators)):
            row = test_with_indicators.iloc[idx]
            prev_row = test_with_indicators.iloc[idx - 1] if idx > 0 else row
            
            current_price = row['close']
            current_time = row['time']
            fast_ma = row['ma_fast']
            slow_ma = row['ma_slow']
            
            if pd.isna(fast_ma) or pd.isna(slow_ma):
                continue
            
            fast_above_slow = fast_ma > slow_ma
            prev_fast_above_slow = prev_row['ma_fast'] > prev_row['ma_slow'] if not pd.isna(prev_row['ma_fast']) and not pd.isna(prev_row['ma_slow']) else False
            
            # Entry: Golden cross (fast crosses above slow)
            if position is None:
                if fast_above_slow and not prev_fast_above_slow:
                    position = Trade(
                        symbol=test_data.iloc[0]['symbol'] if 'symbol' in test_data.columns else "UNKNOWN",
                        entry_date=current_time,
                        entry_price=current_price,
                        signal_type=SignalType.BUY,
                        position_size=1.0
                    )
                    prev_fast_above = True
            
            # Exit logic
            elif position.exit_price is None:
                # Time-based exit
                days_held = (current_time - position.entry_date).days
                if days_held >= hold_days:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
                    prev_fast_above = False
                    continue
                
                # Stop loss
                price_change = (current_price - position.entry_price) / position.entry_price
                if price_change <= -stop_loss:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
                    prev_fast_above = False
                    continue
                
                # Take profit
                if price_change >= take_profit:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
                    prev_fast_above = False
                    continue
                
                # Death cross exit
                if not fast_above_slow and prev_fast_above_slow:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
                    prev_fast_above = False
        
        # Close any open position
        if position is not None and position.exit_price is None:
            last_row = test_with_indicators.iloc[-1]
            position.exit_date = last_row['time']
            position.exit_price = last_row['close']
            trades.append(position)
        
        return trades
    
    return strategy


def create_momentum_strategy(
    lookback: int = 20,
    momentum_threshold: float = 0.05,
    stop_loss: float = 0.05,
    take_profit: float = 0.12,
    hold_days: int = 25
) -> Callable:
    """
    Create Momentum/Trend Following strategy.
    
    Entry: Strong upward momentum (price change > threshold)
    Exit: Momentum reversal OR stop-loss/take-profit
    
    Args:
        lookback: Days to calculate momentum (default 20)
        momentum_threshold: Minimum momentum to enter (default 5%)
        stop_loss: Stop loss percentage (default 5%)
        take_profit: Take profit percentage (default 12%)
        hold_days: Maximum days to hold (default 25)
    
    Returns:
        Strategy function
    """
    def strategy(train_data: pd.DataFrame, test_data: pd.DataFrame) -> List[Trade]:
        """Momentum strategy."""
        analyzer = TechnicalAnalyzer()
        trades = []
        
        # Compute indicators
        test_with_indicators = analyzer.compute_all(test_data)
        
        # Calculate momentum
        test_with_indicators['momentum'] = test_with_indicators['close'].pct_change(lookback)
        test_with_indicators['momentum_signal'] = test_with_indicators['momentum'] > momentum_threshold
        
        position = None
        for idx in range(lookback, len(test_with_indicators)):
            row = test_with_indicators.iloc[idx]
            current_price = row['close']
            current_time = row['time']
            momentum = row.get('momentum', 0)
            momentum_signal = row.get('momentum_signal', False)
            
            if pd.isna(momentum):
                continue
            
            # Entry logic: Strong momentum
            if position is None:
                if momentum_signal and momentum > momentum_threshold:
                    position = Trade(
                        symbol=test_data.iloc[0]['symbol'] if 'symbol' in test_data.columns else "UNKNOWN",
                        entry_date=current_time,
                        entry_price=current_price,
                        signal_type=SignalType.BUY,
                        position_size=1.0
                    )
            
            # Exit logic
            elif position.exit_price is None:
                # Time-based exit
                days_held = (current_time - position.entry_date).days
                if days_held >= hold_days:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
                    continue
                
                # Stop loss
                price_change = (current_price - position.entry_price) / position.entry_price
                if price_change <= -stop_loss:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
                    continue
                
                # Take profit
                if price_change >= take_profit:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
                    continue
                
                # Momentum reversal
                if momentum < 0 or not momentum_signal:
                    position.exit_date = current_time
                    position.exit_price = current_price
                    trades.append(position)
                    position = None
        
        # Close any open position
        if position is not None and position.exit_price is None:
            last_row = test_with_indicators.iloc[-1]
            position.exit_date = last_row['time']
            position.exit_price = last_row['close']
            trades.append(position)
        
        return trades
    
    return strategy


def create_strategy_from_config(config: Dict) -> Callable:
    """
    Create strategy function from user configuration.
    
    Maps UI inputs to strategy parameters and returns
    the appropriate strategy function.
    
    Args:
        config: Dictionary with strategy configuration
            - strategy_type: "my_signals", "rsi", "ma_crossover", "momentum"
            - symbol: Stock symbol (for signal-based)
            - start_date: Start date (for signal-based)
            - end_date: End date (for signal-based)
            - All other strategy-specific parameters
    
    Returns:
        Strategy function compatible with WalkForwardBacktester
    """
    strategy_type = config.get('strategy_type', 'rsi')
    
    if strategy_type == 'my_signals':
        return create_signal_based_strategy(
            symbol=config.get('symbol', 'AAPL'),
            start_date=config.get('start_date'),
            end_date=config.get('end_date'),
            min_confidence=config.get('min_confidence', 0.5)
        )
    elif strategy_type == 'rsi':
        return create_rsi_strategy(
            oversold=config.get('oversold', 30),
            overbought=config.get('overbought', 70),
            stop_loss=config.get('stop_loss', 0.05),
            take_profit=config.get('take_profit', 0.10),
            hold_days=config.get('hold_days', 20)
        )
    elif strategy_type == 'ma_crossover':
        return create_ma_crossover_strategy(
            fast_period=config.get('fast_period', 50),
            slow_period=config.get('slow_period', 200),
            stop_loss=config.get('stop_loss', 0.05),
            take_profit=config.get('take_profit', 0.15),
            hold_days=config.get('hold_days', 30)
        )
    elif strategy_type == 'momentum':
        return create_momentum_strategy(
            lookback=config.get('lookback', 20),
            momentum_threshold=config.get('momentum_threshold', 0.05),
            stop_loss=config.get('stop_loss', 0.05),
            take_profit=config.get('take_profit', 0.12),
            hold_days=config.get('hold_days', 25)
        )
    else:
        logger.warning(f"Unknown strategy type: {strategy_type}, defaulting to RSI")
        return create_rsi_strategy()
