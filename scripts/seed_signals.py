"""
Seed Database with Sample Trading Signals.

Creates 15+ realistic trading signals for demonstration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import random
import uuid

from src.data.models import TradeSignal
from src.data.persistence import get_database
from src.logging_config import get_logger

logger = get_logger(__name__)

# Sample trading signals data
SAMPLE_SIGNALS = [
    # Strong Buy Signals
    {
        "symbol": "AAPL",
        "signal_type": "STRONG_BUY",
        "technical_score": 0.88,
        "sentiment_score": 0.82,
        "ml_score": 0.85,
        "confluence_score": 0.85,
        "price_at_signal": 178.50,
        "var_95": 0.032,
        "cvar_95": 0.045,
        "max_drawdown": 0.08,
        "sharpe_ratio": 1.85,
        "suggested_position_size": 0.08,
        "risk_reward_ratio": 2.5,
        "technical_rationale": "RSI at 45 with bullish MACD crossover. Price above 50-day and 200-day SMA. Strong momentum with increasing volume.",
        "sentiment_rationale": "85% positive news sentiment. Recent earnings beat expectations by 12%. Analyst upgrades from Goldman Sachs and Morgan Stanley.",
        "risk_warning": "Tech sector volatility. Monitor Fed rate decisions.",
        "is_executed": True,
        "execution_price": 178.75,
        "realized_pnl": 425.50,
        "realized_pnl_pct": 4.75,
    },
    {
        "symbol": "NVDA",
        "signal_type": "STRONG_BUY",
        "technical_score": 0.92,
        "sentiment_score": 0.88,
        "ml_score": 0.90,
        "confluence_score": 0.90,
        "price_at_signal": 485.25,
        "var_95": 0.045,
        "cvar_95": 0.062,
        "max_drawdown": 0.12,
        "sharpe_ratio": 2.15,
        "suggested_position_size": 0.06,
        "risk_reward_ratio": 3.2,
        "technical_rationale": "Breakout above resistance at $480. RSI momentum strong at 62. MACD histogram expanding. ATR showing increasing volatility.",
        "sentiment_rationale": "AI demand surge driving record datacenter revenue. FinBERT shows 92% positive sentiment on recent news.",
        "risk_warning": "High valuation multiple. Semiconductor cycle risk.",
        "is_executed": True,
        "execution_price": 486.00,
        "realized_pnl": 1250.00,
        "realized_pnl_pct": 8.25,
    },
    # Buy Signals
    {
        "symbol": "MSFT",
        "signal_type": "BUY",
        "technical_score": 0.75,
        "sentiment_score": 0.72,
        "ml_score": 0.74,
        "confluence_score": 0.74,
        "price_at_signal": 378.90,
        "var_95": 0.028,
        "cvar_95": 0.038,
        "max_drawdown": 0.06,
        "sharpe_ratio": 1.65,
        "suggested_position_size": 0.07,
        "risk_reward_ratio": 2.1,
        "technical_rationale": "Price bouncing off 50-day SMA support. RSI at 48 showing neutral to bullish momentum. MFI indicates accumulation.",
        "sentiment_rationale": "Azure cloud growth exceeding expectations. Copilot AI integration boosting enterprise sales.",
        "risk_warning": "Enterprise spending slowdown risk.",
        "is_executed": False,
    },
    {
        "symbol": "GOOGL",
        "signal_type": "BUY",
        "technical_score": 0.71,
        "sentiment_score": 0.78,
        "ml_score": 0.73,
        "confluence_score": 0.74,
        "price_at_signal": 142.15,
        "var_95": 0.030,
        "cvar_95": 0.042,
        "max_drawdown": 0.07,
        "sharpe_ratio": 1.45,
        "suggested_position_size": 0.065,
        "risk_reward_ratio": 1.9,
        "technical_rationale": "Golden cross forming (50-day crossing above 200-day). Bollinger Bands tightening suggests impending move.",
        "sentiment_rationale": "Gemini AI launch receiving positive reviews. Search market share stable.",
        "risk_warning": "Regulatory pressure on advertising business.",
        "is_executed": True,
        "execution_price": 142.50,
        "realized_pnl": 315.00,
        "realized_pnl_pct": 3.50,
    },
    {
        "symbol": "AMZN",
        "signal_type": "BUY",
        "technical_score": 0.68,
        "sentiment_score": 0.75,
        "ml_score": 0.70,
        "confluence_score": 0.71,
        "price_at_signal": 178.25,
        "var_95": 0.035,
        "cvar_95": 0.048,
        "max_drawdown": 0.09,
        "sharpe_ratio": 1.52,
        "suggested_position_size": 0.055,
        "risk_reward_ratio": 1.8,
        "technical_rationale": "Breaking out of consolidation pattern. Volume surge 2.5x average. RSI at 58.",
        "sentiment_rationale": "AWS revenue acceleration. Prime membership at all-time high.",
        "risk_warning": "E-commerce margin pressure. Labor cost concerns.",
        "is_executed": False,
    },
    {
        "symbol": "META",
        "signal_type": "BUY",
        "technical_score": 0.73,
        "sentiment_score": 0.69,
        "ml_score": 0.72,
        "confluence_score": 0.71,
        "price_at_signal": 485.50,
        "var_95": 0.038,
        "cvar_95": 0.052,
        "max_drawdown": 0.10,
        "sharpe_ratio": 1.38,
        "suggested_position_size": 0.05,
        "risk_reward_ratio": 1.75,
        "technical_rationale": "Ascending triangle pattern. Price above all key moving averages. ADX at 28 showing trend strength.",
        "sentiment_rationale": "Reels monetization improving. Cost efficiency measures paying off.",
        "risk_warning": "Metaverse investments remain uncertain. Regulatory risks.",
        "is_executed": True,
        "execution_price": 486.00,
        "realized_pnl": 540.00,
        "realized_pnl_pct": 5.50,
    },
    # Hold Signals
    {
        "symbol": "TSLA",
        "signal_type": "HOLD",
        "technical_score": 0.52,
        "sentiment_score": 0.48,
        "ml_score": 0.50,
        "confluence_score": 0.50,
        "price_at_signal": 245.75,
        "var_95": 0.055,
        "cvar_95": 0.075,
        "max_drawdown": 0.15,
        "sharpe_ratio": 0.85,
        "suggested_position_size": 0.03,
        "risk_reward_ratio": 1.0,
        "technical_rationale": "Range-bound between $230-$260. RSI neutral at 50. No clear trend direction.",
        "sentiment_rationale": "Mixed sentiment on EV market competition. Cybertruck production ramping.",
        "risk_warning": "High volatility. Competition intensifying from legacy automakers.",
        "is_executed": False,
    },
    {
        "symbol": "JPM",
        "signal_type": "HOLD",
        "technical_score": 0.55,
        "sentiment_score": 0.52,
        "ml_score": 0.53,
        "confluence_score": 0.53,
        "price_at_signal": 195.80,
        "var_95": 0.025,
        "cvar_95": 0.035,
        "max_drawdown": 0.05,
        "sharpe_ratio": 1.05,
        "suggested_position_size": 0.04,
        "risk_reward_ratio": 1.1,
        "technical_rationale": "Trading in tight range. Awaiting next earnings for direction.",
        "sentiment_rationale": "Interest rate environment uncertain. Credit quality stable.",
        "risk_warning": "Bank stress concerns persist in market.",
        "is_executed": False,
    },
    # Sell Signals
    {
        "symbol": "BA",
        "signal_type": "SELL",
        "technical_score": 0.35,
        "sentiment_score": 0.28,
        "ml_score": 0.32,
        "confluence_score": 0.32,
        "price_at_signal": 175.25,
        "var_95": 0.048,
        "cvar_95": 0.065,
        "max_drawdown": 0.14,
        "sharpe_ratio": 0.45,
        "suggested_position_size": 0.02,
        "risk_reward_ratio": 0.6,
        "technical_rationale": "Death cross formed. Price below all major moving averages. RSI at 38 and falling.",
        "sentiment_rationale": "Quality control issues dominating headlines. FAA investigations ongoing.",
        "risk_warning": "Significant operational headwinds. Cash burn concerns.",
        "is_executed": True,
        "execution_price": 174.50,
        "realized_pnl": 125.00,
        "realized_pnl_pct": 2.25,
    },
    {
        "symbol": "INTC",
        "signal_type": "SELL",
        "technical_score": 0.32,
        "sentiment_score": 0.35,
        "ml_score": 0.33,
        "confluence_score": 0.33,
        "price_at_signal": 21.45,
        "var_95": 0.052,
        "cvar_95": 0.072,
        "max_drawdown": 0.18,
        "sharpe_ratio": 0.35,
        "suggested_position_size": 0.02,
        "risk_reward_ratio": 0.5,
        "technical_rationale": "Downtrend intact. Lower highs and lower lows pattern. Volume confirming weakness.",
        "sentiment_rationale": "Market share losses to AMD and NVIDIA. Foundry business struggles.",
        "risk_warning": "Turnaround plan execution risk. High capex requirements.",
        "is_executed": False,
    },
    # Strong Sell Signals
    {
        "symbol": "COIN",
        "signal_type": "STRONG_SELL",
        "technical_score": 0.22,
        "sentiment_score": 0.25,
        "ml_score": 0.23,
        "confluence_score": 0.23,
        "price_at_signal": 155.50,
        "var_95": 0.085,
        "cvar_95": 0.115,
        "max_drawdown": 0.25,
        "sharpe_ratio": 0.15,
        "suggested_position_size": 0.01,
        "risk_reward_ratio": 0.3,
        "technical_rationale": "Breaking down through key support at $160. RSI oversold but momentum negative. MACD bearish crossover.",
        "sentiment_rationale": "SEC lawsuit concerns. Trading volume declining. Regulatory uncertainty.",
        "risk_warning": "High correlation to crypto market. Regulatory existential risk.",
        "is_executed": True,
        "execution_price": 154.00,
        "realized_pnl": 450.00,
        "realized_pnl_pct": 6.50,
    },
    # More diverse signals
    {
        "symbol": "V",
        "signal_type": "BUY",
        "technical_score": 0.76,
        "sentiment_score": 0.74,
        "ml_score": 0.75,
        "confluence_score": 0.75,
        "price_at_signal": 278.90,
        "var_95": 0.022,
        "cvar_95": 0.030,
        "max_drawdown": 0.04,
        "sharpe_ratio": 1.75,
        "suggested_position_size": 0.07,
        "risk_reward_ratio": 2.2,
        "technical_rationale": "Steady uptrend with higher highs. Price pulling back to 20-day SMA support.",
        "sentiment_rationale": "Strong consumer spending trends. Cross-border travel recovery boosting volumes.",
        "risk_warning": "Credit card fee regulation risk.",
        "is_executed": False,
    },
    {
        "symbol": "UNH",
        "signal_type": "STRONG_BUY",
        "technical_score": 0.84,
        "sentiment_score": 0.79,
        "ml_score": 0.82,
        "confluence_score": 0.82,
        "price_at_signal": 515.25,
        "var_95": 0.020,
        "cvar_95": 0.028,
        "max_drawdown": 0.05,
        "sharpe_ratio": 1.95,
        "suggested_position_size": 0.08,
        "risk_reward_ratio": 2.8,
        "technical_rationale": "Breakout to all-time highs. Strong volume confirmation. RSI at 65 with room to run.",
        "sentiment_rationale": "Healthcare demand robust. Optum growth accelerating.",
        "risk_warning": "Medicare advantage rate cuts possible.",
        "is_executed": True,
        "execution_price": 516.00,
        "realized_pnl": 825.00,
        "realized_pnl_pct": 4.25,
    },
    {
        "symbol": "DIS",
        "signal_type": "HOLD",
        "technical_score": 0.48,
        "sentiment_score": 0.45,
        "ml_score": 0.47,
        "confluence_score": 0.47,
        "price_at_signal": 92.50,
        "var_95": 0.035,
        "cvar_95": 0.048,
        "max_drawdown": 0.10,
        "sharpe_ratio": 0.75,
        "suggested_position_size": 0.03,
        "risk_reward_ratio": 0.9,
        "technical_rationale": "Trading sideways in $85-$100 range. No clear breakout direction.",
        "sentiment_rationale": "Streaming losses narrowing but competition intense. Parks segment strong.",
        "risk_warning": "Content spending pressure. Cord-cutting headwind.",
        "is_executed": False,
    },
    {
        "symbol": "COST",
        "signal_type": "BUY",
        "technical_score": 0.78,
        "sentiment_score": 0.81,
        "ml_score": 0.79,
        "confluence_score": 0.79,
        "price_at_signal": 725.50,
        "var_95": 0.018,
        "cvar_95": 0.025,
        "max_drawdown": 0.04,
        "sharpe_ratio": 1.88,
        "suggested_position_size": 0.075,
        "risk_reward_ratio": 2.4,
        "technical_rationale": "Consistent uptrend. New 52-week highs with strong breadth. Low volatility.",
        "sentiment_rationale": "Membership renewal rates at 92%. Value proposition resonating with consumers.",
        "risk_warning": "Premium valuation. Competition from Walmart+.",
        "is_executed": True,
        "execution_price": 726.00,
        "realized_pnl": 580.00,
        "realized_pnl_pct": 3.75,
    },
    {
        "symbol": "AMD",
        "signal_type": "BUY",
        "technical_score": 0.72,
        "sentiment_score": 0.76,
        "ml_score": 0.74,
        "confluence_score": 0.74,
        "price_at_signal": 152.75,
        "var_95": 0.042,
        "cvar_95": 0.058,
        "max_drawdown": 0.11,
        "sharpe_ratio": 1.55,
        "suggested_position_size": 0.055,
        "risk_reward_ratio": 2.0,
        "technical_rationale": "Cup and handle pattern forming. Volume surge on recent rally. RSI at 58.",
        "sentiment_rationale": "AI chip competition heating up. Data center wins from Intel.",
        "risk_warning": "NVIDIA dominance in AI. PC market cyclicality.",
        "is_executed": False,
    },
]


def seed_signals(user_id: str = None):
    """Insert sample trading signals into database."""
    db = get_database()
    
    with db.get_session() as session:
        # Get user if email provided
        if user_id is None:
            # Try to find a test user
            from src.auth.models import User
            user = session.query(User).filter(User.email == "testuser@example.com").first()
            if user:
                user_id = user.id
        
        signals_added = 0
        for i, signal_data in enumerate(SAMPLE_SIGNALS):
            # Create varied timestamps over the past 7 days
            hours_ago = random.randint(1, 168)  # Up to 7 days
            created_at = datetime.utcnow() - timedelta(hours=hours_ago)
            
            # If executed, set execution date
            execution_date = None
            close_date = None
            if signal_data.get("is_executed"):
                execution_date = created_at + timedelta(minutes=random.randint(5, 60))
                if signal_data.get("realized_pnl") is not None:
                    close_date = execution_date + timedelta(hours=random.randint(2, 48))
            
            signal = TradeSignal(
                id=uuid.uuid4(),
                created_at=created_at,
                user_id=user_id,
                symbol=signal_data["symbol"],
                signal_type=signal_data["signal_type"],
                technical_score=signal_data["technical_score"],
                sentiment_score=signal_data["sentiment_score"],
                ml_score=signal_data["ml_score"],
                confluence_score=signal_data["confluence_score"],
                price_at_signal=signal_data["price_at_signal"],
                var_95=signal_data.get("var_95"),
                cvar_95=signal_data.get("cvar_95"),
                max_drawdown=signal_data.get("max_drawdown"),
                sharpe_ratio=signal_data.get("sharpe_ratio"),
                suggested_position_size=signal_data.get("suggested_position_size"),
                risk_reward_ratio=signal_data.get("risk_reward_ratio"),
                technical_rationale=signal_data.get("technical_rationale"),
                sentiment_rationale=signal_data.get("sentiment_rationale"),
                risk_warning=signal_data.get("risk_warning"),
                is_executed=signal_data.get("is_executed", False),
                execution_price=signal_data.get("execution_price"),
                execution_date=execution_date,
                close_price=signal_data.get("execution_price", 0) * (1 + signal_data.get("realized_pnl_pct", 0) / 100) if signal_data.get("is_executed") else None,
                close_date=close_date,
                realized_pnl=signal_data.get("realized_pnl"),
                realized_pnl_pct=signal_data.get("realized_pnl_pct"),
            )
            session.add(signal)
            signals_added += 1
            logger.info(f"Added signal: {signal.symbol} {signal.signal_type}")
        
        session.commit()
        logger.info(f"✓ Added {signals_added} sample trading signals")
        print(f"✓ Added {signals_added} sample trading signals to database")
    
    return signals_added


if __name__ == "__main__":
    seed_signals()
