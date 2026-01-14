"""
Signals API Endpoints.

Handles trading signal retrieval and filtering.
"""

from flask import Blueprint, request, g
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from src.api.utils import success_response, error_response, require_api_key_or_auth
from src.data.persistence import get_database
from src.logging_config import get_logger

logger = get_logger(__name__)

signals_bp = Blueprint('signals', __name__)


@signals_bp.route('/signals', methods=['GET'])
@require_api_key_or_auth
def get_signals():
    """
    Get latest trading signals.
    
    Query parameters:
    - limit: Number of signals (default: 20, max: 100)
    - symbol: Filter by symbol (optional)
    - min_confidence: Minimum confluence score 0-1 (optional)
    - signal_type: Filter by signal type (BUY, SELL, etc.) (optional)
    - days: Number of days to look back (default: 30) (optional)
    """
    try:
        user = g.current_user
        
        # Get query parameters
        limit = min(int(request.args.get('limit', 20)), 100)
        symbol = request.args.get('symbol', '').strip().upper()
        min_confidence = request.args.get('min_confidence')
        signal_type = request.args.get('signal_type', '').strip()
        days = int(request.args.get('days', 30))
        
        # Parse min_confidence
        min_conf = float(min_confidence) if min_confidence else None
        
        # Get signals (show all signals for demo - in production, filter by user portfolio)
        db = get_database()
        signals = db.get_latest_signals(
            limit=limit,
            min_confidence=min_conf,
            user_id=None  # Show all signals for demo purposes
        )
        
        # Filter by symbol
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        # Filter by signal type
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        
        # Filter by date
        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            signals = [s for s in signals if s.created_at and s.created_at >= cutoff_date]
        
        # Serialize signals
        signals_data = []
        for signal in signals:
            signal_dict = {
                "id": str(signal.id),
                "symbol": signal.symbol,
                "signal_type": signal.signal_type,
                "confluence_score": float(signal.confluence_score) if signal.confluence_score else None,
                "technical_score": float(signal.technical_score) if signal.technical_score else None,
                "sentiment_score": float(signal.sentiment_score) if signal.sentiment_score else None,
                "ml_score": float(signal.ml_score) if signal.ml_score else None,
                "price_at_signal": float(signal.price_at_signal) if signal.price_at_signal else None,
                "risk_reward_ratio": float(signal.risk_reward_ratio) if signal.risk_reward_ratio else None,
                "var_95": float(signal.var_95) if signal.var_95 else None,
                "suggested_position_size": float(signal.suggested_position_size) if signal.suggested_position_size else None,
                "created_at": signal.created_at.isoformat() if signal.created_at else None,
                "technical_rationale": signal.technical_rationale,
                "sentiment_rationale": signal.sentiment_rationale,
            }
            signals_data.append(signal_dict)
        
        return success_response(
            data={
                "signals": signals_data,
                "count": len(signals_data)
            }
        )
        
    except ValueError as e:
        return error_response(f"Invalid parameter: {str(e)}", 400)
    except Exception as e:
        logger.error(f"Get signals error: {e}", exc_info=True)
        return error_response("Failed to get signals", 500)


@signals_bp.route('/signals/<signal_id>', methods=['GET'])
@require_api_key_or_auth
def get_signal(signal_id: str):
    """Get detailed information about a specific signal."""
    try:
        from uuid import UUID
        
        # Validate UUID
        try:
            signal_uuid = UUID(signal_id)
        except ValueError:
            return error_response("Invalid signal ID", 400)
        
        # Get signal
        db = get_database()
        with db.get_session() as session:
            from src.data.models import TradeSignal
            signal = session.query(TradeSignal).filter(TradeSignal.id == signal_uuid).first()
            
            if not signal:
                return error_response("Signal not found", 404)
            
            # Serialize signal
            signal_data = {
                "id": str(signal.id),
                "symbol": signal.symbol,
                "signal_type": signal.signal_type,
                "confluence_score": float(signal.confluence_score) if signal.confluence_score else None,
                "technical_score": float(signal.technical_score) if signal.technical_score else None,
                "sentiment_score": float(signal.sentiment_score) if signal.sentiment_score else None,
                "ml_score": float(signal.ml_score) if signal.ml_score else None,
                "price_at_signal": float(signal.price_at_signal) if signal.price_at_signal else None,
                "risk_reward_ratio": float(signal.risk_reward_ratio) if signal.risk_reward_ratio else None,
                "var_95": float(signal.var_95) if signal.var_95 else None,
                "cvar_95": float(signal.cvar_95) if signal.cvar_95 else None,
                "max_drawdown": float(signal.max_drawdown) if signal.max_drawdown else None,
                "sharpe_ratio": float(signal.sharpe_ratio) if signal.sharpe_ratio else None,
                "suggested_position_size": float(signal.suggested_position_size) if signal.suggested_position_size else None,
                "created_at": signal.created_at.isoformat() if signal.created_at else None,
                "technical_rationale": signal.technical_rationale,
                "sentiment_rationale": signal.sentiment_rationale,
                "risk_warning": signal.risk_warning,
                "is_executed": signal.is_executed,
                "execution_price": float(signal.execution_price) if signal.execution_price else None,
                "realized_pnl_pct": float(signal.realized_pnl_pct) if signal.realized_pnl_pct else None,
            }
            
            return success_response(data={"signal": signal_data})
            
    except Exception as e:
        logger.error(f"Get signal error: {e}", exc_info=True)
        return error_response("Failed to get signal", 500)
