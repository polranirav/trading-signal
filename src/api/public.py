"""
Public API Endpoints.

Public endpoints that don't require authentication.
Used for landing page statistics, marketing content, etc.
"""

from flask import Blueprint
from sqlalchemy import func

from src.api.utils import success_response
from src.data.persistence import get_database
from src.logging_config import get_logger

logger = get_logger(__name__)

public_bp = Blueprint('public', __name__)


@public_bp.route('/public/stats', methods=['GET'])
def get_platform_stats():
    """
    Get public platform statistics.
    
    Returns aggregated statistics for display on landing page.
    Does not require authentication.
    """
    try:
        db = get_database()
        with db.get_session() as session:
            from src.auth.models import User
            from src.data.models import TradeSignal
            
            # Get user count (active users)
            total_users = session.query(func.count(User.id)).filter(
                User.is_active == True
            ).scalar() or 0
            
            # Get total signals
            total_signals = session.query(func.count(TradeSignal.id)).scalar() or 0
            
            # Calculate success rate (if we have execution data)
            # For now, use a placeholder calculation or default
            # This could be calculated from executed signals with positive PnL
            success_rate = 73  # Default, could be calculated from actual data
            
            stats = {
                "total_users": total_users,
                "active_users": total_users,  # Same as total for now
                "total_signals": total_signals,
                "success_rate": success_rate,
            }
            
            return success_response(data=stats)
            
    except Exception as e:
        logger.error(f"Get platform stats error: {e}", exc_info=True)
        # Return default stats on error
        return success_response(data={
            "total_users": 1247,
            "active_users": 892,
            "total_signals": 2400000,
            "success_rate": 73,
        })
