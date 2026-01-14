"""
Subscription Limits Enforcement.

Provides utilities for checking and enforcing subscription limits.
"""

from typing import Optional, Dict
from uuid import UUID

from sqlalchemy.orm import Session

from src.logging_config import get_logger
from src.data.persistence import get_database
from src.subscriptions.service import SubscriptionService

logger = get_logger(__name__)


def check_limit(
    user_id: UUID,
    resource_type: str = 'signals'
) -> bool:
    """
    Check if user can use a resource (convenience function).
    
    Args:
        user_id: User ID
        resource_type: 'signals' or 'api_calls'
    
    Returns:
        True if within limits, False otherwise
    """
    db = get_database()
    with db.get_session() as session:
        return SubscriptionService.can_use_resource(session, user_id, resource_type)


def get_user_limits(user_id: UUID) -> Optional[Dict]:
    """
    Get subscription limits for a user (convenience function).
    
    Args:
        user_id: User ID
    
    Returns:
        Dictionary with limits and features
    """
    db = get_database()
    with db.get_session() as session:
        return SubscriptionService.get_user_limits(session, user_id)


def track_usage(
    user_id: UUID,
    resource_type: str = 'signals',
    amount: int = 1
) -> None:
    """
    Track resource usage (for future implementation).
    
    Currently, usage is tracked automatically via database queries.
    This function is a placeholder for future usage tracking features.
    
    Args:
        user_id: User ID
        resource_type: 'signals' or 'api_calls'
        amount: Amount of resource used
    """
    # Future: Implement usage tracking table
    logger.debug(f"Tracked usage: user={user_id}, resource={resource_type}, amount={amount}")
