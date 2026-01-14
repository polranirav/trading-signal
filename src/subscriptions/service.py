"""
Subscription Service.

Provides subscription management logic:
- Tier management
- Usage tracking
- Upgrade/downgrade flows
"""

from typing import Optional, Dict
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from src.logging_config import get_logger
from src.data.persistence import get_database
from src.auth.models import User, Subscription, SubscriptionLimit
from src.auth.service import AuthService

logger = get_logger(__name__)


class SubscriptionService:
    """Service for managing user subscriptions."""
    
    @staticmethod
    def get_user_tier(session: Session, user_id: UUID) -> str:
        """
        Get the current subscription tier for a user.
        
        Args:
            session: Database session
            user_id: User ID
        
        Returns:
            Tier name ('free', 'essential', 'advanced', etc.)
        """
        subscription = AuthService.get_user_subscription(session, user_id)
        return subscription.tier if subscription else 'free'
    
    @staticmethod
    def get_user_limits(session: Session, user_id: UUID) -> Optional[Dict]:
        """
        Get subscription limits for a user.
        
        Args:
            session: Database session
            user_id: User ID
        
        Returns:
            Dictionary with limits and features
        """
        tier = SubscriptionService.get_user_tier(session, user_id)
        limit = AuthService.get_subscription_limit(session, tier)
        
        if not limit:
            logger.warning(f"Subscription limit not found for tier: {tier}")
            return None
        
        return {
            'tier': tier,
            'max_signals_per_day': limit.max_signals_per_day,
            'max_api_calls_per_day': limit.max_api_calls_per_day,
            'features': limit.features,
            'price_monthly': float(limit.price_monthly),
            'price_yearly': float(limit.price_yearly)
        }
    
    @staticmethod
    def check_usage_today(
        session: Session,
        user_id: UUID,
        resource_type: str = 'signals'  # 'signals' or 'api_calls'
    ) -> int:
        """
        Check how much of a resource has been used today.
        
        Args:
            session: Database session
            user_id: User ID
            resource_type: 'signals' or 'api_calls'
        
        Returns:
            Count of usage today
        """
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if resource_type == 'signals':
            # Count signals generated today for this user
            from src.data.models import TradeSignal
            count = session.query(func.count(TradeSignal.id)).filter(
                and_(
                    TradeSignal.user_id == user_id,
                    TradeSignal.created_at >= today_start
                )
            ).scalar()
            return count or 0
        
        elif resource_type == 'api_calls':
            # Count API calls today (if tracking API calls in database)
            # For now, return 0 (can be implemented with API call tracking table)
            return 0
        
        return 0
    
    @staticmethod
    def can_use_resource(
        session: Session,
        user_id: UUID,
        resource_type: str = 'signals'
    ) -> bool:
        """
        Check if user can use a resource (within limits).
        
        Args:
            session: Database session
            user_id: User ID
            resource_type: 'signals' or 'api_calls'
        
        Returns:
            True if within limits, False otherwise
        """
        limits = SubscriptionService.get_user_limits(session, user_id)
        if not limits:
            return False
        
        usage_key = f'max_{resource_type}_per_day'
        limit = limits.get(usage_key, 0)
        
        if limit == 0:
            return False  # Resource not available for this tier
        
        usage = SubscriptionService.check_usage_today(session, user_id, resource_type)
        return usage < limit
    
    @staticmethod
    def upgrade_subscription(
        session: Session,
        user_id: UUID,
        new_tier: str
    ) -> bool:
        """
        Upgrade user subscription tier.
        
        Args:
            session: Database session
            user_id: User ID
            new_tier: New tier name
        
        Returns:
            True if successful, False otherwise
        """
        subscription = AuthService.get_user_subscription(session, user_id)
        
        if not subscription:
            # Create new subscription
            subscription = Subscription(
                user_id=user_id,
                tier=new_tier,
                status='active'
            )
            session.add(subscription)
        else:
            # Update existing subscription
            subscription.tier = new_tier
            subscription.status = 'active'
        
        session.commit()
        logger.info(f"Subscription upgraded for user {user_id} to tier {new_tier}")
        return True
    
    @staticmethod
    def downgrade_subscription(
        session: Session,
        user_id: UUID,
        new_tier: str
    ) -> bool:
        """
        Downgrade user subscription tier.
        
        Args:
            session: Database session
            user_id: User ID
            new_tier: New tier name
        
        Returns:
            True if successful, False otherwise
        """
        subscription = AuthService.get_user_subscription(session, user_id)
        
        if not subscription:
            return False
        
        subscription.tier = new_tier
        subscription.status = 'active'
        session.commit()
        
        logger.info(f"Subscription downgraded for user {user_id} to tier {new_tier}")
        return True
    
    @staticmethod
    def cancel_subscription(
        session: Session,
        user_id: UUID,
        cancel_at_period_end: bool = True
    ) -> bool:
        """
        Cancel user subscription.
        
        Args:
            session: Database session
            user_id: User ID
            cancel_at_period_end: If True, cancel at period end; if False, cancel immediately
        
        Returns:
            True if successful, False otherwise
        """
        subscription = AuthService.get_user_subscription(session, user_id)
        
        if not subscription:
            return False
        
        if subscription.stripe_subscription_id:
            # Cancel via Stripe
            from src.payments.stripe_client import StripeClient
            StripeClient.cancel_subscription(
                subscription.stripe_subscription_id,
                cancel_at_period_end=cancel_at_period_end
            )
        
        if cancel_at_period_end:
            subscription.cancel_at_period_end = True
        else:
            subscription.status = 'cancelled'
            # Downgrade to free
            subscription.tier = 'free'
        
        session.commit()
        logger.info(f"Subscription cancelled for user {user_id}")
        return True
