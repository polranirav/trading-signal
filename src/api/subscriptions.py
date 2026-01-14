"""
Subscriptions API Endpoints.

Handles subscription management and information.
"""

from flask import Blueprint, g

from src.api.utils import success_response, error_response, require_auth_api
from src.data.persistence import get_database
from src.subscriptions.service import SubscriptionService
from src.logging_config import get_logger

logger = get_logger(__name__)

subscriptions_bp = Blueprint('subscriptions', __name__)


@subscriptions_bp.route('/subscription', methods=['GET'])
@require_auth_api
def get_subscription():
    """Get current user's subscription information."""
    try:
        user = g.current_user
        
        db = get_database()
        with db.get_session() as db_session:
            tier = SubscriptionService.get_user_tier(db_session, user.id)
            limits = SubscriptionService.get_user_limits(db_session, user.id)
            
            # Get subscription details
            from src.auth.service import AuthService
            subscription = AuthService.get_user_subscription(db_session, user.id)
            
            subscription_data = {
                "tier": tier,
                "limits": limits,
                "status": subscription.status if subscription else "active",
                "current_period_start": subscription.current_period_start.isoformat() if subscription and subscription.current_period_start else None,
                "current_period_end": subscription.current_period_end.isoformat() if subscription and subscription.current_period_end else None,
            }
            
            return success_response(data={"subscription": subscription_data})
    except Exception as e:
        logger.error(f"Get subscription error: {e}", exc_info=True)
        return error_response("Failed to get subscription", 500)


@subscriptions_bp.route('/subscription/upgrade', methods=['POST'])
@require_auth_api
def upgrade_subscription():
    """Upgrade user subscription (redirects to Stripe checkout)."""
    try:
        from flask import request
        data = request.get_json() or {}
        tier = data.get('tier', '').strip().lower()
        
        if not tier or tier not in ['essential', 'advanced', 'premium']:
            return error_response("Invalid tier. Must be: essential, advanced, or premium", 400)
        
        user = g.current_user
        
        # Redirect to Stripe checkout (frontend will handle this)
        # For API, we return checkout URL
        from src.payments.stripe_client import StripeClient
        from src.config import settings
        
        checkout_url = f"/checkout?tier={tier}&period=monthly"
        
        return success_response(
            data={
                "checkout_url": checkout_url,
                "tier": tier
            },
            message="Redirect to checkout"
        )
    except Exception as e:
        logger.error(f"Upgrade subscription error: {e}", exc_info=True)
        return error_response("Failed to process upgrade", 500)


@subscriptions_bp.route('/subscription/cancel', methods=['POST'])
@require_auth_api
def cancel_subscription():
    """Cancel user subscription."""
    try:
        user = g.current_user
        
        db = get_database()
        with db.get_session() as db_session:
            success = SubscriptionService.cancel_subscription(
                db_session,
                user_id=user.id,
                cancel_at_period_end=True
            )
            
            if not success:
                return error_response("Failed to cancel subscription", 500)
            
            logger.info(f"Subscription cancelled for user: {user.email}")
            
            return success_response(message="Subscription cancelled successfully")
    except Exception as e:
        logger.error(f"Cancel subscription error: {e}", exc_info=True)
        return error_response("Failed to cancel subscription", 500)
