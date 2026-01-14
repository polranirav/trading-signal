"""
Stripe Webhook Handlers.

Handles Stripe webhook events for payment processing.
"""

import stripe
from flask import request
from typing import Dict, Optional
from datetime import datetime

from src.config import settings
from src.logging_config import get_logger
from src.data.persistence import get_database
from src.auth.models import User, Subscription
from src.payments.models import Payment, Invoice

logger = get_logger(__name__)

# Get webhook secret from config
WEBHOOK_SECRET = getattr(settings, 'STRIPE_WEBHOOK_SECRET', None)


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """
    Verify Stripe webhook signature.
    
    Args:
        payload: Raw request body
        signature: Stripe signature header
    
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        if not WEBHOOK_SECRET:
            logger.warning("Webhook secret not configured, skipping verification")
            return True  # Allow in development
        
        stripe.Webhook.construct_event(
            payload,
            signature,
            WEBHOOK_SECRET
        )
        return True
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        return False
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        return False


def handle_checkout_session_completed(event: Dict) -> None:
    """Handle checkout.session.completed event."""
    session = event['data']['object']
    customer_email = session.get('customer_email')
    subscription_id = session.get('subscription')
    metadata = session.get('metadata', {})
    tier = metadata.get('tier', 'essential')
    
    logger.info(f"Checkout completed for {customer_email}, tier: {tier}")
    
    db = get_database()
    with db.get_session() as db_session:
        # Find user by email
        from src.auth.service import AuthService
        user = AuthService.get_user_by_email(db_session, customer_email)
        
        if not user:
            logger.error(f"User not found for email: {customer_email}")
            return
        
        # Update or create subscription
        subscription = db_session.query(Subscription).filter(
            Subscription.user_id == user.id,
            Subscription.status == 'active'
        ).first()
        
        if subscription:
            # Update existing subscription
            subscription.stripe_subscription_id = subscription_id
            subscription.tier = tier
            subscription.status = 'active'
            subscription.cancel_at_period_end = False
            
            # Get subscription details from Stripe
            from src.payments.stripe_client import StripeClient
            sub_data = StripeClient.get_subscription(subscription_id)
            if sub_data:
                subscription.current_period_start = sub_data['current_period_start']
                subscription.current_period_end = sub_data['current_period_end']
        else:
            # Create new subscription
            from src.payments.stripe_client import StripeClient
            sub_data = StripeClient.get_subscription(subscription_id)
            
            subscription = Subscription(
                user_id=user.id,
                tier=tier,
                status='active',
                stripe_subscription_id=subscription_id,
                stripe_customer_id=sub_data['customer'] if sub_data else None,
                current_period_start=sub_data['current_period_start'] if sub_data else None,
                current_period_end=sub_data['current_period_end'] if sub_data else None,
                cancel_at_period_end=False
            )
            db_session.add(subscription)
        
        db_session.commit()
        logger.info(f"Subscription created/updated for user {user.email}")


def handle_customer_subscription_updated(event: Dict) -> None:
    """Handle customer.subscription.updated event."""
    stripe_subscription = event['data']['object']
    subscription_id = stripe_subscription['id']
    status = stripe_subscription['status']
    
    logger.info(f"Subscription updated: {subscription_id}, status: {status}")
    
    db = get_database()
    with db.get_session() as db_session:
        subscription = db_session.query(Subscription).filter(
            Subscription.stripe_subscription_id == subscription_id
        ).first()
        
        if subscription:
            subscription.status = status
            subscription.cancel_at_period_end = stripe_subscription.get('cancel_at_period_end', False)
            subscription.current_period_start = datetime.fromtimestamp(stripe_subscription['current_period_start'])
            subscription.current_period_end = datetime.fromtimestamp(stripe_subscription['current_period_end'])
            db_session.commit()
            logger.info(f"Subscription {subscription_id} updated in database")
        else:
            logger.warning(f"Subscription {subscription_id} not found in database")


def handle_customer_subscription_deleted(event: Dict) -> None:
    """Handle customer.subscription.deleted event."""
    stripe_subscription = event['data']['object']
    subscription_id = stripe_subscription['id']
    
    logger.info(f"Subscription deleted: {subscription_id}")
    
    db = get_database()
    with db.get_session() as db_session:
        subscription = db_session.query(Subscription).filter(
            Subscription.stripe_subscription_id == subscription_id
        ).first()
        
        if subscription:
            subscription.status = 'cancelled'
            db_session.commit()
            logger.info(f"Subscription {subscription_id} marked as cancelled")
        else:
            logger.warning(f"Subscription {subscription_id} not found in database")


def handle_invoice_payment_succeeded(event: Dict) -> None:
    """Handle invoice.payment_succeeded event."""
    invoice = event['data']['object']
    subscription_id = invoice.get('subscription')
    
    logger.info(f"Invoice payment succeeded for subscription: {subscription_id}")
    
    db = get_database()
    with db.get_session() as db_session:
        subscription = db_session.query(Subscription).filter(
            Subscription.stripe_subscription_id == subscription_id
        ).first()
        
        if subscription:
            # Update subscription status to active
            subscription.status = 'active'
            db_session.commit()
            
            # Create invoice record (optional)
            # You can save invoice details here if needed
            logger.info(f"Subscription {subscription_id} activated")


def handle_invoice_payment_failed(event: Dict) -> None:
    """Handle invoice.payment_failed event."""
    invoice = event['data']['object']
    subscription_id = invoice.get('subscription')
    
    logger.warning(f"Invoice payment failed for subscription: {subscription_id}")
    
    db = get_database()
    with db.get_session() as db_session:
        subscription = db_session.query(Subscription).filter(
            Subscription.stripe_subscription_id == subscription_id
        ).first()
        
        if subscription:
            # Mark as past_due (grace period)
            subscription.status = 'past_due'
            db_session.commit()
            logger.warning(f"Subscription {subscription_id} marked as past_due")


def handle_webhook(payload: bytes, signature: str) -> bool:
    """
    Handle Stripe webhook event.
    
    Args:
        payload: Raw request body
        signature: Stripe signature header
    
    Returns:
        True if handled successfully, False otherwise
    """
    # Verify signature
    if not verify_webhook_signature(payload, signature):
        logger.error("Webhook signature verification failed")
        return False
    
    try:
        # Parse event
        event = stripe.Webhook.construct_event(
            payload,
            signature,
            WEBHOOK_SECRET
        )
        
        event_type = event['type']
        logger.info(f"Handling webhook event: {event_type}")
        
        # Route to appropriate handler
        if event_type == 'checkout.session.completed':
            handle_checkout_session_completed(event)
        elif event_type == 'customer.subscription.updated':
            handle_customer_subscription_updated(event)
        elif event_type == 'customer.subscription.deleted':
            handle_customer_subscription_deleted(event)
        elif event_type == 'invoice.payment_succeeded':
            handle_invoice_payment_succeeded(event)
        elif event_type == 'invoice.payment_failed':
            handle_invoice_payment_failed(event)
        else:
            logger.info(f"Unhandled webhook event type: {event_type}")
        
        return True
        
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        return False
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        return False
    except Exception as e:
        logger.error(f"Webhook handling error: {e}")
        return False
