"""
Stripe API Client.

Handles Stripe integration for subscription payments, checkout, and webhooks.
"""

import stripe
from typing import Optional, Dict, List
from decimal import Decimal
from datetime import datetime

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# Initialize Stripe
# Set in config: STRIPE_SECRET_KEY (for production) or STRIPE_TEST_SECRET_KEY (for testing)
stripe.api_key = getattr(settings, 'STRIPE_SECRET_KEY', None) or getattr(settings, 'STRIPE_TEST_SECRET_KEY', 'sk_test_...')


class StripeClient:
    """Stripe API client for payment processing."""
    
    # Subscription tier configuration (matches subscription_limits table)
    TIERS = {
        'free': {
            'price_id_monthly': None,  # Set after creating in Stripe
            'price_id_yearly': None,
            'price_monthly': Decimal('0.00'),
            'price_yearly': Decimal('0.00'),
        },
        'essential': {
            'price_id_monthly': None,  # Set after creating in Stripe
            'price_id_yearly': None,
            'price_monthly': Decimal('29.99'),
            'price_yearly': Decimal('299.99'),  # $300/year (save $60)
        },
        'advanced': {
            'price_id_monthly': None,  # Set after creating in Stripe
            'price_id_yearly': None,
            'price_monthly': Decimal('99.99'),
            'price_yearly': Decimal('999.99'),
        }
    }
    
    @staticmethod
    def create_checkout_session(
        customer_email: str,
        tier: str,
        billing_period: str = 'monthly',  # 'monthly' or 'yearly'
        success_url: str = None,
        cancel_url: str = None
    ) -> Optional[Dict]:
        """
        Create a Stripe Checkout session.
        
        Args:
            customer_email: Customer email
            tier: Subscription tier ('essential', 'advanced', etc.)
            billing_period: 'monthly' or 'yearly'
            success_url: Success redirect URL
            cancel_url: Cancel redirect URL
        
        Returns:
            Checkout session dictionary with 'id' and 'url'
        """
        try:
            tier_config = StripeClient.TIERS.get(tier)
            if not tier_config:
                logger.error(f"Invalid tier: {tier}")
                return None
            
            price_key = f'price_id_{billing_period}'
            price_id = tier_config.get(price_key)
            
            if not price_id:
                logger.error(f"Price ID not set for tier {tier} {billing_period}")
                return None
            
            session = stripe.checkout.Session.create(
                customer_email=customer_email,
                payment_method_types=['card'],
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url or f"{getattr(settings, 'BASE_URL', 'http://localhost:8050')}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=cancel_url or f"{getattr(settings, 'BASE_URL', 'http://localhost:8050')}/checkout/cancel",
                metadata={
                    'tier': tier,
                    'billing_period': billing_period
                }
            )
            
            logger.info(f"Created checkout session for {customer_email} tier {tier}")
            return {
                'id': session.id,
                'url': session.url,
                'customer_email': customer_email
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe checkout error: {e}")
            return None
        except Exception as e:
            logger.error(f"Checkout session creation failed: {e}")
            return None
    
    @staticmethod
    def create_customer(email: str, name: str = None) -> Optional[Dict]:
        """
        Create a Stripe customer.
        
        Args:
            email: Customer email
            name: Optional customer name
        
        Returns:
            Customer dictionary with 'id'
        """
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name
            )
            
            logger.info(f"Created Stripe customer: {email}")
            return {
                'id': customer.id,
                'email': customer.email
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe customer creation error: {e}")
            return None
    
    @staticmethod
    def get_subscription(stripe_subscription_id: str) -> Optional[Dict]:
        """Get subscription details from Stripe."""
        try:
            subscription = stripe.Subscription.retrieve(stripe_subscription_id)
            return {
                'id': subscription.id,
                'status': subscription.status,
                'customer': subscription.customer,
                'current_period_start': datetime.fromtimestamp(subscription.current_period_start),
                'current_period_end': datetime.fromtimestamp(subscription.current_period_end),
                'cancel_at_period_end': subscription.cancel_at_period_end,
                'items': [item.id for item in subscription.items.data]
            }
        except stripe.error.StripeError as e:
            logger.error(f"Stripe subscription retrieval error: {e}")
            return None
    
    @staticmethod
    def cancel_subscription(stripe_subscription_id: str, cancel_at_period_end: bool = True) -> bool:
        """
        Cancel a subscription.
        
        Args:
            stripe_subscription_id: Stripe subscription ID
            cancel_at_period_end: If True, cancel at end of period; if False, cancel immediately
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if cancel_at_period_end:
                subscription = stripe.Subscription.modify(
                    stripe_subscription_id,
                    cancel_at_period_end=True
                )
            else:
                subscription = stripe.Subscription.delete(stripe_subscription_id)
            
            logger.info(f"Cancelled subscription: {stripe_subscription_id}")
            return True
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe subscription cancellation error: {e}")
            return False
    
    @staticmethod
    def update_subscription_tier(
        stripe_subscription_id: str,
        new_tier: str,
        billing_period: str = 'monthly'
    ) -> bool:
        """
        Update subscription to a new tier.
        
        Args:
            stripe_subscription_id: Stripe subscription ID
            new_tier: New tier name
            billing_period: 'monthly' or 'yearly'
        
        Returns:
            True if successful, False otherwise
        """
        try:
            tier_config = StripeClient.TIERS.get(new_tier)
            if not tier_config:
                logger.error(f"Invalid tier: {new_tier}")
                return False
            
            price_key = f'price_id_{billing_period}'
            price_id = tier_config.get(price_key)
            
            if not price_id:
                logger.error(f"Price ID not set for tier {new_tier} {billing_period}")
                return False
            
            # Get current subscription
            subscription = stripe.Subscription.retrieve(stripe_subscription_id)
            
            # Update subscription item
            subscription_item_id = subscription.items.data[0].id
            stripe.SubscriptionItem.modify(
                subscription_item_id,
                price=price_id
            )
            
            logger.info(f"Updated subscription {stripe_subscription_id} to tier {new_tier}")
            return True
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe subscription update error: {e}")
            return False
    
    @staticmethod
    def get_invoice(stripe_invoice_id: str) -> Optional[Dict]:
        """Get invoice details from Stripe."""
        try:
            invoice = stripe.Invoice.retrieve(stripe_invoice_id)
            return {
                'id': invoice.id,
                'status': invoice.status,
                'amount_due': Decimal(str(invoice.amount_due)) / 100,  # Convert cents to dollars
                'amount_paid': Decimal(str(invoice.amount_paid)) / 100,
                'currency': invoice.currency,
                'invoice_pdf': invoice.invoice_pdf,
                'hosted_invoice_url': invoice.hosted_invoice_url,
                'number': invoice.number,
                'period_start': datetime.fromtimestamp(invoice.period_start) if invoice.period_start else None,
                'period_end': datetime.fromtimestamp(invoice.period_end) if invoice.period_end else None,
            }
        except stripe.error.StripeError as e:
            logger.error(f"Stripe invoice retrieval error: {e}")
            return None
