"""
Flask route for Stripe webhooks.

This provides a Flask route endpoint that Stripe can POST to.
"""

from flask import Blueprint, request, jsonify
from src.payments.webhooks import handle_webhook
from src.logging_config import get_logger

logger = get_logger(__name__)

payment_webhook_bp = Blueprint('payment_webhook', __name__)


@payment_webhook_bp.route('/webhooks/stripe', methods=['POST'])
def stripe_webhook():
    """
    Stripe webhook endpoint.
    
    This endpoint receives webhook events from Stripe.
    """
    payload = request.data
    signature = request.headers.get('Stripe-Signature')
    
    if not signature:
        logger.error("Missing Stripe-Signature header")
        return jsonify({'error': 'Missing signature'}), 400
    
    success = handle_webhook(payload, signature)
    
    if success:
        return jsonify({'status': 'success'}), 200
    else:
        return jsonify({'error': 'Webhook processing failed'}), 400


def register_webhook_routes(app):
    """Register webhook routes with Flask/Dash app."""
    if hasattr(app, 'server'):
        app.server.register_blueprint(payment_webhook_bp)
    else:
        app.register_blueprint(payment_webhook_bp)
