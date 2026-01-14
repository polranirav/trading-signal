#!/usr/bin/env python3
"""
Initialize authentication tables in the database.

Creates user authentication tables and initializes subscription limits.
Run this after creating the base database tables.

Usage:
    python scripts/init_auth_tables.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.data.models import get_engine, Base
from src.auth.models import User, Subscription, APIKey, SubscriptionLimit
from src.logging_config import get_logger

logger = get_logger(__name__)


def init_auth_tables(database_url: str = None):
    """
    Create authentication tables.
    
    This creates the User, Subscription, APIKey, and SubscriptionLimit tables.
    """
    engine = get_engine(database_url)
    
    # Import all models to ensure they're registered
    from src.data.models import (
        AssetMetadata, MarketCandle, NewsSentiment,
        TradeSignal, IndicatorCache, BacktestResult
    )
    
    # Create all tables (including auth tables)
    Base.metadata.create_all(bind=engine)
    logger.info("Authentication tables created successfully")
    
    return engine


def init_subscription_limits(database_url: str = None):
    """
    Initialize subscription limits configuration.
    
    Sets up default tiers with their features and pricing.
    """
    from src.data.persistence import get_database
    from datetime import datetime
    
    db = get_database()
    
    # Default subscription limits (Tier 1: Essential)
    limits = [
        {
            'tier': 'free',
            'max_signals_per_day': 3,
            'max_api_calls_per_day': 100,
            'features': {
                'email_alerts': False,
                'api_access': False,
                'performance_tracking': False,
                'priority_support': False
            },
            'price_monthly': 0.0,
            'price_yearly': 0.0,
            'stripe_price_id_monthly': None,
            'stripe_price_id_yearly': None
        },
        {
            'tier': 'essential',
            'max_signals_per_day': 10,
            'max_api_calls_per_day': 1000,
            'features': {
                'email_alerts': True,
                'api_access': True,
                'performance_tracking': True,
                'priority_support': False
            },
            'price_monthly': 29.99,
            'price_yearly': 299.99,  # $300/year (save $60)
            'stripe_price_id_monthly': None,  # Set after creating in Stripe
            'stripe_price_id_yearly': None
        },
        {
            'tier': 'advanced',
            'max_signals_per_day': 50,
            'max_api_calls_per_day': 5000,
            'features': {
                'email_alerts': True,
                'api_access': True,
                'performance_tracking': True,
                'telegram_alerts': True,
                'portfolio_tracking': True,
                'priority_support': True
            },
            'price_monthly': 99.99,
            'price_yearly': 999.99,
            'stripe_price_id_monthly': None,
            'stripe_price_id_yearly': None
        }
    ]
    
    with db.get_session() as session:
        for limit_data in limits:
            # Check if exists
            existing = session.query(SubscriptionLimit).filter(
                SubscriptionLimit.tier == limit_data['tier']
            ).first()
            
            if existing:
                # Update existing
                for key, value in limit_data.items():
                    setattr(existing, key, value)
                logger.info(f"Updated subscription limits for tier: {limit_data['tier']}")
            else:
                # Create new
                limit = SubscriptionLimit(**limit_data)
                session.add(limit)
                logger.info(f"Created subscription limits for tier: {limit_data['tier']}")
        
        session.commit()
    
    logger.info("Subscription limits initialized successfully")


if __name__ == "__main__":
    print("=" * 80)
    print("Initializing Authentication Tables")
    print("=" * 80)
    print()
    
    try:
        # Create tables
        print("Creating authentication tables...")
        engine = init_auth_tables()
        print("✓ Authentication tables created")
        print()
        
        # Initialize subscription limits
        print("Initializing subscription limits...")
        init_subscription_limits()
        print("✓ Subscription limits initialized")
        print()
        
        print("=" * 80)
        print("SUCCESS: Authentication tables initialized")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Update Stripe price IDs in subscription_limits table")
        print("  2. Create admin user (optional)")
        print("  3. Test authentication")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
