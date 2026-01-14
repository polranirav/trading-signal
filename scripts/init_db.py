#!/usr/bin/env python3
"""
Database Initialization Script.

Sets up the trading_signals database with:
1. All table schemas (SQLAlchemy models)
2. TimescaleDB hypertables (for time-series optimization)
3. Subscription tier seed data

Usage:
    python scripts/init_db.py
    
With sample data:
    python scripts/init_db.py --seed
"""

import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from datetime import datetime
from decimal import Decimal


def init_tables():
    """Create all database tables."""
    print("🔧 Creating database tables...")
    
    from src.data.models import init_database
    from src.config import settings
    
    try:
        init_database(settings.DATABASE_URL)
        print("✅ Tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        return False


def init_timescale():
    """Initialize TimescaleDB hypertables."""
    print("🔧 Initializing TimescaleDB hypertables...")
    
    from src.data.models import init_timescaledb
    from src.config import settings
    
    try:
        init_timescaledb(settings.DATABASE_URL)
        print("✅ TimescaleDB initialized")
        return True
    except Exception as e:
        print(f"⚠️ TimescaleDB init skipped (may already exist or not installed): {e}")
        return False


def seed_subscription_tiers():
    """Seed subscription tier limits."""
    print("🔧 Seeding subscription tiers...")
    
    from src.data.persistence import get_database
    from src.auth.models import SubscriptionLimit
    from sqlalchemy.dialects.postgresql import insert
    
    db = get_database()
    
    tiers = [
        {
            "tier": "free",
            "max_signals_per_day": 3,
            "max_api_calls_per_day": 50,
            "features": {
                "email_alerts": False,
                "api_access": False,
                "real_time_signals": False,
                "advanced_analytics": False,
                "priority_support": False,
            },
            "price_monthly": Decimal("0.00"),
            "price_yearly": Decimal("0.00"),
        },
        {
            "tier": "essential",
            "max_signals_per_day": 25,
            "max_api_calls_per_day": 500,
            "features": {
                "email_alerts": True,
                "api_access": True,
                "real_time_signals": False,
                "advanced_analytics": False,
                "priority_support": False,
            },
            "price_monthly": Decimal("29.00"),
            "price_yearly": Decimal("290.00"),
        },
        {
            "tier": "advanced",
            "max_signals_per_day": 100,
            "max_api_calls_per_day": 2000,
            "features": {
                "email_alerts": True,
                "api_access": True,
                "real_time_signals": True,
                "advanced_analytics": True,
                "priority_support": False,
            },
            "price_monthly": Decimal("79.00"),
            "price_yearly": Decimal("790.00"),
        },
        {
            "tier": "premium",
            "max_signals_per_day": -1,  # Unlimited
            "max_api_calls_per_day": -1,  # Unlimited
            "features": {
                "email_alerts": True,
                "api_access": True,
                "real_time_signals": True,
                "advanced_analytics": True,
                "priority_support": True,
            },
            "price_monthly": Decimal("199.00"),
            "price_yearly": Decimal("1990.00"),
        },
    ]
    
    try:
        with db.get_session() as session:
            for tier_data in tiers:
                stmt = insert(SubscriptionLimit).values(**tier_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['tier'],
                    set_={
                        'max_signals_per_day': stmt.excluded.max_signals_per_day,
                        'max_api_calls_per_day': stmt.excluded.max_api_calls_per_day,
                        'features': stmt.excluded.features,
                        'price_monthly': stmt.excluded.price_monthly,
                        'price_yearly': stmt.excluded.price_yearly,
                        'updated_at': datetime.utcnow(),
                    }
                )
                session.execute(stmt)
        
        print(f"✅ Seeded {len(tiers)} subscription tiers")
        return True
    except Exception as e:
        print(f"❌ Failed to seed subscription tiers: {e}")
        return False


def seed_sample_assets():
    """Seed sample assets for testing."""
    print("🔧 Seeding sample assets...")
    
    from src.data.persistence import get_database
    
    db = get_database()
    
    assets = [
        {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics", "is_active": True},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "industry": "Software", "is_active": True},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "industry": "Internet Services", "is_active": True},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "industry": "E-Commerce", "is_active": True},
        {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology", "industry": "Semiconductors", "is_active": True},
        {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology", "industry": "Social Media", "is_active": True},
        {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers", "is_active": True},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services", "industry": "Banks", "is_active": True},
        {"symbol": "V", "name": "Visa Inc.", "sector": "Financial Services", "industry": "Credit Services", "is_active": True},
        {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "industry": "Drug Manufacturers", "is_active": True},
    ]
    
    try:
        for asset in assets:
            db.save_asset(asset)
        
        print(f"✅ Seeded {len(assets)} sample assets")
        return True
    except Exception as e:
        print(f"❌ Failed to seed assets: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Initialize Trading Signals Database")
    parser.add_argument("--seed", action="store_true", help="Seed sample data for testing")
    parser.add_argument("--skip-timescale", action="store_true", help="Skip TimescaleDB initialization")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 TRADING SIGNALS DATABASE INITIALIZATION")
    print("=" * 60)
    print()
    
    # Step 1: Create tables
    if not init_tables():
        print("\n❌ Database initialization failed")
        sys.exit(1)
    
    # Step 2: Initialize TimescaleDB (optional)
    if not args.skip_timescale:
        init_timescale()
    
    # Step 3: Seed subscription tiers (always)
    seed_subscription_tiers()
    
    # Step 4: Seed sample data (optional)
    if args.seed:
        seed_sample_assets()
    
    print()
    print("=" * 60)
    print("✅ DATABASE INITIALIZATION COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Start the dashboard: python src/web/app.py")
    print("  2. Open http://localhost:8050")
    print()


if __name__ == "__main__":
    main()
