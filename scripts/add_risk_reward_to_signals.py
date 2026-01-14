#!/usr/bin/env python3
"""
Migration script to add risk_reward_ratio column to trade_signals table.

Usage:
    python scripts/add_risk_reward_to_signals.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.data.models import get_engine
from src.logging_config import get_logger

logger = get_logger(__name__)


def add_risk_reward_column():
    """Add risk_reward_ratio column to trade_signals table."""
    engine = get_engine()
    
    with engine.connect() as conn:
        try:
            # Add risk_reward_ratio column
            conn.execute(text("""
                ALTER TABLE trade_signals
                ADD COLUMN IF NOT EXISTS risk_reward_ratio DECIMAL(5,2)
            """))
            
            conn.commit()
            logger.info("Added risk_reward_ratio column to trade_signals table")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add risk_reward_ratio column: {e}")
            return False


if __name__ == "__main__":
    print("Adding risk_reward_ratio column to trade_signals table...")
    if add_risk_reward_column():
        print("✓ Successfully added risk_reward_ratio column")
    else:
        print("✗ Failed to add risk_reward_ratio column")
        sys.exit(1)
