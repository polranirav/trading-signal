#!/usr/bin/env python3
"""
Migration script to add user_id column to trade_signals table.

This allows signals to be associated with users for subscription limits.

Usage:
    python scripts/add_user_id_to_signals.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.data.models import get_engine
from src.logging_config import get_logger

logger = get_logger(__name__)


def add_user_id_column():
    """Add user_id column to trade_signals table."""
    engine = get_engine()
    
    with engine.connect() as conn:
        try:
            # Add user_id column (nullable for existing signals)
            conn.execute(text("""
                ALTER TABLE trade_signals
                ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE SET NULL
            """))
            
            # Add index for performance
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_signals_user_id 
                ON trade_signals(user_id)
            """))
            
            conn.commit()
            logger.info("Added user_id column to trade_signals table")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add user_id column: {e}")
            return False


if __name__ == "__main__":
    print("Adding user_id column to trade_signals table...")
    if add_user_id_column():
        print("✓ Successfully added user_id column")
    else:
        print("✗ Failed to add user_id column")
        sys.exit(1)
