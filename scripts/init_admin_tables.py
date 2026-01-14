#!/usr/bin/env python3
"""
Initialize admin database tables.

Usage:
    python scripts/init_admin_tables.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from src.admin.models import AuditLog, SystemSettings, AdminActivity
from src.data.models import Base
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


def init_admin_tables():
    """Create admin tables."""
    try:
        # Create engine
        database_url = settings.DATABASE_URL
        engine = create_engine(database_url)
        
        # Create tables
        Base.metadata.create_all(engine, tables=[
            AuditLog.__table__,
            SystemSettings.__table__,
            AdminActivity.__table__,
        ])
        logger.info("Admin tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create admin tables: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    print("Initializing admin tables...")
    if init_admin_tables():
        print("✓ Admin tables created successfully")
    else:
        print("✗ Failed to create admin tables")
        sys.exit(1)
