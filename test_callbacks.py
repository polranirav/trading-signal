#!/usr/bin/env python3
"""
Quick diagnostic script to test callback registration and dependencies.
"""

import sys
import os
sys.path.insert(0, '.')
sys.path.insert(0, 'src')

def test_imports():
    """Test if all imports work."""
    print("=" * 70)
    print("TESTING IMPORTS")
    print("=" * 70)
    
    try:
        from web.app import app
        print("✅ App imported")
        return app
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_callbacks(app):
    """Test callback registration."""
    print("\n" + "=" * 70)
    print("TESTING CALLBACKS")
    print("=" * 70)
    
    print(f"📊 Total callbacks registered: {len(app.callback_map)}")
    
    # List all callback IDs
    callback_ids = list(app.callback_map.keys())
    print(f"\n📋 Callback IDs:")
    for i, cb_id in enumerate(callback_ids[:20], 1):
        print(f"   {i}. {cb_id}")
    if len(callback_ids) > 20:
        print(f"   ... and {len(callback_ids) - 20} more")

def test_dependencies():
    """Test if critical dependencies are available."""
    print("\n" + "=" * 70)
    print("TESTING DEPENDENCIES")
    print("=" * 70)
    
    # Database
    try:
        from data.persistence import get_database
        db = get_database()
        print("✅ Database module available")
    except Exception as e:
        print(f"⚠️  Database: {e}")
    
    # Redis/Cache
    try:
        from data.cache import get_cache
        cache = get_cache()
        print("✅ Cache module available")
    except Exception as e:
        print(f"⚠️  Cache: {e}")
    
    # Analytics
    try:
        from analytics.confluence import ConfluenceEngine
        print("✅ Analytics modules available")
    except Exception as e:
        print(f"⚠️  Analytics: {e}")

if __name__ == "__main__":
    app = test_imports()
    if app:
        test_callbacks(app)
        test_dependencies()
        print("\n" + "=" * 70)
        print("✅ DIAGNOSTIC COMPLETE")
        print("=" * 70)
