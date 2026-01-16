import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    print("Step 1: Importing database manager...")
    from src.data.persistence import get_database
    db = get_database()
    print("✓ Database manager imported")

    print("Step 2: Importing Auth Service...")
    from src.auth.service import AuthService
    print("✓ Auth Service imported")

    print("Step 3: Creating session...")
    with db.get_session() as session:
        print("✓ Session created")
        
        print("Step 4: Attempting to query user...")
        # Try to find a user (mock email)
        user = AuthService.get_user_by_email(session, "test@example.com")
        print(f"✓ Query successful (User found: {user is not None})")

    print("Step 5: Checking user_api_keys module...")
    from src.api.user_api_keys import user_api_keys_bp
    print("✓ user_api_keys module imported")

    print("SUCCESS: Core auth modules are working.")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
