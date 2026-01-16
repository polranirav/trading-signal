
import sys
import os
sys.path.append('.')

from src.data.persistence import get_database
from src.auth.service import AuthService
from datetime import datetime

def reproduce_full_login():
    print("Step 1: Get DB Session")
    db = get_database()
    
    with db.get_session() as session:
        print("Step 2: Authenticate User")
        # Test credentials
        email = "testuser@example.com"
        password = "password123"
        
        user = AuthService.authenticate_user(session, email, password)
        
        if not user:
            print("   FAILURE: Authentication failed (User not found or password wrong)")
            return

        print(f"   Authentication Successful: User {user.id}")
        
        print("Step 3: Update Last Login & Commit")
        try:
            user.last_login = datetime.utcnow()
            session.commit()
            print("   Commit Successful")
        except Exception as e:
            print(f"   CRITICAL: Commit Failed: {e}")
            raise

        print("Step 4: Get Subscription Tier")
        try:
            from src.subscriptions.service import SubscriptionService
            tier = SubscriptionService.get_user_tier(session, user.id)
            print(f"   Result: Tier = {tier}")
        except Exception as e:
            print(f"   CRITICAL: Subscription Service Failed: {e}")
            raise

if __name__ == "__main__":
    reproduce_full_login()
