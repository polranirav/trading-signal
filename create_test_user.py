import sys
import os

# Ensure src is in python path
sys.path.append(os.getcwd())

from src.data.persistence import DatabaseManager
from src.auth.service import AuthService
from src.auth.models import SubscriptionLimit

def create_test_user():
    print("Connecting to database...")
    db = DatabaseManager()
    
    with db.get_session() as session:
        # Create subscription limits first (required for app logic usually)
        print("Checking subscription limits...")
        tiers = ['free', 'essential', 'advanced', 'premium']
        for tier in tiers:
            limit = session.query(SubscriptionLimit).filter_by(tier=tier).first()
            if not limit:
                print(f"Creating limit for {tier}")
                new_limit = SubscriptionLimit(
                    tier=tier,
                    max_signals_per_day=10 if tier == 'free' else 100,
                    max_api_calls_per_day=100 if tier == 'free' else 1000,
                    features={'email_alerts': tier != 'free'}
                )
                session.add(new_limit)
        session.commit()

        email = "testuser@example.com"
        password = "password123"
        
        print(f"Creating user {email}...")
        user, error = AuthService.create_user(session, email, password, full_name="Test User")
        
        if error:
            print(f"Error creating user: {error}")
        else:
            print(f"User created successfully: {user.email}")
            print(f"ID: {user.id}")

            # Verify password
            if AuthService.verify_password(password, user.password_hash):
                print("Password verification successful")
            else:
                print("Password verification FAILED")

if __name__ == "__main__":
    create_test_user()
