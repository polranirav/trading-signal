
import sys
import os
sys.path.append('.')

from src.data.persistence import get_database
from src.auth.service import AuthService
from src.auth.models import User

def reset_password(email="testuser@example.com", new_password="password123"):
    try:
        db_manager = get_database()
        
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.email == email).first()
            
            if not user:
                print(f"User {email} not found!")
                return

            print(f"Resetting password for {user.email}...")
            user.password_hash = AuthService.hash_password(new_password)
            session.commit()
            print(f"Password reset successfully to '{new_password}'")
        
    except Exception as e:
        print(f"Error resetting password: {e}")

if __name__ == "__main__":
    reset_password()
