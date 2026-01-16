
import sys
import os
sys.path.append('.')

from src.data.persistence import get_database

def list_users():
    try:
        db = get_database()
        users = db.execute_query("SELECT id, email, created_at FROM users")
        print(f"Found {len(users)} users:")
        for user in users:
            print(f"- {user['email']} (ID: {user['id']})")
    except Exception as e:
        print(f"Error checking users: {e}")

if __name__ == "__main__":
    list_users()
