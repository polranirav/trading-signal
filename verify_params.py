
import sys
sys.path.append('.')
from src.data.persistence import get_database

def verify_params():
    db = get_database()
    print("Testing execute_query with tuple params and %s...")
    try:
        # This mirrors the correct usage: using :email and a dict
        
        result = db.execute_query("SELECT email FROM users WHERE email = :email LIMIT 1", {'email': 'testuser@example.com'})
        print("Result:", result)
        print("SUCCESS: Dict params worked!")
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_params()
