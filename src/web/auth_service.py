"""
Authentication service for the Dash dashboard.

Handles user authentication, session management, and password hashing.
"""

import bcrypt
from flask import session
from datetime import datetime
from typing import Optional, Tuple

from src.logging_config import get_logger
from src.data.persistence import get_database

logger = get_logger(__name__)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(
            password.encode('utf-8'),
            password_hash.encode('utf-8')
        )
    except Exception:
        return False


def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[dict], str]:
    """
    Authenticate a user by email and password.
    
    Returns:
        Tuple of (success, user_dict, error_message)
    """
    if not email or not password:
        return False, None, "Email and password are required"
    
    try:
        db = get_database()
        
        with db.get_session() as db_session:
            from src.auth.models import User
            
            user = db_session.query(User).filter(
                User.email == email.lower().strip()
            ).first()
            
            if not user:
                return False, None, "Invalid email or password"
            
            if not user.is_active:
                return False, None, "Account is disabled"
            
            if not verify_password(password, user.password_hash):
                return False, None, "Invalid email or password"
            
            # Update last login
            user.last_login = datetime.utcnow()
            
            user_dict = {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "is_admin": user.is_admin
            }
            
            logger.info("User authenticated", email=email)
            return True, user_dict, ""
            
    except Exception as e:
        logger.error(f"Authentication error: {e}", exc_info=True)
        return False, None, "Authentication service error"


def register_user(email: str, password: str, full_name: str) -> Tuple[bool, str]:
    """
    Register a new user.
    
    Returns:
        Tuple of (success, error_message)
    """
    if not email or not password:
        return False, "Email and password are required"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    try:
        db = get_database()
        
        with db.get_session() as db_session:
            from src.auth.models import User, Subscription
            
            # Check if user exists
            existing = db_session.query(User).filter(
                User.email == email.lower().strip()
            ).first()
            
            if existing:
                return False, "An account with this email already exists"
            
            # Create user
            user = User(
                email=email.lower().strip(),
                password_hash=hash_password(password),
                full_name=full_name.strip() if full_name else None,
                is_active=True,
                email_verified=False
            )
            db_session.add(user)
            db_session.flush()
            
            # Create free subscription
            subscription = Subscription(
                user_id=user.id,
                tier="free",
                status="active"
            )
            db_session.add(subscription)
            
            logger.info("User registered", email=email)
            return True, ""
            
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return False, "Registration failed. Please try again."


def login_user(user_dict: dict, remember: bool = False) -> None:
    """Store user in session."""
    session["user"] = user_dict
    session["authenticated"] = True
    session.permanent = remember


def logout_user() -> None:
    """Clear user session."""
    session.pop("user", None)
    session.pop("authenticated", None)


def get_current_user() -> Optional[dict]:
    """Get the currently authenticated user from session."""
    if session.get("authenticated"):
        return session.get("user")
    return None


def is_authenticated() -> bool:
    """Check if user is authenticated."""
    return session.get("authenticated", False)
