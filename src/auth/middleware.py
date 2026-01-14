"""
Authentication Middleware.

Provides decorators and utilities for protecting routes and API endpoints.
"""

from functools import wraps
from typing import Optional, Callable, Any
from flask import session, request, g, abort
from sqlalchemy.orm import Session

from src.logging_config import get_logger
from src.auth.models import User
from src.auth.service import AuthService
from src.data.persistence import get_database

logger = get_logger(__name__)


def get_current_user() -> Optional[User]:
    """
    Get the current authenticated user from session or API key.
    
    This function checks:
    1. Flask session (for web dashboard)
    2. API key header (for API requests)
    
    Returns:
        User object if authenticated, None otherwise
    """
    db = get_database()
    
    # Check Flask session (web dashboard)
    if 'user_id' in session:
        try:
            with db.get_session() as db_session:
                user = AuthService.get_user_by_id(db_session, session['user_id'])
                if user and user.is_active:
                    # Expunge to detach from session before it closes
                    db_session.expunge(user)
                    return user
        except Exception as e:
            logger.error(f"Error getting user from session: {e}")
    
    # Check API key (API requests)
    api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization', '').replace('Bearer ', '')
    if api_key:
        try:
            with db.get_session() as db_session:
                user = AuthService.verify_api_key(db_session, api_key)
                if user:
                    # Expunge to detach from session before it closes
                    db_session.expunge(user)
                    return user
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
    
    return None


def require_auth(f: Callable = None, api_key_only: bool = False):
    """
    Decorator to require authentication for a route.
    
    Usage:
        @require_auth
        def protected_route():
            user = g.current_user
            return f"Hello {user.email}"
    
    Args:
        f: Function to decorate
        api_key_only: If True, only allow API key auth (not session)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = get_current_user()
            
            if not user:
                if request.is_json or api_key_only:
                    # API request - return 401 JSON
                    abort(401, description="Authentication required")
                else:
                    # Web request - redirect to login
                    from flask import redirect, url_for
                    return redirect('/login?next=' + request.path)
            
            # Store user in Flask g for access in route
            g.current_user = user
            
            return func(*args, **kwargs)
        return wrapper
    
    if f is None:
        return decorator
    else:
        return decorator(f)


def require_subscription(tier: str = 'essential'):
    """
    Decorator to require a specific subscription tier.
    
    Usage:
        @require_subscription(tier='essential')
        def premium_feature():
            user = g.current_user
            return "Premium content"
    
    Args:
        tier: Minimum required tier ('free', 'essential', 'advanced', 'premium')
    """
    tier_levels = {'free': 0, 'essential': 1, 'advanced': 2, 'premium': 3}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @require_auth
        def wrapper(*args, **kwargs):
            user = g.current_user
            db = get_database()
            
            with db.get_session() as session:
                subscription = AuthService.get_user_subscription(session, user.id)
                
                if not subscription:
                    abort(403, description="No active subscription")
                
                user_tier_level = tier_levels.get(subscription.tier, 0)
                required_tier_level = tier_levels.get(tier, 0)
                
                if user_tier_level < required_tier_level:
                    abort(403, description=f"Subscription tier '{tier}' required")
            
            return func(*args, **kwargs)
        return wrapper
    
    return decorator
