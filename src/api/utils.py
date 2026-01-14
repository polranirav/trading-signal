"""
API Utilities.

Common utilities for API responses, error handling, and authentication.
"""

from flask import jsonify
from functools import wraps
from typing import Optional, Dict, Any
from datetime import datetime

from src.auth.middleware import get_current_user
from src.logging_config import get_logger

logger = get_logger(__name__)


def success_response(data: Any = None, message: str = "Success", status_code: int = 200):
    """
    Create a successful JSON response.
    
    Args:
        data: Response data
        message: Success message
        status_code: HTTP status code
    
    Returns:
        Flask JSON response
    """
    response = {
        "success": True,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if data is not None:
        response["data"] = data
    
    return jsonify(response), status_code


def error_response(message: str = "Error", status_code: int = 400, errors: Dict = None):
    """
    Create an error JSON response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        errors: Optional error details
    
    Returns:
        Flask JSON response
    """
    response = {
        "success": False,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if errors:
        response["errors"] = errors
    
    return jsonify(response), status_code


def require_auth_api(f):
    """
    Decorator to require authentication for API endpoints.
    
    Usage:
        @require_auth_api
        def my_endpoint():
            user = g.current_user  # Access current user
            ...
    """
    from flask import g
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        
        if not user:
            return error_response("Authentication required", 401)
        
        if not user.is_active:
            return error_response("Account is inactive", 403)
        
        # Store user in Flask g for easy access
        g.current_user = user
        
        return f(*args, **kwargs)
    
    return decorated_function


def require_api_key_or_auth(f):
    """
    Decorator to allow API key OR session authentication.
    
    Usage:
        @require_api_key_or_auth
        def my_endpoint():
            user = g.current_user
            ...
    """
    from flask import g, request
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Try API key first
        api_key = request.headers.get('X-API-Key')
        
        if api_key:
            from src.auth.service import AuthService
            from src.data.persistence import get_database
            
            db = get_database()
            with db.get_session() as session:
                user = AuthService.verify_api_key(session, api_key)
                
                if user and user.is_active:
                    g.current_user = user
                    return f(*args, **kwargs)
        
        # Fall back to session auth
        user = get_current_user()
        
        if not user:
            return error_response("Authentication required", 401)
        
        if not user.is_active:
            return error_response("Account is inactive", 403)
        
        g.current_user = user
        return f(*args, **kwargs)
    
    return decorated_function
