"""
Admin Utilities.

Provides decorators and helper functions for admin operations:
- require_admin: Decorator to restrict endpoints to admin users
- log_admin_action: Helper to log admin actions to audit log
- get_request_metadata: Extract request metadata for logging
"""

from flask import request, g
from functools import wraps
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from src.api.utils import error_response
from src.auth.middleware import get_current_user
from src.logging_config import get_logger

logger = get_logger(__name__)


def require_admin(f):
    """
    Decorator to require admin authentication for API endpoints.
    
    Usage:
        @require_admin
        def my_admin_endpoint():
            admin = g.current_user  # Access current admin user
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
        
        if not user.is_admin:
            logger.warning(f"Non-admin user {user.email} attempted to access admin endpoint")
            return error_response("Admin access required", 403)
        
        # Store user in Flask g for easy access
        g.current_user = user
        g.is_admin = True
        
        return f(*args, **kwargs)
    
    return decorated_function


def get_request_metadata() -> Dict[str, Any]:
    """
    Extract metadata from current request for audit logging.
    
    Returns:
        Dictionary with request metadata
    """
    return {
        "ip_address": request.remote_addr or request.headers.get("X-Forwarded-For", "unknown"),
        "user_agent": request.headers.get("User-Agent"),
        "request_path": request.path,
        "request_method": request.method,
    }


def log_admin_action(
    db_session,
    admin_id: UUID,
    admin_email: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    changes: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    success: bool = True,
    error_message: Optional[str] = None
) -> None:
    """
    Log an admin action to the audit log.
    
    Args:
        db_session: SQLAlchemy session
        admin_id: ID of admin user performing action
        admin_email: Email of admin (denormalized)
        action: Action name (e.g., "user.update", "signal.delete")
        resource_type: Type of resource (e.g., "user", "signal")
        resource_id: ID of affected resource
        changes: Dictionary with "old" and "new" values
        metadata: Additional context
        success: Whether action succeeded
        error_message: Error message if action failed
    """
    from src.admin.models import AuditLog
    
    try:
        request_meta = get_request_metadata()
        
        audit_log = AuditLog(
            admin_id=admin_id,
            admin_email=admin_email,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=request_meta.get("ip_address"),
            user_agent=request_meta.get("user_agent"),
            request_path=request_meta.get("request_path"),
            request_method=request_meta.get("request_method"),
            changes=changes,
            metadata=metadata,
            success=success,
            error_message=error_message
        )
        
        db_session.add(audit_log)
        db_session.flush()
        
    except Exception as e:
        logger.error(f"Failed to log admin action: {e}", exc_info=True)
        # Don't fail the request if audit logging fails


def log_admin_activity(
    db_session,
    admin_id: UUID,
    activity_type: str,
    description: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Log admin activity (login, logout, page views, etc.).
    
    Args:
        db_session: SQLAlchemy session
        admin_id: ID of admin user
        activity_type: Type of activity (e.g., "login", "dashboard_view")
        description: Human-readable description
        metadata: Additional context
    """
    from src.admin.models import AdminActivity
    
    try:
        request_meta = get_request_metadata()
        
        activity = AdminActivity(
            admin_id=admin_id,
            activity_type=activity_type,
            description=description,
            ip_address=request_meta.get("ip_address"),
            user_agent=request_meta.get("user_agent"),
            metadata=metadata
        )
        
        db_session.add(activity)
        db_session.flush()
        
    except Exception as e:
        logger.error(f"Failed to log admin activity: {e}", exc_info=True)
