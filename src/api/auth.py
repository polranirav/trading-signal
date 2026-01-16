"""
Authentication API Endpoints.

Handles user registration, login, logout, and session management.
"""

from flask import Blueprint, request, session as flask_session, g
from typing import Dict, Optional
from datetime import datetime

from src.api.utils import success_response, error_response, require_auth_api
from src.data.persistence import get_database
from src.auth.service import AuthService
from src.logging_config import get_logger

logger = get_logger(__name__)

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/auth/register', methods=['POST'])
def register():
    """Register a new user."""
    try:
        data = request.get_json()
        
        if not data:
            return error_response("Request body required", 400)
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        
        # Validation
        if not email or '@' not in email:
            return error_response("Valid email required", 400)
        
        if not password or len(password) < 8:
            return error_response("Password must be at least 8 characters", 400)
        
        # Create user
        db = get_database()
        with db.get_session() as db_session:
            user, error = AuthService.create_user(
                db_session,
                email=email,
                password=password,
                full_name=full_name if full_name else None
            )
            
            if error:
                return error_response(error, 400)
            
            if not user:
                return error_response("Failed to create user", 500)
            
            # Create free subscription
            from src.auth.models import Subscription
            # Create free subscription
            free_sub = Subscription(
                user_id=user.id,
                tier='free',
                status='active'
            )
            db_session.add(free_sub)
            db_session.commit()
            
            # Set session
            flask_session['user_id'] = str(user.id)
            flask_session.permanent = True
            
            logger.info(f"User registered: {email}")
            
            return success_response(
                data={
                    "user": {
                        "id": str(user.id),
                        "email": user.email,
                        "full_name": user.full_name,
                        "email_verified": user.email_verified,
                        "is_admin": user.is_admin,
                        "is_active": user.is_active,
                        "created_at": user.created_at.isoformat() if user.created_at else None,
                        "last_login": user.last_login.isoformat() if user.last_login else None
                    }
                },
                message="User registered successfully",
                status_code=201
            )
            
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return error_response("Registration failed", 500)


@auth_bp.route('/auth/login', methods=['POST'])
def login():
    """Authenticate user and create session."""
    with open('/tmp/login_debug.log', 'a') as f:
        f.write(f"\\n[{datetime.utcnow()}] Login request received\\n")
    try:
        data = request.get_json()
        
        if not data:
            return error_response("Request body required", 400)
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        with open('/tmp/login_debug.log', 'a') as f:
            f.write(f"[{datetime.utcnow()}] Attempting login for {email}\\n")
        
        if not email or not password:
            return error_response("Email and password required", 400)
        
        # Authenticate
        db = get_database()
        with db.get_session() as db_session:
            with open('/tmp/login_debug.log', 'a') as f:
                f.write(f"[{datetime.utcnow()}] Session acquired, calling authenticate_user\\n")
                
            user = AuthService.authenticate_user(db_session, email, password)
            
            with open('/tmp/login_debug.log', 'a') as f:
                f.write(f"[{datetime.utcnow()}] Authenticated: {bool(user)}\\n")
            
            if not user:
                return error_response("Invalid email or password", 401)
            
            if not user.is_active:
                return error_response("Account is inactive", 403)
            
            # Update last login
            # Update last login
            user.last_login = datetime.utcnow()
            db_session.commit()
            
            with open('/tmp/login_debug.log', 'a') as f:
                f.write(f"[{datetime.utcnow()}] DB commit success. Setting session.\\n")
            
            # Set session
            flask_session['user_id'] = str(user.id)
            flask_session.permanent = True
            
            logger.info(f"User logged in: {email}")
            
            with open('/tmp/login_debug.log', 'a') as f:
                f.write(f"[{datetime.utcnow()}] Session set. Getting tier.\\n")
            
            # Get subscription info
            try:
                from src.subscriptions.service import SubscriptionService
                tier = SubscriptionService.get_user_tier(db_session, user.id)
                with open('/tmp/login_debug.log', 'a') as f:
                    f.write(f"[{datetime.utcnow()}] Tier: {tier}\\n")
            except Exception as sub_e:
                with open('/tmp/login_debug.log', 'a') as f:
                    f.write(f"[{datetime.utcnow()}] Subscription check failed: {sub_e}\\n")
                # Fallback to free if this fails? Or re-raise?
                # Let's log and re-raise to be sure.
                raise sub_e
            
            return success_response(
                data={
                    "user": {
                        "id": str(user.id),
                        "email": user.email,
                        "full_name": user.full_name,
                        "email_verified": user.email_verified,
                        "is_admin": user.is_admin,
                        "is_active": user.is_active,
                        "tier": tier,
                        "created_at": user.created_at.isoformat() if user.created_at else None,
                        "last_login": user.last_login.isoformat() if user.last_login else None
                    }
                },
                message="Login successful"
            )
            
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        with open('/tmp/login_debug.log', 'a') as f:
            f.write(f"[{datetime.utcnow()}] CRASH: {e}\\n{tb}\\n")
        logger.error(f"Login error: {e}", exc_info=True)
        return error_response("Login failed", 500)


@auth_bp.route('/auth/logout', methods=['POST'])
@require_auth_api
def logout():
    """Logout user and clear session."""
    try:
        flask_session.clear()
        return success_response(message="Logout successful")
    except Exception as e:
        logger.error(f"Logout error: {e}", exc_info=True)
        return error_response("Logout failed", 500)


@auth_bp.route('/auth/me', methods=['GET'])
@require_auth_api
def get_current_user_info():
    """Get current authenticated user information."""
    try:
        user = g.current_user
        
        # Get subscription
        db = get_database()
        with db.get_session() as db_session:
            from src.subscriptions.service import SubscriptionService
            tier = SubscriptionService.get_user_tier(db_session, user.id)
            limits = SubscriptionService.get_user_limits(db_session, user.id)
            
            return success_response(
                data={
                    "user": {
                        "id": str(user.id),
                        "email": user.email,
                        "full_name": user.full_name,
                        "email_verified": user.email_verified,
                        "is_admin": user.is_admin,
                        "is_active": user.is_active,
                        "tier": tier,
                        "limits": limits,
                        "created_at": user.created_at.isoformat() if user.created_at else None,
                        "last_login": user.last_login.isoformat() if user.last_login else None
                    }
                }
            )
    except Exception as e:
        logger.error(f"Get user info error: {e}", exc_info=True)
        return error_response("Failed to get user info", 500)
