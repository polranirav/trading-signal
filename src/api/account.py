"""
Account API Endpoints.

Handles user account management, profile, and API keys.
"""

from flask import Blueprint, request, g
from typing import Dict, Optional
from uuid import UUID

from src.api.utils import success_response, error_response, require_auth_api
from src.data.persistence import get_database
from src.auth.service import AuthService
from src.logging_config import get_logger

logger = get_logger(__name__)

account_bp = Blueprint('account', __name__)


@account_bp.route('/account', methods=['GET'])
@require_auth_api
def get_account():
    """Get current user account information."""
    try:
        user = g.current_user
        
        db = get_database()
        with db.get_session() as db_session:
            # Get subscription
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
        logger.error(f"Get account error: {e}", exc_info=True)
        return error_response("Failed to get account", 500)


@account_bp.route('/account', methods=['PUT'])
@require_auth_api
def update_account():
    """Update user account information."""
    try:
        user = g.current_user
        data = request.get_json()
        
        if not data:
            return error_response("Request body required", 400)
        
        db = get_database()
        with db.get_session() as db_session:
            # Update full name
            if 'full_name' in data:
                user.full_name = data['full_name'].strip() if data['full_name'] else None
            
            db_session.commit()
            
            logger.info(f"Account updated: {user.email}")
            
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
                message="Account updated successfully"
            )
    except Exception as e:
        logger.error(f"Update account error: {e}", exc_info=True)
        return error_response("Failed to update account", 500)


@account_bp.route('/account/api-keys', methods=['GET'])
@require_auth_api
def get_api_keys():
    """Get user's API keys."""
    try:
        user = g.current_user
        
        db = get_database()
        with db.get_session() as db_session:
            from src.auth.models import APIKey
            api_keys = db_session.query(APIKey).filter(
                APIKey.user_id == user.id
            ).order_by(APIKey.created_at.desc()).all()
            
            keys_data = []
            for key in api_keys:
                keys_data.append({
                    "id": str(key.id),
                    "key_prefix": key.key_prefix,
                    "name": key.name,
                    "last_used": key.last_used.isoformat() if key.last_used else None,
                    "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                    "created_at": key.created_at.isoformat() if key.created_at else None,
                })
            
            return success_response(data={"api_keys": keys_data})
    except Exception as e:
        logger.error(f"Get API keys error: {e}", exc_info=True)
        return error_response("Failed to get API keys", 500)


@account_bp.route('/account/api-keys', methods=['POST'])
@require_auth_api
def create_api_key():
    """Create a new API key."""
    try:
        user = g.current_user
        data = request.get_json() or {}
        
        name = data.get('name', '').strip() if data.get('name') else None
        expires_days = int(data.get('expires_days', 365)) if data.get('expires_days') else None
        
        db = get_database()
        with db.get_session() as db_session:
            api_key, error = AuthService.create_api_key(
                db_session,
                user_id=user.id,
                name=name,
                expires_days=expires_days
            )
            
            if error:
                return error_response(error, 400)
            
            if not api_key:
                return error_response("Failed to create API key", 500)
            
            logger.info(f"API key created for user: {user.email}")
            
            # Return the key (only shown once!)
            return success_response(
                data={
                    "api_key": api_key,  # Full key (only time it's shown)
                    "key_prefix": api_key.split('.')[0] if '.' in api_key else api_key[:8],
                    "name": name,
                    "expires_days": expires_days
                },
                message="API key created successfully. Save this key - it won't be shown again!",
                status_code=201
            )
    except ValueError as e:
        return error_response(f"Invalid parameter: {str(e)}", 400)
    except Exception as e:
        logger.error(f"Create API key error: {e}", exc_info=True)
        return error_response("Failed to create API key", 500)


@account_bp.route('/account/api-keys/<key_id>', methods=['DELETE'])
@require_auth_api
def delete_api_key(key_id: str):
    """Delete an API key."""
    try:
        user = g.current_user
        
        try:
            key_uuid = UUID(key_id)
        except ValueError:
            return error_response("Invalid API key ID", 400)
        
        db = get_database()
        with db.get_session() as db_session:
            from src.auth.models import APIKey
            api_key = db_session.query(APIKey).filter(
                APIKey.id == key_uuid,
                APIKey.user_id == user.id
            ).first()
            
            if not api_key:
                return error_response("API key not found", 404)
            
            db_session.delete(api_key)
            db_session.commit()
            
            logger.info(f"API key deleted: {key_id}")
            
            return success_response(message="API key deleted successfully")
    except Exception as e:
        logger.error(f"Delete API key error: {e}", exc_info=True)
        return error_response("Failed to delete API key", 500)
