"""
Admin API Endpoints.

REST API endpoints for system administration.
"""

from flask import Blueprint, request, g
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from uuid import UUID

from src.api.utils import success_response, error_response, require_auth_api
from src.data.persistence import get_database
from src.auth.service import AuthService
from src.logging_config import get_logger

logger = get_logger(__name__)

admin_bp = Blueprint('admin', __name__)


def require_admin(f):
    """
    Decorator to require admin role.
    """
    from functools import wraps
    
    @wraps(f)
    @require_auth_api
    def decorated_function(*args, **kwargs):
        user = g.current_user
        
        # Check if user is admin (you can customize this logic)
        db = get_database()
        with db.get_session() as db_session:
            from src.subscriptions.service import SubscriptionService
            tier = SubscriptionService.get_user_tier(db_session, user.id)
            
            # For now, assume 'advanced' tier is admin
            # You can add a proper role system later
            if tier and tier.lower() not in ['advanced', 'admin']:
                return error_response("Admin access required", 403)
        
        return f(*args, **kwargs)
    
    return decorated_function


@admin_bp.route('/admin/audit-logs', methods=['GET'])
@require_admin
def get_audit_logs():
    """
    Get audit logs.
    
    Query parameters:
    - limit: Number of logs (default: 50, max: 500)
    - user_id: Filter by user ID (optional)
    - action: Filter by action (optional)
    - resource_type: Filter by resource type (optional)
    - days: Number of days to look back (default: 7) (optional)
    """
    try:
        limit = min(int(request.args.get('limit', 50)), 500)
        user_id = request.args.get('user_id', '').strip()
        action = request.args.get('action', '').strip()
        resource_type = request.args.get('resource_type', '').strip()
        days = int(request.args.get('days', 7))
        
        db = get_database()
        with db.get_session() as session:
            from src.admin.models import AuditLog
            
            query = session.query(AuditLog)
            
            # Filter by user
            if user_id:
                try:
                    user_uuid = UUID(user_id)
                    query = query.filter(AuditLog.user_id == user_uuid)
                except ValueError:
                    return error_response("Invalid user_id", 400)
            
            # Filter by action
            if action:
                query = query.filter(AuditLog.action == action)
            
            # Filter by resource type
            if resource_type:
                query = query.filter(AuditLog.resource_type == resource_type)
            
            # Filter by date
            if days:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                query = query.filter(AuditLog.created_at >= cutoff_date)
            
            # Order by date (newest first)
            query = query.order_by(AuditLog.created_at.desc())
            
            # Limit results
            logs = query.limit(limit).all()
            
            # Serialize logs
            logs_data = []
            for log in logs:
                logs_data.append({
                    "id": str(log.id),
                    "user_id": str(log.user_id) if log.user_id else None,
                    "action": log.action,
                    "resource_type": log.resource_type,
                    "resource_id": str(log.resource_id) if log.resource_id else None,
                    "details": log.details,
                    "ip_address": log.ip_address,
                    "user_agent": log.user_agent,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                })
            
            return success_response(
                data={
                    "logs": logs_data,
                    "count": len(logs_data)
                }
            )
            
    except ValueError as e:
        return error_response(f"Invalid parameter: {str(e)}", 400)
    except Exception as e:
        logger.error(f"Get audit logs error: {e}", exc_info=True)
        return error_response("Failed to get audit logs", 500)


@admin_bp.route('/admin/settings', methods=['GET'])
@require_admin
def get_settings():
    """
    Get system settings.
    
    Query parameters:
    - public_only: Only return public settings (default: false)
    """
    try:
        public_only = request.args.get('public_only', 'false').lower() == 'true'
        
        db = get_database()
        with db.get_session() as session:
            from src.admin.models import SystemSettings
            
            query = session.query(SystemSettings)
            
            if public_only:
                query = query.filter(SystemSettings.is_public == True)
            
            settings = query.all()
            
            # Serialize settings
            settings_data = []
            for setting in settings:
                settings_data.append({
                    "id": str(setting.id),
                    "key": setting.key,
                    "value": setting.value,
                    "value_type": setting.value_type,
                    "description": setting.description,
                    "is_public": setting.is_public,
                    "updated_by": str(setting.updated_by) if setting.updated_by else None,
                    "updated_at": setting.updated_at.isoformat() if setting.updated_at else None,
                    "created_at": setting.created_at.isoformat() if setting.created_at else None,
                })
            
            return success_response(data={"settings": settings_data})
            
    except Exception as e:
        logger.error(f"Get settings error: {e}", exc_info=True)
        return error_response("Failed to get settings", 500)


@admin_bp.route('/admin/settings/<key>', methods=['GET'])
@require_admin
def get_setting(key: str):
    """
    Get a specific system setting by key.
    """
    try:
        db = get_database()
        with db.get_session() as session:
            from src.admin.models import SystemSettings
            
            setting = session.query(SystemSettings).filter(SystemSettings.key == key).first()
            
            if not setting:
                return error_response("Setting not found", 404)
            
            setting_data = {
                "id": str(setting.id),
                "key": setting.key,
                "value": setting.value,
                "value_type": setting.value_type,
                "description": setting.description,
                "is_public": setting.is_public,
                "updated_by": str(setting.updated_by) if setting.updated_by else None,
                "updated_at": setting.updated_at.isoformat() if setting.updated_at else None,
                "created_at": setting.created_at.isoformat() if setting.created_at else None,
            }
            
            return success_response(data={"setting": setting_data})
            
    except Exception as e:
        logger.error(f"Get setting error: {e}", exc_info=True)
        return error_response("Failed to get setting", 500)


@admin_bp.route('/admin/settings/<key>', methods=['PUT'])
@require_admin
def update_setting(key: str):
    """
    Update a system setting.
    
    Request body:
    - value: New value (required)
    - description: Description (optional)
    - is_public: Whether setting is public (optional)
    """
    try:
        user = g.current_user
        data = request.get_json()
        
        if not data:
            return error_response("Request body required", 400)
        
        if 'value' not in data:
            return error_response("value is required", 400)
        
        db = get_database()
        with db.get_session() as session:
            from src.admin.models import SystemSettings
            
            setting = session.query(SystemSettings).filter(SystemSettings.key == key).first()
            
            if not setting:
                return error_response("Setting not found", 404)
            
            # Update value
            setting.value = str(data['value'])
            
            if 'description' in data:
                setting.description = data['description']
            
            if 'is_public' in data:
                setting.is_public = bool(data['is_public'])
            
            setting.updated_by = user.id
            setting.updated_at = datetime.utcnow()
            
            session.commit()
            
            logger.info(f"Setting updated: {key} by {user.email}")
            
            setting_data = {
                "id": str(setting.id),
                "key": setting.key,
                "value": setting.value,
                "value_type": setting.value_type,
                "description": setting.description,
                "is_public": setting.is_public,
                "updated_by": str(setting.updated_by) if setting.updated_by else None,
                "updated_at": setting.updated_at.isoformat() if setting.updated_at else None,
            }
            
            return success_response(
                data={"setting": setting_data},
                message="Setting updated successfully"
            )
            
    except Exception as e:
        logger.error(f"Update setting error: {e}", exc_info=True)
        return error_response("Failed to update setting", 500)


@admin_bp.route('/admin/activities', methods=['GET'])
@require_admin
def get_admin_activities():
    """
    Get admin activity logs.
    
    Query parameters:
    - limit: Number of activities (default: 50, max: 500)
    - admin_id: Filter by admin ID (optional)
    - action_type: Filter by action type (optional)
    - days: Number of days to look back (default: 30) (optional)
    """
    try:
        limit = min(int(request.args.get('limit', 50)), 500)
        admin_id = request.args.get('admin_id', '').strip()
        action_type = request.args.get('action_type', '').strip()
        days = int(request.args.get('days', 30))
        
        db = get_database()
        with db.get_session() as session:
            from src.admin.models import AdminActivity
            
            query = session.query(AdminActivity)
            
            # Filter by admin
            if admin_id:
                try:
                    admin_uuid = UUID(admin_id)
                    query = query.filter(AdminActivity.admin_id == admin_uuid)
                except ValueError:
                    return error_response("Invalid admin_id", 400)
            
            # Filter by action type
            if action_type:
                query = query.filter(AdminActivity.action_type == action_type)
            
            # Filter by date
            if days:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                query = query.filter(AdminActivity.created_at >= cutoff_date)
            
            # Order by date (newest first)
            query = query.order_by(AdminActivity.created_at.desc())
            
            # Limit results
            activities = query.limit(limit).all()
            
            # Serialize activities
            activities_data = []
            for activity in activities:
                activities_data.append({
                    "id": str(activity.id),
                    "admin_id": str(activity.admin_id) if activity.admin_id else None,
                    "action_type": activity.action_type,
                    "target_type": activity.target_type,
                    "target_id": str(activity.target_id) if activity.target_id else None,
                    "description": activity.description,
                    "metadata": activity.meta_data,
                    "ip_address": activity.ip_address,
                    "created_at": activity.created_at.isoformat() if activity.created_at else None,
                })
            
            return success_response(
                data={
                    "activities": activities_data,
                    "count": len(activities_data)
                }
            )
            
    except ValueError as e:
        return error_response(f"Invalid parameter: {str(e)}", 400)
    except Exception as e:
        logger.error(f"Get admin activities error: {e}", exc_info=True)
        return error_response("Failed to get admin activities", 500)


@admin_bp.route('/admin/users', methods=['GET'])
@require_admin
def get_users():
    """
    Get list of users (admin only).
    
    Query parameters:
    - limit: Number of users (default: 50, max: 500)
    - email: Filter by email (optional, partial match)
    - tier: Filter by subscription tier (optional)
    """
    try:
        limit = min(int(request.args.get('limit', 50)), 500)
        email = request.args.get('email', '').strip()
        tier = request.args.get('tier', '').strip()
        
        db = get_database()
        with db.get_session() as session:
            from src.auth.models import User
            
            query = session.query(User)
            
            # Filter by email
            if email:
                query = query.filter(User.email.ilike(f'%{email}%'))
            
            # Limit results
            users = query.limit(limit).all()
            
            # Get user tiers and serialize
            from src.subscriptions.service import SubscriptionService
            users_data = []
            for user in users:
                user_tier = SubscriptionService.get_user_tier(session, user.id)
                
                # Filter by tier if specified
                if tier and user_tier != tier:
                    continue
                
                users_data.append({
                    "id": str(user.id),
                    "email": user.email,
                    "full_name": user.full_name,
                    "email_verified": user.email_verified,
                    "tier": user_tier,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                })
            
            return success_response(
                data={
                    "users": users_data,
                    "count": len(users_data)
                }
            )
            
    except ValueError as e:
        return error_response(f"Invalid parameter: {str(e)}", 400)
    except Exception as e:
        logger.error(f"Get users error: {e}", exc_info=True)
        return error_response("Failed to get users", 500)
