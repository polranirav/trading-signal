"""
Admin Service

Service layer for admin operations and audit logging.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID
from flask import request

from src.data.persistence import get_database
from src.admin.models import AuditLog, SystemSettings, AdminActivity
from src.logging_config import get_logger

logger = get_logger(__name__)


class AdminService:
    """
    Service for admin operations.
    """
    
    @staticmethod
    def log_audit(
        db_session,
        action: str,
        user_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditLog:
        """
        Create an audit log entry.
        
        Args:
            db_session: Database session
            action: Action performed (e.g., "user.login", "signal.create")
            user_id: ID of user who performed the action
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            details: Additional details (JSON)
            ip_address: IP address of request
            user_agent: User agent of request
        
        Returns:
            Created AuditLog instance
        """
        try:
            # Get IP and user agent from request if not provided
            if not ip_address and request:
                ip_address = request.remote_addr
            if not user_agent and request:
                user_agent = request.headers.get('User-Agent')
            
            audit_log = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent,
                created_at=datetime.utcnow(),
            )
            
            db_session.add(audit_log)
            db_session.flush()
            
            return audit_log
            
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}", exc_info=True)
            # Don't fail the main operation if audit logging fails
            return None
    
    @staticmethod
    def log_admin_activity(
        db_session,
        admin_id: UUID,
        action_type: str,
        target_type: Optional[str] = None,
        target_id: Optional[UUID] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> AdminActivity:
        """
        Create an admin activity log entry.
        
        Args:
            db_session: Database session
            admin_id: ID of admin who performed the action
            action_type: Type of action (e.g., "user.suspend", "signal.delete")
            target_type: Type of target resource
            target_id: ID of target resource
            description: Human-readable description
            metadata: Additional metadata (JSON)
            ip_address: IP address of request
        
        Returns:
            Created AdminActivity instance
        """
        try:
            # Get IP from request if not provided
            if not ip_address and request:
                ip_address = request.remote_addr
            
            activity = AdminActivity(
                admin_id=admin_id,
                action_type=action_type,
                target_type=target_type,
                target_id=target_id,
                description=description,
                metadata=metadata,
                ip_address=ip_address,
                created_at=datetime.utcnow(),
            )
            
            db_session.add(activity)
            db_session.flush()
            
            return activity
            
        except Exception as e:
            logger.error(f"Failed to create admin activity log: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_setting(db_session, key: str) -> Optional[SystemSettings]:
        """
        Get a system setting by key.
        
        Args:
            db_session: Database session
            key: Setting key
        
        Returns:
            SystemSettings instance or None
        """
        return db_session.query(SystemSettings).filter(SystemSettings.key == key).first()
    
    @staticmethod
    def set_setting(
        db_session,
        key: str,
        value: str,
        value_type: str = 'string',
        description: Optional[str] = None,
        is_public: bool = False,
        updated_by: Optional[UUID] = None,
    ) -> SystemSettings:
        """
        Set or update a system setting.
        
        Args:
            db_session: Database session
            key: Setting key
            value: Setting value
            value_type: Type of value (string, int, float, bool, json)
            description: Description
            is_public: Whether setting is public
            updated_by: ID of user updating the setting
        
        Returns:
            SystemSettings instance
        """
        setting = db_session.query(SystemSettings).filter(SystemSettings.key == key).first()
        
        if setting:
            # Update existing
            setting.value = value
            setting.value_type = value_type
            if description is not None:
                setting.description = description
            if is_public is not None:
                setting.is_public = is_public
            setting.updated_by = updated_by
            setting.updated_at = datetime.utcnow()
        else:
            # Create new
            setting = SystemSettings(
                key=key,
                value=value,
                value_type=value_type,
                description=description,
                is_public=is_public,
                updated_by=updated_by,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db_session.add(setting)
        
        db_session.flush()
        return setting
