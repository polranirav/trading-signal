"""
Admin Models

Database models for system administration, auditing, and settings.
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from src.data.models import Base


class AuditLog(Base):
    """
    Audit log for tracking system events and user actions.
    """
    __tablename__ = 'audit_logs'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)  # e.g., "user.login", "signal.create"
    resource_type = Column(String(100), nullable=True, index=True)  # e.g., "user", "signal", "subscription"
    resource_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    details = Column(JSON, nullable=True)  # Additional context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", foreign_keys=[user_id], lazy="select")

    def __repr__(self):
        return f"<AuditLog {self.action} by {self.user_id}>"


class SystemSettings(Base):
    """
    System-wide configuration settings.
    """
    __tablename__ = 'system_settings'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=True)
    value_type = Column(String(50), nullable=False, default='string')  # string, int, float, bool, json
    description = Column(Text, nullable=True)
    is_public = Column(Boolean, default=False)  # Whether setting is visible to non-admins
    updated_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    # Relationships
    updater = relationship("User", foreign_keys=[updated_by], lazy="select")

    def __repr__(self):
        return f"<SystemSettings {self.key}={self.value}>"


class AdminActivity(Base):
    """
    Tracks administrative actions performed by admin users.
    """
    __tablename__ = 'admin_activities'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    admin_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=False, index=True)
    action_type = Column(String(100), nullable=False, index=True)  # e.g., "user.suspend", "signal.delete"
    target_type = Column(String(100), nullable=True)  # e.g., "user", "signal"
    target_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    description = Column(Text, nullable=True)
    meta_data = Column(JSON, nullable=True)  # Additional details (renamed from 'metadata' as it's reserved in SQLAlchemy)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    admin = relationship("User", foreign_keys=[admin_id], lazy="select")

    def __repr__(self):
        return f"<AdminActivity {self.action_type} by {self.admin_id}>"
