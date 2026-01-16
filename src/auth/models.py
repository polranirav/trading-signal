"""
Authentication-related SQLAlchemy models.

This module extends the base models with user authentication tables.
Models are imported into src/data/models.py for unified database initialization.
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import DECIMAL
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

# Import Base from data.models to ensure all models are in same registry
from src.data.models import Base


class User(Base):
    """
    User accounts for the trading signals platform.
    
    Stores user credentials and profile information.
    """
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)  # bcrypt hash
    full_name = Column(String(255))
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreferences", uselist=False, backref="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.email}>"


class Subscription(Base):
    """
    User subscription tiers and status.
    
    Links users to subscription tiers (free, essential, advanced, premium)
    and tracks subscription status via Stripe.
    """
    __tablename__ = "subscriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Subscription tier
    tier = Column(String(50), nullable=False, index=True)  # 'free', 'essential', 'advanced', 'premium'
    status = Column(String(50), nullable=False, index=True)  # 'active', 'cancelled', 'expired', 'trial', 'past_due'
    
    # Stripe integration
    stripe_subscription_id = Column(String(255), unique=True, index=True)
    stripe_customer_id = Column(String(255), index=True)
    
    # Billing period
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    cancel_at_period_end = Column(Boolean, default=False)
    
    # Trial period
    trial_start = Column(DateTime)
    trial_end = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    
    def __repr__(self):
        return f"<Subscription {self.user_id} {self.tier} ({self.status})>"


class APIKey(Base):
    """
    API keys for programmatic access.
    
    Allows users to access signals via REST API.
    """
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # API key (hashed for storage)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    key_prefix = Column(String(16), nullable=False)  # First 8 chars for display (ts_xxxx...)
    
    # Metadata
    name = Column(String(255))  # User-friendly name
    last_used = Column(DateTime)
    expires_at = Column(DateTime, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<APIKey {self.key_prefix}... (user: {self.user_id})>"


class SubscriptionLimit(Base):
    """
    Feature limits for each subscription tier.
    
    Configuration table that defines what each tier can access.
    """
    __tablename__ = "subscription_limits"
    
    tier = Column(String(50), primary_key=True, nullable=False)  # 'free', 'essential', 'advanced', 'premium'
    
    # Usage limits
    max_signals_per_day = Column(Integer, nullable=False, default=0)
    max_api_calls_per_day = Column(Integer, nullable=False, default=0)
    
    # Feature flags (JSONB for flexibility)
    features = Column(JSON, nullable=False, default=dict)  # {'email_alerts': True, 'api_access': False, ...}
    
    # Pricing
    price_monthly = Column(DECIMAL(10, 2), nullable=False, default=0)
    price_yearly = Column(DECIMAL(10, 2), nullable=False, default=0)
    
    # Stripe Price IDs
    stripe_price_id_monthly = Column(String(255))
    stripe_price_id_yearly = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SubscriptionLimit {self.tier}>"


class UserWatchlist(Base):
    """
    User's personal watchlist of stocks.
    
    Tracks which stocks a user wants to monitor for signals.
    """
    __tablename__ = "user_watchlists"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Alert settings for this stock
    alerts_enabled = Column(Boolean, default=True)
    email_alerts = Column(Boolean, default=False)
    
    # User notes
    notes = Column(String(500))
    
    # Timestamps
    added_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Composite unique constraint
    __table_args__ = (
        {'extend_existing': True},
    )
    
    def __repr__(self):
        return f"<UserWatchlist {self.user_id} -> {self.symbol}>"


class UserPreferences(Base):
    """
    User preferences and onboarding state.
    
    Stores user settings and tracks onboarding completion.
    """
    __tablename__ = "user_preferences"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    
    # Onboarding state
    onboarding_completed = Column(Boolean, default=False)
    onboarding_step = Column(Integer, default=0)
    
    # Preferences
    preferred_sectors = Column(JSON, default=list)  # ['Technology', 'Healthcare', ...]
    risk_tolerance = Column(String(20), default='moderate')  # 'conservative', 'moderate', 'aggressive'
    
    # External API Keys (User Provided)
    api_keys = Column(JSON, default=dict)  # {'alphavantage': '...', 'openai': '...'}
    
    # UI Preferences
    theme = Column(String(20), default='dark')
    default_chart_timeframe = Column(String(10), default='1D')
    
    # Notification preferences
    email_daily_summary = Column(Boolean, default=True)
    email_signal_alerts = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserPreferences {self.user_id}>"

