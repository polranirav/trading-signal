"""
Payment-related SQLAlchemy models.

Payment and Invoice models for tracking transactions.
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, ForeignKey, JSON, Enum, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import DECIMAL
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import enum

from src.data.models import Base


class PaymentStatus(enum.Enum):
    """Payment status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class Payment(Base):
    """
    Payment transactions from Stripe.
    
    Tracks individual payment events and status.
    """
    __tablename__ = "payments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id", ondelete="SET NULL"), index=True)
    
    # Stripe IDs
    stripe_payment_intent_id = Column(String(255), unique=True, index=True)
    stripe_charge_id = Column(String(255), index=True)
    
    # Payment details
    amount = Column(DECIMAL(10, 2), nullable=False)  # Amount in cents
    currency = Column(String(10), default="usd", nullable=False)
    status = Column(String(50), nullable=False, index=True)  # PaymentStatus enum value
    
    # Metadata
    description = Column(String(500))
    metadata_json = Column('metadata', JSON)  # Additional Stripe metadata (renamed to avoid SQLAlchemy conflict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Payment {self.id} {self.status} ${self.amount}>"


class Invoice(Base):
    """
    Invoices for subscriptions.
    
    Tracks billing invoices from Stripe.
    """
    __tablename__ = "invoices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id", ondelete="SET NULL"), index=True)
    
    # Stripe IDs
    stripe_invoice_id = Column(String(255), unique=True, index=True)
    stripe_payment_intent_id = Column(String(255), index=True)
    
    # Invoice details
    amount_due = Column(DECIMAL(10, 2), nullable=False)
    amount_paid = Column(DECIMAL(10, 2), default=0)
    currency = Column(String(10), default="usd", nullable=False)
    status = Column(String(50), nullable=False, index=True)  # 'draft', 'open', 'paid', 'void', 'uncollectible'
    
    # Billing period
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    
    # Invoice number
    number = Column(String(100), unique=True, index=True)
    
    # PDF URL (from Stripe)
    invoice_pdf = Column(String(500))
    hosted_invoice_url = Column(String(500))
    
    # Metadata
    description = Column(Text)
    metadata_json = Column('metadata', JSON)  # Renamed to avoid SQLAlchemy conflict
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Invoice {self.number} {self.status} ${self.amount_due}>"
