"""
Authentication Service.

Provides core authentication logic:
- User registration and login
- Password hashing and verification
- JWT token generation
- API key generation
"""

import bcrypt
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.logging_config import get_logger
from src.auth.models import User, Subscription, APIKey, SubscriptionLimit

logger = get_logger(__name__)


class AuthService:
    """Authentication service for user management."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            bcrypt hash string
        """
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Verify a password against a hash.
        
        Args:
            password: Plain text password
            password_hash: bcrypt hash
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception as e:
            logger.warning(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def create_user(
        session: Session,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> Tuple[Optional[User], Optional[str]]:
        """
        Create a new user account.
        
        Args:
            session: Database session
            email: User email (must be unique)
            password: Plain text password (will be hashed)
            full_name: Optional full name
            
        Returns:
            Tuple of (User object, error_message)
            If successful: (user, None)
            If error: (None, error_message)
        """
        try:
            # Check if user already exists
            existing = session.query(User).filter(User.email == email.lower()).first()
            if existing:
                return None, "Email already registered"
            
            # Hash password
            password_hash = AuthService.hash_password(password)
            
            # Create user
            user = User(
                email=email.lower(),
                password_hash=password_hash,
                full_name=full_name,
                is_active=True,
                is_admin=False,
                email_verified=False
            )
            session.add(user)
            session.flush()  # Get user.id
            
            # Create free subscription by default
            subscription = Subscription(
                user_id=user.id,
                tier='free',
                status='active'
            )
            session.add(subscription)
            session.commit()
            
            logger.info(f"User created: {user.email} (id: {user.id})")
            return user, None
            
        except IntegrityError as e:
            session.rollback()
            logger.error(f"User creation failed (integrity error): {e}")
            return None, "Email already registered"
        except Exception as e:
            session.rollback()
            logger.error(f"User creation failed: {e}")
            return None, f"Registration failed: {str(e)}"
    
    @staticmethod
    def authenticate_user(session: Session, email: str, password: str) -> Optional[User]:
        """
        Authenticate a user by email and password.
        
        Args:
            session: Database session
            email: User email
            password: Plain text password
            
        Returns:
            User object if authenticated, None otherwise
        """
        try:
            user = session.query(User).filter(User.email == email.lower()).first()
            
            if not user:
                logger.warning(f"Authentication failed: user not found ({email})")
                return None
            
            if not user.is_active:
                logger.warning(f"Authentication failed: user inactive ({email})")
                return None
            
            if not AuthService.verify_password(password, user.password_hash):
                logger.warning(f"Authentication failed: invalid password ({email})")
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            session.commit()
            
            logger.info(f"User authenticated: {user.email}")
            return user
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    @staticmethod
    def get_user_by_id(session: Session, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        return session.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_user_by_email(session: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return session.query(User).filter(User.email == email.lower()).first()
    
    @staticmethod
    def generate_api_key() -> Tuple[str, str]:
        """
        Generate a new API key.
        
        Returns:
            Tuple of (full_key, key_prefix)
            full_key: Complete API key (ts_xxxx...)
            key_prefix: First 8 chars for display (ts_xxxx)
        """
        # Generate random bytes
        random_bytes = secrets.token_bytes(32)
        key_part = secrets.token_urlsafe(32)  # URL-safe base64
        
        # Format: ts_xxxx...
        full_key = f"ts_{key_part}"
        key_prefix = full_key[:16]  # ts_xxxx... (first 16 chars for display)
        
        return full_key, key_prefix
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """
        Hash an API key for storage.
        
        Args:
            api_key: Full API key string
            
        Returns:
            SHA256 hash of the key
        """
        return hashlib.sha256(api_key.encode('utf-8')).hexdigest()
    
    @staticmethod
    def create_api_key(
        session: Session,
        user_id: UUID,
        name: Optional[str] = None,
        expires_days: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Create a new API key for a user.
        
        Args:
            session: Database session
            user_id: User ID
            name: Optional friendly name for the key
            expires_days: Optional expiration days (None = no expiration)
            
        Returns:
            Tuple of (api_key, error_message)
            If successful: (full_api_key_string, None)
            If error: (None, error_message)
        """
        try:
            # Generate key
            full_key, key_prefix = AuthService.generate_api_key()
            key_hash = AuthService.hash_api_key(full_key)
            
            # Check for collisions (unlikely but possible)
            existing = session.query(APIKey).filter(APIKey.key_hash == key_hash).first()
            if existing:
                # Retry once
                full_key, key_prefix = AuthService.generate_api_key()
                key_hash = AuthService.hash_api_key(full_key)
            
            # Set expiration
            expires_at = None
            if expires_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_days)
            
            # Create API key
            api_key = APIKey(
                user_id=user_id,
                key_hash=key_hash,
                key_prefix=key_prefix,
                name=name or "Default API Key",
                expires_at=expires_at
            )
            session.add(api_key)
            session.commit()
            
            logger.info(f"API key created for user {user_id} (prefix: {key_prefix})")
            return full_key, None
            
        except Exception as e:
            session.rollback()
            logger.error(f"API key creation failed: {e}")
            return None, f"Failed to create API key: {str(e)}"
    
    @staticmethod
    def verify_api_key(session: Session, api_key: str) -> Optional[User]:
        """
        Verify an API key and return the associated user.
        
        Args:
            session: Database session
            api_key: Full API key string
            
        Returns:
            User object if key is valid, None otherwise
        """
        try:
            key_hash = AuthService.hash_api_key(api_key)
            api_key_obj = session.query(APIKey).filter(APIKey.key_hash == key_hash).first()
            
            if not api_key_obj:
                return None
            
            # Check expiration
            if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
                logger.warning(f"API key expired: {api_key_obj.key_prefix}")
                return None
            
            # Get user
            user = session.query(User).filter(User.id == api_key_obj.user_id).first()
            
            if not user or not user.is_active:
                return None
            
            # Update last used
            api_key_obj.last_used = datetime.utcnow()
            session.commit()
            
            return user
            
        except Exception as e:
            logger.error(f"API key verification error: {e}")
            return None
    
    @staticmethod
    def get_user_subscription(session: Session, user_id: UUID) -> Optional[Subscription]:
        """
        Get active subscription for a user.
        
        Args:
            session: Database session
            user_id: User ID
            
        Returns:
            Active subscription or None
        """
        return session.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.status == 'active'
        ).first()
    
    @staticmethod
    def get_subscription_limit(session: Session, tier: str) -> Optional[SubscriptionLimit]:
        """
        Get subscription limits for a tier.
        
        Args:
            session: Database session
            tier: Subscription tier ('free', 'essential', etc.)
            
        Returns:
            SubscriptionLimit object or None
        """
        return session.query(SubscriptionLimit).filter(SubscriptionLimit.tier == tier).first()
