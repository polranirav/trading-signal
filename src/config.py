"""
Application configuration with Pydantic Settings.

All environment variables are validated at runtime.
Usage: from src.config import settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with runtime validation.
    Loads from .env file. Type-safe, validated.
    """
    
    # ============ API KEYS ============
    ALPHA_VANTAGE_KEY: str = Field(
        default="demo",
        description="Alpha Vantage API key"
    )
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key"
    )
    FMP_API_KEY: Optional[str] = Field(
        default=None,
        description="Financial Modeling Prep API key"
    )
    PINECONE_API_KEY: Optional[str] = Field(
        default=None,
        description="Pinecone vector database API key"
    )
    PINECONE_ENV: str = Field(
        default="us-west1-gcp",
        description="Pinecone environment"
    )
    
    # ============ DATABASE ============
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/trading_signals",
        description="PostgreSQL connection string"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string"
    )
    
    # ============ CELERY CONFIGURATION ============
    CELERY_BROKER: str = Field(
        default="redis://localhost:6379/1",
        description="Celery broker URL"
    )
    CELERY_BACKEND: str = Field(
        default="redis://localhost:6379/2",
        description="Celery result backend URL"
    )
    
    # ============ TRADING PARAMETERS ============
    PORTFOLIO_SIZE: int = Field(
        default=50,
        description="Number of assets to monitor"
    )
    MAX_POSITION_SIZE: float = Field(
        default=0.02,
        description="Maximum position size (2% of portfolio)"
    )
    MIN_CONFIDENCE_SCORE: float = Field(
        default=0.65,
        description="Minimum signal confidence threshold"
    )
    TRADING_LOOKBACK_DAYS: int = Field(
        default=252,
        description="Lookback period in trading days (252 = 1 year)"
    )
    
    # ============ FEATURE FLAGS ============
    ENABLE_LIVE_TRADING: bool = Field(
        default=False,
        description="Enable live trading mode"
    )
    ENABLE_NEWS_SENTIMENT: bool = Field(
        default=True,
        description="Enable news sentiment analysis"
    )
    ENABLE_GPT4_ANALYSIS: bool = Field(
        default=False,
        description="Enable GPT-4 analysis (expensive)"
    )
    ENABLE_BACKTESTING: bool = Field(
        default=True,
        description="Enable backtesting features"
    )
    
    # ============ MONITORING & LOGGING ============
    SENTRY_DSN: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    ENVIRONMENT: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )
    
    # ============ PAYMENTS (Stripe) ============
    STRIPE_SECRET_KEY: Optional[str] = Field(
        default=None,
        description="Stripe secret key (production)"
    )
    STRIPE_TEST_SECRET_KEY: Optional[str] = Field(
        default=None,
        description="Stripe test secret key (development)"
    )
    STRIPE_WEBHOOK_SECRET: Optional[str] = Field(
        default=None,
        description="Stripe webhook signing secret"
    )
    BASE_URL: str = Field(
        default="http://localhost:8050",
        description="Base URL for the application (for Stripe redirects)"
    )
    
    # ============ EMAIL (SendGrid/SES) ============
    SENDGRID_API_KEY: Optional[str] = Field(
        default=None,
        description="SendGrid API key for email delivery"
    )
    EMAIL_FROM: str = Field(
        default="signals@tradingsignals.com",
        description="From email address"
    )
    EMAIL_FROM_NAME: str = Field(
        default="Trading Signals Pro",
        description="From name"
    )
    
    # ============ PATHS ============
    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent
    )
    
    @property
    def DATA_DIR(self) -> Path:
        """Data directory path."""
        path = self.PROJECT_ROOT / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def MODELS_DIR(self) -> Path:
        """Models directory path."""
        path = self.PROJECT_ROOT / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def LOGS_DIR(self) -> Path:
        """Logs directory path."""
        path = self.PROJECT_ROOT / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v.startswith("postgresql://"):
            raise ValueError("DATABASE_URL must be PostgreSQL")
        return v
    
    @field_validator("REDIS_URL")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        if not v.startswith("redis://"):
            raise ValueError("REDIS_URL must start with redis://")
        return v
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",
    }


# Create global settings instance
settings = Settings()
