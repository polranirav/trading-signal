"""
User API Keys Management

Allows users to store their own API keys for external data services.
Keys are encrypted at rest and retrieved per-user for signal intelligence.
"""

from flask import Blueprint, request, g
from cryptography.fernet import Fernet
import os
import json
import asyncio
import aiohttp
from datetime import datetime

from src.api.utils import success_response, error_response, require_api_key_or_auth
from src.data.persistence import get_database
from src.logging_config import get_logger

logger = get_logger(__name__)

user_api_keys_bp = Blueprint('user_api_keys', __name__)

# Encryption key - should be stored securely in production
ENCRYPTION_KEY = os.getenv('API_KEY_ENCRYPTION_KEY', Fernet.generate_key())
if isinstance(ENCRYPTION_KEY, str):
    ENCRYPTION_KEY = ENCRYPTION_KEY.encode()
fernet = Fernet(ENCRYPTION_KEY)


# Supported API services with their metadata
SUPPORTED_SERVICES = {
    'alpha_vantage': {
        'name': 'Alpha Vantage',
        'description': 'Stock prices, technical indicators, news sentiment',
        'url': 'https://www.alphavantage.co/',
        'free_tier': '25 requests/day',
        'test_endpoint': 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey={key}',
        'signals': ['Technical Analysis', 'News Sentiment', 'Price Data'],
        'icon': 'trending_up',
        'color': '#10b981',
    },
    'finnhub': {
        'name': 'Finnhub',
        'description': 'Real-time quotes, analyst ratings, social sentiment, insider transactions',
        'url': 'https://finnhub.io/',
        'free_tier': '60 requests/min',
        'test_endpoint': 'https://finnhub.io/api/v1/quote?symbol=AAPL&token={key}',
        'signals': ['Social Sentiment', 'Analyst Ratings', 'Insider Activity', 'Options Flow'],
        'icon': 'analytics',
        'color': '#3b82f6',
    },
    'newsapi': {
        'name': 'NewsAPI',
        'description': 'News articles from 150,000+ sources worldwide',
        'url': 'https://newsapi.org/',
        'free_tier': '100 requests/day',
        'test_endpoint': 'https://newsapi.org/v2/everything?q=test&pageSize=1&apiKey={key}',
        'signals': ['News Volume', 'Breaking News', 'Headline Sentiment'],
        'icon': 'newspaper',
        'color': '#8b5cf6',
    },
    'fmp': {
        'name': 'Financial Modeling Prep',
        'description': 'Company fundamentals, financial statements, valuation metrics',
        'url': 'https://financialmodelingprep.com/',
        'free_tier': '250 requests/day',
        'test_endpoint': 'https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={key}',
        'signals': ['P/E Ratio', 'EPS', 'Revenue Growth', 'ROE', 'Debt/Equity'],
        'icon': 'bar_chart',
        'color': '#f59e0b',
    },
    'fred': {
        'name': 'FRED (Federal Reserve)',
        'description': 'Macroeconomic data: GDP, inflation, interest rates, employment',
        'url': 'https://fred.stlouisfed.org/',
        'free_tier': 'Unlimited (registration required)',
        'test_endpoint': 'https://api.stlouisfed.org/fred/series?series_id=GDP&api_key={key}&file_type=json',
        'signals': ['Fed Rate', 'CPI Inflation', 'GDP Growth', 'Unemployment'],
        'icon': 'account_balance',
        'color': '#06b6d4',
    },
    'polygon': {
        'name': 'Polygon.io',
        'description': 'Real-time and historical market data, aggregates, options',
        'url': 'https://polygon.io/',
        'free_tier': '5 API calls/min',
        'test_endpoint': 'https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={key}',
        'signals': ['Price Data', 'Volume Analysis', 'Options Data'],
        'icon': 'hexagon',
        'color': '#ec4899',
    },
    'openai': {
        'name': 'OpenAI',
        'description': 'AI-powered analysis, sentiment classification, reasoning',
        'url': 'https://openai.com/',
        'free_tier': 'Pay-as-you-go',
        'test_endpoint': None,  # Don't test OpenAI automatically
        'signals': ['AI Analysis', 'Advanced Reasoning', 'Report Generation'],
        'icon': 'psychology',
        'color': '#a855f7',
    },
}


def encrypt_api_key(key: str) -> str:
    """Encrypt an API key for storage."""
    return fernet.encrypt(key.encode()).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key from storage."""
    try:
        return fernet.decrypt(encrypted_key.encode()).decode()
    except Exception:
        return ""


def get_user_api_keys(user_id: str) -> dict:
    """Get all API keys for a user from the database."""
    db = get_database()
    try:
        # Try to get from user_api_keys table
        result = db.execute_query(
            "SELECT service, encrypted_key, is_valid, last_validated, created_at FROM user_api_keys WHERE user_id = %s",
            (user_id,)
        )
        if result:
            return {row['service']: {
                'encrypted_key': row['encrypted_key'],
                'is_valid': row.get('is_valid', None),
                'last_validated': row.get('last_validated'),
                'created_at': row.get('created_at'),
            } for row in result}
    except Exception as e:
        logger.warning(f"Error getting user API keys: {e}")
    return {}


def save_user_api_key(user_id: str, service: str, encrypted_key: str, is_valid: bool = None):
    """Save or update an API key for a user."""
    db = get_database()
    try:
        # Upsert the API key
        db.execute_query(
            """
            INSERT INTO user_api_keys (user_id, service, encrypted_key, is_valid, last_validated, created_at, updated_at)
            VALUES (%s, %s, %s, %s, NOW(), NOW(), NOW())
            ON CONFLICT (user_id, service) 
            DO UPDATE SET encrypted_key = EXCLUDED.encrypted_key, 
                          is_valid = EXCLUDED.is_valid,
                          last_validated = NOW(),
                          updated_at = NOW()
            """,
            (user_id, service, encrypted_key, is_valid)
        )
        return True
    except Exception as e:
        logger.error(f"Error saving API key: {e}")
        # Try creating the table if it doesn't exist
        try:
            db.execute_query("""
                CREATE TABLE IF NOT EXISTS user_api_keys (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    service VARCHAR(50) NOT NULL,
                    encrypted_key TEXT NOT NULL,
                    is_valid BOOLEAN,
                    last_validated TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(user_id, service)
                )
            """)
            # Retry the insert
            db.execute_query(
                """
                INSERT INTO user_api_keys (user_id, service, encrypted_key, is_valid, last_validated, created_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW(), NOW())
                ON CONFLICT (user_id, service) 
                DO UPDATE SET encrypted_key = EXCLUDED.encrypted_key, 
                              is_valid = EXCLUDED.is_valid,
                              last_validated = NOW(),
                              updated_at = NOW()
                """,
                (user_id, service, encrypted_key, is_valid)
            )
            return True
        except Exception as e2:
            logger.error(f"Error creating table or inserting: {e2}")
            return False


def delete_user_api_key(user_id: str, service: str) -> bool:
    """Delete an API key for a user."""
    db = get_database()
    try:
        db.execute_query(
            "DELETE FROM user_api_keys WHERE user_id = %s AND service = %s",
            (user_id, service)
        )
        return True
    except Exception as e:
        logger.error(f"Error deleting API key: {e}")
        return False


async def test_api_key(service: str, key: str) -> tuple[bool, str]:
    """Test if an API key is valid by making a test request."""
    if service not in SUPPORTED_SERVICES:
        return False, "Unknown service"
    
    service_config = SUPPORTED_SERVICES[service]
    test_url = service_config.get('test_endpoint')
    
    if not test_url:
        # Can't test this service, assume valid
        return True, "Key saved (not tested)"
    
    try:
        url = test_url.format(key=key)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    return True, "Key validated successfully"
                elif resp.status == 401 or resp.status == 403:
                    return False, "Invalid API key"
                elif resp.status == 429:
                    return True, "Key valid (rate limited)"
                else:
                    return False, f"Validation failed (HTTP {resp.status})"
    except asyncio.TimeoutError:
        return False, "Validation timeout"
    except Exception as e:
        logger.warning(f"API key test error: {e}")
        return False, f"Validation error: {str(e)}"


def run_async(coro):
    """Run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@user_api_keys_bp.route('/user-api-keys', methods=['GET'])
@require_api_key_or_auth
def get_api_keys():
    """
    Get all external API keys for the current user.
    Returns service metadata and connection status, not the actual keys.
    """
    try:
        user_id = g.get('user_id', 'default')
        stored_keys = get_user_api_keys(user_id)
        
        # Build response with service info and status
        services = []
        for service_id, config in SUPPORTED_SERVICES.items():
            stored = stored_keys.get(service_id, {})
            has_key = bool(stored.get('encrypted_key'))
            
            services.append({
                'id': service_id,
                'name': config['name'],
                'description': config['description'],
                'url': config['url'],
                'free_tier': config['free_tier'],
                'signals': config['signals'],
                'icon': config['icon'],
                'color': config['color'],
                'connected': has_key,
                'is_valid': stored.get('is_valid') if has_key else None,
                'last_validated': stored.get('last_validated').isoformat() if stored.get('last_validated') else None,
                'created_at': stored.get('created_at').isoformat() if stored.get('created_at') else None,
            })
        
        # Calculate total connected
        connected_count = sum(1 for s in services if s['connected'])
        total_signals = sum(len(s['signals']) for s in services if s['connected'])
        
        return success_response(
            data={
                'services': services,
                'connected_count': connected_count,
                'total_services': len(SUPPORTED_SERVICES),
                'total_signals_enabled': total_signals,
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting API keys: {e}", exc_info=True)
        return error_response("Failed to get API keys", 500)


@user_api_keys_bp.route('/user-api-keys/<service>', methods=['POST'])
@require_api_key_or_auth
def save_api_key(service: str):
    """
    Save an external API key for the current user.
    Validates the key before saving.
    """
    try:
        if service not in SUPPORTED_SERVICES:
            return error_response(f"Unknown service: {service}", 400)
        
        user_id = g.get('user_id', 'default')
        data = request.json or {}
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return error_response("API key is required", 400)
        
        # Test the key
        is_valid, message = run_async(test_api_key(service, api_key))
        
        # Encrypt and save
        encrypted = encrypt_api_key(api_key)
        saved = save_user_api_key(user_id, service, encrypted, is_valid)
        
        if not saved:
            return error_response("Failed to save API key", 500)
        
        return success_response(
            data={
                'service': service,
                'is_valid': is_valid,
                'message': message,
            },
            message=f"API key saved for {SUPPORTED_SERVICES[service]['name']}"
        )
        
    except Exception as e:
        logger.error(f"Error saving API key: {e}", exc_info=True)
        return error_response("Failed to save API key", 500)


@user_api_keys_bp.route('/user-api-keys/<service>', methods=['DELETE'])
@require_api_key_or_auth
def remove_api_key(service: str):
    """Remove an external API key for the current user."""
    try:
        if service not in SUPPORTED_SERVICES:
            return error_response(f"Unknown service: {service}", 400)
        
        user_id = g.get('user_id', 'default')
        deleted = delete_user_api_key(user_id, service)
        
        if not deleted:
            return error_response("Failed to delete API key", 500)
        
        return success_response(
            message=f"API key removed for {SUPPORTED_SERVICES[service]['name']}"
        )
        
    except Exception as e:
        logger.error(f"Error deleting API key: {e}", exc_info=True)
        return error_response("Failed to delete API key", 500)


@user_api_keys_bp.route('/user-api-keys/<service>/test', methods=['POST'])
@require_api_key_or_auth
def test_stored_key(service: str):
    """Test a stored API key by making a validation request."""
    try:
        if service not in SUPPORTED_SERVICES:
            return error_response(f"Unknown service: {service}", 400)
        
        user_id = g.get('user_id', 'default')
        stored_keys = get_user_api_keys(user_id)
        
        if service not in stored_keys or not stored_keys[service].get('encrypted_key'):
            return error_response(f"No API key stored for {service}", 404)
        
        # Decrypt and test
        api_key = decrypt_api_key(stored_keys[service]['encrypted_key'])
        is_valid, message = run_async(test_api_key(service, api_key))
        
        # Update validation status
        save_user_api_key(
            user_id, 
            service, 
            stored_keys[service]['encrypted_key'], 
            is_valid
        )
        
        return success_response(
            data={
                'service': service,
                'is_valid': is_valid,
                'message': message,
            }
        )
        
    except Exception as e:
        logger.error(f"Error testing API key: {e}", exc_info=True)
        return error_response("Failed to test API key", 500)


# Helper function for signal providers to get user's API key
def get_user_decrypted_key(user_id: str, service: str) -> str | None:
    """
    Get a decrypted API key for a user and service.
    Used by signal providers to get user's custom API keys.
    """
    stored_keys = get_user_api_keys(user_id)
    if service in stored_keys and stored_keys[service].get('encrypted_key'):
        return decrypt_api_key(stored_keys[service]['encrypted_key'])
    return None
