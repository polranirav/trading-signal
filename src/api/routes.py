"""
API Routes Registration.

Registers all API Blueprints with Flask app.
"""

from flask import Blueprint

from src.api.auth import auth_bp
from src.api.signals import signals_bp
from src.api.account import account_bp
from src.api.subscriptions import subscriptions_bp
from src.api.public import public_bp
from src.api.portfolio import portfolio_bp
from src.api.signal_intelligence import signal_intelligence_bp
from src.api.user_api_keys import user_api_keys_bp

# Import admin blueprint
try:
    from src.admin.api import admin_bp
    ADMIN_AVAILABLE = True
except ImportError:
    ADMIN_AVAILABLE = False

# Import docs blueprint
try:
    from src.api.docs import docs_bp
    DOCS_AVAILABLE = True
except ImportError:
    DOCS_AVAILABLE = False

# Create main API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Register sub-blueprints
api_bp.register_blueprint(public_bp)
api_bp.register_blueprint(auth_bp)
api_bp.register_blueprint(signals_bp)
api_bp.register_blueprint(account_bp)
api_bp.register_blueprint(subscriptions_bp)
api_bp.register_blueprint(portfolio_bp)
api_bp.register_blueprint(signal_intelligence_bp)
api_bp.register_blueprint(user_api_keys_bp)

# Register admin blueprint if available
if ADMIN_AVAILABLE:
    api_bp.register_blueprint(admin_bp)

# Register docs blueprint if available
if DOCS_AVAILABLE:
    api_bp.register_blueprint(docs_bp)


def register_api_routes(app):
    """
    Register API routes with Flask app.
    
    Args:
        app: Flask application instance (from dash_app.server)
    """
    app.register_blueprint(api_bp)
    
    # Configure CORS
    try:
        from flask_cors import CORS
        CORS(app, 
             origins=[
                 "http://localhost:3000", 
                 "http://localhost:5173", 
                 "http://localhost:3002",
                 "http://127.0.0.1:3002",
                 "http://0.0.0.0:3002"
             ],  # React dev servers
             supports_credentials=True,
             allow_headers=["Content-Type", "Authorization", "X-API-Key"],
             methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    except ImportError:
        pass  # flask-cors not installed
