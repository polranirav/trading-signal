"""
Trading Signals Dashboard - Enterprise Edition.

Entry point for the Dash application.
Structure:
- src/web/app.py: Application initialization
- src/web/layouts.py: UI Component definitions
- src/web/callbacks.py: Interactive logic
"""

import sys
import os
import secrets

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import dash
import dash_bootstrap_components as dbc
from src.logging_config import get_logger
from src.web.layouts import create_layout
from src.web.callbacks import register_callbacks
from src.web.payment_webhooks import register_webhook_routes
from src.api.routes import register_api_routes
from src.config import settings

logger = get_logger(__name__)

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
app.title = "Trading Signals Pro"
server = app.server

# Configure Flask server for sessions and security
server.secret_key = settings.SECRET_KEY
server.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=settings.ENVIRONMENT == 'production',
    PERMANENT_SESSION_LIFETIME=86400 * 30,  # 30 days
)

# Set Main Layout
app.layout = create_layout()

# Register Callbacks
register_callbacks(app)

# Register Webhook Routes
register_webhook_routes(app)

# Register API Routes
register_api_routes(server)

if __name__ == "__main__":
    logger.info("Starting Trading Signals Dashboard (Enterprise Mode)")
    app.run(debug=True, host="0.0.0.0", port=8050)
