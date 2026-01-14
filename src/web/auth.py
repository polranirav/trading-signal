"""
Web Authentication Routes for Dash Application.

Provides login, register, logout, and profile pages.
Integrates with Flask session for authentication.
"""

from flask import session as flask_session
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from src.logging_config import get_logger
from src.data.persistence import get_database
from src.auth.service import AuthService

logger = get_logger(__name__)


def create_login_page():
    """Create the login page layout."""
    return dbc.Container([
        # Hidden location component for redirect
        dcc.Location(id="login-redirect", refresh=True),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3("Sign In", className="mb-0")),
                    dbc.CardBody([
                        dbc.Form([
                            dbc.Label("Email", html_for="login-email"),
                            dbc.Input(
                                id="login-email",
                                type="email",
                                placeholder="your@email.com",
                                className="mb-3"
                            ),
                            dbc.Label("Password", html_for="login-password"),
                            dbc.Input(
                                id="login-password",
                                type="password",
                                placeholder="••••••••",
                                className="mb-3"
                            ),
                            dbc.Checklist(
                                options=[{"label": "Remember me", "value": "remember"}],
                                id="login-remember",
                                className="mb-3"
                            ),
                            # Login button
                            dbc.Button(
                                "Sign In",
                                id="login-submit",
                                color="primary",
                                className="w-100 mb-3"
                            ),
                            # Loading indicator - shows spinner while processing
                            dcc.Loading(
                                id="login-loading",
                                type="circle",
                                color="#3b82f6",
                                children=[
                                    html.Div(id="login-error", className="text-danger mb-2"),
                                ],
                                style={"minHeight": "30px"}
                            ),
                            html.Hr(),
                            html.P([
                                "Don't have an account? ",
                                dcc.Link("Sign up", href="/register", className="text-primary")
                            ], className="text-center mb-0")
                        ])
                    ])
                ], className="glass-card mt-5")
            ], md=6, lg=4, className="mx-auto")
        ])
    ], fluid=True, className="mt-5")


def create_register_page():
    """Create the registration page layout."""
    return dbc.Container([
        # Hidden location component for redirect
        dcc.Location(id="register-redirect", refresh=True),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3("Create Account", className="mb-0")),
                    dbc.CardBody([
                        dbc.Form([
                            dbc.Label("Full Name", html_for="register-name"),
                            dbc.Input(
                                id="register-name",
                                type="text",
                                placeholder="John Doe",
                                className="mb-3"
                            ),
                            dbc.Label("Email", html_for="register-email"),
                            dbc.Input(
                                id="register-email",
                                type="email",
                                placeholder="your@email.com",
                                className="mb-3"
                            ),
                            dbc.Label("Password", html_for="register-password"),
                            dbc.Input(
                                id="register-password",
                                type="password",
                                placeholder="••••••••",
                                className="mb-3"
                            ),
                            dbc.Label("Confirm Password", html_for="register-password-confirm"),
                            dbc.Input(
                                id="register-password-confirm",
                                type="password",
                                placeholder="••••••••",
                                className="mb-3"
                            ),
                            dbc.Button(
                                "Create Account",
                                id="register-submit",
                                color="primary",
                                className="w-100 mb-3"
                            ),
                            # Loading indicator - shows spinner while processing
                            dcc.Loading(
                                id="register-loading",
                                type="circle",
                                color="#3b82f6",
                                children=[
                                    html.Div(id="register-error", className="text-danger mb-2"),
                                    html.Div(id="register-success", className="text-success mb-2"),
                                ],
                                style={"minHeight": "30px"}
                            ),
                            html.Hr(),
                            html.P([
                                "Already have an account? ",
                                dcc.Link("Sign in", href="/login", className="text-primary")
                            ], className="text-center mb-0")
                        ])
                    ])
                ], className="glass-card mt-5")
            ], md=6, lg=4, className="mx-auto")
        ])
    ], fluid=True, className="mt-5")


def register_auth_callbacks(dash_app):
    """
    Register authentication callbacks with the Dash app.
    
    This handles login, register, and logout functionality.
    """
    import dash
    
    @dash_app.callback(
        [Output("login-error", "children"),
         Output("login-error", "style"),
         Output("login-redirect", "href")],
        Input("login-submit", "n_clicks"),
        [State("login-email", "value"),
         State("login-password", "value")],
        prevent_initial_call=True
    )
    def handle_login(n_clicks, email, password):
        """Handle user login."""
        import dash
        from dash.exceptions import PreventUpdate
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # No redirect by default
        no_redirect = dash.no_update
        
        if not email or not password:
            return "Please enter email and password", {"display": "block", "color": "#ef4444"}, no_redirect
        
        db = get_database()
        try:
            with db.get_session() as db_session:
                user = AuthService.authenticate_user(db_session, email, password)
                
                if user:
                    # Set Flask session (accessed via dash_app.server)
                    with dash_app.server.app_context():
                        flask_session['user_id'] = str(user.id)
                        flask_session['user_email'] = user.email
                        flask_session['user_name'] = user.full_name
                        flask_session.permanent = True
                    
                    logger.info(f"User logged in: {user.email}")
                    
                    # Show success message and redirect
                    return (
                        html.Span([
                            html.I(className="fas fa-check-circle me-2"),
                            "Login successful! Redirecting..."
                        ], style={"color": "#10b981"}),
                        {"display": "block"},
                        "/overview"  # Redirect to overview page
                    )
                else:
                    return "Invalid email or password", {"display": "block", "color": "#ef4444"}, no_redirect
                    
        except Exception as e:
            logger.error(f"Login error: {e}")
            return f"Login failed: {str(e)}", {"display": "block", "color": "#ef4444"}, no_redirect

    
    @dash_app.callback(
        [Output("register-error", "children"),
         Output("register-error", "style"),
         Output("register-success", "children"),
         Output("register-success", "style"),
         Output("register-redirect", "href")],
        Input("register-submit", "n_clicks"),
        [State("register-name", "value"),
         State("register-email", "value"),
         State("register-password", "value"),
         State("register-password-confirm", "value")],
        prevent_initial_call=True
    )
    def handle_register(n_clicks, name, email, password, password_confirm):
        """Handle user registration."""
        import dash
        from dash.exceptions import PreventUpdate
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # No redirect by default
        no_redirect = dash.no_update
        
        # Validation
        if not email or not password:
            return "Please enter email and password", {"display": "block", "color": "#ef4444"}, "", {"display": "none"}, no_redirect
        
        if password != password_confirm:
            return "Passwords do not match", {"display": "block", "color": "#ef4444"}, "", {"display": "none"}, no_redirect
        
        if len(password) < 8:
            return "Password must be at least 8 characters", {"display": "block", "color": "#ef4444"}, "", {"display": "none"}, no_redirect
        
        db = get_database()
        try:
            with db.get_session() as db_session:
                user, error = AuthService.create_user(db_session, email, password, name)
                
                if user:
                    logger.info(f"User registered: {user.email}")
                    
                    # Set Flask session (accessed via dash_app.server)
                    with dash_app.server.app_context():
                        flask_session['user_id'] = str(user.id)
                        flask_session['user_email'] = user.email
                        flask_session.permanent = True
                    
                    return (
                        "",
                        {"display": "none"},
                        html.Span([
                            html.I(className="fas fa-check-circle me-2"),
                            "Account created! Redirecting..."
                        ], style={"color": "#10b981"}),
                        {"display": "block"},
                        "/overview"  # Redirect to overview page
                    )
                else:
                    return error or "Registration failed", {"display": "block", "color": "#ef4444"}, "", {"display": "none"}, no_redirect
                    
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return f"Registration failed: {str(e)}", {"display": "block", "color": "#ef4444"}, "", {"display": "none"}, no_redirect
    
    # Logout callback - commented out to avoid duplicate callback errors
    # Using client-side redirect or dcc.Location instead
    # @dash_app.callback(
    #     Output("url", "pathname", allow_duplicate=True),
    #     Input("logout-button", "n_clicks"),
    #     prevent_initial_call=True
    # )
    # def handle_logout(n_clicks):
    #     """Handle user logout."""
    #     with dash_app.server.app_context():
    #         flask_session.clear()
    #     logger.info("User logged out")
    #     return "/login"
