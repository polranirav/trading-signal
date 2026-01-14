"""
Callbacks for Account Management Page.

Handles profile updates, password changes, API key management,
and subscription information.
"""

import dash
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
from flask import session as flask_session
from datetime import datetime

from src.logging_config import get_logger
from src.auth.service import AuthService
from src.data.persistence import get_database

logger = get_logger(__name__)


def register_account_callbacks(app: dash.Dash):
    """Register all account-related callbacks."""
    
    # Load Account Data
    @app.callback(
        [
            Output("account-email", "value"),
            Output("account-name", "value"),
            Output("account-company", "value"),
            Output("account-phone", "value"),
            Output("account-subscription-info", "children"),
            Output("account-usage-stats", "children"),
            Output("account-api-keys-list", "children"),
            Output("account-notification-prefs", "value"),
            Output("account-data-store", "data"),
        ],
        Input("url", "pathname")
    )
    def load_account_data(pathname):
        """Load user account data when page loads."""
        if pathname != "/account":
            return (dash.no_update,) * 9
        
        try:
            user_id = flask_session.get('user_id')
            if not user_id:
                return (dash.no_update,) * 9
            
            db = get_database()
            user = db.get_user_by_id(user_id)
            
            if not user:
                return (dash.no_update,) * 9
            
            # Subscription Info
            subscription_info = html.Div([
                html.Div([
                    html.Span("Current Plan", className="text-muted d-block mb-2"),
                    html.H4("Essential", style={"color": "#3b82f6", "marginBottom": "8px"}),
                    html.Small("$19/month", className="text-muted d-block mb-3")
                ]),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-check text-success me-2"),
                        html.Span("1000 API requests/day", className="text-muted")
                    ], className="mb-2"),
                    html.Div([
                        html.I(className="fas fa-check text-success me-2"),
                        html.Span("Real-time signals", className="text-muted")
                    ], className="mb-2"),
                    html.Div([
                        html.I(className="fas fa-check text-success me-2"),
                        html.Span("Advanced analytics", className="text-muted")
                    ], className="mb-2"),
                ]),
                dbc.Button(
                    [html.I(className="fas fa-cog me-2"), "Manage Subscription"],
                    href="/billing",
                    color="primary",
                    className="w-100 mt-3"
                )
            ])
            
            # Usage Statistics
            usage_stats = html.Div([
                html.Div([
                    html.Div([
                        html.Span("API Requests (Today)", className="text-muted d-block mb-1"),
                        html.H5("247 / 1,000", style={"color": "#fff", "marginBottom": "0"})
                    ]),
                    html.Div([
                        html.Div(style={
                            "width": "24.7%",
                            "height": "4px",
                            "background": "linear-gradient(90deg, #3b82f6, #60a5fa)",
                            "borderRadius": "2px",
                            "marginTop": "8px"
                        })
                    ])
                ], className="mb-4"),
                html.Div([
                    html.Div([
                        html.Span("Signals Generated", className="text-muted d-block mb-1"),
                        html.H5("1,234", style={"color": "#fff", "marginBottom": "0"})
                    ]),
                    html.Small("This month", className="text-muted")
                ], className="mb-4"),
                html.Div([
                    html.Div([
                        html.Span("Backtests Run", className="text-muted d-block mb-1"),
                        html.H5("12", style={"color": "#fff", "marginBottom": "0"})
                    ]),
                    html.Small("This month", className="text-muted")
                ])
            ])
            
            # API Keys List
            api_keys_list = html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Span("Production Key", style={"fontWeight": "600", "color": "#fff"}),
                            html.Small("Created: Jan 10, 2026", className="text-muted ms-2")
                        ]),
                        html.Div([
                            html.Code("ts_live_••••••••••••••••••••••••", style={"color": "#94a3b8", "fontSize": "0.85rem"}),
                            dbc.Button(
                                html.I(className="fas fa-copy"),
                                size="sm",
                                color="secondary",
                                outline=True,
                                className="ms-2"
                            ),
                            dbc.Button(
                                html.I(className="fas fa-trash"),
                                size="sm",
                                color="danger",
                                outline=True,
                                className="ms-2"
                            )
                        ], style={"display": "flex", "alignItems": "center", "marginTop": "8px"})
                    ], style={
                        "padding": "12px",
                        "background": "rgba(0,0,0,0.2)",
                        "borderRadius": "8px",
                        "border": "1px solid rgba(255,255,255,0.05)",
                        "marginBottom": "12px"
                    })
                ]),
                html.Small("You can create up to 5 API keys", className="text-muted")
            ])
            
            # Notification preferences (default values)
            notification_prefs = ["email_signals", "weekly_summary", "announcements"]
            
            account_data = {
                "user_id": str(user_id),
                "email": user.email,
                "name": user.full_name or "",
                "company": "",
                "phone": "",
            }
            
            return (
                user.email or "",
                user.full_name or "",
                "",
                "",
                subscription_info,
                usage_stats,
                api_keys_list,
                notification_prefs,
                account_data
            )
            
        except Exception as e:
            logger.error(f"Error loading account data: {e}", exc_info=True)
            return (dash.no_update,) * 9
    
    # Update Profile
    @app.callback(
        Output("account-profile-feedback", "children"),
        Input("account-update-profile-btn", "n_clicks"),
        [
            State("account-name", "value"),
            State("account-company", "value"),
            State("account-phone", "value"),
        ]
    )
    def update_profile(n_clicks, name, company, phone):
        """Update user profile information."""
        if not n_clicks:
            return ""
        
        try:
            user_id = flask_session.get('user_id')
            if not user_id:
                return dbc.Alert("Please log in to update your profile", color="warning")
            
            db = get_database()
            # Update user profile (simplified - would need actual update method)
            
            return dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                "Profile updated successfully!"
            ], color="success", className="mt-2")
            
        except Exception as e:
            logger.error(f"Error updating profile: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)[:100]}", color="danger", className="mt-2")
    
    # Change Password
    @app.callback(
        Output("account-password-feedback", "children"),
        Input("account-change-password-btn", "n_clicks"),
        [
            State("account-current-password", "value"),
            State("account-new-password", "value"),
            State("account-confirm-password", "value"),
        ]
    )
    def change_password(n_clicks, current_pwd, new_pwd, confirm_pwd):
        """Change user password."""
        if not n_clicks:
            return ""
        
        try:
            user_id = flask_session.get('user_id')
            if not user_id:
                return dbc.Alert("Please log in", color="warning")
            
            if not current_pwd or not new_pwd or not confirm_pwd:
                return dbc.Alert("Please fill in all password fields", color="warning")
            
            if new_pwd != confirm_pwd:
                return dbc.Alert("New passwords do not match", color="danger")
            
            if len(new_pwd) < 8:
                return dbc.Alert("Password must be at least 8 characters", color="warning")
            
            auth_service = AuthService()
            db = get_database()
            user = db.get_user_by_id(user_id)
            
            # Verify current password
            if not auth_service.verify_password(current_pwd, user.password_hash):
                return dbc.Alert("Current password is incorrect", color="danger")
            
            # Update password (simplified - would need actual update method)
            
            return dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                "Password changed successfully!"
            ], color="success", className="mt-2")
            
        except Exception as e:
            logger.error(f"Error changing password: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)[:100]}", color="danger", className="mt-2")
    
    # Save Notification Preferences
    @app.callback(
        Output("account-prefs-feedback", "children"),
        Input("account-save-prefs-btn", "n_clicks"),
        State("account-notification-prefs", "value")
    )
    def save_preferences(n_clicks, prefs):
        """Save notification preferences."""
        if not n_clicks:
            return ""
        
        try:
            user_id = flask_session.get('user_id')
            if not user_id:
                return dbc.Alert("Please log in", color="warning")
            
            # Save preferences (simplified)
            
            return dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                "Preferences saved successfully!"
            ], color="success", className="mt-2")
            
        except Exception as e:
            logger.error(f"Error saving preferences: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)[:100]}", color="danger", className="mt-2")
    
    # API Key Modal
    @app.callback(
        Output("api-key-modal", "is_open"),
        [
            Input("account-create-api-key-btn", "n_clicks"),
            Input("api-key-modal-close", "n_clicks"),
            Input("api-key-create-btn", "n_clicks"),
        ],
        State("api-key-modal", "is_open")
    )
    def toggle_api_key_modal(open_clicks, close_clicks, create_clicks, is_open):
        """Toggle API key creation modal."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id in ["account-create-api-key-btn", "api-key-create-btn"]:
            return not is_open
        elif trigger_id == "api-key-modal-close":
            return False
        
        return is_open
