"""
Payment and Billing Pages for Dash Application.

Provides checkout, billing, and subscription management pages.
"""

from flask import session as flask_session
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from src.logging_config import get_logger
from src.config import settings
from src.data.persistence import get_database
from src.auth.service import AuthService
from src.payments.stripe_client import StripeClient
from src.auth.middleware import get_current_user

logger = get_logger(__name__)


def create_pricing_page():
    """Create the pricing page layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Pricing", className="text-white mb-4"),
                html.P("Choose the plan that's right for you", className="text-muted mb-5")
            ], width=12)
        ]),
        
        dbc.Row([
            # Free Tier
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Free", className="mb-0")),
                    dbc.CardBody([
                        html.H2("$0", className="mb-3"),
                        html.P("/month", className="text-muted mb-4"),
                        html.Ul([
                            html.Li("3 signals per day"),
                            html.Li("Basic dashboard"),
                            html.Li("Email alerts"),
                            html.Li("No API access"),
                            html.Li("No performance tracking"),
                        ], className="mb-4"),
                        dcc.Link(
                            dbc.Button("Sign Up Free", color="secondary", className="w-100", outline=True),
                            href="/register"
                        )
                    ])
                ], className="h-100")
            ], md=4, className="mb-4"),
            
            # Essential Tier (Recommended)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Essential", className="mb-0"),
                        dbc.Badge("Recommended", color="success", className="ms-2")
                    ]),
                    dbc.CardBody([
                        html.H2("$29.99", className="mb-3"),
                        html.P("/month", className="text-muted mb-4"),
                        html.Ul([
                            html.Li("10 signals per day"),
                            html.Li("Full dashboard"),
                            html.Li("Email alerts"),
                            html.Li("API access"),
                            html.Li("Performance tracking"),
                            html.Li("7-day free trial"),
                        ], className="mb-4"),
                        dcc.Link(
                            dbc.Button("Start Free Trial", color="primary", className="w-100"),
                            href="/checkout?tier=essential&period=monthly",
                            id="checkout-essential-monthly"
                        ),
                        html.P([
                            "or ",
                            dcc.Link("$299.99/year", href="/checkout?tier=essential&period=yearly", className="text-primary")
                        ], className="text-center mt-2 small")
                    ])
                ], className="h-100 border-primary", style={"border-width": "2px"})
            ], md=4, className="mb-4"),
            
            # Advanced Tier
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Advanced", className="mb-0")),
                    dbc.CardBody([
                        html.H2("$99.99", className="mb-3"),
                        html.P("/month", className="text-muted mb-4"),
                        html.Ul([
                            html.Li("50 signals per day"),
                            html.Li("All Essential features"),
                            html.Li("Telegram alerts"),
                            html.Li("Portfolio tracking"),
                            html.Li("Priority support"),
                        ], className="mb-4"),
                        dbc.Button(
                            "Coming Soon",
                            color="secondary",
                            className="w-100",
                            disabled=True
                        )
                    ])
                ], className="h-100")
            ], md=4, className="mb-4"),
        ], className="mb-5"),
        
        # Comparison Table
        dbc.Row([
            dbc.Col([
                html.H3("Feature Comparison", className="text-white mb-4"),
                html.Div([
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Feature"),
                            html.Th("Free"),
                            html.Th("Essential"),
                            html.Th("Advanced"),
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td("Signals per day"),
                                html.Td("3"),
                                html.Td("10"),
                                html.Td("50"),
                            ]),
                            html.Tr([
                                html.Td("Email alerts"),
                                html.Td("❌"),
                                html.Td("✅"),
                                html.Td("✅"),
                            ]),
                            html.Tr([
                                html.Td("API access"),
                                html.Td("❌"),
                                html.Td("✅"),
                                html.Td("✅"),
                            ]),
                            html.Tr([
                                html.Td("Performance tracking"),
                                html.Td("❌"),
                                html.Td("✅"),
                                html.Td("✅"),
                            ]),
                            html.Tr([
                                html.Td("Telegram alerts"),
                                html.Td("❌"),
                                html.Td("❌"),
                                html.Td("✅"),
                            ]),
                            html.Tr([
                                html.Td("Portfolio tracking"),
                                html.Td("❌"),
                                html.Td("❌"),
                                html.Td("✅"),
                            ]),
                            html.Tr([
                                html.Td("Priority support"),
                                html.Td("❌"),
                                html.Td("❌"),
                                html.Td("✅"),
                            ]),
                        ])
                    ], className="table table-dark table-striped")
                ])
            ], width=12)
        ])
    ], fluid=True, className="mt-5")


def create_checkout_page():
    """Create the checkout redirect page (redirects to Stripe)."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Redirecting to checkout...", className="text-center mb-4"),
                        html.Div(id="checkout-status"),
                        dcc.Interval(id="checkout-redirect", interval=1000, n_intervals=0, max_intervals=1)
                    ])
                ])
            ], md=6, className="mx-auto mt-5")
        ])
    ])


def create_checkout_success_page():
    """Create the checkout success page."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-check-circle fa-3x text-success mb-4"),
                            html.H2("Payment Successful!", className="mb-3"),
                            html.P("Your subscription has been activated.", className="mb-4"),
                            dcc.Link(
                                dbc.Button("Go to Dashboard", color="primary"),
                                href="/dashboard"
                            )
                        ], className="text-center")
                    ])
                ])
            ], md=6, className="mx-auto mt-5")
        ])
    ])


def create_checkout_cancel_page():
    """Create the checkout cancel page."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-times-circle fa-3x text-danger mb-4"),
                            html.H2("Payment Cancelled", className="mb-3"),
                            html.P("Your payment was cancelled. No charges were made.", className="mb-4"),
                            dcc.Link(
                                dbc.Button("Back to Pricing", color="secondary"),
                                href="/pricing"
                            )
                        ], className="text-center")
                    ])
                ])
            ], md=6, className="mx-auto mt-5")
        ])
    ])


def create_billing_page():
    """Create the billing/subscription management page."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Billing & Subscription", className="text-white mb-4"),
                
                # Current Subscription
                dbc.Card([
                    dbc.CardHeader(html.H5("Current Subscription", className="mb-0")),
                    dbc.CardBody(id="subscription-details")
                ], className="mb-4"),
                
                # Payment Methods
                dbc.Card([
                    dbc.CardHeader(html.H5("Payment Methods", className="mb-0")),
                    dbc.CardBody(id="payment-methods")
                ], className="mb-4"),
                
                # Billing History
                dbc.Card([
                    dbc.CardHeader(html.H5("Billing History", className="mb-0")),
                    dbc.CardBody(id="billing-history")
                ])
            ], width=12)
        ])
    ], fluid=True, className="mt-4")


def register_payment_callbacks(dash_app):
    """Register payment-related callbacks."""
    import dash
    from flask import request
    
    @dash_app.callback(
        Output("checkout-status", "children"),
        [Input("checkout-redirect", "n_intervals"),
         Input("url", "search")],
        prevent_initial_call=True
    )
    def handle_checkout_redirect(n_intervals, search):
        """Handle checkout redirect to Stripe."""
        from dash.exceptions import PreventUpdate
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # Parse query parameters
        from urllib.parse import parse_qs, urlparse
        if search:
            params = parse_qs(urlparse(f"?{search}").query)
            tier = params.get('tier', ['essential'])[0]
            period = params.get('period', ['monthly'])[0]
        else:
            tier = 'essential'
            period = 'monthly'
        
        # Get current user
        user = get_current_user()
        if not user:
            return "Please log in first"
        
        # Get user email
        db = get_database()
        with db.get_session() as db_session:
            user_obj = AuthService.get_user_by_id(db_session, user.id)
            if not user_obj:
                return "User not found"
            
            # Create Stripe checkout session
            checkout_session = StripeClient.create_checkout_session(
                customer_email=user_obj.email,
                tier=tier,
                billing_period=period,
                success_url=f"{getattr(settings, 'BASE_URL', 'http://localhost:8050')}/checkout/success",
                cancel_url=f"{getattr(settings, 'BASE_URL', 'http://localhost:8050')}/checkout/cancel"
            )
            
            if checkout_session and checkout_session.get('url'):
                # Redirect to Stripe checkout
                return html.Div([
                    html.P("Redirecting to Stripe..."),
                    dcc.Location(id="stripe-redirect", href=checkout_session['url'], refresh=True)
                ])
            else:
                return html.Div([
                    html.P("Failed to create checkout session. Please try again."),
                    dcc.Link("Back to Pricing", href="/pricing")
                ])
    
    @dash_app.callback(
        Output("subscription-details", "children"),
        Input("url", "pathname"),
        prevent_initial_call=True
    )
    def update_subscription_details(pathname):
        """Update subscription details display."""
        user = get_current_user()
        if not user:
            return html.P("Please log in to view subscription details.")
        
        db = get_database()
        with db.get_session() as db_session:
            from src.auth.service import AuthService
            subscription = AuthService.get_user_subscription(db_session, user.id)
            
            if not subscription:
                return html.Div([
                    html.P("No active subscription."),
                    dcc.Link(
                        dbc.Button("Subscribe Now", color="primary"),
                        href="/pricing"
                    )
                ])
            
            # Format subscription details
            return html.Div([
                html.P([
                    html.Strong("Tier: "),
                    subscription.tier.upper()
                ]),
                html.P([
                    html.Strong("Status: "),
                    subscription.status.upper()
                ]),
                html.P([
                    html.Strong("Renewal Date: "),
                    subscription.current_period_end.strftime("%B %d, %Y") if subscription.current_period_end else "N/A"
                ]),
                html.Hr(),
                dbc.Button(
                    "Cancel Subscription",
                    color="danger",
                    outline=True,
                    id="cancel-subscription-btn"
                )
            ])
