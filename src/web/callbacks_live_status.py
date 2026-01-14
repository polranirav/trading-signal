"""
Live Status Callbacks.

Callbacks for live data status updates and refresh functionality.
"""

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from datetime import datetime

from src.web.components.live_status import get_live_status
from src.web.components.dashboard_components import (
    create_live_status_bar,
    create_data_sources_panel
)


def register_live_status_callbacks(app):
    """
    Register callbacks for live status components.
    
    Callbacks:
    - Update live status bar every minute
    - Refresh data sources on button click
    - Health check button
    """
    
    @app.callback(
        Output('live-status-bar', 'children'),
        Input('interval-live-status', 'n_intervals')
    )
    def update_live_status_bar(n):
        """Update live status bar every minute."""
        try:
            status = get_live_status()
            
            active_count = status['active_count']
            total_count = status['total_count']
            last_update = status['last_update']
            
            # Format last update time
            if last_update:
                time_str = last_update.strftime('%I:%M %p')
            else:
                time_str = "Never"
            
            # Build status indicators
            status_indicators = []
            for key, source_status in status['statuses'].items():
                if source_status.status == 'unknown':
                    continue
                
                # Status icon
                if source_status.status == 'live':
                    icon = "✅"
                elif source_status.status == 'slow':
                    icon = "⚠️"
                else:
                    icon = "❌"
                
                status_indicators.append(
                    dbc.Badge(
                        f"{source_status.name} {icon}",
                        color="success" if source_status.status == 'live' else "warning" if source_status.status == 'slow' else "danger",
                        className="me-2"
                    )
                )
            
            return dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.Span("🔴 LIVE", style={'color': '#10b981', 'fontWeight': 'bold', 'marginRight': '15px'}),
                        html.Span(f"API Keys Active: {active_count}/{total_count}", style={'marginRight': '15px'}),
                        html.Span(f"Last Update: {time_str}", style={'marginRight': '15px'}),
                        html.Div(status_indicators, style={'display': 'inline-block', 'marginLeft': '20px'})
                    ], color="dark", className="mb-0")
                ], width=12)
            ])
        except Exception as e:
            # Return minimal error state (don't show intrusive alert)
            # Just show a small indicator or empty state
            return dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Status: ", style={'marginRight': '10px', 'color': '#9ca3af'}),
                        html.Span("⚠️ Update unavailable", style={'color': '#9ca3af', 'fontSize': '0.85em'})
                    ], style={
                        'padding': '8px 20px',
                        'backgroundColor': '#1e293b',
                        'borderRadius': '5px',
                        'marginBottom': '20px',
                        'fontSize': '0.9em'
                    })
                ], width=12)
            ])
    
    @app.callback(
        Output('data-sources-panel', 'children'),
        Input('refresh-data-sources-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def refresh_data_sources(n_clicks):
        """Refresh data sources on button click."""
        if n_clicks is None:
            raise PreventUpdate
        
        try:
            # Recreate the panel (forces refresh)
            panel = create_data_sources_panel()
            return panel.children
        except Exception as e:
            return dbc.Alert("Failed to refresh data sources", color="danger")
    
    @app.callback(
        Output('health-check-output', 'children'),
        Input('health-check-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def run_health_check(n_clicks):
        """Run API health check."""
        if n_clicks is None:
            raise PreventUpdate
        
        try:
            status = get_live_status()
            
            # Build health report
            report = []
            for key, source_status in status['statuses'].items():
                status_color = "success" if source_status.status == 'live' else "warning" if source_status.status == 'slow' else "danger"
                report.append(
                    dbc.ListGroupItem([
                        html.Strong(source_status.name),
                        dbc.Badge(source_status.status.upper(), color=status_color, className="ms-2"),
                        html.Br(),
                        html.Small(f"Last update: {source_status.freshness}")
                    ])
                )
            
            return dbc.ListGroup(report)
        except Exception as e:
            return dbc.Alert(f"Health check failed: {str(e)}", color="danger")
