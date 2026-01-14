"""
Dashboard UI Components.

Reusable components for the live dashboard:
- Live status bar
- Prediction points card
- Data sources panel
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
from datetime import datetime
from typing import Dict, List

from src.web.components.live_status import get_live_status, DataSourceStatus
from src.web.components.prediction_points import PredictionPoint, get_prediction_points


def create_live_status_bar() -> dbc.Row:
    """
    Create live status bar (top of dashboard).
    
    Shows:
    - API keys active count
    - Last update time
    - Data sources status
    """
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
            html.Span(
                f"{source_status.name} {icon}",
                style={'marginRight': '15px', 'fontSize': '0.9em'}
            )
        )
    
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("🔴 LIVE", style={'color': '#10b981', 'fontWeight': 'bold', 'marginRight': '15px'}),
                html.Span(f"API Keys Active: {active_count}/{total_count}", style={'marginRight': '15px'}),
                html.Span(f"Last Update: {time_str}", style={'marginRight': '15px'}),
                html.Div(status_indicators, style={'display': 'inline-block', 'marginLeft': '20px'})
            ], style={
                'padding': '10px 20px',
                'backgroundColor': '#1e293b',
                'borderRadius': '5px',
                'marginBottom': '20px',
                'fontSize': '0.9em'
            })
        ], width=12)
    ], id='live-status-bar')


def create_prediction_points_card(symbol: str, current_price: float) -> dbc.Card:
    """
    Create prediction points card (Perplexity style).
    
    Shows discrete price targets with confidence levels.
    """
    try:
        points = get_prediction_points(symbol, current_price)
    except Exception as e:
        # Return empty card on error
        points = []
    
    # Build prediction point rows
    point_rows = []
    for point in points:
        # Color based on confidence
        color = point.confidence_color
        if color == 'green':
            text_color = '#10b981'
        elif color == 'yellow':
            text_color = '#f59e0b'
        else:
            text_color = '#ef4444'
        
        # Format probability change
        prob_change_str = f"{point.direction}{abs(point.probability_change):.1f}%"
        
        point_rows.append(
            html.Div([
                html.Span(f"Target: ${point.target_price:.2f}", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                html.Span("→", style={'marginRight': '10px', 'color': '#64748b'}),
                html.Span(f"Confidence: {point.confidence:.0f}%", style={'marginRight': '10px', 'color': text_color}),
                html.Span(prob_change_str, style={'marginRight': '10px', 'color': text_color}),
                html.Span(f"({point.timeframe})", style={'color': '#64748b', 'fontSize': '0.9em'})
            ], style={
                'padding': '8px 0',
                'borderBottom': '1px solid #334155' if point != points[-1] else 'none'
            })
        )
    
    if not points:
        point_rows = [
            html.Div("No prediction points available", style={'color': '#64748b', 'fontStyle': 'italic'})
        ]
    
    # Get volume/metrics (would come from database)
    from src.web.components.prediction_points import PredictionPointsGenerator
    generator = PredictionPointsGenerator()
    volume_info = generator.get_prediction_volume(symbol)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("📊 Price Predictions (Next 30 Days)", style={'margin': 0})
        ]),
        dbc.CardBody([
            html.Div(point_rows),
            html.Hr(style={'margin': '15px 0', 'borderColor': '#334155'}),
            html.Div([
                html.Span(f"Volume: {volume_info['volume']}", style={'marginRight': '15px', 'fontSize': '0.9em', 'color': '#64748b'}),
                html.Span(f"Model: {volume_info['model']}", style={'marginRight': '15px', 'fontSize': '0.9em', 'color': '#64748b'}),
                html.Span(f"Updated: {datetime.utcnow().strftime('%I:%M %p')}", style={'fontSize': '0.9em', 'color': '#64748b'})
            ])
        ])
    ], className="mb-4", id='prediction-points-card')


def create_data_sources_panel() -> dbc.Card:
    """
    Create data sources panel showing status of each API.
    
    Shows:
    - Source name
    - Status (LIVE, SLOW, DOWN)
    - Last update time
    - Data freshness
    """
    status = get_live_status()
    
    # Build source rows
    source_rows = []
    for key, source_status in status['statuses'].items():
        # Status badge
        if source_status.status == 'live':
            badge = dbc.Badge("LIVE", color="success", className="me-2")
        elif source_status.status == 'slow':
            badge = dbc.Badge("SLOW", color="warning", className="me-2")
        elif source_status.status == 'down':
            badge = dbc.Badge("DOWN", color="danger", className="me-2")
        else:
            badge = dbc.Badge("UNKNOWN", color="secondary", className="me-2")
        
        # Last update time
        if source_status.last_update:
            last_update_str = source_status.last_update.strftime('%I:%M %p')
            freshness = source_status.freshness
        else:
            last_update_str = "Never"
            freshness = "Never"
        
        # Response time
        response_time_str = ""
        if source_status.response_time_ms:
            response_time_str = f" ({source_status.response_time_ms:.0f}ms)"
        
        source_rows.append(
            html.Div([
                html.Div([
                    html.Strong(source_status.name, style={'marginRight': '10px'}),
                    badge,
                    html.Span(f"Last: {last_update_str}", style={'marginRight': '10px', 'fontSize': '0.9em', 'color': '#64748b'}),
                    html.Span(f"({freshness})", style={'fontSize': '0.9em', 'color': '#64748b'}),
                    html.Span(response_time_str, style={'fontSize': '0.9em', 'color': '#64748b'})
                ])
            ], style={
                'padding': '10px 0',
                'borderBottom': '1px solid #334155' if key != list(status['statuses'].keys())[-1] else 'none'
            })
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("📡 Live Data Sources", style={'margin': 0})
        ]),
        dbc.CardBody([
            html.Div(source_rows),
            html.Hr(style={'margin': '15px 0', 'borderColor': '#334155'}),
            html.Div([
                dbc.Button("Refresh All", color="primary", size="sm", className="me-2", id='refresh-data-sources-btn'),
                dbc.Button("API Health Check", color="secondary", size="sm", id='health-check-btn')
            ]),
            html.Div(id='health-check-output', className="mt-3")
        ])
    ], className="mb-4", id='data-sources-panel')
