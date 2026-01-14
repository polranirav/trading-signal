"""
Premium Charting Module.

Defines custom Plotly templates and chart factories for a high-end financial aesthetic.
"""

import plotly.graph_objects as go
import plotly.io as pio

# Define Premium Dark Theme
premium_dark = go.layout.Template(
    layout=go.Layout(
        # Backgrounds
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        
        # Fonts
        font={
            'family': 'Inter, sans-serif',
            'color': '#cbd5e1',
            'size': 12
        },
        
        # Colorway (Neon Palette)
        colorway=['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#06b6d4'],
        
        # Axes
        xaxis={
            'showgrid': True,
            'gridcolor': 'rgba(255, 255, 255, 0.05)',
            'gridwidth': 1,
            'zeroline': False,
            'showline': False,
            'tickfont': {'color': '#64748b'}
        },
        yaxis={
            'showgrid': True,
            'gridcolor': 'rgba(255, 255, 255, 0.05)',
            'gridwidth': 1,
            'zeroline': False,
            'showline': False,
            'tickfont': {'color': '#64748b'}
        },
        
        # Legend
        legend={
            'bgcolor': 'rgba(0,0,0,0)',
            'bordercolor': 'rgba(0,0,0,0)',
            'font': {'color': '#cbd5e1'}
        },
        
        # Hover
        hoverlabel={
            'bgcolor': '#1e293b',
            'bordercolor': '#334155',
            'font': {'family': 'Inter, sans-serif', 'color': '#fff'}
        },
        
        # Margins (Minimalist)
        margin={'t': 30, 'l': 10, 'r': 10, 'b': 10}
    )
)

# Register template
pio.templates['premium_dark'] = premium_dark
pio.templates.default = 'premium_dark'


def create_sparkline(data, color='#3b82f6'):
    """Create a minimalist sparkline chart."""
    fig = go.Figure(go.Scatter(
        y=data,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"
    ))
    
    fig.update_layout(
        template='premium_dark',
        showlegend=False,
        xaxis={'visible': False, 'fixedrange': True},
        yaxis={'visible': False, 'fixedrange': True},
        margin={'t': 0, 'l': 0, 'r': 0, 'b': 0},
        height=40,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def apply_glow_effect(fig):
    """Add glow effect to lines (simulated with shadow traces)."""
    # Note: Real glow in Plotly requires multiple traces with increasing width and transparency
    # This is a future enhancement
    return fig
