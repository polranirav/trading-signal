"""
Monitoring module for Trading Signals.

Provides:
- Prometheus metrics (metrics.py)
- Model performance tracking (performance.py)
- Alerting system (alerts.py)
- Grafana dashboard templates (dashboards/)
"""

from src.monitoring.metrics import (
    MetricsCollector,
    record_signal,
    record_error,
    record_api_call,
    record_ml_prediction,
    update_model_accuracy,
    update_market_regime
)

from src.monitoring.performance import (
    ModelPerformanceTracker,
    get_performance_tracker
)

from src.monitoring.alerts import (
    AlertManager,
    AlertType,
    AlertSeverity,
    fire_alert,
    get_alert_manager
)

__all__ = [
    # Metrics
    'MetricsCollector',
    'record_signal',
    'record_error',
    'record_api_call',
    'record_ml_prediction',
    'update_model_accuracy',
    'update_market_regime',
    
    # Performance tracking
    'ModelPerformanceTracker',
    'get_performance_tracker',
    
    # Alerts
    'AlertManager',
    'AlertType',
    'AlertSeverity',
    'fire_alert',
    'get_alert_manager',
]
