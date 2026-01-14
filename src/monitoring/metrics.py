"""
Prometheus Metrics for Trading Signals.

Exposes metrics for:
- Signal generation performance
- ML model accuracy
- Data pipeline health
- API latency
- System health

Metrics Categories:
1. Counters: signals_generated_total, errors_total
2. Gauges: active_symbols, model_accuracy
3. Histograms: processing_duration, api_latency
4. Summaries: signal_score_distribution

Usage:
    from src.monitoring.metrics import MetricsCollector
    MetricsCollector.start_server(port=8000)
    
    # Record metrics
    SIGNALS_GENERATED.labels(signal_type='BUY').inc()
    CONFLUENCE_SCORE.observe(0.72)
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    start_http_server, REGISTRY, generate_latest
)
from functools import wraps
import time
from typing import Callable
from datetime import datetime

from src.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# COUNTERS
# ============================================================================

# Signal generation
SIGNALS_GENERATED = Counter(
    'trading_signals_generated_total',
    'Total trading signals generated',
    ['symbol', 'signal_type']
)

SIGNALS_EXECUTED = Counter(
    'trading_signals_executed_total',
    'Signals that were executed (paper or live)',
    ['symbol', 'signal_type', 'outcome']
)

# Errors
ERRORS_TOTAL = Counter(
    'trading_errors_total',
    'Total errors by type',
    ['component', 'error_type']
)

# API calls
API_CALLS = Counter(
    'trading_api_calls_total',
    'External API calls',
    ['api', 'status']
)

# ML predictions
ML_PREDICTIONS = Counter(
    'trading_ml_predictions_total',
    'ML model predictions made',
    ['model', 'outcome']
)


# ============================================================================
# GAUGES
# ============================================================================

# Active tracking
ACTIVE_SYMBOLS = Gauge(
    'trading_active_symbols',
    'Number of symbols being tracked'
)

MODEL_ACCURACY = Gauge(
    'trading_model_accuracy',
    'Current model accuracy',
    ['model']
)

ENSEMBLE_AGREEMENT = Gauge(
    'trading_ensemble_agreement',
    'Agreement score across ensemble models'
)

MARKET_REGIME = Gauge(
    'trading_market_regime',
    'Current market regime (0=BULL, 1=BEAR, 2=SIDEWAYS, 3=HIGH_VOL)'
)

VIX_LEVEL = Gauge(
    'trading_vix_level',
    'Current VIX level'
)

# Portfolio metrics
PORTFOLIO_VALUE = Gauge(
    'trading_portfolio_value_usd',
    'Current portfolio value in USD'
)

PORTFOLIO_VAR = Gauge(
    'trading_portfolio_var_95',
    'Portfolio 95% Value at Risk'
)

OPEN_POSITIONS = Gauge(
    'trading_open_positions',
    'Number of open positions'
)


# ============================================================================
# HISTOGRAMS
# ============================================================================

# Processing latency
SIGNAL_PROCESSING_TIME = Histogram(
    'trading_signal_processing_seconds',
    'Time to generate a signal',
    ['symbol'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

DATA_FETCH_TIME = Histogram(
    'trading_data_fetch_seconds',
    'Time to fetch data from external sources',
    ['source'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

ML_INFERENCE_TIME = Histogram(
    'trading_ml_inference_seconds',
    'ML model inference time',
    ['model'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

SENTIMENT_ANALYSIS_TIME = Histogram(
    'trading_sentiment_analysis_seconds',
    'Sentiment analysis processing time',
    ['analyzer'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0]
)


# ============================================================================
# SUMMARIES
# ============================================================================

CONFLUENCE_SCORE = Summary(
    'trading_confluence_score',
    'Distribution of confluence scores'
)

TECHNICAL_SCORE = Summary(
    'trading_technical_score',
    'Distribution of technical scores'
)

SENTIMENT_SCORE = Summary(
    'trading_sentiment_score',
    'Distribution of sentiment scores'
)

POSITION_SIZE = Summary(
    'trading_position_size_percent',
    'Distribution of position sizes'
)


# ============================================================================
# INFO
# ============================================================================

SYSTEM_INFO = Info(
    'trading_system',
    'Trading system information'
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def track_time(metric: Histogram, label_values: dict = None):
    """
    Decorator to track function execution time.
    
    Usage:
        @track_time(SIGNAL_PROCESSING_TIME, {'symbol': 'AAPL'})
        def generate_signal(symbol):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            labels = label_values or {}
            with metric.labels(**labels).time():
                return func(*args, **kwargs)
        return wrapper
    return decorator


def record_signal(symbol: str, signal_type: str, confluence_score: float):
    """Record a generated trading signal."""
    SIGNALS_GENERATED.labels(symbol=symbol, signal_type=signal_type).inc()
    CONFLUENCE_SCORE.observe(confluence_score)


def record_error(component: str, error_type: str):
    """Record an error."""
    ERRORS_TOTAL.labels(component=component, error_type=error_type).inc()


def record_api_call(api: str, status: str):
    """Record an API call."""
    API_CALLS.labels(api=api, status=status).inc()


def record_ml_prediction(model: str, prediction: float, actual: float = None):
    """Record an ML prediction."""
    outcome = 'unknown'
    if actual is not None:
        # Check if prediction direction matches actual
        pred_direction = 'up' if prediction > 0.5 else 'down'
        actual_direction = 'up' if actual > 0 else 'down'
        outcome = 'correct' if pred_direction == actual_direction else 'incorrect'
    
    ML_PREDICTIONS.labels(model=model, outcome=outcome).inc()


def update_model_accuracy(model: str, accuracy: float):
    """Update model accuracy gauge."""
    MODEL_ACCURACY.labels(model=model).set(accuracy)


def update_market_regime(regime: str, vix: float = None):
    """Update market regime gauge."""
    regime_map = {'BULL': 0, 'BEAR': 1, 'SIDEWAYS': 2, 'HIGH_VOL': 3}
    MARKET_REGIME.set(regime_map.get(regime, 2))
    
    if vix is not None:
        VIX_LEVEL.set(vix)


# ============================================================================
# METRICS COLLECTOR CLASS
# ============================================================================

class MetricsCollector:
    """
    Central metrics collection and management.
    
    Provides:
    - Prometheus HTTP server
    - Metric registration
    - Periodic metric updates
    """
    
    _server_started = False
    
    @classmethod
    def start_server(cls, port: int = 9090):
        """
        Start Prometheus metrics HTTP server.
        
        Args:
            port: Port to expose metrics on (default 9090)
        """
        if cls._server_started:
            logger.warning("Metrics server already started")
            return
        
        try:
            start_http_server(port)
            cls._server_started = True
            
            # Set system info
            SYSTEM_INFO.info({
                'version': '1.0.0',
                'component': 'trading-signals',
                'started_at': datetime.utcnow().isoformat()
            })
            
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    @staticmethod
    def get_metrics() -> bytes:
        """Get current metrics as Prometheus format."""
        return generate_latest(REGISTRY)
    
    @staticmethod
    def update_active_symbols(count: int):
        """Update active symbols gauge."""
        ACTIVE_SYMBOLS.set(count)
    
    @staticmethod
    def update_portfolio_metrics(value: float, var: float, positions: int):
        """Update portfolio metrics."""
        PORTFOLIO_VALUE.set(value)
        PORTFOLIO_VAR.set(var)
        OPEN_POSITIONS.set(positions)


# ============================================================================
# FLASK INTEGRATION
# ============================================================================

def create_metrics_endpoint():
    """
    Create Flask blueprint for metrics endpoint.
    
    Usage:
        from src.monitoring.metrics import create_metrics_endpoint
        app.register_blueprint(create_metrics_endpoint())
    """
    from flask import Blueprint, Response
    
    metrics_bp = Blueprint('metrics', __name__)
    
    @metrics_bp.route('/metrics')
    def metrics():
        return Response(
            MetricsCollector.get_metrics(),
            mimetype='text/plain'
        )
    
    return metrics_bp


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_metrics_collector() -> MetricsCollector:
    """Get MetricsCollector instance."""
    return MetricsCollector()


if __name__ == "__main__":
    # Test metrics
    print("=== Prometheus Metrics Test ===\n")
    
    # Start server
    MetricsCollector.start_server(port=9090)
    
    # Record some test metrics
    record_signal("AAPL", "BUY", 0.72)
    record_signal("MSFT", "HOLD", 0.55)
    record_signal("GOOGL", "SELL", 0.35)
    
    record_api_call("alpha_vantage", "success")
    record_api_call("alpha_vantage", "success")
    record_api_call("yahoo", "error")
    
    record_ml_prediction("ensemble", 0.65, 0.02)
    record_ml_prediction("ensemble", 0.45, -0.01)
    
    update_model_accuracy("ensemble", 0.59)
    update_market_regime("BULL", vix=14.5)
    
    MetricsCollector.update_active_symbols(50)
    MetricsCollector.update_portfolio_metrics(100000, 0.05, 10)
    
    print("Metrics recorded successfully!")
    print("View metrics at http://localhost:9090/metrics")
    print("\nPress Ctrl+C to exit...")
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
