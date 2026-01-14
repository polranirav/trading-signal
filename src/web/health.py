"""
Health check and metrics endpoints.

Provides:
- /health - Basic health check
- /ready - Readiness check (database, redis)
- /metrics - Prometheus metrics (future)
"""

from flask import Blueprint, jsonify
import time

from src.logging_config import get_logger

logger = get_logger(__name__)

health_bp = Blueprint('health', __name__)

# Track start time
START_TIME = time.time()


@health_bp.route('/health')
def health():
    """
    Basic health check endpoint.
    
    Returns 200 if the service is running.
    """
    return jsonify({
        "status": "healthy",
        "uptime_seconds": int(time.time() - START_TIME),
        "service": "trading-signals"
    }), 200


@health_bp.route('/ready')
def ready():
    """
    Readiness check endpoint.
    
    Verifies database and Redis connections are active.
    """
    checks = {
        "database": False,
        "redis": False
    }
    
    # Check database
    try:
        from src.data.persistence import get_database
        db = get_database()
        # Simple query to verify connection
        with db.SessionLocal() as session:
            session.execute("SELECT 1")
        checks["database"] = True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
    
    # Check Redis
    try:
        from src.data.cache import get_cache
        cache = get_cache()
        cache.client.ping()
        checks["redis"] = True
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return jsonify({
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks
    }), status_code


@health_bp.route('/metrics')
def metrics():
    """
    Prometheus-compatible metrics endpoint.
    
    Returns basic metrics in Prometheus format.
    """
    # Basic metrics for now
    # In production, use prometheus_client library
    
    try:
        from src.data.cache import get_cache
        cache = get_cache()
        
        # Get some basic stats
        info = cache.client.info()
        
        metrics_text = f"""# HELP trading_uptime_seconds Time since service start
# TYPE trading_uptime_seconds gauge
trading_uptime_seconds {int(time.time() - START_TIME)}

# HELP redis_connected_clients Number of connected Redis clients
# TYPE redis_connected_clients gauge
redis_connected_clients {info.get('connected_clients', 0)}

# HELP redis_used_memory_bytes Redis memory usage
# TYPE redis_used_memory_bytes gauge
redis_used_memory_bytes {info.get('used_memory', 0)}
"""
        return metrics_text, 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return f"# Error collecting metrics: {e}\n", 500, {'Content-Type': 'text/plain'}


def register_health_routes(app):
    """Register health check routes with a Flask/Dash app."""
    # For Dash, we need to access the underlying Flask server
    if hasattr(app, 'server'):
        app.server.register_blueprint(health_bp)
    else:
        app.register_blueprint(health_bp)
