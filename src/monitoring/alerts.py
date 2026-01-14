"""
Alerting and Notification System.

Provides:
- Alert rules for signal quality drift
- Threshold-based alerts
- Notification channels (Slack, email placeholder)
- Alert aggregation

Alert Types:
1. Model Drift: Accuracy drops below threshold
2. Data Quality: Missing or stale data
3. System Health: High latency, errors
4. Trading: Unusual signals, high VaR
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import json
from pathlib import Path

from src.logging_config import get_logger

logger = get_logger(__name__)

ALERTS_DIR = Path("logs/alerts")
ALERTS_DIR.mkdir(parents=True, exist_ok=True)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Categories of alerts."""
    MODEL_DRIFT = "MODEL_DRIFT"
    DATA_QUALITY = "DATA_QUALITY"
    SYSTEM_HEALTH = "SYSTEM_HEALTH"
    TRADING = "TRADING"
    PERFORMANCE = "PERFORMANCE"


@dataclass
class Alert:
    """Single alert instance."""
    id: str
    type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata
        }


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    type: AlertType
    severity: AlertSeverity
    condition: Callable[[], bool]  # Returns True if alert should fire
    message_template: str
    cooldown_minutes: int = 60  # Don't re-alert within this time


class AlertManager:
    """
    Central alert management system.
    
    Features:
    - Register alert rules
    - Check conditions and fire alerts
    - Aggregate similar alerts
    - Send notifications
    """
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_fired: Dict[str, datetime] = {}
        self.notification_handlers: List[Callable[[Alert], None]] = []
        
        # Register default rules
        self._register_default_rules()
        self._load_history()
    
    def _load_history(self):
        """Load alert history from disk."""
        history_file = ALERTS_DIR / "alert_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                for alert_data in data[-100:]:  # Keep last 100
                    self.alert_history.append(Alert(
                        id=alert_data['id'],
                        type=AlertType(alert_data['type']),
                        severity=AlertSeverity(alert_data['severity']),
                        message=alert_data['message'],
                        timestamp=datetime.fromisoformat(alert_data['timestamp']),
                        resolved=alert_data.get('resolved', False),
                        metadata=alert_data.get('metadata', {})
                    ))
            except Exception as e:
                logger.warning(f"Failed to load alert history: {e}")
    
    def _save_history(self):
        """Save alert history to disk."""
        history_file = ALERTS_DIR / "alert_history.json"
        
        with open(history_file, 'w') as f:
            json.dump([a.to_dict() for a in self.alert_history[-100:]], f, indent=2)
    
    def _register_default_rules(self):
        """Register default alert rules."""
        
        # Model drift rule
        def check_model_drift():
            try:
                from src.monitoring.performance import get_performance_tracker
                tracker = get_performance_tracker()
                drift = tracker.detect_drift("ensemble")
                return drift.get("action_required", False)
            except:
                return False
        
        self.register_rule(AlertRule(
            name="model_accuracy_drift",
            type=AlertType.MODEL_DRIFT,
            severity=AlertSeverity.WARNING,
            condition=check_model_drift,
            message_template="Model accuracy has drifted below threshold. Recent accuracy may be degraded.",
            cooldown_minutes=360  # 6 hours
        ))
        
        # High VaR rule
        def check_high_var():
            # Placeholder: would check actual VaR
            return False
        
        self.register_rule(AlertRule(
            name="high_portfolio_var",
            type=AlertType.TRADING,
            severity=AlertSeverity.CRITICAL,
            condition=check_high_var,
            message_template="Portfolio VaR exceeds risk limits. Consider reducing exposure.",
            cooldown_minutes=60
        ))
    
    def register_rule(self, rule: AlertRule):
        """Register a new alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler (e.g., Slack, email)."""
        self.notification_handlers.append(handler)
    
    def fire_alert(
        self,
        type: AlertType,
        severity: AlertSeverity,
        message: str,
        metadata: Dict = None
    ) -> Alert:
        """
        Fire an alert manually.
        
        Args:
            type: Alert type
            severity: Alert severity
            message: Alert message
            metadata: Additional context
        
        Returns:
            Created Alert
        """
        alert_id = f"{type.value}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        
        alert = Alert(
            id=alert_id,
            type=type,
            severity=severity,
            message=message,
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self._save_history()
        
        # Log
        logger.warning(f"ALERT [{severity.value}] {type.value}: {message}")
        
        # Notify handlers
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
        
        # Update Prometheus metrics
        try:
            from src.monitoring.metrics import ERRORS_TOTAL
            ERRORS_TOTAL.labels(component="alerts", error_type=type.value).inc()
        except ImportError:
            pass
        
        return alert
    
    def check_rules(self):
        """Check all registered rules and fire alerts as needed."""
        for name, rule in self.rules.items():
            # Check cooldown
            last = self.last_fired.get(name)
            if last and datetime.utcnow() - last < timedelta(minutes=rule.cooldown_minutes):
                continue
            
            # Check condition
            try:
                should_fire = rule.condition()
            except Exception as e:
                logger.error(f"Rule check failed for {name}: {e}")
                continue
            
            if should_fire:
                self.fire_alert(
                    type=rule.type,
                    severity=rule.severity,
                    message=rule.message_template,
                    metadata={"rule": name}
                )
                self.last_fired[name] = datetime.utcnow()
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            self._save_history()
            logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alert status."""
        return {
            "active_count": len(self.active_alerts),
            "critical": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
            "warning": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.WARNING]),
            "recent_history": [a.to_dict() for a in self.alert_history[-10:]]
        }


# Notification Handlers

def slack_notification_handler(alert: Alert):
    """
    Send alert to Slack (placeholder).
    
    In production, use actual Slack webhook.
    """
    logger.info(f"[SLACK] Would send alert: {alert.message}")
    # webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    # requests.post(webhook_url, json={"text": f"🚨 {alert.severity.value}: {alert.message}"})


def email_notification_handler(alert: Alert):
    """
    Send alert via email (placeholder).
    
    In production, use SMTP or email service.
    """
    logger.info(f"[EMAIL] Would send alert: {alert.message}")


# Convenience functions
def get_alert_manager() -> AlertManager:
    """Get AlertManager singleton."""
    if not hasattr(get_alert_manager, '_instance'):
        get_alert_manager._instance = AlertManager()
    return get_alert_manager._instance


def fire_alert(type: AlertType, severity: AlertSeverity, message: str, metadata: Dict = None) -> Alert:
    """Convenience function to fire an alert."""
    return get_alert_manager().fire_alert(type, severity, message, metadata)


if __name__ == "__main__":
    print("=== Alert System Test ===\n")
    
    manager = AlertManager()
    
    # Add handlers
    manager.add_notification_handler(slack_notification_handler)
    
    # Fire some test alerts
    manager.fire_alert(
        type=AlertType.SYSTEM_HEALTH,
        severity=AlertSeverity.INFO,
        message="System startup complete"
    )
    
    manager.fire_alert(
        type=AlertType.MODEL_DRIFT,
        severity=AlertSeverity.WARNING,
        message="Model accuracy dropped 5% in last 7 days",
        metadata={"model": "ensemble", "accuracy_drop": 0.05}
    )
    
    # Check rules
    manager.check_rules()
    
    # Summary
    summary = manager.get_alert_summary()
    print(f"\nAlert Summary:")
    print(f"  Active: {summary['active_count']}")
    print(f"  Critical: {summary['critical']}")
    print(f"  Warning: {summary['warning']}")
