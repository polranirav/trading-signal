"""
Notification Tasks for Celery.

Handles async email sending for signal alerts.
"""

from celery import shared_task
from typing import Dict, List, Optional

from src.tasks.celery_app import app
from src.notifications.email import get_email_service
from src.logging_config import get_logger
from src.config import settings

logger = get_logger(__name__)


@app.task(
    bind=True,
    name='src.tasks.notification_tasks.send_signal_email',
    max_retries=3,
    default_retry_delay=60
)
def send_signal_email(
    self,
    user_email: str,
    signal_data: Dict,
    dashboard_url: Optional[str] = None
) -> Dict:
    """
    Send signal alert email to a user.
    
    Args:
        user_email: Recipient email address
        signal_data: Signal data dictionary
        dashboard_url: Optional dashboard URL
    
    Returns:
        Result dictionary with status
    """
    logger.info(f"Sending signal email to {user_email}", symbol=signal_data.get('symbol'))
    
    try:
        email_service = get_email_service()
        
        if not dashboard_url:
            dashboard_url = f"{getattr(settings, 'BASE_URL', 'http://localhost:8050')}/dashboard"
        
        success = email_service.send_signal_alert(
            to_email=user_email,
            signal_data=signal_data,
            dashboard_url=dashboard_url
        )
        
        if success:
            logger.info(f"Signal email sent successfully to {user_email}")
            return {
                "status": "success",
                "email": user_email,
                "symbol": signal_data.get('symbol')
            }
        else:
            logger.warning(f"Failed to send email to {user_email}")
            return {
                "status": "failed",
                "email": user_email,
                "reason": "email_service_not_configured"
            }
            
    except Exception as e:
        logger.error(f"Email sending error: {e}", exc_info=True)
        # Retry on failure
        raise self.retry(exc=e)


@app.task(
    bind=True,
    name='src.tasks.notification_tasks.send_signal_emails_batch',
    max_retries=2
)
def send_signal_emails_batch(
    self,
    user_emails: List[str],
    signal_data: Dict,
    dashboard_url: Optional[str] = None
) -> Dict:
    """
    Send signal alerts to multiple users.
    
    Args:
        user_emails: List of email addresses
        signal_data: Signal data dictionary
        dashboard_url: Optional dashboard URL
    
    Returns:
        Result dictionary with sent/failed counts
    """
    logger.info(f"Sending batch signal emails to {len(user_emails)} users")
    
    try:
        email_service = get_email_service()
        
        if not dashboard_url:
            dashboard_url = f"{getattr(settings, 'BASE_URL', 'http://localhost:8050')}/dashboard"
        
        results = email_service.send_batch_alerts(
            emails=user_emails,
            signal_data=signal_data,
            dashboard_url=dashboard_url
        )
        
        logger.info(f"Batch email completed: {results['sent']} sent, {results['failed']} failed")
        return {
            "status": "success",
            "total": len(user_emails),
            "sent": results['sent'],
            "failed": results['failed']
        }
        
    except Exception as e:
        logger.error(f"Batch email error: {e}", exc_info=True)
        raise
