"""
Email Service for Signal Alerts.

Handles email delivery via SendGrid or AWS SES.
"""

from typing import Dict, List, Optional
from datetime import datetime

from src.config import settings
from src.logging_config import get_logger
from src.notifications.templates import EmailTemplates

logger = get_logger(__name__)


class EmailService:
    """Email service for sending signal alerts."""
    
    def __init__(self):
        """Initialize email service."""
        # Try to initialize SendGrid or SES
        self.sendgrid_api_key = getattr(settings, 'SENDGRID_API_KEY', None)
        self.from_email = getattr(settings, 'EMAIL_FROM', 'signals@tradingsignals.com')
        self.from_name = getattr(settings, 'EMAIL_FROM_NAME', 'Trading Signals Pro')
        
        # Initialize SendGrid if available
        if self.sendgrid_api_key:
            try:
                import sendgrid
                self.sg = sendgrid.SendGridAPIClient(api_key=self.sendgrid_api_key)
                self.use_sendgrid = True
                logger.info("Email service initialized with SendGrid")
            except ImportError:
                logger.warning("SendGrid not installed, email sending disabled")
                self.sg = None
                self.use_sendgrid = False
        else:
            self.sg = None
            self.use_sendgrid = False
            logger.warning("SENDGRID_API_KEY not set, email sending disabled")
    
    def send_signal_alert(
        self,
        to_email: str,
        signal_data: Dict,
        dashboard_url: str = None
    ) -> bool:
        """
        Send signal alert email.
        
        Args:
            to_email: Recipient email address
            signal_data: Signal data dictionary
            dashboard_url: Optional dashboard URL for links
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.use_sendgrid or not self.sg:
            logger.warning("Email service not configured, skipping email")
            return False
        
        try:
            from sendgrid.helpers.mail import Mail, Email, To, Content
            
            # Add dashboard URL to signal data
            if dashboard_url:
                signal_data = signal_data.copy()
                signal_data['dashboard_url'] = dashboard_url
                signal_data['unsubscribe_url'] = f"{dashboard_url}/account/notifications"
                signal_data['preferences_url'] = f"{dashboard_url}/account/notifications"
            
            # Generate email content
            html_content = EmailTemplates.signal_alert_html(signal_data)
            text_content = EmailTemplates.signal_alert_text(signal_data)
            
            # Create email
            message = Mail(
                from_email=Email(self.from_email, self.from_name),
                to_emails=To(to_email),
                subject=f"🚀 New Signal: {signal_data.get('symbol', 'N/A')} - {signal_data.get('signal_type', 'HOLD')}",
                html_content=Content("text/html", html_content),
                plain_text_content=Content("text/plain", text_content)
            )
            
            # Send email
            response = self.sg.send(message)
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Signal alert email sent to {to_email}", symbol=signal_data.get('symbol'))
                return True
            else:
                logger.error(f"Failed to send email: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Email sending error: {e}")
            return False
    
    def send_batch_alerts(
        self,
        emails: List[str],
        signal_data: Dict,
        dashboard_url: str = None
    ) -> Dict:
        """
        Send signal alerts to multiple recipients.
        
        Args:
            emails: List of email addresses
            signal_data: Signal data dictionary
            dashboard_url: Optional dashboard URL
        
        Returns:
            Dictionary with success/failure counts
        """
        results = {'sent': 0, 'failed': 0}
        
        for email in emails:
            success = self.send_signal_alert(email, signal_data, dashboard_url)
            if success:
                results['sent'] += 1
            else:
                results['failed'] += 1
        
        logger.info(f"Batch email sent: {results['sent']} succeeded, {results['failed']} failed")
        return results


def get_email_service() -> EmailService:
    """Get email service singleton."""
    global _email_service
    if '_email_service' not in globals():
        _email_service = EmailService()
    return _email_service
