"""
Notifications Module.

Handles signal delivery via email, Telegram, etc.
"""

from src.notifications.email import EmailService
from src.notifications.templates import EmailTemplates

__all__ = [
    "EmailService",
    "EmailTemplates",
]
