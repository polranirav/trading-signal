"""
Scheduled Celery Tasks for Signal Generation.

Tasks that run on a schedule (via Celery Beat) during market hours.
"""

from celery import shared_task
from typing import List
from datetime import datetime

from src.tasks.celery_app import app
from src.data.persistence import get_database
from src.tasks.analysis_tasks import generate_signal
from src.tasks.scheduler import TIER_RATE_LIMITS
from src.subscriptions.service import SubscriptionService
from src.logging_config import get_logger

logger = get_logger(__name__)


@app.task(
    bind=True,
    name='src.tasks.scheduled_tasks.generate_signals_for_watchlist',
    max_retries=3
)
def generate_signals_for_watchlist(self, user_id: str = None, tier: str = None):
    """
    Generate signals for all active assets in watchlist.
    
    Respects subscription tier rate limits.
    
    Args:
        user_id: Optional user ID (for user-specific watchlist)
        tier: Optional subscription tier (for rate limiting)
    """
    try:
        db = get_database()
        
        # Get active assets
        assets = db.get_active_assets()
        symbols = [asset.symbol for asset in assets]
        
        logger.info(f"Generating signals for {len(symbols)} symbols")
        
        # If user_id provided, filter by user's watchlist (TODO: implement watchlist feature)
        # For now, generate for all active assets
        
        # Rate limiting per tier
        max_signals = None
        if tier and tier in TIER_RATE_LIMITS:
            max_signals = TIER_RATE_LIMITS[tier]["signals_per_day"]
            if max_signals == -1:
                max_signals = None  # Unlimited
        
        signals_generated = 0
        
        for symbol in symbols:
            try:
                # Check rate limit
                if max_signals and signals_generated >= max_signals:
                    logger.info(f"Rate limit reached for tier {tier}: {max_signals} signals")
                    break
                
                # Generate signal (pass user_id if provided)
                if user_id:
                    result = generate_signal.delay(symbol, user_id=user_id)
                else:
                    result = generate_signal.delay(symbol)
                
                signals_generated += 1
                
                if signals_generated % 10 == 0:
                    logger.info(f"Generated {signals_generated} signals...")
                    
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
                continue
        
        logger.info(f"Signal generation complete: {signals_generated} signals queued")
        return {
            "status": "success",
            "signals_generated": signals_generated,
            "symbols_processed": len(symbols)
        }
        
    except Exception as e:
        logger.error(f"Error in generate_signals_for_watchlist: {e}", exc_info=True)
        raise self.retry(exc=e)


@app.task(
    bind=True,
    name='src.tasks.scheduled_tasks.generate_weekly_summary',
    max_retries=2
)
def generate_weekly_summary(self):
    """
    Generate weekly performance summary for all users.
    
    Sends email summary of signal performance for the past week.
    """
    try:
        from datetime import timedelta
        from src.data.persistence import get_database
        from src.subscriptions.service import SubscriptionService
        from src.tasks.notification_tasks import send_signal_email
        
        db = get_database()
        
        # Get all active users with email alerts
        users = db.get_users_with_email_alerts()
        
        week_ago = datetime.utcnow() - timedelta(days=7)
        
        summaries_sent = 0
        
        for user in users:
            try:
                # Get user's signals from past week
                signals = db.get_latest_signals(
                    limit=1000,
                    user_id=str(user.id)
                )
                
                week_signals = [
                    s for s in signals
                    if s.created_at and s.created_at >= week_ago
                ]
                
                if not week_signals:
                    continue
                
                # Calculate performance metrics
                from src.web.history import calculate_performance_metrics
                metrics = calculate_performance_metrics(week_signals)
                
                # TODO: Send weekly summary email
                # For now, just log
                logger.info(
                    f"Weekly summary for {user.email}: "
                    f"{metrics['total_signals']} signals, "
                    f"{metrics['win_rate']:.1f}% win rate"
                )
                
                summaries_sent += 1
                
            except Exception as e:
                logger.error(f"Error generating summary for user {user.id}: {e}", exc_info=True)
                continue
        
        logger.info(f"Weekly summary generation complete: {summaries_sent} summaries")
        return {
            "status": "success",
            "summaries_sent": summaries_sent
        }
        
    except Exception as e:
        logger.error(f"Error in generate_weekly_summary: {e}", exc_info=True)
        raise self.retry(exc=e)
