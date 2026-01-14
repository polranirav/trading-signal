"""
Celery Beat Scheduler Configuration.

Defines scheduled tasks for signal generation during market hours.
"""

from celery.schedules import crontab
from datetime import time

# Market hours (ET/EDT)
# Pre-market: 4:00 AM - 9:30 AM
# Regular: 9:30 AM - 4:00 PM
# After-hours: 4:00 PM - 8:00 PM

beat_schedule = {
    # Daily market open analysis (9:30 AM ET)
    "daily-market-open-analysis": {
        "task": "src.tasks.analysis_tasks.run_daily_analysis",
        "schedule": crontab(hour=9, minute=30, day_of_week="mon-fri"),
        "options": {"timezone": "America/New_York", "queue": "analysis"}
    },
    
    # Mid-day signal generation (12:00 PM ET)
    "midday-signal-generation": {
        "task": "src.tasks.scheduled_tasks.generate_signals_for_watchlist",
        "schedule": crontab(hour=12, minute=0, day_of_week="mon-fri"),
        "options": {"timezone": "America/New_York", "queue": "analysis"}
    },
    
    # End-of-day analysis (4:15 PM ET, after market close)
    "end-of-day-analysis": {
        "task": "src.tasks.analysis_tasks.run_daily_analysis",
        "schedule": crontab(hour=16, minute=15, day_of_week="mon-fri"),
        "options": {"timezone": "America/New_York", "queue": "analysis"}
    },
    
    # Weekly summary (Sunday 8:00 PM ET)
    "weekly-summary": {
        "task": "src.tasks.scheduled_tasks.generate_weekly_summary",
        "schedule": crontab(hour=20, minute=0, day_of_week="sun"),
        "options": {"timezone": "America/New_York", "queue": "default"}
    },
    
    # Indicator update (before market open, 8:00 AM ET)
    "pre-market-indicator-update": {
        "task": "src.tasks.analysis_tasks.update_all_indicators",
        "schedule": crontab(hour=8, minute=0, day_of_week="mon-fri"),
        "options": {"timezone": "America/New_York", "queue": "analysis"}
    },
}

# Rate limiting per subscription tier
# Signals per day limits
TIER_RATE_LIMITS = {
    "free": {
        "signals_per_day": 5,
        "api_calls_per_day": 100,
    },
    "essential": {
        "signals_per_day": 50,
        "api_calls_per_day": 1000,
    },
    "advanced": {
        "signals_per_day": 200,
        "api_calls_per_day": 10000,
    },
    "premium": {
        "signals_per_day": -1,  # Unlimited
        "api_calls_per_day": -1,  # Unlimited
    },
}
