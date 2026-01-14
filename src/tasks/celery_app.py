"""
Celery application configuration.

Configures the Celery app with:
- Redis broker and backend
- Task routing to specialized queues
- Retry policies with exponential backoff
- Scheduled tasks (Celery Beat)

Usage:
    celery -A src.tasks.celery_app worker -Q ingestion -l INFO
    celery -A src.tasks.celery_app beat -l INFO
"""

from celery import Celery
from celery.schedules import crontab
from kombu import Queue

from src.config import settings

# Create Celery app
app = Celery(
    'trading_signals',
    broker=settings.CELERY_BROKER,
    backend=settings.CELERY_BACKEND,
    include=[
        'src.tasks.ingestion_tasks',
        'src.tasks.analysis_tasks',
        'src.tasks.sentiment_tasks',
        'src.tasks.confluence_tasks',
        'src.tasks.ml_tasks',
    ]
)

# Celery configuration
app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Timezone
    timezone='America/New_York',
    enable_utc=True,
    
    # Task options
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,  # Requeue if worker dies
    task_time_limit=300,  # 5 minute hard limit
    task_soft_time_limit=270,  # 4.5 minute soft limit
    
    # Worker options
    worker_prefetch_multiplier=1,  # One task at a time
    worker_concurrency=2,  # 2 concurrent tasks
    
    # Result backend
    result_expires=3600,  # Results expire in 1 hour
    
    # Retry policy (default for all tasks)
    task_default_retry_delay=60,  # 1 minute initial delay
    task_retry_backoff=True,  # Exponential backoff
    task_retry_backoff_max=3600,  # Max 1 hour between retries
    task_max_retries=5,
    
    # Queue routing
    task_queues=(
        Queue('ingestion', routing_key='ingestion.#'),
        Queue('analysis', routing_key='analysis.#'),
        Queue('alerts', routing_key='alerts.#'),
        Queue('default', routing_key='default.#'),
    ),
    task_default_queue='default',
    task_default_routing_key='default.task',
)

# Task routing - send tasks to their designated queues
app.conf.task_routes = {
    'src.tasks.ingestion_tasks.*': {'queue': 'ingestion'},
    'src.tasks.analysis_tasks.*': {'queue': 'analysis'},
}

# Celery Beat schedule (scheduled tasks)
# Merge with scheduler.py schedule
try:
    from src.tasks.scheduler import beat_schedule as signal_schedule
    app.conf.beat_schedule.update(signal_schedule)
except ImportError:
    pass  # Scheduler module not available

# Existing scheduled tasks
app.conf.beat_schedule.update({
    # Fetch market data every 5 minutes during market hours
    'fetch-prices-every-5-min': {
        'task': 'src.tasks.ingestion_tasks.fetch_all_prices',
        'schedule': crontab(minute='*/5', hour='9-16', day_of_week='mon-fri'),
        'options': {'queue': 'ingestion'}
    },
    
    # Fetch news every 15 minutes
    'fetch-news-every-15-min': {
        'task': 'src.tasks.ingestion_tasks.fetch_all_news',
        'schedule': crontab(minute='*/15'),
        'options': {'queue': 'ingestion'}
    },
    
    # Run full analysis at market close (4:30 PM ET)
    'daily-analysis-market-close': {
        'task': 'src.tasks.analysis_tasks.run_daily_analysis',
        'schedule': crontab(minute=30, hour=16, day_of_week='mon-fri'),
        'options': {'queue': 'analysis'}
    },
    
    # Update indicators every hour during market hours
    'update-indicators-hourly': {
        'task': 'src.tasks.analysis_tasks.update_all_indicators',
        'schedule': crontab(minute=0, hour='9-16', day_of_week='mon-fri'),
        'options': {'queue': 'analysis'}
    },
})


if __name__ == '__main__':
    app.start()
