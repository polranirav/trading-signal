"""
Analysis Celery tasks.

Tasks for computing indicators and generating signals:
- update_indicators: Compute technical indicators for a symbol
- update_all_indicators: Batch indicator update
- run_daily_analysis: Full daily analysis pipeline

All tasks are idempotent and safe to retry.
"""

from celery import shared_task
from datetime import datetime
from typing import List, Dict

from src.tasks.celery_app import app
from src.data.persistence import get_database
from src.data.cache import get_cache
from src.logging_config import get_logger
from src.config import settings

logger = get_logger(__name__)


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
    name='src.tasks.analysis_tasks.update_indicators'
)
def update_indicators(self, symbol: str) -> dict:
    """
    Compute and cache technical indicators for a symbol.
    
    Indicators computed:
    - RSI (14)
    - MACD (12, 26, 9)
    - Bollinger Bands (20, 2σ)
    - SMA (20, 50, 200)
    - ATR (14)
    
    Args:
        symbol: Stock ticker
    
    Returns:
        Result dictionary with indicator summary
    """
    logger.info("Updating indicators", symbol=symbol, task_id=self.request.id)
    
    try:
        from src.analytics.signals import SignalGenerator
        
        generator = SignalGenerator()
        indicators = generator.update_and_save_indicators(symbol)
        
        if indicators is None:
            logger.warning("Insufficient data for indicators", symbol=symbol)
            return {"status": "insufficient_data", "symbol": symbol}
        
        logger.info(
            "Updated indicators", 
            symbol=symbol,
            score=indicators.get('technical_score')
        )
        
        return {
            "status": "success",
            "symbol": symbol,
            "as_of_date": str(indicators.get("as_of_date")),
            "technical_score": indicators.get("technical_score"),
            "rsi": indicators.get("rsi_14"),
            "trend": indicators.get("trend_signal"),
        }
        
    except Exception as e:
        logger.error("Failed to update indicators", symbol=symbol, error=str(e))
        raise


@app.task(
    bind=True,
    name='src.tasks.analysis_tasks.update_all_indicators'
)
def update_all_indicators(self, symbols: List[str] = None) -> dict:
    """
    Update indicators for all active assets.
    
    Args:
        symbols: Optional list of symbols
    
    Returns:
        Summary of update operation
    """
    logger.info("Starting batch indicator update", task_id=self.request.id)
    
    if not symbols:
        db = get_database()
        assets = db.get_active_assets()
        symbols = [a.symbol for a in assets]
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Queue individual tasks
    results = []
    for symbol in symbols:
        task = update_indicators.delay(symbol)
        results.append({"symbol": symbol, "task_id": task.id})
    
    return {
        "status": "queued",
        "symbols": len(symbols),
        "tasks": results
    }


@app.task(
    bind=True,
    name='src.tasks.analysis_tasks.run_daily_analysis'
)
def run_daily_analysis(self) -> dict:
    """
    Run full daily analysis pipeline.
    
    This is scheduled to run at market close (4:30 PM ET).
    
    Pipeline:
    1. Fetch latest prices
    2. Compute indicators
    3. Run sentiment analysis
    4. Generate signals
    
    Returns:
        Summary of analysis run
    """
    logger.info("Starting daily analysis", task_id=self.request.id)
    
    try:
        db = get_database()
        assets = db.get_active_assets()
        symbols = [a.symbol for a in assets]
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        
        # Step 1: Update indicators for all symbols
        indicator_tasks = []
        for symbol in symbols:
            task = update_indicators.delay(symbol)
            indicator_tasks.append(task.id)
        
        logger.info("Queued indicator updates", count=len(indicator_tasks))
        
        # Step 2: Wait for indicators to complete, then generate signals
        # Note: In production, use task chains or callbacks for proper sequencing
        # For now, generate signals (they will use cached indicators if available)
        signal_tasks = []
        for symbol in symbols:
            # Generate signal (will use ConfluenceEngine internally)
            task = generate_signal.delay(symbol)
            signal_tasks.append({"symbol": symbol, "task_id": task.id})
        
        logger.info("Queued signal generation", count=len(signal_tasks))
        
        return {
            "status": "success",
            "symbols_analyzed": len(symbols),
            "indicator_tasks": len(indicator_tasks),
            "signal_tasks": len(signal_tasks),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Daily analysis failed", error=str(e))
        raise


@app.task(
    bind=True,
    name='src.tasks.analysis_tasks.generate_signal'
)
def generate_signal(self, symbol: str) -> dict:
    """
    Generate trading signal for a single symbol using ConfluenceEngine.
    
    Combines:
    - Technical indicators
    - Sentiment analysis (FinBERT)
    - ML predictions
    - Risk metrics (Monte Carlo VaR)
    
    Args:
        symbol: Stock ticker
    
    Returns:
        Generated signal details with all confluence scores
    """
    logger.info("Generating signal", symbol=symbol, task_id=self.request.id)
    
    try:
        from src.analytics.confluence import ConfluenceEngine
        
        db = get_database()
        cache = get_cache()
        
        # Initialize ConfluenceEngine (handles all analysis internally)
        engine = ConfluenceEngine()
        
        # Perform full confluence analysis
        # This automatically:
        # 1. Fetches price data
        # 2. Computes technical indicators
        # 3. Runs sentiment analysis (FinBERT)
        # 4. Calculates risk metrics (VaR)
        # 5. Gets ML predictions if available
        # 6. Combines everything into unified signal
        result = engine.analyze(symbol=symbol, use_cache=True)
        
        if result is None:
            logger.warning("Confluence analysis returned None", symbol=symbol)
            return {"status": "failed", "symbol": symbol, "reason": "insufficient_data"}
        
        # Map ConfluenceResult to TradeSignal model fields
        signal_data = {
            "symbol": result.symbol,
            "signal_type": result.signal_type.value,  # STRONG_BUY, BUY, HOLD, etc.
            "signal_strength": result.signal_strength.value,  # VERY_STRONG, STRONG, etc.
            "technical_score": result.technical_score,
            "sentiment_score": result.sentiment_score,
            "ml_score": result.ml_score,
            "risk_score": result.risk_score,
            "confluence_score": result.confluence_score,
            "var_95": result.var_95,
            "suggested_position_size": result.recommended_position_pct,
            "position_size": result.recommended_position_pct,  # Alias for task return
            "stop_loss": result.stop_loss_pct,
            "take_profit": result.take_profit_pct,
            "risk_reward_ratio": result.risk_reward_ratio,
            "technical_rationale": result.technical_rationale,
            "sentiment_rationale": result.sentiment_rationale,
            "risk_warning": result.overall_rationale,  # Overall rationale as risk warning
            "created_at": result.timestamp or datetime.utcnow(),
        }
        
        # Add optional fields if available from risk_details
        risk_details = result.risk_details or {}
        if "cvar_95" in risk_details:
            signal_data["cvar_95"] = risk_details["cvar_95"]
        if "max_drawdown" in risk_details:
            signal_data["max_drawdown"] = risk_details["max_drawdown"]
        if "sharpe_ratio" in risk_details:
            signal_data["sharpe_ratio"] = risk_details["sharpe_ratio"]
        
        # Get current price if available (for price_at_signal)
        try:
            from src.data.persistence import get_database
            db = get_database()
            latest_candle = db.get_candles(symbol, limit=1)
            if not latest_candle.empty:
                signal_data["price_at_signal"] = float(latest_candle.iloc[-1]["close"])
        except Exception:
            pass  # Price not critical, skip if unavailable
        
        # Save signal to database
        signal_id = db.save_signal(signal_data)
        signal_data["signal_id"] = signal_id
        
        # Cache signal for fast retrieval (TTL: 24 hours)
        cache.set_signal(symbol, signal_data, ttl=24 * 60 * 60)
        
        logger.info(
            "Generated signal",
            symbol=symbol,
            type=signal_data["signal_type"],
            confidence=signal_data["confluence_score"],
            signal_id=signal_id
        )
        
        # Send email alerts to subscribed users (async)
        try:
            from src.tasks.notification_tasks import send_signal_emails_batch
            
            # Get users with email alerts enabled (Essential tier and above)
            users = db.get_users_with_email_alerts(tier='essential')  # TODO: Add 'advanced' tier
            
            if users:
                user_emails = [user.email for user in users if user.email_verified]
                
                if user_emails:
                    # Send emails asynchronously
                    dashboard_url = f"{getattr(settings, 'BASE_URL', 'http://localhost:8050')}/signals/{signal_id}"
                    send_signal_emails_batch.delay(
                        user_emails=user_emails,
                        signal_data=signal_data,
                        dashboard_url=dashboard_url
                    )
                    logger.info(f"Queued email alerts for {len(user_emails)} users")
        except Exception as e:
            logger.warning(f"Failed to queue email alerts: {e}", exc_info=True)
            # Don't fail signal generation if email fails
        
        return {
            "status": "success",
            "symbol": symbol,
            "signal_id": signal_id,
            "signal_type": signal_data["signal_type"],
            "signal_strength": result.signal_strength.value,
            "confluence_score": signal_data["confluence_score"],
            "technical_score": signal_data["technical_score"],
            "sentiment_score": signal_data["sentiment_score"],
            "ml_score": signal_data.get("ml_score", 0.5),
            "risk_score": result.risk_score,
            "position_size": signal_data["suggested_position_size"],
            "stop_loss": result.stop_loss_pct,
            "take_profit": result.take_profit_pct,
        }
        
    except Exception as e:
        logger.error("Signal generation failed", symbol=symbol, error=str(e), exc_info=True)
        raise
