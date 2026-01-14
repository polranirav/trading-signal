"""
Confluence Celery tasks.

Tasks for running confluence analysis:
- analyze_confluence: Full confluence analysis for a symbol
- analyze_all_confluence: Batch analysis for all active symbols
- generate_trading_report: Generate comprehensive trading report
"""

from celery import shared_task
from datetime import datetime
from typing import List, Dict

from src.tasks.celery_app import app
from src.logging_config import get_logger

logger = get_logger(__name__)


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
    name='src.tasks.confluence_tasks.analyze_confluence'
)
def analyze_confluence(self, symbol: str) -> dict:
    """
    Run full confluence analysis for a symbol.
    
    Combines:
    - Technical analysis (40%)
    - Sentiment analysis (35%)
    - ML predictions (15%)
    - Risk metrics (10%)
    
    Args:
        symbol: Stock ticker
    
    Returns:
        Confluence result dictionary
    """
    logger.info("Running confluence analysis", symbol=symbol, task_id=self.request.id)
    
    try:
        from src.analytics.confluence import ConfluenceEngine
        from src.data.cache import get_cache
        
        engine = ConfluenceEngine()
        cache = get_cache()
        
        # Run analysis
        result = engine.analyze(symbol)
        
        # Cache the result
        cache_key = f"confluence:{symbol}"
        cache.client.setex(
            cache_key,
            3600,  # 1 hour TTL
            str(result.to_dict())
        )
        
        logger.info(
            "Confluence analysis complete",
            symbol=symbol,
            signal=result.signal_type.value,
            score=result.confluence_score
        )
        
        return {
            "status": "success",
            "symbol": symbol,
            "confluence_score": result.confluence_score,
            "signal_type": result.signal_type.value,
            "signal_strength": result.signal_strength.value,
            "recommended_position_pct": result.recommended_position_pct * 100,
            "rationale": result.overall_rationale
        }
        
    except Exception as e:
        logger.error("Confluence analysis failed", symbol=symbol, error=str(e))
        raise


@app.task(
    bind=True,
    name='src.tasks.confluence_tasks.analyze_all_confluence'
)
def analyze_all_confluence(self, symbols: List[str] = None) -> dict:
    """
    Run confluence analysis for all active symbols.
    
    Args:
        symbols: Optional list of symbols (defaults to active assets)
    
    Returns:
        Summary of analysis
    """
    logger.info("Starting batch confluence analysis", task_id=self.request.id)
    
    if not symbols:
        from src.data.persistence import get_database
        db = get_database()
        assets = db.get_active_assets()
        symbols = [a.symbol for a in assets] if assets else ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Queue individual tasks
    results = []
    for symbol in symbols:
        task = analyze_confluence.delay(symbol)
        results.append({"symbol": symbol, "task_id": task.id})
    
    logger.info("Queued confluence analysis", count=len(symbols))
    
    return {
        "status": "queued",
        "symbols_count": len(symbols),
        "tasks": results
    }


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    max_retries=2,
    name='src.tasks.confluence_tasks.generate_trading_report'
)
def generate_trading_report(self, symbols: List[str] = None) -> dict:
    """
    Generate comprehensive trading report with confluence analysis.
    
    Includes:
    - Top buy signals
    - Top sell signals
    - Risk summary
    - Market overview
    
    Args:
        symbols: Optional list of symbols
    
    Returns:
        Report dictionary
    """
    logger.info("Generating trading report", task_id=self.request.id)
    
    try:
        from src.analytics.confluence import ConfluenceEngine
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'UNH']
        
        engine = ConfluenceEngine()
        results = engine.analyze_batch(symbols)
        
        # Categorize signals
        buy_signals = [r for r in results if r.signal_type.value in ['STRONG_BUY', 'BUY']]
        sell_signals = [r for r in results if r.signal_type.value in ['STRONG_SELL', 'SELL']]
        hold_signals = [r for r in results if r.signal_type.value == 'HOLD']
        
        # Build report
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "symbols_analyzed": len(results),
            
            "summary": {
                "buy_count": len(buy_signals),
                "sell_count": len(sell_signals),
                "hold_count": len(hold_signals),
                "market_sentiment": "BULLISH" if len(buy_signals) > len(sell_signals) else "BEARISH" if len(sell_signals) > len(buy_signals) else "NEUTRAL"
            },
            
            "top_buys": [
                {
                    "symbol": r.symbol,
                    "score": r.confluence_score,
                    "signal": r.signal_type.value,
                    "strength": r.signal_strength.value,
                    "position_pct": r.recommended_position_pct * 100
                }
                for r in buy_signals[:5]
            ],
            
            "top_sells": [
                {
                    "symbol": r.symbol,
                    "score": r.confluence_score,
                    "signal": r.signal_type.value,
                    "strength": r.signal_strength.value
                }
                for r in sell_signals[:5]
            ],
            
            "risk_metrics": {
                "avg_var_95": sum(r.var_95 for r in results) / len(results) * 100 if results else 0,
                "high_risk_count": sum(1 for r in results if r.var_95 > 0.08),
            }
        }
        
        logger.info(
            "Trading report generated",
            buys=len(buy_signals),
            sells=len(sell_signals),
            holds=len(hold_signals)
        )
        
        return {
            "status": "success",
            "report": report
        }
        
    except Exception as e:
        logger.error("Report generation failed", error=str(e))
        raise
