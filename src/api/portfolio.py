"""
Portfolio API Routes.

Endpoints for managing user's portfolio holdings:
- List holdings
- Add holding
- Update holding
- Delete holding
- Import from CSV
- Get portfolio summary with P&L
"""

from flask import Blueprint, request, jsonify, g
from sqlalchemy.orm import Session
from sqlalchemy import and_
from datetime import datetime
import uuid
import csv
from io import StringIO

from src.auth.middleware import require_auth
from src.data.models import PortfolioHolding, PortfolioTransaction
from src.data.persistence import get_database
from src.logging_config import get_logger

logger = get_logger(__name__)

portfolio_bp = Blueprint("portfolio", __name__, url_prefix="/portfolio")


@portfolio_bp.route("", methods=["GET"])
@require_auth
def get_holdings():
    """Get all holdings for the current user."""
    user = g.current_user
    
    db = get_database()
    with db.get_session() as session:
        holdings = session.query(PortfolioHolding).filter(
            and_(
                PortfolioHolding.user_id == user.id,
                PortfolioHolding.is_active == True
            )
        ).all()
        
        result = []
        for h in holdings:
            result.append({
                "id": str(h.id),
                "symbol": h.symbol,
                "shares": h.shares,
                "avg_cost": h.avg_cost,
                "cost_basis": h.shares * h.avg_cost,
                "purchase_date": h.purchase_date.isoformat() if h.purchase_date else None,
                "source": h.source,
                "notes": h.notes,
                "created_at": h.created_at.isoformat() if h.created_at else None,
            })
        
        return jsonify({
            "holdings": result,
            "count": len(result)
        })


@portfolio_bp.route("/summary", methods=["GET"])
@require_auth
def get_portfolio_summary():
    """Get portfolio summary with total value and P&L."""
    user = g.current_user
    
    db = get_database()
    with db.get_session() as session:
        holdings = session.query(PortfolioHolding).filter(
            and_(
                PortfolioHolding.user_id == user.id,
                PortfolioHolding.is_active == True
            )
        ).all()
        
        total_cost_basis = 0
        total_current_value = 0
        holdings_list = []
        
        for h in holdings:
            cost_basis = h.shares * h.avg_cost
            total_cost_basis += cost_basis
            
            # Real-time price simulation
            # Use mock realistic prices if available, otherwise +/- 20% of cost basis
            mock_prices = {
                'AAPL': 185.50, 'NVDA': 650.25, 'MSFT': 420.80, 'GOOGL': 145.20,
                'META': 475.30, 'AMZN': 175.40, 'TSLA': 215.60, 'CMG': 3210.00,
                'HLT': 195.40, 'LOW': 198.50, 'HHH': 72.40, 'COST': 780.00,
                'JPM': 185.20, 'V': 285.50, 'UNH': 520.40, 'LLY': 850.20,
                'XOM': 105.30, 'WMT': 175.80, 'SPY': 495.50, 'QQQ': 430.20
            }

            import random
            if h.symbol in mock_prices:
                 # Add small random fluctuation (0.5%) to simulate live market
                current_price = mock_prices[h.symbol] * (1 + (random.random() - 0.5) * 0.005)
            else:
                current_price = h.avg_cost * (1 + (random.random() - 0.5) * 0.2)
            
            current_value = h.shares * current_price
            total_current_value += current_value
            
            holdings_list.append({
                "id": str(h.id),
                "symbol": h.symbol,
                "shares": h.shares,
                "avg_cost": h.avg_cost,
                "cost_basis": cost_basis,
                "current_price": round(current_price, 2),
                "current_value": round(current_value, 2),
                "pnl": round(current_value - cost_basis, 2),
                "pnl_pct": round((current_value - cost_basis) / cost_basis * 100, 2) if cost_basis > 0 else 0,
                "source": h.source,
            })
        
        total_pnl = total_current_value - total_cost_basis
        total_pnl_pct = (total_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0
        
        return jsonify({
            "holdings": holdings_list,
            "summary": {
                "total_holdings": len(holdings_list),
                "total_cost_basis": round(total_cost_basis, 2),
                "total_current_value": round(total_current_value, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl_pct, 2),
            }
        })


@portfolio_bp.route("/add", methods=["POST"])
@require_auth
def add_holding():
    """Add a new holding to portfolio."""
    user = g.current_user
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required = ["symbol", "shares", "avg_cost"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
    
    symbol = data["symbol"].upper().strip()
    shares = float(data["shares"])
    avg_cost = float(data["avg_cost"])
    
    if shares <= 0:
        return jsonify({"error": "Shares must be greater than 0"}), 400
    if avg_cost <= 0:
        return jsonify({"error": "Average cost must be greater than 0"}), 400
    
    db = get_database()
    with db.get_session() as session:
        # Check if holding already exists
        existing = session.query(PortfolioHolding).filter(
            and_(
                PortfolioHolding.user_id == user.id,
                PortfolioHolding.symbol == symbol,
                PortfolioHolding.is_active == True
            )
        ).first()
        
        if existing:
            # Update existing holding (average in)
            total_shares = existing.shares + shares
            total_cost = (existing.shares * existing.avg_cost) + (shares * avg_cost)
            new_avg_cost = total_cost / total_shares
            
            existing.shares = total_shares
            existing.avg_cost = new_avg_cost
            existing.updated_at = datetime.utcnow()
            session.commit()
            
            return jsonify({
                "message": f"Updated {symbol} position",
                "holding": {
                    "id": str(existing.id),
                    "symbol": existing.symbol,
                    "shares": existing.shares,
                    "avg_cost": existing.avg_cost,
                }
            })
        
        # Create new holding
        purchase_date = None
        if data.get("purchase_date"):
            try:
                purchase_date = datetime.fromisoformat(data["purchase_date"].replace('Z', '+00:00'))
            except:
                pass
        
        holding = PortfolioHolding(
            user_id=user.id,
            symbol=symbol,
            shares=shares,
            avg_cost=avg_cost,
            purchase_date=purchase_date,
            source=data.get("source", "manual"),
            notes=data.get("notes"),
        )
        
        session.add(holding)
        session.commit()
        
        logger.info(f"Added holding {symbol} for user {user.id}")
        
        return jsonify({
            "message": f"Added {symbol} to portfolio",
            "holding": {
                "id": str(holding.id),
                "symbol": holding.symbol,
                "shares": holding.shares,
                "avg_cost": holding.avg_cost,
            }
        }), 201


@portfolio_bp.route("/<holding_id>", methods=["PUT"])
@require_auth
def update_holding(holding_id):
    """Update an existing holding."""
    user = g.current_user
    data = request.get_json()
    
    db = get_database()
    with db.get_session() as session:
        holding = session.query(PortfolioHolding).filter(
            and_(
                PortfolioHolding.id == uuid.UUID(holding_id),
                PortfolioHolding.user_id == user.id,
                PortfolioHolding.is_active == True
            )
        ).first()
        
        if not holding:
            return jsonify({"error": "Holding not found"}), 404
        
        if "shares" in data:
            holding.shares = float(data["shares"])
        if "avg_cost" in data:
            holding.avg_cost = float(data["avg_cost"])
        if "notes" in data:
            holding.notes = data["notes"]
        
        holding.updated_at = datetime.utcnow()
        session.commit()
        
        return jsonify({
            "message": "Holding updated",
            "holding": {
                "id": str(holding.id),
                "symbol": holding.symbol,
                "shares": holding.shares,
                "avg_cost": holding.avg_cost,
            }
        })


@portfolio_bp.route("/<holding_id>", methods=["DELETE"])
@require_auth
def delete_holding(holding_id):
    """Delete (soft delete) a holding."""
    user = g.current_user
    
    db = get_database()
    with db.get_session() as session:
        holding = session.query(PortfolioHolding).filter(
            and_(
                PortfolioHolding.id == uuid.UUID(holding_id),
                PortfolioHolding.user_id == user.id,
                PortfolioHolding.is_active == True
            )
        ).first()
        
        if not holding:
            return jsonify({"error": "Holding not found"}), 404
        
        holding.is_active = False
        holding.updated_at = datetime.utcnow()
        session.commit()
        
        return jsonify({"message": f"Removed {holding.symbol} from portfolio"})


@portfolio_bp.route("/import", methods=["POST"])
@require_auth
def import_csv():
    """Import holdings from CSV file."""
    user = g.current_user
    
    if "file" not in request.files:
        # Try to parse CSV from request body
        data = request.get_json()
        if not data or "csv_content" not in data:
            return jsonify({"error": "No file or csv_content provided"}), 400
        csv_content = data["csv_content"]
    else:
        file = request.files["file"]
        csv_content = file.read().decode("utf-8")
    
    try:
        reader = csv.DictReader(StringIO(csv_content))
        imported = []
        errors = []
        
        db = get_database()
        with db.get_session() as session:
            for i, row in enumerate(reader, 1):
                try:
                    symbol = row.get("Symbol", row.get("symbol", "")).upper().strip()
                    shares = float(row.get("Shares", row.get("shares", 0)))
                    avg_cost = float(row.get("Avg Cost", row.get("avg_cost", row.get("Cost", 0))))
                    
                    if not symbol or shares <= 0 or avg_cost <= 0:
                        errors.append(f"Row {i}: Invalid data")
                        continue
                    
                    # Check for existing
                    existing = session.query(PortfolioHolding).filter(
                        and_(
                            PortfolioHolding.user_id == user.id,
                            PortfolioHolding.symbol == symbol,
                            PortfolioHolding.is_active == True
                        )
                    ).first()
                    
                    if existing:
                        # Average in
                        total_shares = existing.shares + shares
                        total_cost = (existing.shares * existing.avg_cost) + (shares * avg_cost)
                        existing.shares = total_shares
                        existing.avg_cost = total_cost / total_shares
                        existing.updated_at = datetime.utcnow()
                        imported.append({"symbol": symbol, "action": "updated"})
                    else:
                        holding = PortfolioHolding(
                            user_id=user.id,
                            symbol=symbol,
                            shares=shares,
                            avg_cost=avg_cost,
                            source="csv",
                        )
                        session.add(holding)
                        imported.append({"symbol": symbol, "action": "added"})
                        
                except Exception as e:
                    errors.append(f"Row {i}: {str(e)}")
            
            session.commit()
        
        return jsonify({
            "message": f"Imported {len(imported)} holdings",
            "imported": imported,
            "errors": errors
        })
        
    except Exception as e:
        logger.error(f"CSV import error: {e}")
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400


@portfolio_bp.route("/signals", methods=["GET"])
@require_auth
def get_portfolio_signals():
    """Get AI signals for portfolio holdings with dynamic signal generation."""
    user = g.current_user
    
    db = get_database()
    with db.get_session() as session:
        from src.data.models import TradeSignal
        import random
        from datetime import timedelta
        
        holdings = session.query(PortfolioHolding).filter(
            and_(
                PortfolioHolding.user_id == user.id,
                PortfolioHolding.is_active == True
            )
        ).all()
        
        symbols = [h.symbol for h in holdings]
        
        # Get latest signals for each symbol from DB
        signals_by_symbol = {}
        for symbol in symbols:
            signal = session.query(TradeSignal).filter(
                TradeSignal.symbol == symbol
            ).order_by(TradeSignal.created_at.desc()).first()
            
            if signal:
                signals_by_symbol[symbol] = {
                    "signal_type": signal.signal_type,
                    "confluence_score": signal.confluence_score,
                    "technical_score": signal.technical_score,
                    "sentiment_score": signal.sentiment_score,
                    "risk_reward_ratio": signal.risk_reward_ratio,
                    "price_at_signal": signal.price_at_signal,
                    "created_at": signal.created_at.isoformat() if signal.created_at else None,
                    "rationale": signal.technical_rationale,
                }
        
        # Mock stock data for dynamic signal generation (same as signals.py)
        mock_stock_signals = {
            'AAPL': {'signal': 'STRONG_BUY', 'conf': 0.91, 'rr': 3.5, 'rationale': 'AI momentum, services growth'},
            'NVDA': {'signal': 'STRONG_BUY', 'conf': 0.93, 'rr': 4.0, 'rationale': 'AI demand, data center +400% YoY'},
            'MSFT': {'signal': 'BUY', 'conf': 0.85, 'rr': 2.8, 'rationale': 'Azure growth, Copilot monetization'},
            'GOOGL': {'signal': 'HOLD', 'conf': 0.70, 'rr': 2.0, 'rationale': 'Antitrust concerns, AI investments'},
            'META': {'signal': 'STRONG_BUY', 'conf': 0.89, 'rr': 3.3, 'rationale': 'Reels driving ad revenue'},
            'AMZN': {'signal': 'BUY', 'conf': 0.82, 'rr': 2.7, 'rationale': 'AWS recovery, margin improvement'},
            'TSLA': {'signal': 'SELL', 'conf': 0.65, 'rr': 1.5, 'rationale': 'EV competition, margin pressure'},
            'CMG': {'signal': 'BUY', 'conf': 0.88, 'rr': 3.2, 'rationale': 'Strong same-store sales growth'},
            'HLT': {'signal': 'HOLD', 'conf': 0.72, 'rr': 2.1, 'rationale': 'Travel normalized, leisure stable'},
            'LOW': {'signal': 'SELL', 'conf': 0.68, 'rr': 1.8, 'rationale': 'Housing slowdown, DIY spending down'},
            'HHH': {'signal': 'BUY', 'conf': 0.75, 'rr': 2.5, 'rationale': 'Land value unlocking'},
            'COST': {'signal': 'BUY', 'conf': 0.83, 'rr': 2.6, 'rationale': 'Membership growth strong'},
            'JPM': {'signal': 'BUY', 'conf': 0.78, 'rr': 2.4, 'rationale': 'Strong NII, credit quality holding'},
            'V': {'signal': 'BUY', 'conf': 0.80, 'rr': 2.6, 'rationale': 'Consumer spending resilient'},
            'UNH': {'signal': 'STRONG_BUY', 'conf': 0.86, 'rr': 3.0, 'rationale': 'Healthcare secular growth'},
            'LLY': {'signal': 'STRONG_BUY', 'conf': 0.92, 'rr': 3.8, 'rationale': 'GLP-1 drugs exploding'},
            'XOM': {'signal': 'HOLD', 'conf': 0.70, 'rr': 1.9, 'rationale': 'Oil range-bound'},
            'WMT': {'signal': 'BUY', 'conf': 0.80, 'rr': 2.5, 'rationale': 'E-commerce, ad revenue scaling'},
        }
        
        # Mock real-time market prices (approximate as of Jan 2026)
        mock_prices = {
            'AAPL': 185.50, 'NVDA': 650.25, 'MSFT': 420.80, 'GOOGL': 145.20,
            'META': 475.30, 'AMZN': 175.40, 'TSLA': 215.60, 'CMG': 3210.00,
            'HLT': 195.40, 'LOW': 198.50, 'HHH': 72.40, 'COST': 780.00,
            'JPM': 185.20, 'V': 285.50, 'UNH': 520.40, 'LLY': 850.20,
            'XOM': 105.30, 'WMT': 175.80, 'SPY': 495.50, 'QQQ': 430.20
        }

        # Build response with holdings + signals
        result = []
        for h in holdings:
            # P&L Calculation: Use mock real price if available, else vary from cost
            if h.symbol in mock_prices:
                # Add small random fluctuation to simulate live market
                current_price = mock_prices[h.symbol] * (1 + (random.random() - 0.5) * 0.01)
            else:
                current_price = h.avg_cost * (1 + (random.random() - 0.5) * 0.2)
            
            cost_basis = h.shares * h.avg_cost
            current_value = h.shares * current_price
            pnl = current_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            holding_data = {
                "id": str(h.id),
                "symbol": h.symbol,
                "shares": h.shares,
                "avg_cost": h.avg_cost,
                "cost_basis": cost_basis,
                "current_price": round(current_price, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            }
            
            # Priority 1: Use DB signal if available
            if h.symbol in signals_by_symbol:
                holding_data["signal"] = signals_by_symbol[h.symbol]
            # Priority 2: Use known mock signal data (with logic checks)
            elif h.symbol in mock_stock_signals:
                mock = mock_stock_signals[h.symbol]
                
                # Sanity Check: Don't show "BUY" if we are down >15% unless it's a deep value play
                # For HHH (Howard Hughes), if down significantly, justify it or change signal
                if h.symbol == 'HHH' and pnl_pct < -5:
                    final_signal = 'HOLD'
                    rationale = 'Long-term value play, but near-term pressure remains'
                    conf = 0.65
                else:
                    final_signal = mock['signal']
                    rationale = mock['rationale']
                    conf = mock['conf']

                holding_data["signal"] = {
                    "signal_type": final_signal,
                    "confluence_score": conf,
                    "risk_reward_ratio": mock['rr'],
                    "rationale": rationale,
                    "technical_score": conf - 0.03,
                    "sentiment_score": conf + 0.02,
                }
            # Priority 3: Generate dynamic signal based on P&L
            else:
                # Dynamic signal based on P&L and volatility
                if pnl_pct >= 10:
                    signal_type = "TAKE_PROFIT"
                    conf = 0.78
                    rationale = f"Position up {pnl_pct:.1f}%, consider taking profits"
                elif pnl_pct >= 5:
                    signal_type = "HOLD"
                    conf = 0.72
                    rationale = f"Position performing well (+{pnl_pct:.1f}%), let it run"
                elif pnl_pct >= -3:
                    signal_type = "HOLD"
                    conf = 0.65
                    rationale = "Position within normal range"
                elif pnl_pct >= -8:
                    signal_type = "REVIEW"
                    conf = 0.60
                    rationale = f"Position down {abs(pnl_pct):.1f}%, review thesis"
                else:
                    signal_type = "STOP_LOSS"
                    conf = 0.55
                    rationale = f"Position down {abs(pnl_pct):.1f}%, consider stop-loss"
                
                holding_data["signal"] = {
                    "signal_type": signal_type,
                    "confluence_score": conf,
                    "risk_reward_ratio": 2.0 if pnl_pct > 0 else 1.2,
                    "rationale": rationale,
                    "technical_score": conf - 0.05,
                    "sentiment_score": conf + 0.02,
                }
            
            result.append(holding_data)
        
        return jsonify({
            "holdings_with_signals": result,
            "count": len(result)
        })


@portfolio_bp.route("/famous-investors", methods=["GET"])
def get_famous_investors():
    """Get list of available famous investors to import."""
    from src.services.sec_portfolio_fetcher import get_sec_fetcher
    
    fetcher = get_sec_fetcher()
    investors = fetcher.get_famous_investors()
    
    result = []
    for inv_id, inv_data in investors.items():
        result.append({
            "id": inv_id,
            "name": inv_data["name"],
            "fund": inv_data["fund"],
            "description": inv_data["description"],
            "avatar": inv_data["avatar"],
        })
    
    return jsonify({
        "investors": result,
        "count": len(result)
    })


@portfolio_bp.route("/preview-investor/<investor_id>", methods=["GET"])
def preview_investor_portfolio(investor_id):
    """Preview a famous investor's holdings before importing."""
    from src.services.sec_portfolio_fetcher import get_sec_fetcher
    
    try:
        fetcher = get_sec_fetcher()
        holdings, filing = fetcher.import_investor_portfolio(investor_id)
        
        investor = fetcher.get_investor_info(investor_id)
        
        holdings_list = []
        total_value = 0
        for h in holdings:
            total_value += h.value_usd
            holdings_list.append({
                "symbol": h.symbol,
                "name": h.name,
                "shares": h.shares,
                "value_usd": h.value_usd,
                "pct_portfolio": h.pct_portfolio,
            })
        
        return jsonify({
            "investor": {
                "id": investor_id,
                "name": investor["name"],
                "fund": investor["fund"],
            },
            "filing": {
                "date": filing.filing_date,
                "accession_number": filing.accession_number,
                "form_type": filing.form_type,
            },
            "holdings": holdings_list,
            "summary": {
                "total_holdings": len(holdings_list),
                "total_value_usd": total_value,
            }
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to preview investor portfolio: {e}")
        return jsonify({"error": "Failed to fetch portfolio"}), 500


@portfolio_bp.route("/import-investor", methods=["POST"])
@require_auth
def import_famous_investor():
    """Import a famous investor's portfolio holdings."""
    user = g.current_user
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    investor_id = data.get("investor_id")
    sec_url = data.get("sec_url")
    
    if not investor_id and not sec_url:
        return jsonify({"error": "Either investor_id or sec_url required"}), 400
    
    from src.services.sec_portfolio_fetcher import get_sec_fetcher
    
    try:
        fetcher = get_sec_fetcher()
        
        if sec_url:
            # Parse URL to get CIK
            parsed = fetcher.parse_sec_url(sec_url)
            if not parsed:
                return jsonify({"error": "Could not parse SEC URL"}), 400
            cik, accession = parsed
            # For custom URLs, fetch with CIK directly
            holdings = fetcher.fetch_13f_holdings(cik, accession or "")
            investor_name = f"SEC Filing ({cik})"
            source = f"sec:{cik}"
        else:
            # Import from known investor
            holdings, filing = fetcher.import_investor_portfolio(investor_id)
            investor = fetcher.get_investor_info(investor_id)
            investor_name = investor["name"]
            source = f"investor:{investor_id}"
        
        # Add holdings to user's portfolio
        db = get_database()
        imported = []
        
        with db.get_session() as session:
            for h in holdings:
                # Calculate average cost from total value
                avg_cost = h.value_usd / h.shares if h.shares > 0 else 0
                
                # Check if holding already exists
                existing = session.query(PortfolioHolding).filter(
                    and_(
                        PortfolioHolding.user_id == user.id,
                        PortfolioHolding.symbol == h.symbol,
                        PortfolioHolding.is_active == True
                    )
                ).first()
                
                if existing:
                    # Update existing holding
                    existing.shares = h.shares
                    existing.avg_cost = avg_cost
                    existing.source = source
                    existing.updated_at = datetime.utcnow()
                    imported.append({
                        "symbol": h.symbol,
                        "action": "updated",
                        "shares": h.shares,
                    })
                else:
                    # Create new holding
                    holding = PortfolioHolding(
                        user_id=user.id,
                        symbol=h.symbol,
                        shares=h.shares,
                        avg_cost=avg_cost,
                        source=source,
                        notes=f"Imported from {investor_name}",
                    )
                    session.add(holding)
                    imported.append({
                        "symbol": h.symbol,
                        "action": "added",
                        "shares": h.shares,
                    })
            
            session.commit()
        
        logger.info(f"Imported {len(imported)} holdings from {investor_name} for user {user.id}")
        
        return jsonify({
            "message": f"Successfully imported {len(imported)} holdings from {investor_name}",
            "imported": imported,
            "source": source,
            "count": len(imported),
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to import investor portfolio: {e}")
        return jsonify({"error": f"Import failed: {str(e)}"}), 500


# ============ TRANSACTION HISTORY ENDPOINTS ============

@portfolio_bp.route("/transactions", methods=["GET"])
@require_auth
def get_transactions():
    """Get all transactions for the current user."""
    user = g.current_user
    symbol = request.args.get("symbol")  # Optional filter
    
    db = get_database()
    with db.get_session() as session:
        query = session.query(PortfolioTransaction).filter(
            PortfolioTransaction.user_id == user.id
        )
        
        if symbol:
            query = query.filter(PortfolioTransaction.symbol == symbol.upper())
        
        transactions = query.order_by(PortfolioTransaction.transaction_date.desc()).all()
        
        result = []
        for t in transactions:
            result.append({
                "id": str(t.id),
                "symbol": t.symbol,
                "transaction_type": t.transaction_type,
                "shares": t.shares,
                "price": t.price,
                "total_value": t.total_value or (t.shares * t.price),
                "transaction_date": t.transaction_date.isoformat() if t.transaction_date else None,
                "source": t.source,
                "notes": t.notes,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            })
        
        return jsonify({
            "transactions": result,
            "count": len(result)
        })


@portfolio_bp.route("/transactions/import", methods=["POST"])
@require_auth
def import_transactions():
    """
    Import transaction history from CSV.
    
    Expected CSV format:
    Symbol,Type,Date,Shares,Price,Notes
    AAPL,BUY,2024-01-15,100,185.50,Initial purchase
    AAPL,SELL,2024-06-20,50,195.25,Taking profits
    NVDA,BUY,2024-02-01,25,650.00,
    """
    user = g.current_user
    
    if "file" not in request.files:
        data = request.get_json()
        if not data or "csv_content" not in data:
            return jsonify({"error": "No file or csv_content provided"}), 400
        csv_content = data["csv_content"]
    else:
        file = request.files["file"]
        csv_content = file.read().decode("utf-8")
    
    try:
        reader = csv.DictReader(StringIO(csv_content))
        imported = []
        errors = []
        
        db = get_database()
        with db.get_session() as session:
            for i, row in enumerate(reader, 1):
                try:
                    # Parse fields (support multiple column name formats)
                    symbol = row.get("Symbol", row.get("symbol", row.get("Ticker", ""))).upper().strip()
                    
                    trans_type = row.get("Type", row.get("type", row.get("Transaction", row.get("Action", "")))).upper().strip()
                    if trans_type in ["B", "BUY", "BOUGHT", "PURCHASE"]:
                        trans_type = "buy"
                    elif trans_type in ["S", "SELL", "SOLD", "SALE"]:
                        trans_type = "sell"
                    elif trans_type in ["D", "DIV", "DIVIDEND"]:
                        trans_type = "dividend"
                    else:
                        trans_type = trans_type.lower()
                    
                    shares = float(row.get("Shares", row.get("shares", row.get("Quantity", 0))))
                    price = float(row.get("Price", row.get("price", row.get("Cost", row.get("Amount", 0)))))
                    
                    # Parse date
                    date_str = row.get("Date", row.get("date", row.get("Transaction Date", "")))
                    trans_date = None
                    if date_str:
                        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
                            try:
                                trans_date = datetime.strptime(date_str.strip(), fmt)
                                break
                            except:
                                continue
                    
                    if not symbol or shares <= 0 or price <= 0:
                        errors.append(f"Row {i}: Invalid data (symbol={symbol}, shares={shares}, price={price})")
                        continue
                    
                    if trans_type not in ["buy", "sell", "dividend"]:
                        errors.append(f"Row {i}: Invalid transaction type '{trans_type}'")
                        continue
                    
                    # Create transaction
                    transaction = PortfolioTransaction(
                        user_id=user.id,
                        symbol=symbol,
                        transaction_type=trans_type,
                        shares=shares,
                        price=price,
                        total_value=shares * price,
                        transaction_date=trans_date or datetime.utcnow(),
                        source="csv",
                        notes=row.get("Notes", row.get("notes", "")),
                    )
                    
                    session.add(transaction)
                    imported.append({
                        "symbol": symbol,
                        "type": trans_type,
                        "shares": shares,
                        "price": price,
                        "date": trans_date.isoformat() if trans_date else None,
                    })
                    
                except Exception as e:
                    errors.append(f"Row {i}: {str(e)}")
            
            session.commit()
        
        # After importing transactions, recalculate holdings
        if imported:
            _recalculate_holdings(user.id)
        
        return jsonify({
            "message": f"Imported {len(imported)} transactions",
            "imported": imported,
            "errors": errors
        })
        
    except Exception as e:
        logger.error(f"Transaction import error: {e}")
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400


@portfolio_bp.route("/transactions/add", methods=["POST"])
@require_auth
def add_transaction():
    """Add a single transaction."""
    user = g.current_user
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required = ["symbol", "transaction_type", "shares", "price"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
    
    symbol = data["symbol"].upper().strip()
    trans_type = data["transaction_type"].lower()
    shares = float(data["shares"])
    price = float(data["price"])
    
    if trans_type not in ["buy", "sell", "dividend"]:
        return jsonify({"error": "Invalid transaction_type. Must be: buy, sell, dividend"}), 400
    
    # Parse date
    trans_date = datetime.utcnow()
    if data.get("transaction_date"):
        try:
            trans_date = datetime.fromisoformat(data["transaction_date"].replace('Z', '+00:00'))
        except:
            pass
    
    db = get_database()
    with db.get_session() as session:
        transaction = PortfolioTransaction(
            user_id=user.id,
            symbol=symbol,
            transaction_type=trans_type,
            shares=shares,
            price=price,
            total_value=shares * price,
            transaction_date=trans_date,
            source="manual",
            notes=data.get("notes", ""),
        )
        
        session.add(transaction)
        session.commit()
        
        logger.info(f"Added transaction {trans_type} {symbol} for user {user.id}")
    
    # Recalculate holdings
    _recalculate_holdings(user.id)
    
    return jsonify({
        "message": f"Added {trans_type} transaction for {symbol}",
        "transaction": {
            "id": str(transaction.id),
            "symbol": symbol,
            "transaction_type": trans_type,
            "shares": shares,
            "price": price,
            "date": trans_date.isoformat(),
        }
    }), 201


@portfolio_bp.route("/transactions/timeline", methods=["GET"])
@require_auth
def get_transaction_timeline():
    """
    Get transaction timeline with calculated P&L for each transaction.
    Returns transactions grouped by symbol with running totals.
    """
    user = g.current_user
    
    db = get_database()
    with db.get_session() as session:
        transactions = session.query(PortfolioTransaction).filter(
            PortfolioTransaction.user_id == user.id
        ).order_by(PortfolioTransaction.transaction_date.asc()).all()
        
        # Calculate running totals and P&L for each symbol
        positions = {}  # symbol -> {shares, cost_basis, realized_pnl}
        timeline = []
        
        for t in transactions:
            if t.symbol not in positions:
                positions[t.symbol] = {"shares": 0, "cost_basis": 0, "realized_pnl": 0}
            
            pos = positions[t.symbol]
            pnl = None
            
            if t.transaction_type == "buy":
                # Add to position
                pos["cost_basis"] += t.shares * t.price
                pos["shares"] += t.shares
                
            elif t.transaction_type == "sell":
                # Calculate P&L on sell
                if pos["shares"] > 0:
                    avg_cost = pos["cost_basis"] / pos["shares"]
                    pnl = (t.price - avg_cost) * t.shares
                    pos["realized_pnl"] += pnl
                    
                    # Reduce position
                    pos["cost_basis"] -= avg_cost * t.shares
                    pos["shares"] -= t.shares
                    
            elif t.transaction_type == "dividend":
                pnl = t.shares * t.price  # For dividends, this is just income
                pos["realized_pnl"] += pnl
            
            timeline.append({
                "id": str(t.id),
                "symbol": t.symbol,
                "transaction_type": t.transaction_type,
                "shares": t.shares,
                "price": t.price,
                "total_value": t.total_value or (t.shares * t.price),
                "transaction_date": t.transaction_date.isoformat() if t.transaction_date else None,
                "pnl": round(pnl, 2) if pnl is not None else None,
                "running_shares": round(pos["shares"], 4),
                "running_cost_basis": round(pos["cost_basis"], 2),
                "notes": t.notes,
            })
        
        # Summary by symbol
        symbol_summary = {}
        for symbol, pos in positions.items():
            symbol_summary[symbol] = {
                "current_shares": round(pos["shares"], 4),
                "total_cost_basis": round(pos["cost_basis"], 2),
                "realized_pnl": round(pos["realized_pnl"], 2),
                "avg_cost": round(pos["cost_basis"] / pos["shares"], 2) if pos["shares"] > 0 else 0,
            }
        
        return jsonify({
            "timeline": timeline,
            "summary_by_symbol": symbol_summary,
            "total_realized_pnl": round(sum(p["realized_pnl"] for p in positions.values()), 2),
            "count": len(timeline)
        })


def _recalculate_holdings(user_id):
    """
    Recalculate holdings from transaction history.
    This ensures holdings always match the transaction log.
    """
    db = get_database()
    with db.get_session() as session:
        # Get all transactions for user
        transactions = session.query(PortfolioTransaction).filter(
            PortfolioTransaction.user_id == user_id
        ).order_by(PortfolioTransaction.transaction_date.asc()).all()
        
        # Calculate positions from transactions
        positions = {}  # symbol -> {shares, cost_basis}
        
        for t in transactions:
            if t.symbol not in positions:
                positions[t.symbol] = {"shares": 0, "cost_basis": 0}
            
            pos = positions[t.symbol]
            
            if t.transaction_type == "buy":
                pos["cost_basis"] += t.shares * t.price
                pos["shares"] += t.shares
            elif t.transaction_type == "sell":
                if pos["shares"] > 0:
                    avg_cost = pos["cost_basis"] / pos["shares"]
                    pos["cost_basis"] -= avg_cost * t.shares
                    pos["shares"] -= t.shares
        
        # Update holdings to match positions
        for symbol, pos in positions.items():
            if pos["shares"] > 0:
                avg_cost = pos["cost_basis"] / pos["shares"]
                
                # Find or create holding
                existing = session.query(PortfolioHolding).filter(
                    and_(
                        PortfolioHolding.user_id == user_id,
                        PortfolioHolding.symbol == symbol,
                        PortfolioHolding.is_active == True
                    )
                ).first()
                
                if existing:
                    existing.shares = pos["shares"]
                    existing.avg_cost = avg_cost
                    existing.updated_at = datetime.utcnow()
                else:
                    holding = PortfolioHolding(
                        user_id=user_id,
                        symbol=symbol,
                        shares=pos["shares"],
                        avg_cost=avg_cost,
                        source="transactions",
                    )
                    session.add(holding)
            else:
                # Position is 0, soft delete the holding
                existing = session.query(PortfolioHolding).filter(
                    and_(
                        PortfolioHolding.user_id == user_id,
                        PortfolioHolding.symbol == symbol,
                        PortfolioHolding.is_active == True
                    )
                ).first()
                
                if existing:
                    existing.is_active = False
                    existing.updated_at = datetime.utcnow()
        
        session.commit()
        logger.info(f"Recalculated holdings from transactions for user {user_id}")


@portfolio_bp.route("/lessons", methods=["GET"])
@require_auth
def get_trading_lessons():
    """
    Get AI-powered lessons learned from trading history.
    
    Uses LLM to analyze transaction patterns and provide personalized insights.
    """
    user = g.current_user
    
    db = get_database()
    with db.get_session() as session:
        transactions = session.query(PortfolioTransaction).filter(
            PortfolioTransaction.user_id == user.id
        ).order_by(PortfolioTransaction.transaction_date.asc()).all()
        
        if not transactions:
            return jsonify({
                "lessons": "# 📚 No Trading History Yet\n\nImport your transaction history to get personalized trading lessons!\n\nGo to **My Portfolio** → **Import Transactions** to get started.",
                "has_data": False
            })
        
        # Calculate running totals and P&L for each symbol
        positions = {}
        timeline = []
        
        for t in transactions:
            if t.symbol not in positions:
                positions[t.symbol] = {"shares": 0, "cost_basis": 0, "realized_pnl": 0}
            
            pos = positions[t.symbol]
            pnl = None
            
            if t.transaction_type == "buy":
                pos["cost_basis"] += t.shares * t.price
                pos["shares"] += t.shares
                
            elif t.transaction_type == "sell":
                if pos["shares"] > 0:
                    avg_cost = pos["cost_basis"] / pos["shares"]
                    pnl = (t.price - avg_cost) * t.shares
                    pos["realized_pnl"] += pnl
                    pos["cost_basis"] -= avg_cost * t.shares
                    pos["shares"] -= t.shares
                    
            elif t.transaction_type == "dividend":
                pnl = t.shares * t.price
                pos["realized_pnl"] += pnl
            
            timeline.append({
                "symbol": t.symbol,
                "transaction_type": t.transaction_type,
                "shares": t.shares,
                "price": t.price,
                "transaction_date": t.transaction_date.isoformat() if t.transaction_date else None,
                "pnl": round(pnl, 2) if pnl is not None else None,
            })
        
        # Summary by symbol
        summary_by_symbol = {}
        for symbol, pos in positions.items():
            summary_by_symbol[symbol] = {
                "current_shares": round(pos["shares"], 4),
                "total_cost_basis": round(pos["cost_basis"], 2),
                "realized_pnl": round(pos["realized_pnl"], 2),
                "avg_cost": round(pos["cost_basis"] / pos["shares"], 2) if pos["shares"] > 0 else 0,
            }
        
        total_realized_pnl = round(sum(p["realized_pnl"] for p in positions.values()), 2)
    
    # Get LLM analysis
    try:
        from src.analytics.llm_analysis import get_rag_engine
        engine = get_rag_engine()
        lessons = engine.analyze_trading_history(
            transactions=timeline,
            summary_by_symbol=summary_by_symbol,
            total_realized_pnl=total_realized_pnl
        )
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        lessons = f"""# 📚 Trading Summary

## Overview
- Total Transactions: {len(timeline)}
- Symbols Traded: {len(summary_by_symbol)}
- Total Realized P&L: ${total_realized_pnl:,.2f}

*AI-powered analysis unavailable. Check OpenAI API configuration.*
"""
    
    return jsonify({
        "lessons": lessons,
        "has_data": True,
        "transaction_count": len(timeline),
        "symbol_count": len(summary_by_symbol),
        "total_realized_pnl": total_realized_pnl,
    })
