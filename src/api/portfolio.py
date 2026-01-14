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
            
            # TODO: Integrate real-time price API (for now, mock with slight variation)
            # Mock current price as +/- 10% of cost basis
            import random
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
    """Get AI signals for portfolio holdings."""
    user = g.current_user
    
    db = get_database()
    with db.get_session() as session:
        from src.data.models import TradeSignal
        
        holdings = session.query(PortfolioHolding).filter(
            and_(
                PortfolioHolding.user_id == user.id,
                PortfolioHolding.is_active == True
            )
        ).all()
        
        symbols = [h.symbol for h in holdings]
        
        # Get latest signals for each symbol
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
                }
        
        # Build response with holdings + signals
        result = []
        for h in holdings:
            holding_data = {
                "id": str(h.id),
                "symbol": h.symbol,
                "shares": h.shares,
                "avg_cost": h.avg_cost,
                "cost_basis": h.shares * h.avg_cost,
            }
            
            if h.symbol in signals_by_symbol:
                holding_data["signal"] = signals_by_symbol[h.symbol]
            else:
                # Default signal if none exists
                holding_data["signal"] = {
                    "signal_type": "HOLD",
                    "confluence_score": 0.5,
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

