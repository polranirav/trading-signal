# Trading Signals Pro

**AI-Powered Trading Signal Generation Platform**

A professional-grade trading signals system combining technical analysis, FinBERT sentiment analysis, and machine learning for high-conviction trading decisions.

---

## 🚀 Quick Start

### 1. Start Infrastructure (Docker)
```bash
docker compose up -d db redis
```

### 2. Initialize Database
```bash
source venv/bin/activate
pip install -r requirements.txt  # or: pip install -e .
python scripts/init_db.py --seed
```

### 3. Start Dashboard
```bash
python src/web/app.py
```
Open http://localhost:8050

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Trading Signals Pro                         │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│  Technical  │  Sentiment  │     ML      │    Risk     │  LLM   │
│  Analysis   │  (FinBERT)  │  (Ensemble) │  (VaR/CVaR) │  (RAG) │
├─────────────┴─────────────┴─────────────┴─────────────┴────────┤
│                    Confluence Engine                            │
│         (Weighted combination of all signal sources)           │
├─────────────────────────────────────────────────────────────────┤
│                    Flask API + Dash Dashboard                   │
├────────────────┬────────────────┬────────────────┬─────────────┤
│   PostgreSQL   │     Redis      │     Celery     │  React UI   │
│  (TimescaleDB) │    (Cache)     │    (Tasks)     │  (Optional) │
└────────────────┴────────────────┴────────────────┴─────────────┘
```

---

## 🔑 Key Features

| Feature | Description |
|---------|-------------|
| **Technical Analysis** | RSI, MACD, Bollinger Bands, SMA, ATR, OBV, MFI |
| **FinBERT Sentiment** | Time-weighted sentiment with Days 6-30 peak window |
| **Walk-Forward Backtesting** | López de Prado methodology to prevent overfitting |
| **Monte Carlo VaR** | Fat-tail risk management with t-distribution |
| **Confluence Engine** | 4-layer signal combination (Tech 40%, Sent 35%, ML 15%, Risk 10%) |
| **Real-time Dashboard** | Dash/Plotly interactive UI with live updates |
| **REST API** | Full API with authentication and rate limiting |

---

## 🛠️ Development

### Prerequisites
- Python 3.10+
- PostgreSQL 15+ (or Docker)
- Redis 7+ (or Docker)

### Full Stack (Docker Compose)
```bash
# Start everything
docker compose up -d

# View logs
docker compose logs -f dashboard

# Stop
docker compose down
```

### Services
| Service | Port | Description |
|---------|------|-------------|
| Dashboard | 8050 | Main Dash/Plotly UI |
| Frontend | 3000 | React admin UI |
| Flower | 5555 | Celery monitoring |
| Grafana | 3000 | Metrics visualization |
| Prometheus | 9090 | Metrics collection |

---

## 📁 Project Structure

```
trading-signals/
├── src/
│   ├── analytics/       # Core analysis engines
│   │   ├── technical.py     # Technical indicators
│   │   ├── sentiment.py     # FinBERT sentiment
│   │   ├── confluence.py    # Signal combination
│   │   ├── risk.py          # VaR/CVaR calculations
│   │   └── backtesting.py   # Walk-forward testing
│   ├── api/             # Flask REST API
│   ├── auth/            # Authentication models
│   ├── data/            # Database models & persistence
│   ├── tasks/           # Celery background tasks
│   └── web/             # Dash dashboard
├── frontend/            # React admin UI (optional)
├── scripts/             # Utility scripts
│   └── init_db.py           # Database initialization
├── tests/               # Unit & integration tests
└── docker-compose.yml   # Full stack deployment
```

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/signals` | GET | Get trading signals |
| `/api/v1/signals/{id}` | GET | Get signal details |
| `/api/v1/auth/login` | POST | User login |
| `/api/v1/auth/register` | POST | User registration |
| `/api/v1/account/me` | GET | Current user info |

See full API docs at http://localhost:8050/api/v1/docs

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and configure:

```env
# Required
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/trading_signals
REDIS_URL=redis://localhost:6379/0

# API Keys (optional but recommended)
ALPHA_VANTAGE_KEY=your_key
OPENAI_API_KEY=your_key
```

---

## 📊 Research Foundation

This system implements findings from:
- **FinBERT** (Huang et al., 2022) - Financial sentiment with 92.1% accuracy
- **López de Prado** - Walk-forward testing to prevent overfitting
- **Monte Carlo VaR** - Fat-tail risk management
- **Temporal Fusion Transformer** - Interpretable time-series forecasting

---

## License

MIT License - See LICENSE file for details.
