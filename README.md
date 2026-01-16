<div align="center">

# ⚡ TradingPro

### AI-Powered Institutional-Grade Trading Signal Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A full-stack trading intelligence platform combining Technical Analysis, NLP Sentiment (FinBERT), Machine Learning, and Institutional Risk Management.**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Tech Stack](#-tech-stack) • [API](#-api-reference)

</div>

---

## 🌟 Features

<table>
<tr>
<td width="50%">

### 📊 Technical Analysis Engine
- **15+ Indicators:** RSI, MACD, Bollinger Bands, ATR, OBV, MFI, Ichimoku Cloud
- **Pattern Detection:** RSI Divergence, MACD Crossovers, Bollinger Squeezes
- **Multi-Timeframe:** Analyze 1m, 5m, 1h, 4h, 1D, 1W simultaneously

</td>
<td width="50%">

### 🧠 AI & Sentiment Intelligence
- **FinBERT NLP:** Financial sentiment analysis on 50k+ news sources
- **GPT-4 Integration:** LLM-powered market event reasoning
- **Transformer Models:** Deep learning for time-series forecasting

</td>
</tr>
<tr>
<td width="50%">

### 🛡️ Institutional Risk Management
- **Monte Carlo VaR:** 10,000 path simulations for risk assessment
- **Dynamic Stop-Loss:** ATR-based adaptive trailing stops
- **Kelly Criterion:** Optimal position sizing based on signal confidence

</td>
<td width="50%">

### 🔄 Confluence Engine
- **Multi-Signal Synthesis:** Weighted combination of all data sources
- **Confidence Scoring:** 0-100% probability-based trade signals
- **Walk-Forward Backtesting:** López de Prado methodology to prevent overfitting

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites
- **Docker & Docker Compose** (recommended)
- Python 3.10+ (for local development)
- Node.js 18+ (for React frontend)

### 1. Clone & Configure
```bash
git clone https://github.com/polranirav/trading-signal.git
cd trading-signal
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start with Docker (Recommended)
```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f dashboard
```

### 3. Access the Platform
| Service | URL | Description |
|---------|-----|-------------|
| **React Frontend** | http://localhost:3002 | Modern UI Dashboard |
| **Dash Analytics** | http://localhost:8050 | Advanced Charts & Analysis |
| **Flower** | http://localhost:5555 | Celery Task Monitor |
| **Grafana** | http://localhost:3000 | Metrics & Observability |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRADING PRO PLATFORM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │  TECHNICAL  │ │  SENTIMENT  │ │     ML      │ │    RISK     │       │
│  │   ENGINE    │ │   ENGINE    │ │   ENGINE    │ │   ENGINE    │       │
│  │ (TA-Lib)    │ │ (FinBERT)   │ │ (PyTorch)   │ │ (VaR/CVaR)  │       │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘       │
│         │               │               │               │               │
│         └───────────────┴───────┬───────┴───────────────┘               │
│                                 │                                       │
│                    ┌────────────▼────────────┐                          │
│                    │    CONFLUENCE ENGINE    │                          │
│                    │  (Signal Aggregation)   │                          │
│                    └────────────┬────────────┘                          │
│                                 │                                       │
│  ┌──────────────────────────────┼──────────────────────────────┐       │
│  │                              │                              │       │
│  ▼                              ▼                              ▼       │
│  ┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐       │
│  │  Flask API  │     │   Celery Workers    │     │  React UI   │       │
│  │  (REST)     │     │   (Async Tasks)     │     │  (MUI)      │       │
│  └─────────────┘     └─────────────────────┘     └─────────────┘       │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  PostgreSQL/TimescaleDB  │  Redis (Cache)  │  Prometheus  │  Grafana   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | React 18, Material-UI, TypeScript, Vite, Recharts |
| **Backend** | Python 3.10, Flask, Dash/Plotly, Celery |
| **AI/ML** | PyTorch, Scikit-Learn, FinBERT, LangChain, OpenAI |
| **Database** | PostgreSQL 15 + TimescaleDB (Hypertables) |
| **Cache** | Redis 7 |
| **DevOps** | Docker Compose, Prometheus, Grafana |

---

## 📁 Project Structure

```
trading-signals/
├── src/                      # Python Backend
│   ├── analytics/            # Core Analysis Engines
│   │   ├── technical.py      # TA-Lib Indicators
│   │   ├── sentiment.py      # FinBERT Sentiment
│   │   ├── confluence.py     # Signal Aggregation
│   │   ├── risk.py           # VaR/CVaR Calculations
│   │   └── backtesting.py    # Walk-Forward Testing
│   ├── api/                  # Flask REST API
│   ├── auth/                 # JWT Authentication
│   ├── data/                 # Database Models (SQLAlchemy)
│   ├── tasks/                # Celery Background Jobs
│   └── web/                  # Dash Dashboard
│
├── frontend/                 # React Application
│   ├── src/
│   │   ├── pages/            # Route Pages
│   │   ├── components/       # Reusable UI Components
│   │   └── services/         # API Client Layer
│   └── package.json
│
├── scripts/                  # Utility Scripts
├── tests/                    # Unit & Integration Tests
├── docker-compose.yml        # Multi-Container Orchestration
└── pyproject.toml            # Python Dependencies
```

---

## 📡 API Reference

### Authentication
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/register` | POST | Create new account |
| `/api/v1/auth/login` | POST | Get access token |
| `/api/v1/auth/me` | GET | Current user profile |

### Signals
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/signals` | GET | List all signals |
| `/api/v1/signals/{id}` | GET | Signal details |
| `/api/v1/signals/generate` | POST | Generate new signal |

### Portfolio
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/portfolio` | GET | User portfolio |
| `/api/v1/portfolio/import` | POST | Import holdings |

---

## ⚙️ Configuration

Create a `.env` file (copy from `.env.example`):

```env
# Database
DATABASE_URL=postgresql://postgres:postgres@db:5432/trading_signals

# Cache
REDIS_URL=redis://redis:6379/0

# API Keys (Required for full functionality)
ALPHA_VANTAGE_KEY=your_key_here
OPENAI_API_KEY=sk-...

# Feature Flags
ENABLE_LIVE_TRADING=false
ENABLE_NEWS_SENTIMENT=true
```

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

---

## � Research Foundation

This platform is built on peer-reviewed research and industry best practices:

| Concept | Source | Implementation |
|---------|--------|----------------|
| **FinBERT** | Huang et al. (2022) | Financial sentiment with 92.1% accuracy |
| **Walk-Forward Testing** | López de Prado | Prevents look-ahead bias and overfitting |
| **Monte Carlo VaR** | Industry Standard | Fat-tail risk management with t-distribution |
| **Temporal Fusion Transformer** | Google AI | Interpretable multi-horizon forecasting |
| **Kelly Criterion** | Bell Labs | Optimal bankroll / position sizing |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Nirav Polara](https://github.com/polranirav)**

</div>
