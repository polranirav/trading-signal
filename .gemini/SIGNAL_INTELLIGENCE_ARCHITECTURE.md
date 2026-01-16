# Signal Intelligence System - Complete Architecture

## Overview

The Signal Intelligence System is a comprehensive trading signal analysis platform that aggregates **360+ signals** from multiple data sources, processes them through tiered computation, and presents them in a sophisticated dashboard. This document describes the complete architecture and implementation.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SIGNAL INTELLIGENCE ENGINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│   │  Technical  │   │  Sentiment  │   │Fundamentals │   │    Macro    │        │
│   │  Provider   │   │  Provider   │   │  Provider   │   │  Provider   │        │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘        │
│          │                 │                 │                 │               │
│   ┌──────▼─────────────────▼─────────────────▼─────────────────▼──────┐        │
│   │                    Signal Aggregation Layer                       │        │
│   │         - Tiered Caching (Redis)                                  │        │
│   │         - Weighted Confluence Calculation                         │        │
│   │         - Real-time Event Generation                              │        │
│   └───────────────────────────────────────────────────────────────────┘        │
│                                     │                                          │
│                                     ▼                                          │
│   ┌───────────────────────────────────────────────────────────────────┐        │
│   │                  REST API Layer                                   │        │
│   │   GET /api/v1/signal-intelligence/{symbol}                        │        │
│   │   GET /api/v1/signal-intelligence/{symbol}/category/{category}    │        │
│   └───────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND DASHBOARD                                       │
│                                                                                 │
│   ┌─────────────────┐  ┌─────────────────────────────────────────────┐          │
│   │ Confluence Score│  │         Component Scores                   │          │
│   │  (Large Gauge)  │  │  Technical | Sentiment | Fundamentals | Macro         │
│   └─────────────────┘  └─────────────────────────────────────────────┘          │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐           │
│   │                    Signal Categories (Expandable)               │           │
│   │  ├─ Technical Analysis (15+ signals)                            │           │
│   │  ├─ Sentiment & News (12+ signals)                              │           │
│   │  ├─ Fundamentals (10+ signals)                                  │           │
│   │  ├─ Market Structure (8+ signals)                               │           │
│   │  ├─ Macroeconomics (9+ signals)                                 │           │
│   │  ├─ Correlations (7+ signals)                                   │           │
│   │  ├─ Market Regime (7+ signals)                                  │           │
│   │  └─ External & Tail Risk (10+ signals)                          │           │
│   └─────────────────────────────────────────────────────────────────┘           │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐           │
│   │                    Live Signal Feed                             │           │
│   │  Real-time events with POSITIVE/NEGATIVE impact indicators      │           │
│   └─────────────────────────────────────────────────────────────────┘           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Signal Categories & Sources

### 1. Technical Analysis (40+ signals)
- **Computed from price/volume data**
- Update Frequency: **Near Real-time (5 min)**

| Signal | Description | Source |
|--------|-------------|--------|
| RSI (14) | Relative Strength Index | Computed |
| MACD Crossover | Moving Average Convergence Divergence | Computed |
| Bollinger Bands | Price position relative to bands | Computed |
| SMA 50/200 Cross | Golden/Death cross indicator | Computed |
| ADX | Average Directional Index (trend strength) | Computed |
| Stochastic Oscillator | %K/%D momentum | Computed |
| Williams %R | Overbought/oversold | Computed |
| OBV | On-Balance Volume trend | Computed |
| Parabolic SAR | Stop and Reverse | Computed |
| ATR | Average True Range (volatility) | Computed |
| Ichimoku Cloud | Cloud position and signals | Computed |
| Fibonacci Levels | Support/resistance levels | Computed |
| Candlestick Patterns | Doji, Hammer, Engulfing, etc. | Computed |
| EMA 12/26 | Exponential Moving Average trend | Computed |
| ROC | Rate of Change | Computed |

### 2. Sentiment & News (25+ signals)
- **External APIs + NLP**
- Update Frequency: **Periodic (15-60 min)**

| Signal | Description | API Source |
|--------|-------------|------------|
| News Sentiment (FinBERT) | AI-powered news analysis | Alpha Vantage News |
| News Volume | Article count (7d) | NewsAPI / Finnhub |
| Social Media Buzz | Twitter/Reddit mentions | Finnhub Social |
| Analyst Ratings | Wall Street consensus | Finnhub |
| Insider Activity | Insider buying/selling | Finnhub Insider |
| Options Flow | Put/Call ratio, unusual activity | Computed |
| Short Interest | Short selling pressure | TBD |
| Reddit WSB Mentions | Retail trader interest | Finnhub |
| Earnings Sentiment | Earnings call tone | TBD |

### 3. Fundamentals (20+ signals)
- **Financial APIs**
- Update Frequency: **Daily**

| Signal | Description | API Source |
|--------|-------------|------------|
| P/E Ratio | Valuation relative to earnings | FMP / Alpha Vantage |
| EPS Growth | Earnings trajectory | FMP |
| Revenue Growth | Top-line momentum | FMP |
| Gross Margin | Profitability health | FMP |
| Free Cash Flow | Cash generation | FMP |
| Debt/Equity | Balance sheet leverage | FMP |
| ROE | Return on Equity | FMP / Alpha Vantage |
| Current Ratio | Liquidity indicator | FMP |
| DCF Score | Intrinsic value assessment | FMP |

### 4. Macroeconomics (15+ signals)
- **FRED API**
- Update Frequency: **Daily/Weekly**

| Signal | Description | API Source |
|--------|-------------|------------|
| Fed Funds Rate | Interest rate environment | FRED (FEDFUNDS) |
| CPI Inflation | Consumer price effects | FRED (CPIAUCSL) |
| GDP Growth | Economic expansion | FRED (GDP) |
| Unemployment | Labor market health | FRED (UNRATE) |
| Consumer Confidence | Spending outlook | FRED (UMCSENT) |
| Yield Curve | 10Y-2Y spread (recession indicator) | FRED (T10Y2Y) |
| VIX | Market fear gauge | FRED (VIXCLS) |
| Dollar Strength | DXY currency impact | TBD |

---

## API Endpoints

### Get Complete Signal Intelligence
```
GET /api/v1/signal-intelligence/{symbol}
```

**Query Parameters:**
- `categories` (optional): Comma-separated list (technical,sentiment,fundamentals,macro)
- `include_details` (optional): Include individual signals (default: true)

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "confluence_score": 0.7234,
    "signal_type": "BUY",
    "total_signals": 78,
    "categories": {
      "technical": {
        "avg_score": 0.68,
        "bullish_count": 10,
        "bearish_count": 4,
        "signals": [...]
      },
      "sentiment": {...},
      "fundamentals": {...},
      "macro": {...}
    },
    "component_scores": {
      "technical": 0.68,
      "sentiment": 0.55,
      "fundamentals": 0.62,
      "macro": 0.58
    },
    "live_feed": [...],
    "generated_at": "2024-01-15T10:30:00Z"
  }
}
```

### Get Specific Category
```
GET /api/v1/signal-intelligence/{symbol}/category/{category}
```

---

## Files Created/Modified

### Backend (Python/Flask)

| File | Description |
|------|-------------|
| `src/services/signal_intelligence.py` | Main Signal Intelligence Engine with TechnicalSignalProvider |
| `src/services/sentiment_provider.py` | Sentiment provider with NewsAPI, Finnhub, Alpha Vantage integration |
| `src/services/fundamentals_provider.py` | Fundamentals & Macro providers with FMP, FRED integration |
| `src/api/signal_intelligence.py` | REST API endpoints for Signal Intelligence |
| `src/api/routes.py` | Updated to register signal_intelligence blueprint |

### Frontend (React/TypeScript)

| File | Description |
|------|-------------|
| `frontend/src/pages/dashboard/SignalIntelligencePage.tsx` | Full Signal Intelligence dashboard |
| `frontend/src/services/api.ts` | Added getSignalIntelligence API methods |
| `frontend/src/App.tsx` | Added route for /dashboard/signals |
| `frontend/src/pages/dashboard/OverviewPage.tsx` | Added "Signals" button to table |

---

## Environment Variables Required

```bash
# Optional - APIs will use fallback mock data if not set
ALPHA_VANTAGE_KEY=your_key   # News sentiment, fundamentals
FINNHUB_API_KEY=your_key     # Social sentiment, analyst ratings
FMP_API_KEY=your_key         # Financial Modeling Prep (fundamentals)
FRED_API_KEY=your_key        # Federal Reserve Economic Data
NEWS_API_KEY=your_key        # NewsAPI.org
```

---

## Confluence Score Calculation

The overall confluence score is calculated as a weighted average:

```python
weights = {
    'technical': 0.35,      # 35% - Primary driver
    'sentiment': 0.25,      # 25% - Market mood
    'fundamentals': 0.20,   # 20% - Long-term value
    'macro': 0.10,          # 10% - Economic context
    'correlations': 0.05,   # 5%  - Market relationships
    'external': 0.05,       # 5%  - Risk factors
}

confluence = Σ (category_avg_score × weight) / Σ weights
```

**Signal Type Mapping:**
- `STRONG_BUY`: confluence >= 0.75
- `BUY`: confluence >= 0.60
- `HOLD`: 0.40 < confluence < 0.60
- `SELL`: confluence <= 0.40
- `STRONG_SELL`: confluence <= 0.25

---

## Caching Strategy

| Tier | Update Frequency | TTL | Use Case |
|------|-----------------|-----|----------|
| Real-time | < 1 min | 30s | Price, quotes |
| Near Real-time | 5 min | 5 min | Technical indicators |
| Periodic | 15-60 min | 15 min | News, sentiment |
| Daily | 24 hours | 24h | Fundamentals, macro |
| Weekly | 7 days | 7d | Regime detection |

---

## Future Enhancements

1. **WebSocket Real-time Updates**: Stream signal changes live
2. **Alert System**: Notify on significant signal changes
3. **Historical Signal Timeline**: Track signals over time
4. **ML Model Integration**: Use TFT/Transformer for predictions
5. **Sector-specific Adjustments**: Adjust weights by industry
6. **Portfolio-level Analysis**: Aggregate signals across holdings
7. **Backtesting Integration**: Test signal effectiveness historically

---

## Access Points

| Method | URL |
|--------|-----|
| Dashboard | http://localhost:3002/dashboard/signals?symbol=AAPL |
| API | http://localhost:5001/api/v1/signal-intelligence/AAPL |
| From Overview | Click "Signals" button on any stock |
