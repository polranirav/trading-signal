# Dynamic Predictions & Signal Enhancement

## Summary
Fixed the prediction system to be **truly dynamic** based on actual signal data. The frontend was previously showing hardcoded predictions that never changed regardless of which stock was selected.

## Problems Fixed

### 1. ❌ Predictions Were STATIC (CRITICAL)
**Before**: Predictions always showed the same values:
```tsx
{ period: '+1H', direction: 'UP', confidence: 64 },
{ period: '+2H', direction: 'UP', confidence: 58 },
{ period: '+3H', direction: 'UP', confidence: 55 },
{ period: '+4H', direction: 'DOWN', confidence: 52 },
{ period: '+5H', direction: 'UP', confidence: 61 },
```

**After**: Predictions are generated dynamically using:
- Signal's `confluence_score` → Base confidence
- Signal's `signal_type` → Direction bias (BUY=UP, SELL=DOWN)
- Signal's `technical_score` → Fine-tune probabilities
- Signal's `sentiment_score` → Confidence boost/penalty
- Symbol hash → Consistent per-stock variation

### 2. ❌ Signal Limit Was 50 (Now 500)
Increased the maximum signal limit from 100 to 500 to support larger portfolios.

### 3. ❌ Only 20 Mock Stocks (Now 60+)
Expanded mock signal data to include 60+ stocks across all sectors:
- **Technology**: AAPL, NVDA, MSFT, GOOGL, META, AMZN, TSLA, AMD, INTC, CRM, ORCL, ADBE
- **Consumer Discretionary**: CMG, NFLX, DIS, SBUX, MCD, NKE, TGT, COST
- **Financials**: JPM, V, MA, BAC, GS, BRK.B, SCHW
- **Healthcare**: UNH, JNJ, LLY, PFE, ABBV, MRK
- **Industrials**: CAT, UPS, HLT, LOW, HHH, HD, DE, RTX
- **Energy**: XOM, CVX, NEE
- **Consumer Staples**: PG, KO, PEP, WMT
- **Communication**: T, VZ, TMUS
- **Real Estate**: AMT, PLD, EQIX
- **ETFs**: SPY, QQQ, IWM

## How Dynamic Predictions Work Now

### Direction Generation
```typescript
if (isBullish) {
    // 70-85% chance of UP for BUY signals
    return pseudoRandom < (0.70 + techScore * 0.15) ? 'UP' : 'DOWN'
} else if (isBearish) {
    // 65-80% chance of DOWN for SELL signals 
    return pseudoRandom < (0.65 + (1 - techScore) * 0.15) ? 'DOWN' : 'UP'
} else {
    // 50/50 for HOLD signals
    return pseudoRandom < 0.5 ? 'UP' : 'DOWN'
}
```

### Confidence Generation
```typescript
// Base from confluence score (0.5-0.95 → 50-95%)
let conf = baseConfidence + variation + timeframeDecay[index]

// Sentiment boost/penalty
if (sentScore > 0.7) conf += 5
if (sentScore < 0.3) conf -= 5

return Math.max(45, Math.min(95, Math.round(conf)))
```

## Files Changed

### Frontend
- `frontend/src/pages/dashboard/ChartsPage.tsx`
  - Dynamic predictions based on `currentSignal` data
  - Increased signal fetch limit to 100
  - Dynamic live data (price, change, volume) based on signal
  - AI Reasoning includes actual `technical_rationale` and `sentiment_rationale`

### Backend
- `src/api/signals.py`
  - Increased max limit from 100 to 500
  - Expanded mock data to 60+ stocks
  - Better rationale text per stock

## Verification
1. Go to **Charts** page
2. Switch between different stocks (e.g., CMG → TSLA → NVDA)
3. Observe that:
   - **Predictions change** based on stock
   - **Recommendation changes** (BUY/SELL/HOLD)
   - **Confidence levels differ** per stock
   - **AI Reasoning** is stock-specific
   - **Direction distribution** matches signal type (BUY stocks show more UPs)

## Future Enhancements (Out of Scope)
For a production system, you would want:
1. **Real News API** - Financial news from Alpha Vantage, Polygon, or NewsAPI
2. **Weather API** - For shipping/agriculture stocks (OpenWeatherMap)
3. **Earnings Calendar** - Scheduled events affecting volatility
4. **Macroeconomic Data** - Fed decisions, CPI, jobs report
5. **Sector Correlation** - How industry events affect related stocks
6. **Real-time ML Predictions** - TFT/LSTM models running on live data
