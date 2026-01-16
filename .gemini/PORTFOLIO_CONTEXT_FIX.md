# Portfolio Context Fix - Market Overview Integration

## Problem
When users imported stocks into "My Portfolio" (e.g., CMG, HLT, GOOGL, LOW, HHH), the "Market Overview" page was still showing market-wide signals (NVDA, AAPL, UNH) instead of filtering to their portfolio stocks.

## Root Cause
The `PortfolioContext` was calling a non-existent API endpoint:
- **Incorrect**: `/portfolio/holdings` (404 Not Found)
- **Correct**: `/portfolio/summary` (returns both holdings and summary)

This caused `hasPortfolio` to always be `false` and `portfolioSymbols` to be empty, making the filtering logic ineffective.

## Solution
Updated `frontend/src/context/PortfolioContext.tsx`:
1. Changed API call from `/portfolio/holdings` to `/portfolio/summary`
2. Properly extract and transform holdings data from the response
3. Map backend response fields to frontend interface:
   - `h.pnl_pct` → `pnl_percent`
   - `summary.total_current_value` → `total_value`
   - etc.

## Expected Behavior After Fix

### When User HAS Portfolio (e.g., CMG, HLT, GOOGL, LOW, HHH):
- **Header**: "My Portfolio Dashboard"
- **Subtitle**: "Monitoring 5 active assets in your portfolio"
- **Top Opportunities**: Only shows signals for CMG, HLT, GOOGL, LOW, HHH
- **Analysis Table**: Only shows signals for portfolio stocks (unless user searches for specific symbol)
- **Stats Cards**: Reflect only portfolio stocks' signal counts

### When User Has NO Portfolio:
- **Header**: "Market Dashboard"
- **Shows**: "💡 Tip: Import your portfolio to see personalized signals"
- **All sections**: Show market-wide data

## Verification Steps
1. Go to **My Portfolio** page → Confirm you have stocks (CMG, HLT, GOOGL, LOW, HHH)
2. Go to **Market Overview** page
3. Verify:
   - Header says "My Portfolio Dashboard"
   - "Top Opportunities" shows ONLY your portfolio stocks
   - "Market Analysis & Filters" table shows ONLY your portfolio stocks
   - If you type a specific symbol in the filter (e.g., "AAPL"), it will show that stock even if not in portfolio
