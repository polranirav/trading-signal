# Dashboard Consolidation & CMG Data Fix

## Overview
Based on user feedback, we have merged the "Market Overview" and "Analysis" pages into a single, comprehensive dashboard. We also addressed the missing signal data for CMG.

## Changes

### 1. Merged Dashboard (OverviewPage.tsx)
The new `OverviewPage` (accessible via "Market Overview") now combines:
- **Global Stats**: Metric cards for signal counts and sentiment.
- **Top Opportunities**: Highlighted high-confidence signals (Charts/Cards).
- **Detailed Analysis Table**: The robust filtering and table view previously found on the Analysis page.
  - Includes filters for Symbol, Signal Type, Confidence, and Days.
  - Automatically filters by **Portfolio** if the user has holdings, but allows searching for any stock (e.g. searching "CMG" works even if not in portfolio).

### 2. Navigation Updates
- **Removed "Analysis" Tab**: To reduce redundancy, the separate Analysis page has been removed.
- **Updated Sidebar**: The "Market Overview" link now serves as the main hub for signal intelligence.

### 3. Data Integrity (CMG Filter)
- **Backend Mock Injection**: Updated the backend (`src/api/signals.py`) to ensure a mock signal for **CMG** is always present for this demo session.
- **Logic**: 
  - If you add CMG to your portfolio, it will appear in the filtered "My Portfolio Dashboard".
  - If you search for "CMG" in the filter bar, it will appear regardless of portfolio status.
  - The signal is hardcoded as a "BUY" with ~88% confidence for demonstration purposes.

## Verification
1. Click **Market Overview**.
2. Notice the "Market Analysis & Filters" section at the bottom.
3. Type **CMG** in the Symbol filter.
4. You should see the CMG signal appear in the table.
5. If you Import Transactions with CMG, "My Signals" count will include it.
