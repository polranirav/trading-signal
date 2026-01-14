# Stock Navigation Fix - Analysis Page

## Problem
When users navigate to a stock (e.g., Tesla) and click "Analysis", the Analysis page was showing a different stock or not preserving the selected stock symbol. The symbol wasn't being passed correctly between pages.

## Root Cause
The `AnalysisPage` component was using local state (`useState`) for the symbol without reading from URL query parameters. When navigating from another page (e.g., Overview) to Analysis, the symbol wasn't being passed, so the page would start with an empty symbol.

## Solution
Updated `AnalysisPage.tsx` to:
1. **Read symbol from URL query params** using `useSearchParams` hook
2. **Initialize state from URL** on component mount
3. **Sync state with URL** when URL changes (e.g., when navigating with a symbol in the URL)
4. **Update URL when user changes symbol** manually in the input field

## Changes Made

### 1. Added URL Query Parameter Support
- Imported `useSearchParams` from `react-router-dom`
- Read `symbol` from URL query params: `?symbol=TSLA`
- Initialize state from URL on mount

### 2. Two-Way Sync
- **URL → State**: When URL changes (e.g., navigation with symbol), update state
- **State → URL**: When user types in symbol field, update URL query params

### 3. Symbol Handling
- Convert symbol to uppercase automatically
- Trim whitespace
- Update URL immediately when user changes symbol (no debounce needed for better UX)

## Usage

### Navigate to Analysis with Symbol
```typescript
// From any page, navigate with symbol in URL:
navigate('/dashboard/analysis?symbol=TSLA');

// Or using Link:
<Link to="/dashboard/analysis?symbol=TSLA">View Analysis</Link>
```

### Example: Adding Link from Overview Page
```typescript
// In OverviewPage.tsx or any signal list:
import { Link } from 'react-router-dom';

<Link to={`/dashboard/analysis?symbol=${signal.symbol}`}>
  View Analysis
</Link>
```

## Benefits

1. **Shareable URLs**: Users can share links with specific stocks pre-selected
2. **Browser Back/Forward**: Works correctly with browser navigation
3. **Bookmarkable**: Users can bookmark analysis pages for specific stocks
4. **Consistent State**: Symbol is preserved across page refreshes
5. **Better UX**: No confusion about which stock is being analyzed

## Testing

1. Navigate to `/dashboard/analysis?symbol=TSLA`
   - Should show Tesla analysis
   - Symbol field should be pre-filled with "TSLA"

2. Change symbol in the input field
   - URL should update to reflect new symbol
   - Analysis should update for new symbol

3. Navigate from another page with symbol in URL
   - Symbol should be preserved correctly
   - No race conditions or wrong stock displayed

## Files Modified

- `frontend/src/pages/dashboard/AnalysisPage.tsx`
  - Added `useSearchParams` import
  - Added URL query param reading
  - Added URL/state synchronization
  - Updated `handleSymbolChange` to update URL

## Next Steps (Optional)

1. **Add navigation links** from OverviewPage or signal lists to Analysis page with symbol
2. **Add click handlers** on stock symbols to navigate to analysis
3. **Add "View Analysis" buttons** on signal cards
4. **Preserve other filters** (signal type, confidence) in URL params as well
