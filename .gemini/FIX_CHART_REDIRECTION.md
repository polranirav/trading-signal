# Bug Fix: Chart Redirection & Data Display

## Issue
Clicking the "Chart" button in Analysis or Overview pages redirected to the Charts page but did not update the displayed chart to the selected stock (it often stuck on default AAPL).

## Resolution
1. **URL Parameter Handling (`ChartsPage.tsx`)**:
   - Implemented `useSearchParams` to read `?symbol=XYZ` from the URL.
   - Added `useEffect` to trigger a `selectedSymbol` state update whenever the URL parameter changes.
   - Removed the fallback to `signals[0]` (often AAPL) when the selected symbol's signal isn't found, protecting against misleading data display.

2. **Overview Page Links (`OverviewPage.tsx`)**:
   - Added `onClick` handlers to `OpportunityCard` components.
   - Using `useNavigate` to link directly to `/dashboard/charts?symbol=...`.
   - Cleaned up import mess caused during editing.

3. **User Experience**:
   - Navigation from Analysis table -> Chart now works.
   - Navigation from Dashboard Overview -> Chart now works.
   - Deep linking (refreshing the page with a symbol in URL) works.

## Verification
- Navigate to **History/Analysis** page.
- Click "Chart" next to **NVDA**.
- Charts page should load with **NVDA** selected and "Symbol: NVDA" displayed.
- Go to **Overview** page.
- Click on a Top Opportunity card.
- Charts page should load with that stock selected.
