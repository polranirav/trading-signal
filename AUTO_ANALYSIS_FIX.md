# Auto-Load Analysis Fix

## Problem
When users click "Analyze" on a stock:
1. They see an empty analysis page (first screen) - confusing
2. They have to click "Analyze" button again to see the analysis (second screen)
3. They don't understand why there are two screens

## Solution
Make the analysis automatically load when the page opens with a symbol in the URL, so users go directly to the analysis (no empty screen).

## Changes Made

### 1. Updated Analysis Callback (`src/web/callbacks.py`)
- Changed callback to listen to URL `search` parameter (in addition to analyze button)
- When URL has `?symbol=STOCK`, analysis automatically triggers
- Removed `prevent_initial_call=True` so it can run on page load
- Symbol is extracted from URL if present, otherwise uses input field value

**Before:**
```python
@app.callback(
    [...],
    Input("analyze-btn", "n_clicks"),
    State("symbol-input", "value"),
    prevent_initial_call=True  # Won't run on page load
)
```

**After:**
```python
@app.callback(
    [...],
    [Input("analyze-btn", "n_clicks"),
     Input("url", "search")],  # Also triggers on URL change
    [State("symbol-input", "value"),
     State("url", "pathname")]
)
```

### 2. Symbol Extraction
- Checks URL query parameter first
- Falls back to input field value
- Ensures symbol is uppercase

## Workflow Now

1. **User clicks "Analyze" on stock JPM**
   - Navigates to `/analysis?symbol=JPM`

2. **Page loads**
   - Symbol input is pre-filled with "JPM"
   - Analysis callback automatically triggers (because URL `search` changed)
   - Analysis runs immediately

3. **User sees analysis directly**
   - No empty screen
   - Charts load automatically
   - Results appear immediately

## Benefits

✅ **No empty screen** - Analysis loads automatically
✅ **Better UX** - Users see results immediately
✅ **Clearer flow** - One screen instead of two
✅ **Still works manually** - Users can still click "Analyze" button if they change symbol

## Status
✅ **Fixed** - Analysis now auto-loads when clicking from stocks page!
