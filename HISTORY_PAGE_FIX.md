# History Page Fix

## Problem
The history page at `/history` was not working - nothing was happening when the page loaded. The table and data were not showing.

## Root Causes

1. **Callback Not Triggering on Page Load**: The callback only listened to button clicks (`history-filter-button.n_clicks`) and store data, but didn't trigger when the page first loaded.

2. **Incorrect Store Update**: The code tried to update the store incorrectly using `dash.callback_context.states`, which doesn't work. Stores must be updated via Output in callbacks.

3. **Deprecated Import**: Using `dash_html_components` which is deprecated in newer Dash versions.

## Solution

### 1. Added URL Pathname as Input
- Added `Input("url", "pathname")` to trigger callback when page loads
- Now callback triggers both on page load and button click

### 2. Fixed Store Update
- Added `Output("history-filters-store", "data")` to callback outputs
- Removed incorrect `dash.callback_context.states` assignment
- Store is now properly updated via return value

### 3. Fixed Imports
- Changed `import dash_html_components as html` to `from dash import html`
- Updated imports to use modern Dash syntax

### 4. Improved Initial Load Logic
- On page load (`url.pathname` trigger), uses default form values
- On button click, uses form values from State
- On other triggers, uses stored filters

## Changes Made

### `src/web/history_callbacks.py`

**Before:**
```python
@app.callback(
    [Output("history-signals-table-container", "children"),
     Output("history-performance-summary", "children"),
     Output("history-pagination", "children")],
    [Input("history-filter-button", "n_clicks"),
     Input("history-filters-store", "data")],
    [...]
)
```

**After:**
```python
@app.callback(
    [Output("history-signals-table-container", "children"),
     Output("history-performance-summary", "children"),
     Output("history-pagination", "children"),
     Output("history-filters-store", "data")],  # Added store output
    [Input("history-filter-button", "n_clicks"),
     Input("url", "pathname")],  # Added URL trigger
    [...]
)
```

**Store Update:**
- Before: `dash.callback_context.states["history-filters-store"] = {"data": stored_filters}` (WRONG)
- After: `return table, summary, pagination, new_stored_filters` (CORRECT)

## Workflow Now

1. **User navigates to `/history`**
   - URL pathname changes
   - Callback triggers automatically

2. **Callback executes**
   - Uses default form values (30 days, all types, 50% confidence)
   - Fetches signals from database
   - Formats table and metrics
   - Updates store with current filters

3. **Page displays**
   - Table shows signals
   - Performance summary shows metrics
   - Filters are ready to use

4. **User clicks "Apply Filters"**
   - Button click triggers callback
   - Uses new filter values from form
   - Updates table and store

## Status
✅ **Fixed** - History page now loads data automatically on page load!
