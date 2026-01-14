# History Page Fix - Complete

## Problem
The history page at `/history` was not working - nothing was happening when the page loaded. The table and data were not showing.

## Root Causes Found

1. **Callback Not Triggering on Page Load**: The callback only listened to:
   - Button clicks (`history-filter-button.n_clicks`)
   - Store data changes (`history-filters-store.data`)
   - But NOT page load - so data never loaded when you first visited `/history`

2. **Incorrect Store Update**: The code tried to update the store incorrectly:
   ```python
   dash.callback_context.states["history-filters-store"] = {"data": stored_filters}  # WRONG!
   ```
   - You can't modify states like this
   - Stores must be updated via Output in callbacks

3. **Deprecated Import**: Using `dash_html_components` which is deprecated

## Solution Implemented

### 1. Added URL Pathname as Input
- Added `Input("url", "pathname")` to trigger callback when page loads
- Now callback triggers both:
  - When page loads (pathname changes to `/history`)
  - When button is clicked

### 2. Fixed Store Update
- Added `Output("history-filters-store", "data")` to callback outputs
- Removed incorrect `dash.callback_context.states` assignment
- Store is now properly updated via return value

### 3. Fixed Imports
- Changed `import dash_html_components as html` → `from dash import html`
- Updated to modern Dash syntax

### 4. Improved Initial Load Logic
- On page load (`url.pathname` trigger): Uses default form values
- On button click: Uses form values from State
- Properly handles None/empty values

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
def update_history_table(n_clicks, stored_filters, ...):
    # Store update (WRONG):
    dash.callback_context.states["history-filters-store"] = {"data": stored_filters}
    return table, summary, pagination
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
def update_history_table(n_clicks, pathname, ...):
    # Store update (CORRECT):
    return table, summary, pagination, new_stored_filters
```

## Workflow Now

1. **User navigates to `/history`**
   - URL pathname changes to `/history`
   - Callback triggers automatically (because `Input("url", "pathname")` changed)

2. **Callback executes on page load**
   - Uses default form values (30 days, all types, 50% confidence)
   - Fetches signals from database
   - Filters and formats data
   - Returns table, summary, pagination, and store update

3. **Page displays immediately**
   - Table shows signals
   - Performance summary shows metrics
   - Filters are ready to use

4. **User clicks "Apply Filters"**
   - Button click triggers callback again
   - Uses new filter values from form
   - Updates table and store

## Status
✅ **FIXED** - History page now loads data automatically on page load!

The page should now work correctly - data loads when you navigate to `/history`.
