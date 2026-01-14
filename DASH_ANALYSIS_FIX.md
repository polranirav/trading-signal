# Dash Analysis Page Fix - URL Query Parameter Support

## Problem
When users click "Analyze" button on a stock from the stocks page, it navigates to `/analysis?symbol=JPM` (correct), but the analysis page doesn't read the `symbol` query parameter. The page always shows "AAPL" as the default symbol, not the stock that was clicked.

## Root Cause
1. The route handler in `src/web/callbacks.py` only listened to `Input("url", "pathname")` but not `Input("url", "search")`
2. The `create_analysis_page()` function didn't accept a `symbol` parameter
3. The symbol input field was hardcoded to "AAPL" default value

## Solution

### 1. Updated Route Handler (`src/web/callbacks.py`)
- Changed callback to listen to both `pathname` and `search` properties of the URL Location component
- Added query parameter parsing using `urllib.parse.parse_qs`
- Extracts `symbol` from query string and passes it to `create_analysis_page()`

**Changes:**
```python
# Before:
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    elif pathname == "/analysis": 
        return require_auth(create_analysis_page)

# After:
@app.callback(Output("page-content", "children"), [Input("url", "pathname"), Input("url", "search")])
def display_page(pathname, search):
    elif pathname == "/analysis": 
        # Extract symbol from query parameter
        symbol = None
        if search:
            from urllib.parse import parse_qs
            query_params = parse_qs(search.lstrip('?'))
            symbol = query_params.get('symbol', [None])[0]
        return require_auth(lambda: create_analysis_page(symbol=symbol))
```

### 2. Updated Analysis Page Function (`src/web/layouts.py`)
- Added `symbol` parameter to `create_analysis_page(symbol=None)`
- Uses the provided symbol as default value for the input field
- Falls back to "AAPL" if no symbol provided

**Changes:**
```python
# Before:
def create_analysis_page():
    dbc.Input(id="symbol-input", ..., value="AAPL", ...)

# After:
def create_analysis_page(symbol=None):
    default_symbol = symbol.upper() if symbol else "AAPL"
    dbc.Input(id="symbol-input", ..., value=default_symbol, ...)
```

## Workflow Now

1. **User clicks "Analyze" on a stock** (e.g., JPM from stocks page)
   - Link: `/analysis?symbol=JPM` (from `src/web/stocks.py` line 114)

2. **Route handler receives the URL**
   - `pathname = "/analysis"`
   - `search = "?symbol=JPM"`

3. **Query parameter is parsed**
   - `symbol = "JPM"` extracted from query string

4. **Analysis page is created with symbol**
   - `create_analysis_page(symbol="JPM")` is called
   - Input field is pre-filled with "JPM"

5. **User sees analysis page**
   - Symbol input shows "JPM" (not "AAPL")
   - User can click "Analyze" button to see JPM analysis
   - Or change symbol and analyze different stock

## Files Modified

1. **`src/web/callbacks.py`**
   - Updated `display_page` callback to listen to URL `search` property
   - Added query parameter parsing logic
   - Pass symbol to `create_analysis_page()` function

2. **`src/web/layouts.py`**
   - Updated `create_analysis_page()` to accept `symbol` parameter
   - Use symbol as default value for input field

## Testing

1. Navigate to stocks page: `/stocks`
2. Click "Analyze" button on any stock (e.g., JPM)
3. Should navigate to `/analysis?symbol=JPM`
4. Analysis page should load with "JPM" in the symbol input field
5. Click "Analyze" button to see JPM analysis
6. Try different stocks - symbol should change correctly

## Status
✅ **Fixed** - Analysis page now correctly reads and uses symbol from URL query parameter!
