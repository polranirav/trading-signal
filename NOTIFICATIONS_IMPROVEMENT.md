# Notifications Improvement

## Problem
The user reported that notifications/alerts on the history page (and other pages) "not looks good" and are too intrusive. Specifically:

1. **"Status update failed"** - Large orange alert banner displayed when status updates fail
2. **"Please log in to access this page."** - Large orange alert banner displayed above login form

These alerts were visually intrusive and disrupted the user experience.

## Solution

### 1. Status Update Failed Notification ✅
**Location:** `src/web/callbacks_live_status.py`

**Before:**
- Large orange `dbc.Alert` with `color="warning"`
- Full-width banner that was very prominent

**After:**
- Small, subtle status indicator
- Gray text with minimal styling
- Integrated into the status bar design
- Less intrusive - just shows "⚠️ Update unavailable" in muted text

**Changes:**
```python
# Before:
return [
    dbc.Col([
        dbc.Alert("Status update failed", color="warning")
    ], width=12)
]

# After:
return dbc.Row([
    dbc.Col([
        html.Div([
            html.Span("Status: ", style={'marginRight': '10px', 'color': '#9ca3af'}),
            html.Span("⚠️ Update unavailable", style={'color': '#9ca3af', 'fontSize': '0.85em'})
        ], style={
            'padding': '8px 20px',
            'backgroundColor': '#1e293b',
            'borderRadius': '5px',
            'marginBottom': '20px',
            'fontSize': '0.9em'
        })
    ], width=12)
])
```

### 2. Login Required Notification ✅
**Location:** `src/web/callbacks.py`

**Before:**
- Large orange `dbc.Alert` saying "Please log in to access this page."
- Displayed above the login form
- Redundant since the login form is already visible

**After:**
- Removed the alert entirely
- The login form is self-explanatory
- Cleaner, less cluttered interface

**Changes:**
```python
# Before:
return html.Div([
    dbc.Alert(
        "Please log in to access this page.",
        color="warning",
        className="mb-3"
    ),
    create_login_page()
])

# After:
return create_login_page()
```

## Benefits

1. **Less Visual Noise** - Removed large, attention-grabbing orange banners
2. **Better UX** - Notifications are now subtle and don't interrupt the user experience
3. **Cleaner Design** - Login page is cleaner without redundant messages
4. **Professional Appearance** - Status errors shown as subtle indicators rather than alerts

## Status
✅ **COMPLETE** - Notifications are now less intrusive and better designed!
