# Onboarding Flow Design

## User Journey After Registration

### Current Flow (Problematic)
```
Register → Dashboard → Confusion → Leave
```

### Improved Flow
```
Register 
  → Welcome Screen
  → Onboarding Quiz (Preferences)
  → Watchlist Setup
  → Dashboard Tour
  → First Action Prompt
  → Success State
```

## Detailed Onboarding Steps

### Step 1: Welcome Screen (Immediate)
```tsx
// Show immediately after registration
- Welcome message: "Welcome! Let's set up your account"
- Progress: "Step 1 of 5"
- Skip option: "I'll do this later" (but recommended)
- Visual: Celebratory animation
- Duration: Auto-advance after 2 seconds or click
```

### Step 2: Trading Style Quiz
```tsx
// Collect preferences for personalization
Questions:
1. "What's your trading experience?"
   - Beginner
   - Intermediate  
   - Advanced
   - Professional

2. "What's your risk tolerance?"
   - Conservative (safer, lower returns)
   - Moderate (balanced)
   - Aggressive (higher risk, higher returns)

3. "What sectors interest you?" (multi-select)
   - Technology
   - Healthcare
   - Finance
   - Energy
   - Consumer
   - etc.

4. "What's your primary goal?"
   - Day Trading
   - Swing Trading
   - Long-term Investing
   - Portfolio Diversification

5. "How often do you check signals?"
   - Multiple times daily
   - Daily
   - Weekly
   - Monthly

Benefits:
- Personalizes dashboard
- Sets default risk parameters
- Pre-filters signals
- Improves relevance
```

### Step 3: Watchlist Setup
```tsx
// Let users add stocks to watchlist
- Search bar: "Search for stocks to add"
- Popular stocks: Quick-add buttons
- Categories: "Tech Stocks", "Blue Chips", etc.
- Skip option: "I'll add stocks later"
- Minimum: Suggest at least 3-5 stocks
- Auto-suggest: Based on preferences from Step 2
```

### Step 4: Dashboard Tour
```tsx
// Interactive walkthrough of dashboard
- Highlight key sections with tooltips
- Show where to find signals
- Explain watchlist
- Show settings
- Skip option available
- Progress: "Step 4 of 5"
```

### Step 5: First Action Prompt
```tsx
// Guide to first meaningful action
- Suggested action: "View your first signal"
- Alternative: "Explore your watchlist"
- Help: "Need help? Click here"
- Success state: Celebrate completion
- Next: Redirect to dashboard with highlights
```

## Implementation Details

### Backend Support Needed

1. **UserPreferences Model** (✅ Already added by user)
   - `onboarding_completed`: Boolean
   - `onboarding_step`: Integer
   - `risk_tolerance`: String
   - `preferred_sectors`: JSON array
   - Other preferences

2. **API Endpoints Needed**
   ```python
   POST /api/v1/onboarding/preferences
   - Save user preferences
   
   POST /api/v1/onboarding/complete
   - Mark onboarding as complete
   
   GET /api/v1/onboarding/status
   - Get onboarding progress
   ```

3. **Watchlist API** (✅ Already have UserWatchlist model)
   ```python
   POST /api/v1/watchlist
   - Add stock to watchlist
   
   GET /api/v1/watchlist
   - Get user's watchlist
   ```

### Frontend Components Needed

1. `OnboardingFlow.tsx` - Main onboarding container
2. `WelcomeStep.tsx` - Welcome screen
3. `TradingStyleQuiz.tsx` - Preference collection
4. `WatchlistSetup.tsx` - Stock selection
5. `DashboardTour.tsx` - Interactive tour (using react-joyride or similar)
6. `FirstActionPrompt.tsx` - Guided first action
7. `OnboardingProgress.tsx` - Progress indicator

### User Preferences Storage

```typescript
interface UserPreferences {
  onboarding_completed: boolean
  onboarding_step: number
  risk_tolerance: 'conservative' | 'moderate' | 'aggressive'
  preferred_sectors: string[]
  trading_experience: 'beginner' | 'intermediate' | 'advanced' | 'professional'
  primary_goal: 'day_trading' | 'swing_trading' | 'long_term' | 'diversification'
  check_frequency: 'multiple_daily' | 'daily' | 'weekly' | 'monthly'
}
```

## Personalization Based on Onboarding

### Dashboard Customization
- **Risk Tolerance**: Filter signals by risk level
- **Sectors**: Highlight preferred sectors
- **Experience Level**: Show/hide advanced features
- **Goals**: Tailor recommendations
- **Frequency**: Default notification settings

### Signal Filtering
- Pre-filter by preferred sectors
- Match risk tolerance
- Highlight relevant signals
- Customize default view

### Notifications
- Set default notification preferences
- Email frequency based on check_frequency
- Alert settings based on risk tolerance

## Empty States After Onboarding

### First Dashboard Visit
```tsx
// Show helpful empty states
- Watchlist empty: "Add stocks to your watchlist to get started"
  → CTA: "Add Stocks"
  
- No signals yet: "We're analyzing your watchlist..."
  → Info: "Signals will appear here as we detect opportunities"
  → Show sample signal structure
  
- No history: "Your trading history will appear here"
  → Info: "Track your performance over time"
```

## Success Metrics

- **Onboarding Completion Rate**: % who complete all steps (target: 70%+)
- **Time to Complete**: Average time (target: 2-3 minutes)
- **Watchlist Size**: Average stocks added (target: 5+)
- **First Action Time**: Time to first meaningful action (target: <5 minutes)
- **Retention**: % who return after onboarding (target: 60%+)
