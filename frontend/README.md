# Trading Signals Pro - Frontend

React + TypeScript + Vite frontend application.

## Setup

```bash
# Install dependencies (already done)
npm install

# Create .env file
cp .env.example .env

# Start development server
npm run dev
```

## Development

The frontend runs on `http://localhost:5173` (Vite default port).

The backend API should be running on `http://localhost:8050`.

## Project Structure

- `src/pages/` - Page components
- `src/components/` - Reusable components
- `src/services/` - API client services
- `src/store/` - Zustand state management
- `src/types/` - TypeScript type definitions

## Features Implemented

- ✅ Authentication (Login, Register)
- ✅ Protected routes
- ✅ API client service
- ✅ Auth store (Zustand)
- ✅ Dashboard overview page (basic)
- ✅ Landing page (marketing)

## Next Steps

- Build full dashboard pages
- Build marketing pages (Features, Pricing)
- Build account management pages
- Add charts and visualizations
- Add more components
