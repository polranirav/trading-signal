#!/bin/bash
# Start Frontend Development Server

echo "🚀 Starting Trading Signals Pro Frontend..."
echo ""
echo "📍 Frontend will run on: http://localhost:3002"
echo "📍 Backend API: http://localhost:8050 (via Docker)"
echo ""
echo "⚠️  Make sure Docker containers are running:"
echo "   docker-compose ps"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
  echo "📝 Creating .env file..."
  echo "VITE_API_BASE_URL=/api/v1" > .env
fi

# Display current .env
echo "📋 Current API configuration:"
cat .env
echo ""

# Start dev server
npm run dev
