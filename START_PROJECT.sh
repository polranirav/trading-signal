#!/bin/bash
# Start Trading Signals Dashboard

cd "$(dirname "$0")"
source venv/bin/activate

echo "🚀 Starting Trading Signals Dashboard..."
echo "📍 URL: http://localhost:8050"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 src/web/app.py
