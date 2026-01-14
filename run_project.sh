#!/bin/bash
# Run Trading Signals Dashboard
# Usage: ./run_project.sh [--docker|--local]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_banner() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}                    ${GREEN}🚀 TRADING SIGNALS PRO${NC}                          ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}           ${YELLOW}Institutional-Grade AI Trading Signals${NC}                    ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

run_with_docker() {
    echo -e "${BLUE}[Docker]${NC} Starting full stack with Docker Compose..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
    
    # Copy .env.example to .env if .env doesn't exist
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}Creating .env from .env.example...${NC}"
        cp .env.example .env 2>/dev/null || echo "# Auto-generated .env file" > .env
    fi
    
    # Start services
    echo -e "${GREEN}Starting services...${NC}"
    docker-compose up -d
    
    echo ""
    echo -e "${GREEN}✅ Services started!${NC}"
    echo ""
    echo -e "${BLUE}Service URLs:${NC}"
    echo "  📊 Dashboard:   http://localhost:8050"
    echo "  🌸 Flower:      http://localhost:5555"
    echo "  📈 Grafana:     http://localhost:3000 (admin/trading123)"
    echo "  📉 Prometheus:  http://localhost:9090"
    echo ""
    echo -e "${YELLOW}To view logs: docker-compose logs -f${NC}"
    echo -e "${YELLOW}To stop:      docker-compose down${NC}"
}

run_locally() {
    echo -e "${BLUE}[Local]${NC} Running dashboard locally..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
        python3 -m venv venv
    fi

    # Activate virtual environment
    echo -e "${GREEN}🔧 Activating virtual environment...${NC}"
    source venv/bin/activate

    # Upgrade pip
    echo -e "${GREEN}⬆️  Upgrading pip...${NC}"
    pip install --upgrade pip setuptools wheel --quiet

    # Install essential dependencies
    echo -e "${GREEN}📥 Installing dependencies (this may take a minute)...${NC}"
    pip install dash dash-bootstrap-components pandas numpy requests yfinance pydantic pydantic-settings structlog plotly bcrypt flask-cors redis sqlalchemy psycopg2-binary httpx aiohttp --quiet

    echo ""
    echo -e "${GREEN}✅ Dependencies installed!${NC}"
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${GREEN}🎯 STARTING DASHBOARD${NC}                                              ${BLUE}║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC}  📍 Dashboard URL: ${YELLOW}http://localhost:8050${NC}                           ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  🛑 Press Ctrl+C to stop the server                                ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Run the app
    python3 src/web/app.py
}

print_banner

# Parse arguments
if [ "$1" == "--docker" ]; then
    run_with_docker
elif [ "$1" == "--local" ]; then
    run_locally
else
    # Default: ask user
    echo "How would you like to run the project?"
    echo ""
    echo "  1) 🐳 Docker (full stack with database, Redis, Celery)"
    echo "  2) 💻 Local  (dashboard only, minimal dependencies)"
    echo ""
    read -p "Enter choice [1/2]: " choice
    
    case $choice in
        1)
            run_with_docker
            ;;
        2)
            run_locally
            ;;
        *)
            echo "Invalid choice. Running locally..."
            run_locally
            ;;
    esac
fi
