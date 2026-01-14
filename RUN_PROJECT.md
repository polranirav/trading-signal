# How to Run the Trading Signals Project

## Prerequisites

1. **Python 3.9+** (you have Python 3.9.6 ✅)
2. **Node.js & npm** (you have npm 10.9.2 ✅)
3. **Docker & Docker Compose** (for database and Redis)
4. **Virtual Environment** (recommended)

## Quick Start

### Step 1: Start Database and Redis Services

```bash
# Start Docker services (PostgreSQL + TimescaleDB, Redis)
docker-compose up -d db redis
```

Wait for services to be healthy (about 10-15 seconds).

### Step 2: Set Up Backend

```bash
# Navigate to project root
cd /Users/niravpolara/Desktop/Project\ For\ Resume/trading-signals

# Create virtual environment (if not exists)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
# venv\Scripts\activate  # On Windows

# Install Python dependencies
pip install -e .

# OR if you have a requirements.txt:
# pip install -r requirements.txt

# Initialize database (create tables)
python -c "from src.data.models import init_database; init_database()"
```

### Step 3: Create Admin User (Optional but Recommended)

```bash
python -c "
from src.data.persistence import get_database
from src.auth.models import User
from src.auth.service import AuthService

db = get_database()
with db.get_session() as session:
    # Check if admin exists
    admin = session.query(User).filter(User.email == 'admin@example.com').first()
    if not admin:
        admin = User(
            email='admin@example.com',
            password_hash=AuthService.hash_password('admin123'),
            full_name='Admin User',
            is_admin=True,
            is_active=True,
            email_verified=True
        )
        session.add(admin)
        session.commit()
        print('Admin user created: admin@example.com / admin123')
    else:
        print('Admin user already exists')
"
```

### Step 4: Start Backend Server

```bash
# Make sure virtual environment is activated
python -m src.web.app
```

Backend will run on: **http://localhost:8050**

### Step 5: Set Up Frontend (in a new terminal)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (if not already installed)
npm install

# Start development server
npm run dev
```

Frontend will run on: **http://localhost:5173** (or port shown in terminal)

## Access Points

- **Frontend (React)**: http://localhost:5173
- **Backend API**: http://localhost:8050/api/v1
- **Dash Dashboard**: http://localhost:8050 (if using Dash UI)
- **Admin Panel**: http://localhost:5173/admin (after logging in as admin)

## Running in Production Mode

### Using Docker Compose (All Services)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Environment Variables

Make sure your `.env` file is configured. Key variables:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/trading_signals
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER=redis://localhost:6379/1
CELERY_BACKEND=redis://localhost:6379/2
```

## Troubleshooting

### Database Connection Issues

```bash
# Check if PostgreSQL is running
docker-compose ps db

# Check database logs
docker-compose logs db

# Restart database
docker-compose restart db
```

### Port Already in Use

```bash
# Backend (port 8050)
lsof -ti:8050 | xargs kill -9  # macOS/Linux
# OR change port in src/web/app.py

# Frontend (port 5173)
lsof -ti:5173 | xargs kill -9  # macOS/Linux
# OR change port in frontend/vite.config.ts (if exists)
```

### Python Dependencies Issues

```bash
# Upgrade pip
pip install --upgrade pip

# Reinstall dependencies
pip install -e . --force-reinstall
```

### Frontend Dependencies Issues

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Development Workflow

1. **Backend Development**:
   ```bash
   # Terminal 1: Backend server
   python -m src.web.app
   ```

2. **Frontend Development**:
   ```bash
   # Terminal 2: Frontend server
   cd frontend && npm run dev
   ```

3. **Database Management**:
   ```bash
   # Terminal 3: Database console (if needed)
   docker-compose exec db psql -U postgres -d trading_signals
   ```

## Next Steps

1. Login as admin: http://localhost:5173/login
   - Email: `admin@example.com`
   - Password: `admin123` (or the password you set)

2. Access admin panel: http://localhost:5173/admin

3. Test API endpoints: http://localhost:8050/api/v1/admin/analytics/dashboard
