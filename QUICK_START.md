# Quick Start Guide

## 🚀 Running the Project

Your services are already running! Here's what's available:

### ✅ Currently Running Services

1. **Backend (Flask/Dash)**: http://localhost:8050
   - API Endpoints: http://localhost:8050/api/v1
   - Admin API: http://localhost:8050/api/v1/admin

2. **Database (PostgreSQL + TimescaleDB)**: localhost:5432
   - Database: `trading_signals`
   - User: `postgres`
   - Password: `postgres`

3. **Redis**: localhost:6379

4. **Frontend (React)**: Starting on http://localhost:5173
   - Admin Panel: http://localhost:5173/admin
   - User Dashboard: http://localhost:5173/dashboard

### 📋 Quick Access

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8050/api/v1
- **Admin Panel**: http://localhost:5173/admin (after login)

### 🔐 Login Credentials

**Admin User** (if created):
- Email: `admin@example.com`
- Password: `admin123` (or whatever you set)

**Regular User**: Register at http://localhost:5173/register

### 🛠️ Common Commands

**View Backend Logs:**
```bash
docker-compose logs -f dashboard
```

**View Frontend Logs:**
Check the terminal where `npm run dev` is running

**Restart Services:**
```bash
# Restart backend
docker-compose restart dashboard

# Restart frontend (stop with Ctrl+C, then restart)
cd frontend && npm run dev
```

**Stop All Services:**
```bash
docker-compose down
```

**Start All Services:**
```bash
docker-compose up -d
```

### 🧪 Test the Setup

1. **Check Backend API:**
   ```bash
   curl http://localhost:8050/api/v1/health
   ```

2. **Check Frontend:**
   - Open http://localhost:5173 in your browser
   - You should see the landing page

3. **Login as Admin:**
   - Go to http://localhost:5173/login
   - Use admin credentials
   - Navigate to http://localhost:5173/admin

### 📝 Next Steps

1. **Create Admin User** (if not exists):
   ```bash
   python -c "
   from src.data.persistence import get_database
   from src.auth.models import User
   from src.auth.service import AuthService
   
   db = get_database()
   with db.get_session() as session:
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
           print('✅ Admin user created!')
       else:
           print('ℹ️ Admin user already exists')
   "
   ```

2. **Access Admin Panel:**
   - Login at http://localhost:5173/login
   - Go to http://localhost:5173/admin

3. **Explore Features:**
   - Dashboard: View platform statistics
   - Users: Manage user accounts
   - Signals: View and manage trading signals
   - Subscriptions: Manage user subscriptions
   - Settings: Configure system settings
   - Audit Logs: View audit trail

### 🐛 Troubleshooting

**Frontend not starting:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

**Backend errors:**
```bash
docker-compose logs dashboard
```

**Database connection issues:**
```bash
docker-compose restart db
# Wait 10 seconds, then check
docker-compose ps db
```

**Port conflicts:**
```bash
# Kill process on port 5173
lsof -ti:5173 | xargs kill -9

# Kill process on port 8050
lsof -ti:8050 | xargs kill -9
```

### 📚 Documentation

- **Backend Architecture**: See `ADMIN_BACKEND_ARCHITECTURE.md`
- **UI Implementation**: See `ADMIN_UI_IMPLEMENTATION_SUMMARY.md`
- **Full Run Guide**: See `RUN_PROJECT.md`
