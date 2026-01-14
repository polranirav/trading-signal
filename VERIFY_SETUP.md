# Complete Setup Verification Guide

## ✅ Step-by-Step Setup

### 1. Start Backend (Docker)
```bash
cd /Users/niravpolara/Desktop/Project\ For\ Resume/trading-signals
docker-compose up -d
```

**Verify backend is running:**
```bash
docker-compose ps
# Should show "trading-dashboard" as "Up" and "healthy"
```

**Test backend API:**
```bash
curl http://localhost:8050/api/v1/health
# Should return HTML (Dash app) or JSON
```

### 2. Start Frontend
```bash
cd frontend
npm run dev
```

**OR use the helper script:**
```bash
cd frontend
./start-dev.sh
```

**Verify frontend is running:**
- Open browser: http://localhost:3002
- Should see the landing page

### 3. Test Registration
1. Go to http://localhost:3002/register
2. Fill in:
   - Email: `demo@example.com`
   - Password: `demo12345`
   - Full Name: `Demo User`
3. Click "Create Account"
4. Should redirect to dashboard

### 4. Test Login
1. Go to http://localhost:3002/login
2. Use credentials from step 3
3. Click "Sign In"
4. Should redirect to dashboard

## 🔧 Configuration Files

### Frontend `.env` (frontend/.env)
```
VITE_API_BASE_URL=/api/v1
```

### Vite Proxy (frontend/vite.config.ts)
- Proxies `/api/*` → `http://localhost:8050/api/*`
- Port: 3002

### Backend CORS (src/api/routes.py)
- Allows: `http://localhost:3002`
- Credentials: enabled

## 🐛 Troubleshooting

### "Network Error" on Login/Register
1. **Check backend is running:**
   ```bash
   docker-compose ps dashboard
   curl http://localhost:8050/api/v1/health
   ```

2. **Check frontend .env:**
   ```bash
   cat frontend/.env
   # Should be: VITE_API_BASE_URL=/api/v1
   ```

3. **Restart frontend:**
   - Stop dev server (Ctrl+C)
   - Run `npm run dev` again

4. **Check browser console:**
   - Open DevTools → Network tab
   - Try login/register
   - Look for failed requests
   - Check if requests go to `http://localhost:3002/api/v1/...` (correct)
   - NOT `http://localhost:8050/api/v1/...` (wrong)

### "Invalid email or password"
- This means backend is working!
- The credentials don't exist in database
- Create a new account first, then login

### Database Connection Error
- Backend can't connect to database
- Check: `docker-compose ps db` (should be "healthy")
- Check: `.env` file has `DATABASE_URL=postgresql://postgres:postgres@db:5432/trading_signals`

## 📊 Service Status Check

```bash
# Check all services
docker-compose ps

# Check backend logs
docker-compose logs dashboard --tail=20

# Check database
docker-compose exec db psql -U postgres -d trading_signals -c "SELECT COUNT(*) FROM users;"
```

## 🎯 Quick Test Commands

```bash
# Test registration
curl -X POST http://localhost:8050/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test12345"}'

# Test login
curl -X POST http://localhost:8050/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test12345"}' \
  -c /tmp/cookies.txt
```
