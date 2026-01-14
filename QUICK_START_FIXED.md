# 🚀 Quick Start - Complete Setup

## ✅ Everything is Fixed and Ready!

### Current Status:
- ✅ Backend API: Running in Docker on port 8050
- ✅ Database: Running and healthy
- ✅ Frontend Config: Fixed (uses proxy)
- ✅ CORS: Updated to allow all localhost variants
- ✅ Error Handling: Improved with better messages

---

## 📋 Step-by-Step Instructions

### 1. Verify Backend is Running
```bash
cd /Users/niravpolara/Desktop/Project\ For\ Resume/trading-signals
docker-compose ps
```

**You should see:**
- `trading-dashboard` - Status: `Up (healthy)`
- `trading-db` - Status: `Up (healthy)`
- `trading-redis` - Status: `Up (healthy)`

**If not running, start it:**
```bash
docker-compose up -d
```

### 2. Start Frontend Dev Server
```bash
cd frontend
npm run dev
```

**OR use the helper script:**
```bash
cd frontend
./start-dev.sh
```

**Expected output:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3002/
  ➜  Network: use --host to expose
```

### 3. Test the Application

#### A. Open Browser
Go to: **http://localhost:3002**

#### B. Create Account
1. Click "Sign Up Free" or go to `/register`
2. Fill in:
   - Email: `yourname@example.com`
   - Password: `yourpassword123` (min 8 chars)
   - Full Name: (optional)
3. Click "Create Account"
4. **Should redirect to dashboard** ✅

#### C. Login
1. Go to `/login`
2. Enter your email and password
3. Click "Sign In"
4. **Should redirect to dashboard** ✅

---

## 🔧 Configuration Summary

### Frontend `.env` (frontend/.env)
```
VITE_API_BASE_URL=/api/v1
```
✅ **This is correct!** Uses Vite proxy.

### Vite Proxy (frontend/vite.config.ts)
- ✅ Port: 3002
- ✅ Proxy: `/api` → `http://localhost:8050/api`
- ✅ CORS: Handled by proxy

### Backend CORS
- ✅ Allows: `http://localhost:3002`
- ✅ Credentials: Enabled
- ✅ Methods: GET, POST, PUT, DELETE, OPTIONS

---

## 🐛 If Something Doesn't Work

### Issue: "Network Error"
**Check:**
1. Is backend running?
   ```bash
   docker-compose ps dashboard
   curl http://localhost:8050/api/v1/health
   ```

2. Is frontend running?
   - Check terminal for `npm run dev` output
   - Should show: `Local: http://localhost:3002/`

3. Check browser console (F12 → Console)
   - Look for error messages
   - Check Network tab for failed requests

4. **Restart frontend:**
   - Stop: `Ctrl+C` in frontend terminal
   - Start: `npm run dev`

### Issue: "Invalid email or password"
✅ **This means backend IS working!**
- The credentials don't exist
- **Solution:** Create a new account first

### Issue: Frontend shows white screen
- Check browser console for errors
- Verify frontend is running on port 3002
- Clear browser cache and reload

---

## 🧪 Quick Test Commands

### Test Backend API Directly
```bash
# Test registration
curl -X POST http://localhost:8050/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test12345"}'

# Test login
curl -X POST http://localhost:8050/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test12345"}'
```

### Test Frontend Proxy
```bash
# This should work if frontend is running
curl http://localhost:3002/api/v1/health
```

---

## 📊 Service URLs

| Service | URL | Status |
|---------|-----|--------|
| Frontend | http://localhost:3002 | Start with `npm run dev` |
| Backend API | http://localhost:8050/api/v1 | Docker container |
| Dash Dashboard | http://localhost:8050 | Docker container |
| Database | localhost:5432 | Docker container |
| Redis | localhost:6379 | Docker container |
| Grafana | http://localhost:3000 | Docker container |
| Prometheus | http://localhost:9090 | Docker container |

---

## ✅ Verification Checklist

- [ ] Docker containers running (`docker-compose ps`)
- [ ] Backend accessible (`curl http://localhost:8050/api/v1/health`)
- [ ] Frontend `.env` set to `/api/v1`
- [ ] Frontend dev server running (`npm run dev`)
- [ ] Browser opens http://localhost:3002
- [ ] Can register new account
- [ ] Can login with registered account
- [ ] Redirects to dashboard after login

---

## 🎯 What Was Fixed

1. ✅ **Frontend `.env`** - Changed to use proxy (`/api/v1`)
2. ✅ **CORS Configuration** - Added all localhost variants
3. ✅ **Error Handling** - Better network error messages
4. ✅ **Backend Status** - Verified running and healthy
5. ✅ **Database Connection** - Working (tested registration)

---

## 🚀 Ready to Go!

**Start the frontend:**
```bash
cd frontend
npm run dev
```

**Then open:** http://localhost:3002

Everything should work now! 🎉
