# Google OAuth Setup Guide

Follow these steps to enable "Sign in with Google" functionality.

## Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **"Select a Project"** → **"New Project"**
3. Enter project name: `arpeggiator-ai` (or any name)
4. Click **"Create"**

## Step 2: Enable Google+ API

1. In the search bar, type **"Google+ API"** or **"People API"**
2. Click on **"Google+ API"** or **"People API"**
3. Click **"Enable"**

## Step 3: Configure OAuth Consent Screen

1. Go to **"APIs & Services"** → **"OAuth consent screen"**
2. Select **"External"** (for testing with any Google account)
3. Click **"Create"**
4. Fill in required fields:
   - **App name**: `ArpeggiatorAI`
   - **User support email**: Your email
   - **Developer contact**: Your email
5. Click **"Save and Continue"**
6. **Scopes**: Click **"Add or Remove Scopes"**
   - Add: `email`
   - Add: `profile`
   - Add: `openid`
7. Click **"Save and Continue"**
8. **Test users** (for testing): Add your email
9. Click **"Save and Continue"**

## Step 4: Create OAuth 2.0 Credentials

1. Go to **"APIs & Services"** → **"Credentials"**
2. Click **"+ Create Credentials"** → **"OAuth client ID"**
3. Application type: **"Web application"**
4. Name: `ArpeggiatorAI Web Client`
5. **Authorized JavaScript origins**:
   - Add: `http://localhost:3000`
   - Add: `http://localhost:8006`
6. **Authorized redirect URIs**:
   - Add: `http://localhost:8006/api/auth/google/callback`
7. Click **"Create"**
8. **Copy** the **Client ID** and **Client Secret** that appear

## Step 5: Configure Your Application

Create a `.env` file in the project root:

```bash
cd /Users/abel/Documents/Music-Arpeggiator
cat > .env << 'EOF'
# Database
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/arpeggiator

# JWT Secret
SECRET_KEY=super-secret-key-change-in-production-$(openssl rand -hex 32)

# Google OAuth Credentials (paste your values here)
GOOGLE_CLIENT_ID=your-client-id-here.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret-here
GOOGLE_REDIRECT_URI=http://localhost:8006/api/auth/google/callback

# CORS
CORS_ORIGINS=["http://localhost:3000"]
EOF
```

**Or manually create `.env` and paste:**

```env
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/arpeggiator
SECRET_KEY=your-super-secret-key-change-this
GOOGLE_CLIENT_ID=paste-your-client-id-here
GOOGLE_CLIENT_SECRET=paste-your-client-secret-here
GOOGLE_REDIRECT_URI=http://localhost:8006/api/auth/google/callback
```

## Step 6: Restart the Backend

```bash
# Stop current servers
lsof -ti:8006 | xargs kill -9

# Restart
./run.sh
```

## Step 7: Test Google Sign-In

1. Go to http://localhost:3000/signup or http://localhost:3000/login
2. Click **"Continue with Google"**
3. You should see Google's OAuth consent screen
4. Sign in and authorize
5. You'll be redirected back and logged in!

## Troubleshooting

### Error: "redirect_uri_mismatch"
- Make sure `http://localhost:8006/api/auth/google/callback` is **exactly** in your authorized redirect URIs
- No trailing slash!

### Error: "Access blocked: This app's request is invalid"
- Check that Google+ API or People API is enabled
- Make sure your OAuth consent screen is configured

### Can't see the Sign-In button
- Make sure GOOGLE_CLIENT_ID is set in `.env`
- Check backend logs for errors
- Restart the backend after adding credentials

## For Production

When deploying to production:

1. Update redirect URI to your production URL
2. Add production URL to authorized origins
3. Change OAuth consent screen to "Production" mode
4. Update `.env` with production values
