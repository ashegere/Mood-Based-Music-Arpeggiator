# PostgreSQL Database Setup Guide

This guide will help you set up PostgreSQL for the AI Arpeggiator application.

## Option 1: Install PostgreSQL Locally (macOS)

### Using Homebrew:

```bash
# Install PostgreSQL
brew install postgresql@14

# Start PostgreSQL service
brew services start postgresql@14

# Create database
createdb arpeggiator

# (Optional) Set password for postgres user
psql postgres
\password postgres
# Enter: postgres (or your preferred password)
\q
```

### Using Postgres.app:

1. Download Postgres.app from https://postgresapp.com/
2. Open the app and click "Initialize" to start the server
3. Open terminal and create the database:
   ```bash
   /Applications/Postgres.app/Contents/Versions/latest/bin/createdb arpeggiator
   ```

## Option 2: Use Docker

```bash
# Run PostgreSQL in Docker
docker run --name arpeggiator-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=arpeggiator \
  -p 5432:5432 \
  -d postgres:14

# Check if it's running
docker ps
```

## Verify Connection

Test the database connection:

```bash
psql -h localhost -U postgres -d arpeggiator
# Password: postgres
```

If successful, you should see:
```
psql (14.x)
Type "help" for help.

arpeggiator=#
```

Type `\q` to exit.

## Environment Configuration

The application is currently configured with these defaults in `backend/app/config.py`:

```python
DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/arpeggiator"
```

### To use custom credentials:

Create a `.env` file in the project root:

```bash
# .env
DATABASE_URL=postgresql://your_user:your_password@localhost:5432/your_database
SECRET_KEY=your-super-secret-key-change-this
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

## Verify Database Tables

After starting the backend server, the tables should be created automatically. To verify:

```bash
psql -h localhost -U postgres -d arpeggiator -c "\dt"
```

You should see a `users` table.

## Common Issues

### Port 5432 already in use
```bash
# Check what's using the port
lsof -i :5432

# Stop conflicting PostgreSQL service
brew services stop postgresql
# or
docker stop arpeggiator-db
```

### Connection refused
- Make sure PostgreSQL is running: `brew services list` or `docker ps`
- Check firewall settings
- Verify the host and port are correct

### Permission denied
- Make sure the user has correct permissions
- Try creating a new superuser:
  ```bash
  createuser -s your_username
  ```

## Google OAuth Setup (Optional)

To enable Google Sign-In:

1. Go to https://console.cloud.google.com/
2. Create a new project or select existing one
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add authorized redirect URI: `http://localhost:8006/api/auth/google/callback`
6. Copy Client ID and Client Secret to your `.env` file
7. Restart the backend server

## Testing Authentication

Once the database is set up, you can test authentication:

1. Navigate to http://localhost:3000/signup
2. Create a new account
3. Check the database:
   ```bash
   psql -h localhost -U postgres -d arpeggiator
   SELECT * FROM users;
   ```

You should see your newly created user!
