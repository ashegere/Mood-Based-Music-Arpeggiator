#!/bin/bash

# AI Arpeggiator Launcher
echo "🎵 Starting AI Arpeggiator..."

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to check if a port is in use
check_port() {
    # Try lsof first (most reliable)
    if command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "❌ Port $1 is already in use"
            return 1
        else
            echo "✅ Port $1 is available"
            return 0
        fi
    # Fallback to netstat
    elif command -v netstat >/dev/null 2>&1; then
        if netstat -tln 2>/dev/null | grep -q ":$1 "; then
            echo "❌ Port $1 is already in use"
            return 1
        else
            echo "✅ Port $1 is available"
            return 0
        fi
    # Last resort - assume port is available
    else
        echo "⚠️  Cannot check port availability (lsof/netstat not available), assuming port $1 is available"
        return 0
    fi
}

# Check backend port
if ! check_port 8006; then
    echo "Backend port 8006 is in use. Please stop the conflicting service or change the port."
    exit 1
fi

# Check frontend port
if ! check_port 3000; then
    echo "Frontend port 3000 is in use. Please stop the conflicting service or change the port."
    exit 1
fi

# Start backend server
echo "🚀 Starting backend server on port 8006..."
cd backend

# Try to activate virtual environment first
if [ -d "../venv" ] && [ -f "../venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source ../venv/bin/activate 2>/dev/null || echo "⚠️  Could not activate virtual environment, using system Python"
fi

# Start FastAPI server
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8006 --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo "✅ Backend server started successfully (PID: $BACKEND_PID)"
else
    echo "❌ Backend server failed to start"
    exit 1
fi

cd ..

# Start frontend server
echo "🎨 Starting frontend server on port 3000..."
cd frontend

# Check if npm is available
if command -v npm >/dev/null 2>&1; then
    npm run dev &
    FRONTEND_PID=$!
elif command -v yarn >/dev/null 2>&1; then
    yarn dev &
    FRONTEND_PID=$!
else
    echo "❌ Neither npm nor yarn found. Please install Node.js and npm."
    exit 1
fi

# Wait for frontend to start
sleep 5

if kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "✅ Frontend server started successfully (PID: $FRONTEND_PID)"
else
    echo "❌ Frontend server failed to start"
    exit 1
fi

cd ..

echo ""
echo "🎉 AI Arpeggiator is now running!"
echo ""
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8006"
echo "📚 API Docs: http://localhost:8006/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user interrupt
trap "echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
