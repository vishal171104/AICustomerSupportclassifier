#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "--------------------------------------------------------"
echo "🚀 Customer Support Ticket AI - Research Demo Launcher"
echo "--------------------------------------------------------"

# Start the FastAPI backend in the background
echo "[1/2] Starting FastAPI backend on http://localhost:8000..."
python3 "$PROJECT_ROOT/api/main.py" &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 3

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null
then
   echo "✅ Backend is running (PID: $BACKEND_PID)"
else
   echo "❌ Backend failed to start. Check if port 8000 is occupied."
   exit 1
fi

# Open the UI in the default browser
UI_PATH="file://$PROJECT_ROOT/ui/index.html"
echo "[2/2] Opening UI in browser: $UI_PATH"
open "$UI_PATH"

echo "--------------------------------------------------------"
echo "Press Ctrl+C to stop the backend."
echo "--------------------------------------------------------"

# Wait for user to terminate
wait $BACKEND_PID
