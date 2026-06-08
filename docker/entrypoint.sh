#!/bin/bash
set -e

# FaceGuard Docker Entrypoint Script
# Usage: docker run ... faceguard [backend|frontend|all]

MODE=${1:-all}

echo "=== FaceGuard Startup ==="
echo "Mode: $MODE"
echo "CUDA Available: $(python -c 'import onnxruntime as ort; print(ort.get_device())' 2>/dev/null || echo 'Unknown')"

# Function to download ONNX models if not present
download_models_if_needed() {
    if [ ! -f /app/ai_core/face_detection/onnx/retinaface_best.onnx ]; then
        echo "ONNX models not found. Downloading from Kaggle..."
        if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
            python /app/scripts/download_onnx_files.py || echo "Warning: Model download failed. Some features may not work."
        else
            echo "Warning: KAGGLE_USERNAME or KAGGLE_KEY not set. Skipping model download."
        fi
    else
        echo "ONNX models already present."
    fi
}

# Function to start backend
start_backend() {
    echo "Starting FastAPI Backend on port 8000..."
    
    # Download models if needed
    download_models_if_needed
    
    cd /app/webapp/backend
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
}

# Function to start frontend
start_frontend() {
    echo "Starting Next.js Frontend on port 3000..."
    cd /app/webapp/frontend
    # Ensure backend URL is configured
    export NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL:-"http://localhost:8000/api"}
    exec npx next start -H 0.0.0.0 -p 3000
}

# Function to start both (background backend, foreground frontend)
start_all() {
    echo "Starting both Backend and Frontend..."
    
    # Start backend in background
    start_backend &
    BACKEND_PID=$!
    echo "Backend started with PID: $BACKEND_PID"
    
    # Wait for backend to be ready
    echo "Waiting for backend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "Backend is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "Warning: Backend may not be ready yet"
        fi
        sleep 2
    done
    
    # Start frontend in foreground
    start_frontend
    
    # Wait for backend process
    wait $BACKEND_PID
}

case "$MODE" in
    backend)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    all)
        start_all
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [backend|frontend|all]"
        exit 1
        ;;
esac
