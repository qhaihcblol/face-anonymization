# 🚀 Quick Deploy Script for Vast.ai
# Run this script to prepare and deploy FaceGuard to Vast.ai

set -e

echo "========================================"
echo "  FaceGuard Vast.ai Deployment Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker installed${NC}"
    
    if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
        echo -e "${YELLOW}⚠ Docker Compose not found (optional for local testing)${NC}"
    else
        echo -e "${GREEN}✓ Docker Compose available${NC}"
    fi
    
    if ! command -v vastai &> /dev/null; then
        echo -e "${YELLOW}⚠ Vast.ai CLI not installed (optional, can deploy via web UI)${NC}"
        echo "   Install with: pip install vastai"
    else
        echo -e "${GREEN}✓ Vast.ai CLI available${NC}"
    fi
    
    echo ""
}

# Create .env file from template
setup_env() {
    echo "Setting up environment variables..."
    
    if [ -f ".env" ]; then
        echo -e "${YELLOW}⚠ .env file already exists.${NC}"
        read -p "Do you want to overwrite it? (y/N): " confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            echo "Skipping .env setup."
            echo ""
            return
        fi
    fi
    
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env from template${NC}"
    
    echo ""
    echo -e "${YELLOW}Please edit .env file with your credentials:${NC}"
    echo "  - SECRET_KEY (generate with: openssl rand -hex 32)"
    echo "  - DATABASE_URL"
    echo "  - R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET"
    echo "  - KAGGLE_USERNAME, KAGGLE_KEY"
    echo ""
    read -p "Press Enter after you've updated .env..."
    
    echo ""
}

# Build Docker image
build_image() {
    echo "Building Docker image..."
    echo "This may take 10-20 minutes on first build."
    echo ""
    
    IMAGE_NAME="${DOCKER_IMAGE_NAME:-faceguard:latest}"
    
    docker build --platform linux/amd64 -t "$IMAGE_NAME" .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Docker image built successfully: $IMAGE_NAME${NC}"
    else
        echo -e "${RED}❌ Docker build failed${NC}"
        exit 1
    fi
    
    echo ""
}

# Push to registry
push_image() {
    echo "Pushing image to registry..."
    
    read -p "Enter your Docker Hub username (or press Enter to skip): " dockerhub_user
    
    if [ -z "$dockerhub_user" ]; then
        echo "Skipping push step."
        echo ""
        return
    fi
    
    TAGGED_IMAGE="$dockerhub_user/faceguard:latest"
    
    echo "Tagging image as: $TAGGED_IMAGE"
    docker tag faceguard:latest "$TAGGED_IMAGE"
    
    echo "Pushing to Docker Hub..."
    docker push "$TAGGED_IMAGE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Image pushed successfully: $TAGGED_IMAGE${NC}"
        echo ""
        echo "You can now deploy this image on Vast.ai:"
        echo "  Image: $TAGGED_IMAGE"
    else
        echo -e "${RED}❌ Docker push failed${NC}"
        echo "Make sure you're logged in: docker login"
    fi
    
    echo ""
}

# Generate Vast.ai deployment command
generate_vast_command() {
    echo "Generating Vast.ai deployment command..."
    echo ""
    
    cat << 'EOF'
================================================================================
  VAST.AI DEPLOYMENT INSTRUCTIONS
================================================================================

Option 1: Deploy via Web UI (Recommended for first time)
---------------------------------------------------------
1. Go to: https://cloud.vast.ai/create/
2. Search for GPU: RTX 3090, RTX 4090, or A100
3. Click "Rent" on a suitable instance
4. Configure:
   - Image: your-dockerhub-username/faceguard:latest
   - Disk: 100GB+ NVMe
   - Ports: 8000:8000 3000:3000 5432:5432
   - Environment Variables: Copy contents of your .env file
   - Extra Args: --gpus all
5. Click "Launch"

Option 2: Deploy via CLI
------------------------
Run this command (replace YOUR_DOCKERHUB_USERNAME):

vastai create instance \\
  --image YOUR_DOCKERHUB_USERNAME/faceguard:latest \\
  --gpu-ram 24 \\
  --ram 32 \\
  --disk 100 \\
  --cuda 12.4 \\
  --direct-port-count 3 \\
  --ports "8000:8000 3000:3000 5432:5432" \\
  --env SECRET_KEY="$(grep SECRET_KEY .env | cut -d'=' -f2)" \\
  --env DATABASE_URL="$(grep DATABASE_URL .env | cut -d'=' -f2)" \\
  --env R2_ENDPOINT_URL="$(grep R2_ENDPOINT_URL .env | cut -d'=' -f2)" \\
  --env R2_ACCESS_KEY_ID="$(grep R2_ACCESS_KEY_ID .env | cut -d'=' -f2)" \\
  --env R2_SECRET_ACCESS_KEY="$(grep R2_SECRET_ACCESS_KEY .env | cut -d'=' -f2)" \\
  --env R2_BUCKET="$(grep R2_BUCKET .env | cut -d'=' -f2)" \\
  --env KAGGLE_USERNAME="$(grep KAGGLE_USERNAME .env | cut -d'=' -f2)" \\
  --env KAGGLE_KEY="$(grep KAGGLE_KEY .env | cut -d'=' -f2)" \\
  --env CORS_ORIGINS="http://localhost:3000,http://0.0.0.0:3000" \\
  --onstart "sleep 30 && curl http://localhost:8000/health"

Note: First run will download ONNX models (~5-10 minutes)

================================================================================
EOF
    
    echo ""
}

# Test locally (optional)
test_locally() {
    echo "Do you want to test locally with Docker Compose first? (y/N)"
    read -p "> " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        echo "Starting local test environment..."
        
        if command -v docker compose &> /dev/null; then
            docker compose up -d postgres
            echo -e "${GREEN}✓ PostgreSQL started${NC}"
            echo ""
            echo "Now run: docker compose up backend frontend"
            echo "Or: docker run --gpus all -p 8000:8000 -p 3000:3000 --env-file .env faceguard:latest"
        else
            echo -e "${YELLOW}Docker Compose not available. Use: docker run --gpus all -p 8000:8000 -p 3000:3000 --env-file .env faceguard:latest${NC}"
        fi
    fi
    
    echo ""
}

# Main execution
main() {
    check_prerequisites
    setup_env
    build_image
    push_image
    generate_vast_command
    test_locally
    
    echo "========================================"
    echo -e "${GREEN}✓ Deployment preparation complete!${NC}"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "1. Follow the Vast.ai deployment instructions above"
    echo "2. Wait for instance to start (5-10 minutes)"
    echo "3. Access your app at: http://<vast-ip>:3000"
    echo "4. API docs: http://<vast-ip>:8000/docs"
    echo ""
    echo "Need help? Check DEPLOYMENT.md for detailed guide."
    echo ""
}

# Run main function
main "$@"
