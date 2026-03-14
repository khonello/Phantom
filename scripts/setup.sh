#!/bin/bash
# Setup script for roop-cam on Linux and macOS
# Creates virtual environment and installs dependencies

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}roop-cam Setup Script${NC}"
echo "===================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    echo -e "${RED}Error: Python 3.9 or higher required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Check FFmpeg
echo "Checking FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}⚠ FFmpeg not found${NC}"
    echo "Install with:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  brew install ffmpeg"
    else
        echo "  sudo apt-get install ffmpeg"
    fi
    echo ""
    read -p "Continue without FFmpeg? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ FFmpeg found${NC}"
fi
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install dependencies
echo "Installing dependencies..."
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    REQUIREMENTS="requirements-pipeline-gpu.txt"
    echo -e "${GREEN}CUDA detected — using GPU requirements${NC}"
else
    REQUIREMENTS="requirements-pipeline-cpu.txt"
    echo -e "${YELLOW}No CUDA detected — using CPU requirements${NC}"
fi

pip install -r "$REQUIREMENTS"
echo -e "${GREEN}✓ Dependencies installed from $REQUIREMENTS${NC}"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import insightface; print('InsightFace: OK')"
echo -e "${GREEN}✓ Key dependencies verified${NC}"
echo ""

# Report CUDA status
echo "Checking CUDA support..."
python3 -c "import torch; cuda_available = torch.cuda.is_available(); print(f'CUDA available: {cuda_available}')"
if [ "$REQUIREMENTS" = "requirements-pipeline-gpu.txt" ]; then
    echo -e "${GREEN}✓ GPU requirements installed${NC}"
else
    echo -e "${YELLOW}⚠ CPU-only mode (use requirements-pipeline-gpu.txt for GPU support)${NC}"
fi
echo ""

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Quick start:"
echo "  source venv/bin/activate    # Activate virtual environment"
echo "  python pipeline.py               # Launch GUI"
echo "  python pipeline.py --help        # See CLI options"
echo ""
