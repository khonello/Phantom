#!/bin/bash
# Run all tests and checks for roop-cam
# 1. Linting with flake8
# 2. Type checking with mypy
# 3. Integration test with example files

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}roop-cam Test Suite${NC}"
echo "==================="
echo ""

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo "Consider running: source venv/bin/activate"
    echo ""
fi

# Run flake8
echo "1. Running flake8 (linting)..."
if flake8 pipeline.py roop; then
    echo -e "${GREEN}✓ flake8 passed${NC}"
else
    echo -e "${RED}✗ flake8 failed${NC}"
    exit 1
fi
echo ""

# Run mypy
echo "2. Running mypy (type checking)..."
if mypy pipeline.py roop; then
    echo -e "${GREEN}✓ mypy passed${NC}"
else
    echo -e "${RED}✗ mypy failed${NC}"
    exit 1
fi
echo ""

# Run integration test
echo "3. Running integration test..."
if [ ! -f ".github/examples/source.jpg" ] || [ ! -f ".github/examples/target.mp4" ]; then
    echo -e "${YELLOW}⚠ Example files not found, skipping integration test${NC}"
    echo "  Expected: .github/examples/source.jpg and .github/examples/target.mp4"
else
    TEST_OUTPUT=".test_output.mp4"
    echo "Processing example files..."
    if python pipeline.py \
        -s .github/examples/source.jpg \
        -t .github/examples/target.mp4 \
        -o "$TEST_OUTPUT"; then
        echo -e "${GREEN}✓ Integration test passed${NC}"

        # Clean up test output
        if [ -f "$TEST_OUTPUT" ]; then
            rm "$TEST_OUTPUT"
        fi
    else
        echo -e "${RED}✗ Integration test failed${NC}"
        exit 1
    fi
fi
echo ""

echo -e "${GREEN}All tests passed! ✓${NC}"
echo ""
