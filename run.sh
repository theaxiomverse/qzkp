#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Setting up virtual environment...${NC}"
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo -e "${GREEN}Installing dependencies...${NC}"
pip install -r requirements.txt
pip install -e .

echo -e "${GREEN}Running tests...${NC}"
pytest tests/test_qzkp.py -v

echo -e "${GREEN}Complete! Check the test results above.${NC}"
