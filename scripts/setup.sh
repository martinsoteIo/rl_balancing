#!/bin/bash

# Exit immediately if a command fails
set -e

GREEN='\033[1;32m'
CYAN='\033[1;36m'
DARK_CYAN='\033[0;34m'
NC='\033[0m'
LINE="================================================================================================================="

# Navigate to your workspace
cd ~/workspace

# Install dependencies
echo -e "\n${CYAN}${LINE}"
echo "⚙️  Installing dependencies..."
echo -e "${CYAN}${LINE}${DARK_CYAN}"
pip install ~/src/rl_policies -e .

echo -e "\n${GREEN}${LINE}"
echo "🎉 Build complete!"
echo -e "${GREEN}${LINE}${NC}"