#!/bin/bash
set -euo pipefail

# Work from the root directory
LINTING_DIR=$(realpath $(dirname $0))
ROOT_DIR=$(realpath $LINTING_DIR/..)
cd "${ROOT_DIR}"

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'

echo -e "${BLUE}Setting up main project environment with dev dependencies..."

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo -e "${GREEN}Using uv package manager"
else
    echo -e "${YELLOW}uv package manager not found."
    echo -e "${YELLOW}Please install uv via"
    echo -e "${YELLOW}https://github.com/astral-sh/uv"
    exit 1
fi

install_deps() {
    echo -e "${BLUE}Installing main project with dev and test dependencies..."
    uv sync --group dev --group test
    echo -e "${GREEN}Main project environment ready with linting tools"
}

setup_permissions() {
    echo -e "${BLUE}Setting up script permissions..."
    chmod +x ./linting/lint.sh
    chmod +x ./lcp.sh
    echo -e "${GREEN}Script permissions set successfully"
}

install_deps
setup_permissions