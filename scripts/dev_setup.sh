#!/usr/bin/env bash
set -euo pipefail

# Simple developer setup for Zephon
# - Creates a local env via uv
# - Installs base + dev + test deps
# - Ensures linting scripts are ready
#
# Usage:
#   bash scripts/dev_setup.sh           # install dev+test groups
#   bash scripts/dev_setup.sh --with-docs  # also install docs group

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

want_docs=false
if [[ ${1:-} == "--with-docs" ]]; then
  want_docs=true
fi

# Resolve repo root
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd -P)
cd "${ROOT_DIR}"

echo -e "${BLUE}Zephon developer setup starting...${NC}"

# Check uv
if ! command -v uv >/dev/null 2>&1; then
  echo -e "${RED}uv not found.${NC}"
  echo -e "${YELLOW}Install uv: https://github.com/astral-sh/uv${NC}"
  exit 1
fi

# Ensure usable Python: prefer system 3.10+, else use uv-managed Python
UV_SYNC_PY_ARGS=()
sys_pyver=$(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "0.0")
sys_ok=$(python3 - <<'PY'
import sys
print(int((sys.version_info.major, sys.version_info.minor) >= (3,10)))
PY
)
if [[ "$sys_ok" != "1" ]]; then
  want_py="3.12"
  echo -e "${YELLOW}System Python ${sys_pyver} < 3.10. Using uv-managed Python ${want_py}.${NC}"
  uv python install "${want_py}"
  UV_SYNC_PY_ARGS=(-p "${want_py}")
fi

echo -e "${BLUE}Installing dependencies (base + dev + test)...${NC}"
uv sync ${UV_SYNC_PY_ARGS[@]:-} --group dev --group test

if [[ "$want_docs" == true ]]; then
  echo -e "${BLUE}Also installing docs dependencies...${NC}"
  uv sync ${UV_SYNC_PY_ARGS[@]:-} --group docs
fi

# Ensure lint scripts are ready
if [[ -f ./linting/install_linter.sh ]]; then
  chmod +x ./linting/install_linter.sh || true
  chmod +x ./linting/lint.sh || true
  chmod +x ./lcp.sh || true
  echo -e "${BLUE}Initializing lint tools...${NC}"
  ./linting/install_linter.sh || true
fi

echo -e "${GREEN}Setup complete!${NC}"
echo
echo -e "${BLUE}Next steps:${NC}"
echo "  - Run tests:        uv run pytest -q"
echo "  - Lint/format:      ./linting/lint.sh  (or --check)"
echo "  - Build docs:       uv sync --group docs && make -C docs clean html"
echo "  - Serve docs local: make -C docs host"