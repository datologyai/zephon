#!/bin/bash
# NOTE: We don't `set -e` because we want to collect and report all errors in
# one run. Instead, we track each tool's exit status in OVERALL_STATUS and
# exit with that at the end so CI / make / callers see a real failure.
set -uo pipefail

OVERALL_STATUS=0

# Run a command, preserve its behaviour, but OR its exit code into
# OVERALL_STATUS so we keep going while still failing at the end.
run_step() {
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        OVERALL_STATUS=$rc
    fi
}

# Start in the root directory  
LINTING_DIR=$(realpath $(dirname $0))
ROOT_DIR=$(realpath $LINTING_DIR/..)
cd "${ROOT_DIR}"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD_RED='\033[1;31m'
BOLD_GREEN='\033[1;32m'
BOLD_BLUE='\033[1;34m'
BOLD_YELLOW='\033[1;33m'
NC='\033[0m'

TOOL_ENV_ROOT_DIR="${LINTING_DIR}/.tools_env"

# Display help information if requested
if [ "$#" -gt 0 ] && [ "$1" == "--help" ]; then
    echo "Usage: ./linting/lint.sh [--check] [--changed-only] [--commit]"
    echo "  --check         Run in check mode (don't modify files)"
    echo "  --changed-only  Only lint files that have been changed"
    echo "  --commit        Automatically commit linting changes"
    exit 0
fi

# Check for arguments
CHECK_MODE=false
CHANGED_ONLY=false
AUTO_COMMIT=false

for arg in "$@"; do
    case $arg in
        --check)
            CHECK_MODE=true
            ;;
        --changed-only)
            CHANGED_ONLY=true
            ;;
        --commit)
            AUTO_COMMIT=true
            ;;
        --help)
            # Already handled above
            ;;
        *)
            echo -e "${RED}Invalid argument: $arg${NC}"
            echo "Run ./lint.sh --help for usage information."
            exit 1
            ;;
    esac
done

# Use main project environment
echo -e "${BOLD_YELLOW}Using main project environment...${NC}"
# Ensure dev dependencies are installed  
if [ ! -f "uv.lock" ]; then
    echo -e "${RED}Main project uv.lock not found. Please run 'uv sync --group dev' first.${NC}"
    exit 1
fi

# Set flags based on mode
if [ "$CHECK_MODE" = true ]; then
    echo -e "${YELLOW}Running in ${BOLD_RED}check${YELLOW} mode (will not modify files)...${NC}"
    RUFF_FORMAT_FLAGS="format --check"
    RUFF_CHECK_FLAGS="check"
else
    echo -e "${YELLOW}Running in ${BOLD_RED}fix${YELLOW} mode (will modify files)...${NC}"
    RUFF_FORMAT_FLAGS="format"
    RUFF_CHECK_FLAGS="check --fix"
fi

# Get changed files if needed
CHANGED_FILES=""
if [ "$CHANGED_ONLY" = true ]; then
    if [ -n "${PRE_COMMIT_FROM_REF:-}" ] && [ -n "${PRE_COMMIT_TO_REF:-}" ]; then
        # We're in a pre-push hook, get files from commits that will be pushed
        echo -e "${YELLOW}Running in pre-push mode, checking files in commits to be pushed${NC}"
        CHANGED_FILES=$(git diff --name-only --diff-filter=ACMRT "${PRE_COMMIT_FROM_REF}" "${PRE_COMMIT_TO_REF}" | grep "\.py$" || true)
    else
        # Running manually, get files from working directory
        echo -e "${YELLOW}Running in manual mode, checking files in working directory${NC}"
        # Get changed Python files
        UNCOMMITTED=$(git diff --name-only --diff-filter=ACMR | grep "\.py$" || true)
        STAGED=$(git diff --name-only --cached --diff-filter=ACMR | grep "\.py$" || true)
        UNTRACKED=$(git ls-files --others --exclude-standard | grep "\.py$" || true)
        
        # Combine all files
        CHANGED_FILES=$(echo -e "${UNCOMMITTED}\n${STAGED}\n${UNTRACKED}" | sort -u | grep -v "^\s*$" || true)
    fi
    
    if [ -z "$CHANGED_FILES" ]; then
        echo -e "${YELLOW}No Python files have been changed. Nothing to lint.${NC}"
        exit 0
    fi
    
    echo -e "${BLUE}Found $(echo "$CHANGED_FILES" | wc -l | tr -d '[:space:]') changed Python files to lint.${NC}"
fi

# Run `ruff` and `pyright` on the root directory using main project config
echo -e "${BLUE}Linting on ${ROOT_DIR}${NC}"
# shellcheck disable=SC2164
cd "${ROOT_DIR}"

# If we're in changed-only mode, then only run it on the changed files
if [ "$CHANGED_ONLY" = true ]; then
    if [ -z "$CHANGED_FILES" ]; then
        echo -e "${YELLOW}No changed Python files. Skipping.${NC}"
        exit 0
    fi

    # Run `ruff` on the changed files using main project config
    echo -e "${BLUE}Formatting changed files...${NC}"
    run_step uv run ruff $RUFF_FORMAT_FLAGS $CHANGED_FILES
    echo -e "${BLUE}Linting changed files...${NC}"
    run_step uv run ruff $RUFF_CHECK_FLAGS $CHANGED_FILES

    # Run `pyright` on the changed files using main project config
    echo -e "${BLUE}Checking changed files...${NC}"
    run_step uv run pyright $CHANGED_FILES
else
    # Run on all files using main project config
    echo -e "${BLUE}Formatting all files...${NC}"
    run_step uv run ruff $RUFF_FORMAT_FLAGS . --exclude zephon/_version.py --exclude docs/
    echo -e "${BLUE}Linting all files...${NC}"
    run_step uv run ruff $RUFF_CHECK_FLAGS . --exclude zephon/_version.py --exclude docs/
    # Run `pyright` on all files using main project config
    echo -e "${BLUE}Checking all files...${NC}"
    run_step uv run pyright
fi

echo -e "${BOLD_BLUE}Finished running all linting component(s).${NC}"

if [ $OVERALL_STATUS -ne 0 ]; then
    echo -e "${BOLD_RED}One or more linting steps reported errors (see above).${NC}"
    exit $OVERALL_STATUS
fi

if [ "$CHECK_MODE" = true ]; then
    echo -e "${BOLD_GREEN}Linting checks completed successfully!${NC}"
else
    echo -e "${BOLD_GREEN}Linting and formatting completed successfully!${NC}"

    # Auto-commit changes if requested
    if [ "$AUTO_COMMIT" = true ]; then
        echo -e "${BLUE}Staging and committing linted files...${NC}"
        
        if [ "$CHANGED_ONLY" = true ]; then
            # Stage only the files we linted
            if [ -n "$CHANGED_FILES" ]; then
                echo "$CHANGED_FILES" | while IFS= read -r file; do
                    if [ -f "$file" ]; then
                        git add "$file" || echo -e "${YELLOW}Warning: Could not add $file${NC}"
                    fi
                done
            fi
        else
            # Find all Python files that were modified by the linting
            MODIFIED_FILES=$(git diff --name-only | grep "\.py$" || true)
            if [ -n "$MODIFIED_FILES" ]; then
                echo "$MODIFIED_FILES" | while IFS= read -r file; do
                    if [ -f "$file" ]; then
                        git add "$file" || echo -e "${YELLOW}Warning: Could not add $file${NC}"
                    fi
                done
            fi
        fi
        
        # Commit if there are staged changes
        if ! git diff --cached --quiet; then
            git commit -m "Auto-lint: Apply formatting and fix style issues" || {
                echo -e "${YELLOW}Failed to commit changes.${NC}"
            }
            echo -e "${BOLD_GREEN}Linted files committed successfully!${NC}"
        else
            echo -e "${YELLOW}No changes to commit.${NC}"
        fi
    fi
fi
exit $OVERALL_STATUS