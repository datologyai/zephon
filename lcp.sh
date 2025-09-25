#!/bin/bash

set -e

if [ $# -eq 0 ]; then
    echo "Error: Commit message is required"
    echo "Usage: $0 \"commit message\""
    exit 1
fi

COMMIT_MESSAGE="$1"

# Store the original directory
ORIGINAL_DIR=$(pwd)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting lint_commit_push workflow..."
echo "Original directory: $ORIGINAL_DIR"
echo "Script directory: $SCRIPT_DIR"

# Change to the script directory (src/python)
cd "$SCRIPT_DIR"

# Function to return to original directory on exit
cleanup() {
    if [ "$ORIGINAL_DIR" != "$SCRIPT_DIR" ]; then
        echo "Returning to original directory: $ORIGINAL_DIR"
        cd "$ORIGINAL_DIR"
    fi
}

# Set trap to ensure we return to original directory even on error
trap cleanup EXIT

echo "Step 1: Running linting/lint.sh..."
if [ ! -f "./linting/lint.sh" ]; then
    echo "Error: linting/lint.sh not found in $SCRIPT_DIR"
    exit 1
fi

if ! ./linting/lint.sh; then
    echo "Error: Linting failed"
    exit 1
fi

echo "Step 2: Adding changes to git..."
if ! git add .; then
    echo "Error: Failed to add changes to git"
    exit 1
fi

if git diff --cached --quiet; then
    echo "No changes to commit"
    exit 0
fi

echo "Step 3: Committing changes..."
if ! git commit -m "$COMMIT_MESSAGE"; then
    echo "Error: Failed to commit changes"
    exit 1
fi

echo "Step 4: Pushing to remote..."
CURRENT_BRANCH=$(git branch --show-current)
if [ -z "$CURRENT_BRANCH" ]; then
    echo "Error: Could not determine current branch"
    exit 1
fi

if ! git push origin "$CURRENT_BRANCH"; then
    echo "Error: Failed to push to remote branch '$CURRENT_BRANCH'"
    exit 1
fi

echo "✅ Successfully completed: lint → add → commit → push"
echo "Branch: $CURRENT_BRANCH"
echo "Commit message: $COMMIT_MESSAGE"