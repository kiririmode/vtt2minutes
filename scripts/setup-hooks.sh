#!/bin/bash
# Setup script for git hooks
# This script copies the pre-commit hook to the .git/hooks directory

set -e

echo "🔧 Setting up git hooks for VTT2Minutes..."

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Pre-commit hook content
HOOK_CONTENT="#!/bin/bash
# Pre-commit hook for VTT2Minutes
# Runs code quality checks before each commit

set -e

echo \"🔍 Running pre-commit quality checks...\"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e \"\${GREEN}✓\${NC} \$1\"
}

print_error() {
    echo -e \"\${RED}✗\${NC} \$1\"
}

print_warning() {
    echo -e \"\${YELLOW}⚠\${NC} \$1\"
}

# Check if uv is available
if ! command -v uv &> /dev/null; then
    print_error \"uv is not installed or not in PATH\"
    echo \"Please install uv: https://docs.astral.sh/uv/\"
    exit 1
fi

# Change to repository root
cd \"\$(git rev-parse --show-toplevel)\"

# Ensure dependencies are installed
echo \"📦 Checking dependencies...\"
if ! uv sync --extra dev --quiet; then
    print_error \"Failed to sync dependencies\"
    exit 1
fi
print_status \"Dependencies synced\"

# 1. Code formatting check
echo \"\"
echo \"🎨 Checking code formatting with ruff...\"
if ! uv run --frozen ruff format --check .; then
    print_error \"Code formatting check failed\"
    echo \"Run: uv run ruff format .\"
    exit 1
fi
print_status \"Code formatting passed\"

# 2. Linting check
echo \"\"
echo \"🔍 Running linting with ruff...\"
if ! uv run --frozen ruff check .; then
    print_error \"Linting check failed\"
    echo \"Run: uv run ruff check . --fix\"
    exit 1
fi
print_status \"Linting passed\"

# 3. Type checking
echo \"\"
echo \"🏷️  Running type check with pyright...\"
if ! uv run --frozen pyright; then
    print_error \"Type checking failed\"
    echo \"Fix type errors and try again\"
    exit 1
fi
print_status \"Type checking passed\"

# 4. Run tests
echo \"\"
echo \"🧪 Running tests with pytest...\"
if ! PYTEST_DISABLE_PLUGIN_AUTOLOAD=\"\" uv run --frozen pytest --tb=short --quiet; then
    print_error \"Tests failed\"
    echo \"Fix failing tests and try again\"
    exit 1
fi
print_status \"All tests passed\"

# 5. Cyclomatic complexity check
echo \"\"
echo \"📊 Checking cyclomatic complexity with lizard...\"
if ! uv run --frozen lizard src/vtt2minutes --CCN 10; then
    print_error \"Cyclomatic complexity check failed\"
    echo \"Functions with CCN > 10 found. Please refactor complex functions.\"
    echo \"Run: uv run lizard src/vtt2minutes --CCN 10\"
    exit 1
fi
print_status \"Complexity check passed\"

# Success message
echo \"\"
echo -e \"\${GREEN}🎉 All quality checks passed! Proceeding with commit...\${NC}\"
echo \"\"
"

# Create hooks directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/.git/hooks"

# Write the pre-commit hook
echo "$HOOK_CONTENT" > "$PROJECT_ROOT/.git/hooks/pre-commit"

# Make it executable
chmod +x "$PROJECT_ROOT/.git/hooks/pre-commit"

echo "✅ Pre-commit hook installed successfully!"
echo ""
echo "The hook will now run automatically before each commit and check:"
echo "  - Code formatting (ruff format)"
echo "  - Linting (ruff check)"
echo "  - Type checking (pyright)"
echo "  - Tests (pytest)"
echo "  - Cyclomatic complexity (lizard, CCN ≤ 10)"
echo ""
echo "To bypass the hook temporarily, use: git commit --no-verify"
echo ""