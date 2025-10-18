#!/bin/bash
# Setup script for git hooks using pre-commit framework
# This script installs pre-commit hooks for code quality checks

set -e

echo "🔧 Setting up git hooks for VTT2Minutes..."
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed or not in PATH"
    echo "Please install uv: https://docs.astral.sh/uv/"
    exit 1
fi

# Install dependencies including pre-commit
echo "📦 Installing dependencies..."
if ! uv sync --extra dev --quiet; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
if ! uv run pre-commit install; then
    echo "❌ Failed to install pre-commit hooks"
    exit 1
fi

echo ""
echo "✅ Pre-commit hooks installed successfully!"
echo ""
echo "The hooks will now run automatically before each commit and check:"
echo "  - Code formatting (prettier for YAML/JSON/Markdown, ruff for Python)"
echo "  - Linting (ruff check)"
echo "  - Type checking (pyright)"
echo "  - Tests (pytest with testmon - only changed tests)"
echo "  - Cyclomatic complexity (lizard, CCN ≤ 10)"
echo "  - Code similarity (similarity-py)"
echo ""
echo "Note: Python checks only run when Python files are modified"
echo ""
echo "To bypass the hook temporarily, use: git commit --no-verify"
echo "To run hooks manually on all files: uv run pre-commit run --all-files"
echo ""
