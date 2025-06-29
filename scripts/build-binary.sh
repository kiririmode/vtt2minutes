#!/bin/bash
# Binary build script for VTT2Minutes
# Builds standalone executables using PyInstaller

set -e

echo "🔨 Building VTT2Minutes binary..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if uv is available
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed or not in PATH"
    echo "Please install uv: https://docs.astral.sh/uv/"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Parse command line arguments
CLEAN=false
PLATFORM=""
OUTPUT_DIR="dist"

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean         Clean build artifacts before building"
            echo "  --platform STR  Target platform (linux, windows, current)"
            echo "  --output-dir    Output directory for binaries (default: dist)"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Build for current platform"
            echo "  $0 --clean           # Clean build for current platform"
            echo "  $0 --platform linux  # Build for Linux"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Detect platform if not specified
if [[ -z "$PLATFORM" ]]; then
    case "$(uname -s)" in
        Linux*)     PLATFORM="linux";;
        MINGW*|CYGWIN*|MSYS*) PLATFORM="windows";;
        Darwin*)    PLATFORM="macos";;
        *)          PLATFORM="current";;
    esac
    print_info "Auto-detected platform: $PLATFORM"
fi

# Clean build artifacts if requested
if [[ "$CLEAN" == "true" ]]; then
    print_info "Cleaning build artifacts..."
    rm -rf build dist __pycache__ *.spec.bak
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    print_status "Cleaned build artifacts"
fi

# Ensure dependencies are installed
print_info "Installing dependencies..."
if ! uv sync --extra dev --quiet; then
    print_error "Failed to sync dependencies"
    exit 1
fi
print_status "Dependencies installed"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build binary
print_info "Building binary for $PLATFORM..."
echo ""

# Set platform-specific options
PYINSTALLER_ARGS=""
BINARY_NAME="vtt2minutes"

case "$PLATFORM" in
    windows)
        BINARY_NAME="vtt2minutes.exe"
        PYINSTALLER_ARGS="--target-arch=x86_64"
        ;;
    linux)
        PYINSTALLER_ARGS="--target-arch=x86_64"
        ;;
    macos)
        PYINSTALLER_ARGS="--target-arch=x86_64"
        ;;
esac

# Run PyInstaller
print_info "Running PyInstaller..."
if ! uv run pyinstaller vtt2minutes.spec $PYINSTALLER_ARGS --distpath "$OUTPUT_DIR" --workpath build; then
    print_error "PyInstaller build failed"
    exit 1
fi

# Check if binary was created
BINARY_PATH="$OUTPUT_DIR/$BINARY_NAME"
if [[ ! -f "$BINARY_PATH" ]]; then
    print_error "Binary not found at $BINARY_PATH"
    exit 1
fi

# Make binary executable (Linux/macOS)
if [[ "$PLATFORM" != "windows" ]]; then
    chmod +x "$BINARY_PATH"
fi

# Get binary size
BINARY_SIZE=$(du -h "$BINARY_PATH" | cut -f1)

# Test binary
print_info "Testing binary..."
if "$BINARY_PATH" --help > /dev/null 2>&1; then
    print_status "Binary test passed"
else
    print_warning "Binary test failed, but binary was created"
fi

# Success message
echo ""
print_status "Binary build completed successfully!"
echo ""
echo -e "${BLUE}📦 Binary details:${NC}"
echo "  Platform: $PLATFORM"
echo "  Location: $BINARY_PATH"
echo "  Size: $BINARY_SIZE"
echo ""
echo -e "${BLUE}🚀 Usage:${NC}"
echo "  $BINARY_PATH --help"
echo "  $BINARY_PATH meeting.vtt"
echo ""