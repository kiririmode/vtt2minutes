#!/bin/bash

# VTT2Minutes - Code Similarity Checker
# Uses similarity-py to detect duplicate or similar functions in the codebase

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Default values
THRESHOLD=0.7
MIN_LINES=8
PRINT_CODE=false
SOURCE_DIRS=("src/vtt2minutes" "tests")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -m|--min-lines)
            MIN_LINES="$2"
            shift 2
            ;;
        -p|--print)
            PRINT_CODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --threshold FLOAT    Similarity threshold (0.0-1.0) [default: 0.7]"
            echo "  -m, --min-lines INT      Minimum lines for functions [default: 8]"
            echo "  -p, --print              Print code in output"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                       # Run with default settings"
            echo "  $0 -t 0.8 -p            # Use 0.8 threshold and print code"
            echo "  $0 --threshold 0.9       # Use high similarity threshold"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if similarity-py is installed
if ! command -v similarity-py &> /dev/null; then
    print_error "similarity-py is not installed"
    echo ""
    echo "To install similarity-py:"
    echo "  cargo install similarity-py"
    exit 1
fi

# Ensure we're in the project root
for SOURCE_DIR in "${SOURCE_DIRS[@]}"; do
    if [ ! -d "$SOURCE_DIR" ]; then
        print_error "Source directory '$SOURCE_DIR' not found"
        echo "Please run this script from the project root directory"
        exit 1
    fi
done

echo "🔍 VTT2Minutes Code Similarity Analysis"
echo "======================================="
echo ""
print_info "Analyzing directories: ${SOURCE_DIRS[*]}"
print_info "Similarity threshold: $THRESHOLD"
print_info "Minimum function lines: $MIN_LINES"
print_info "Print code: $PRINT_CODE"
echo ""

# Run similarity analysis for each directory
OVERALL_SUCCESS=true

for SOURCE_DIR in "${SOURCE_DIRS[@]}"; do
    print_info "Analyzing directory: $SOURCE_DIR"
    echo ""
    
    # Build similarity-py command
    SIMILARITY_CMD="similarity-py $SOURCE_DIR --threshold $THRESHOLD --min-lines $MIN_LINES"
    
    if [ "$PRINT_CODE" = true ]; then
        SIMILARITY_CMD="$SIMILARITY_CMD --print"
    fi
    
    # Execute the similarity check
    if ! $SIMILARITY_CMD; then
        print_error "Similarity analysis failed for $SOURCE_DIR"
        OVERALL_SUCCESS=false
    fi
    echo ""
done

if [ "$OVERALL_SUCCESS" = true ]; then
    print_status "Similarity analysis completed successfully for all directories"
    echo ""
    print_info "Summary:"
    echo "  - Functions with similarity ≥ $THRESHOLD were reported above"
    echo "  - Consider refactoring highly similar functions to reduce duplication"
    echo "  - Some similarity is expected in error handling and boilerplate code"
else
    print_error "Similarity analysis failed for one or more directories"
    exit 1
fi

echo ""
print_info "Code quality recommendations:"
echo "  1. Review functions with >85% similarity for potential refactoring"
echo "  2. Extract common patterns into shared utility functions"
echo "  3. Consider using design patterns to reduce code duplication"
echo "  4. Maintain similarity below 80% for better maintainability"