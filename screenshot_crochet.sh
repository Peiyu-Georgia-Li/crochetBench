#!/bin/bash
# Convenience wrapper for capturing CrochetPARADE screenshots

# Default values
WAIT_TIME=10
ADDITIONAL_WAIT=5
OUTPUT="crochet_screenshot.png"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Show usage
usage() {
    echo "Usage: $0 PATTERN_FILE [OPTIONS]"
    echo ""
    echo "Capture screenshot of CrochetPARADE rendering"
    echo ""
    echo "Arguments:"
    echo "  PATTERN_FILE           Path to the crochet pattern txt file (required)"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE      Output screenshot filename (default: crochet_screenshot.png)"
    echo "  -w, --wait TIME        Wait time for rendering in seconds (default: 10)"
    echo "  -a, --additional TIME  Additional wait time in seconds (default: 5)"
    echo "  -f, --full-page        Capture full page instead of canvas only"
    echo "  --headless             Run browser in headless mode (default: visible)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 example_pattern_work.txt"
    echo "  $0 pattern.txt -o output.png -w 20"
    echo "  $0 pattern.txt --full-page"
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

# Parse arguments
PATTERN_FILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -w|--wait)
            WAIT_TIME="$2"
            shift 2
            ;;
        -a|--additional)
            ADDITIONAL_WAIT="$2"
            shift 2
            ;;
        -f|--full-page)
            EXTRA_ARGS+=("--full-page")
            shift
            ;;
        --headless)
            EXTRA_ARGS+=("--headless")
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            if [ -z "$PATTERN_FILE" ]; then
                PATTERN_FILE="$1"
            else
                echo "Error: Multiple pattern files specified"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if pattern file is provided
if [ -z "$PATTERN_FILE" ]; then
    echo "Error: Pattern file not specified"
    usage
    exit 1
fi

# Check if pattern file exists
if [ ! -f "$PATTERN_FILE" ]; then
    echo "Error: Pattern file not found: $PATTERN_FILE"
    exit 1
fi

# Run the Python script
echo "Capturing screenshot from CrochetPARADE..."
python3 "$SCRIPT_DIR/capture_crochetparade_playwright.py" \
    "$PATTERN_FILE" \
    -o "$OUTPUT" \
    -w "$WAIT_TIME" \
    -a "$ADDITIONAL_WAIT" \
    "${EXTRA_ARGS[@]}"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Success! Screenshot saved to: $OUTPUT"
else
    echo "Error: Screenshot capture failed"
    exit 1
fi
