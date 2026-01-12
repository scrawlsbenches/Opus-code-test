#!/bin/bash
# Bootstrap script for Cognitive Agent cold-start
#
# This script handles the tiered storage model:
# - Tier 1 (committed): tokenizer/ + training_manifest.json (~1.3MB)
# - Tier 2 (derived): bridge/ (~152MB) - rebuilt from Tier 1 + source files
#
# Usage:
#   ./scripts/bootstrap_cognitive.sh           # Rebuild if needed
#   ./scripts/bootstrap_cognitive.sh --force   # Force full rebuild
#   ./scripts/bootstrap_cognitive.sh --check   # Check status only
#
# This script is designed to be run:
# - On fresh clone (cold-start)
# - From git post-checkout hook
# - Before running tests that need the cognitive agent

set -e

MODEL_DIR="models/cognitive_agent"
BRIDGE_DIR="$MODEL_DIR/bridge"
TOKENIZER_DIR="$MODEL_DIR/tokenizer"
MANIFEST_FILE="$MODEL_DIR/training_manifest.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_status() {
    echo "Cognitive Agent Model Status"
    echo "============================="

    if [ -f "$TOKENIZER_DIR/meta.json" ]; then
        vocab_size=$(python3 -c "import json; print(json.load(open('$TOKENIZER_DIR/meta.json'))['vocab_size'])" 2>/dev/null || echo "unknown")
        print_status "Tokenizer: $vocab_size words"
    else
        print_error "Tokenizer: NOT FOUND"
    fi

    if [ -f "$MANIFEST_FILE" ]; then
        doc_count=$(python3 -c "import json; print(len(json.load(open('$MANIFEST_FILE'))['documents']))" 2>/dev/null || echo "unknown")
        print_status "Manifest: $doc_count documents"
    else
        print_error "Manifest: NOT FOUND"
    fi

    if [ -f "$BRIDGE_DIR/meta.json" ]; then
        atom_count=$(python3 -c "import json; print(json.load(open('$BRIDGE_DIR/meta.json'))['total_atoms'])" 2>/dev/null || echo "unknown")
        print_status "Bridge: $atom_count atoms"
    else
        print_warning "Bridge: NOT BUILT (run bootstrap to rebuild)"
    fi
}

# Parse arguments
FORCE=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--force] [--check]"
            exit 1
            ;;
    esac
done

# Check only mode
if [ "$CHECK_ONLY" = true ]; then
    check_status
    exit 0
fi

echo "Cognitive Agent Bootstrap"
echo "========================="

# Check if bridge already exists
if [ -f "$BRIDGE_DIR/meta.json" ] && [ "$FORCE" = false ]; then
    print_status "Model already built."
    check_status
    exit 0
fi

# Check prerequisites (Tier 1 must exist)
if [ ! -f "$TOKENIZER_DIR/meta.json" ]; then
    print_error "Tokenizer not found. Need full training."
    echo ""
    echo "Run full training:"
    echo "  python -m cortical.cognitive train cortical/ --pattern '*.py'"
    echo "  python -m cortical.cognitive train samples/ --pattern '*.txt' '*.md'"
    exit 1
fi

if [ ! -f "$MANIFEST_FILE" ]; then
    print_error "Training manifest not found. Need full training."
    echo ""
    echo "Run full training:"
    echo "  python -m cortical.cognitive train cortical/ --pattern '*.py'"
    exit 1
fi

# Rebuild links from committed vocabulary
if [ "$FORCE" = true ] && [ -d "$BRIDGE_DIR" ]; then
    print_warning "Force rebuild - removing existing bridge..."
    rm -rf "$BRIDGE_DIR"
fi

echo ""
print_status "Rebuilding links from committed vocabulary..."
echo ""

python -m cortical.cognitive rebuild-links --metrics

echo ""
print_status "Bootstrap complete!"
check_status
