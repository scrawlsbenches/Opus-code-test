#!/bin/bash
# Pre-commit cognitive load reduction checklist
# Run before committing to catch common issues

echo "═══════════════════════════════════════════"
echo "  Pre-Commit Checklist"
echo "═══════════════════════════════════════════"
echo ""

issues=0

# Check staged files
STAGED=$(git diff --cached --name-only 2>/dev/null)

if [ -z "$STAGED" ]; then
    echo "No files staged for commit."
    exit 0
fi

# 1. Check for source files without corresponding tests
SRC_FILES=$(echo "$STAGED" | grep -E "^cortical/.*\.py$" | grep -v "__pycache__" | grep -v "\.pyc$")
TEST_FILES=$(echo "$STAGED" | grep -E "^tests/.*\.py$")

if [ -n "$SRC_FILES" ] && [ -z "$TEST_FILES" ]; then
    echo "⚠️  Source files staged without tests:"
    echo "$SRC_FILES" | sed 's/^/   /'
    echo ""
    issues=$((issues + 1))
else
    echo "✓  Test coverage check passed"
fi

# 2. Check commit message format reminder
echo ""
echo "📝 Commit message format:"
echo "   type: short description"
echo ""
echo "   Types: feat, fix, docs, refactor, test, chore, perf"
echo ""

# 3. Check for large diffs
LINES_CHANGED=$(git diff --cached --stat | tail -1 | grep -oE '[0-9]+ insertion' | grep -oE '[0-9]+')
if [ -n "$LINES_CHANGED" ] && [ "$LINES_CHANGED" -gt 500 ]; then
    echo "⚠️  Large change detected: $LINES_CHANGED lines"
    echo "   Consider splitting into smaller commits"
    issues=$((issues + 1))
else
    echo "✓  Change size is manageable"
fi

# 4. Check for debug code
DEBUG_PATTERNS="print\(|console\.log|debugger|pdb\.set_trace|breakpoint\(\)"
DEBUG_HITS=$(git diff --cached -G "$DEBUG_PATTERNS" --name-only 2>/dev/null)

if [ -n "$DEBUG_HITS" ]; then
    echo ""
    echo "⚠️  Possible debug code detected in:"
    echo "$DEBUG_HITS" | sed 's/^/   /'
    issues=$((issues + 1))
else
    echo "✓  No debug code detected"
fi

# 5. Summary
echo ""
echo "═══════════════════════════════════════════"
if [ "$issues" -gt 0 ]; then
    echo "  $issues issue(s) to review before committing"
    echo "═══════════════════════════════════════════"
    exit 1
else
    echo "  All checks passed - ready to commit"
    echo "═══════════════════════════════════════════"
    exit 0
fi
