#!/bin/bash
#
# merge-all-branches.sh - Aggressively merge all branches into current branch
#
# Philosophy: Better to have duplicate code we can find than lost code on branches.
# - Merges all branches with unmerged commits
# - Auto-resolves conflicts by keeping BOTH versions
# - Runs smoke tests after each merge
# - Creates detailed report of what was merged
#
# Usage:
#   ./scripts/merge-all-branches.sh [--dry-run] [--no-test]
#

set -e

CURRENT_BRANCH=$(git branch --show-current)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_BRANCH="backup-before-mass-merge-$TIMESTAMP"
REPORT_FILE=".git-ml/merge-report-$TIMESTAMP.md"
DRY_RUN=false
SKIP_TESTS=false

# Parse args
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --no-test) SKIP_TESTS=true ;;
    esac
done

echo "========================================"
echo "AGGRESSIVE BRANCH MERGE SCRIPT"
echo "========================================"
echo "Current branch: $CURRENT_BRANCH"
echo "Timestamp: $TIMESTAMP"
echo "Dry run: $DRY_RUN"
echo ""

# Create report directory
mkdir -p .git-ml

# Start report
cat > "$REPORT_FILE" << EOF
# Branch Merge Report

**Date:** $(date)
**Current Branch:** $CURRENT_BRANCH
**Backup Branch:** $BACKUP_BRANCH

## Summary

EOF

# Create backup branch
if [ "$DRY_RUN" = false ]; then
    echo "Creating backup branch: $BACKUP_BRANCH"
    git branch "$BACKUP_BRANCH"
    echo "- Backup created: \`$BACKUP_BRANCH\`" >> "$REPORT_FILE"
else
    echo "[DRY RUN] Would create backup: $BACKUP_BRANCH"
fi

echo ""
echo "Scanning for branches with unmerged commits..."
echo ""

# Collect all branches
MERGED_COUNT=0
CONFLICT_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0

echo "## Branches Processed" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Get all remote branches
for branch in $(git branch -r | grep -v HEAD | grep -v "$CURRENT_BRANCH" | sed 's/origin\///' | sort -u); do
    # Skip if it's the current branch
    if [ "$branch" = "$CURRENT_BRANCH" ]; then
        continue
    fi

    # Count unmerged commits
    UNMERGED=$(git rev-list --count "$CURRENT_BRANCH".."origin/$branch" 2>/dev/null || echo "0")

    if [ "$UNMERGED" -gt 0 ]; then
        echo "----------------------------------------"
        echo "Branch: $branch ($UNMERGED unmerged commits)"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would merge: origin/$branch"
            echo "- [ ] \`$branch\` - $UNMERGED commits (dry run)" >> "$REPORT_FILE"
            continue
        fi

        # Attempt merge with keep-both strategy for conflicts
        echo "Attempting merge..."

        if git merge "origin/$branch" --no-edit -m "Merge branch '$branch' (auto-merge)" 2>/dev/null; then
            echo "  Clean merge!"
            MERGED_COUNT=$((MERGED_COUNT + 1))
            echo "- [x] \`$branch\` - $UNMERGED commits (clean merge)" >> "$REPORT_FILE"
        else
            # Merge has conflicts - resolve by keeping both
            echo "  Conflicts detected - keeping both versions..."

            CONFLICTED_FILES=$(git diff --name-only --diff-filter=U 2>/dev/null || true)

            if [ -n "$CONFLICTED_FILES" ]; then
                for file in $CONFLICTED_FILES; do
                    echo "    Resolving: $file"

                    # For text files, keep both versions (the whole file)
                    # This creates a file with both sets of changes including conflict markers
                    # which we can find later with grep

                    # Accept both by keeping the merged file as-is (with conflict markers)
                    # Then remove conflict markers but keep both versions
                    if [ -f "$file" ]; then
                        # Option 1: Keep conflict markers (findable with grep)
                        # git add "$file"

                        # Option 2: Remove markers, keep both versions
                        if grep -q "<<<<<<< HEAD" "$file" 2>/dev/null; then
                            # Remove the conflict markers but keep all content
                            sed -i 's/<<<<<<< HEAD//g' "$file"
                            sed -i 's/=======//g' "$file"
                            sed -i '/^>>>>>>> /d' "$file"

                            # Add comment marking this was a merge conflict
                            # (for Python files)
                            if [[ "$file" == *.py ]]; then
                                sed -i "1i# MERGE_CONFLICT_RESOLVED: From branch $branch on $TIMESTAMP" "$file"
                            fi
                        fi
                        git add "$file"
                    fi
                done

                git commit -m "Merge branch '$branch' (conflicts resolved - kept both)"
                CONFLICT_COUNT=$((CONFLICT_COUNT + 1))
                echo "- [x] \`$branch\` - $UNMERGED commits (conflicts resolved)" >> "$REPORT_FILE"
                echo "  - Conflicted files: $CONFLICTED_FILES" >> "$REPORT_FILE"
            else
                # No actual conflicts, something else went wrong
                git merge --abort 2>/dev/null || true
                FAILED_COUNT=$((FAILED_COUNT + 1))
                echo "- [ ] \`$branch\` - $UNMERGED commits (FAILED)" >> "$REPORT_FILE"
                echo "  Failed to merge!"
                continue
            fi
        fi

        # Run smoke tests unless skipped
        if [ "$SKIP_TESTS" = false ]; then
            echo "  Running smoke tests..."
            if python -m pytest tests/smoke/ -q --tb=no 2>/dev/null; then
                echo "  Tests passed!"
            else
                echo "  Tests FAILED - but keeping merge (can fix later)"
                echo "  - WARNING: Smoke tests failed after this merge" >> "$REPORT_FILE"
            fi
        fi

        echo ""
    fi
done

# Also check local branches
echo ""
echo "Checking local branches..."
for branch in $(git branch | grep -v "$CURRENT_BRANCH" | sed 's/^[* ]*//' | sort -u); do
    UNMERGED=$(git rev-list --count "$CURRENT_BRANCH".."$branch" 2>/dev/null || echo "0")

    if [ "$UNMERGED" -gt 0 ]; then
        echo "Local branch: $branch ($UNMERGED unmerged)"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would merge: $branch"
            continue
        fi

        if git merge "$branch" --no-edit -m "Merge local branch '$branch'" 2>/dev/null; then
            MERGED_COUNT=$((MERGED_COUNT + 1))
            echo "- [x] \`$branch\` (local) - $UNMERGED commits" >> "$REPORT_FILE"
        fi
    fi
done

# Summary
echo ""
echo "========================================"
echo "MERGE COMPLETE"
echo "========================================"
echo "Merged cleanly: $MERGED_COUNT"
echo "Conflicts resolved: $CONFLICT_COUNT"
echo "Failed: $FAILED_COUNT"
echo ""
echo "Report saved to: $REPORT_FILE"
echo "Backup branch: $BACKUP_BRANCH"
echo ""

# Add summary to report
cat >> "$REPORT_FILE" << EOF

## Statistics

| Metric | Count |
|--------|-------|
| Merged cleanly | $MERGED_COUNT |
| Conflicts resolved | $CONFLICT_COUNT |
| Failed | $FAILED_COUNT |

## Next Steps

1. Search for merge markers: \`grep -r "MERGE_CONFLICT_RESOLVED" cortical/ scripts/\`
2. Run full test suite: \`python -m pytest tests/ -v\`
3. If issues, restore from backup: \`git reset --hard $BACKUP_BRANCH\`

EOF

echo "To find resolved conflicts later:"
echo "  grep -r 'MERGE_CONFLICT_RESOLVED' cortical/ scripts/"
echo ""
echo "To rollback if needed:"
echo "  git reset --hard $BACKUP_BRANCH"
