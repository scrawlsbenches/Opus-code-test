#!/bin/bash
# Reset workspace to reduce visual noise and cognitive load
# Run at the start of a session for a clean mental slate

clear

echo "═══════════════════════════════════════════"
echo "  Workspace Status"
echo "═══════════════════════════════════════════"
echo ""

# Git status (concise)
echo "📁 Working Directory:"
git status --short 2>/dev/null || echo "   Not a git repository"
echo ""

# Recent commits
echo "📜 Recent Activity:"
git log --oneline -5 2>/dev/null | sed 's/^/   /' || echo "   No commits"
echo ""

# Current branch
BRANCH=$(git branch --show-current 2>/dev/null)
if [ -n "$BRANCH" ]; then
    echo "🌿 Branch: $BRANCH"
    echo ""
fi

# Active tasks (if task system available)
if [ -f "scripts/task_utils.py" ]; then
    echo "📋 Active Tasks:"
    python scripts/task_utils.py list --status in_progress 2>/dev/null | head -5 | sed 's/^/   /' || echo "   No active tasks"
    echo ""
fi

# Sprint context (if available)
if [ -f "tasks/CURRENT_SPRINT.md" ]; then
    echo "🎯 Sprint Goal:"
    grep -A1 "## Goals" tasks/CURRENT_SPRINT.md 2>/dev/null | tail -1 | sed 's/^/   /' || echo "   No sprint defined"
    echo ""
fi

echo "═══════════════════════════════════════════"
echo "  Ready to work"
echo "═══════════════════════════════════════════"
