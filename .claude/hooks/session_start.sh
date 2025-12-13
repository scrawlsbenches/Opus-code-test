#!/usr/bin/env bash
#
# Session Start Hook for Cortical Text Processor
# Automatically indexes codebase and generates AI metadata when starting Claude Code
#
# This hook runs:
# 1. Incremental corpus indexing if corpus_dev.pkl is missing
# 2. Incremental AI metadata generation if .ai_meta files are missing
#
# Silent on success, only outputs during first-time setup.

# Get the project root (parent of .claude/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Check if corpus needs indexing
if [ ! -f "corpus_dev.pkl" ]; then
    echo "Indexing codebase (first-time setup)..." >&2
    python scripts/index_codebase.py --incremental >/dev/null 2>&1 || true
fi

# Check if AI metadata needs generation
# Look for at least one .ai_meta file in cortical/
if ! find cortical/ -name "*.ai_meta" -print -quit 2>/dev/null | grep -q .; then
    echo "Generating AI metadata (first-time setup)..." >&2
    python scripts/generate_ai_metadata.py --incremental >/dev/null 2>&1 || true
fi
