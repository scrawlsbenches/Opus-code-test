# Dog-Fooding Guide

This guide explains how the Cortical Text Processor is used to build and improve itself - a practice known as "dog-fooding." The system indexes its own codebase, enabling semantic search during development.

---

## Overview

**Dog-fooding** means using your own product to develop it. The Cortical Text Processor can:

1. **Index its own source code** - Build a searchable semantic model of the codebase
2. **Search semantically** - Find relevant code by meaning, not just keywords
3. **Update incrementally** - Keep the index current as code changes
4. **Integrate with Claude** - Provide semantic search via Claude skills

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dog-Fooding Workflow                          │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Index   │───▶│  Search  │───▶│ Develop  │───▶│ Re-index │  │
│  │ Codebase │    │   Code   │    │   Code   │    │  Changes │  │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘  │
│       ▲                                               │        │
│       └───────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Index the Codebase

```bash
# First time: Full index (~2-3 seconds)
python scripts/index_codebase.py

# After changes: Incremental update (~1 second)
python scripts/index_codebase.py --incremental
```

### 2. Search for Code

```bash
# Basic search
python scripts/search_codebase.py "PageRank algorithm"

# See query expansion
python scripts/search_codebase.py "bigram separator" --expand

# Interactive mode
python scripts/search_codebase.py --interactive
```

### 3. Use Claude Skills

When using Claude Code in this project:
- **codebase-search**: Search for code patterns and implementations
- **corpus-indexer**: Re-index after making changes

---

## The Indexing System

### What Gets Indexed

The indexer processes these files:

| Category | Pattern | Purpose |
|----------|---------|---------|
| Source code | `cortical/*.py` | Core library implementation |
| Tests | `tests/*.py` | Test cases and examples |
| Documentation | `CLAUDE.md`, `README.md` | Project documentation |
| Intelligence | `docs/*.md` | Architecture docs |
| Task tracking | `TASK_LIST.md` | Development tasks |

### Index Files Created

| File | Purpose |
|------|---------|
| `corpus_dev.pkl` | Serialized processor state (searchable index) |
| `corpus_dev.manifest.json` | File modification times for incremental updates |

### Indexer Options

```bash
# Show what would be indexed without doing it
python scripts/index_codebase.py --status

# Force full rebuild even if nothing changed
python scripts/index_codebase.py --force

# See per-file progress
python scripts/index_codebase.py --verbose

# Log to file for debugging
python scripts/index_codebase.py --log indexer.log

# Set timeout (default 300s)
python scripts/index_codebase.py --timeout 60

# Full semantic analysis (slower, more accurate)
python scripts/index_codebase.py --full-analysis
```

---

## Incremental Indexing

Incremental indexing is the key to efficient dog-fooding. Instead of rebuilding the entire index, it only processes changes.

### How It Works

```
1. Load manifest (file modification times from last index)
2. Scan current files
3. Detect changes:
   - ADDED: Files that didn't exist before
   - MODIFIED: Files with newer modification times
   - DELETED: Files in manifest but no longer exist
4. Update only changed files:
   - Remove deleted docs from index
   - Re-index modified files (remove old, add new)
   - Add new files
5. Recompute analysis (PageRank, TF-IDF, etc.)
6. Save updated index and manifest
```

### Performance

| Operation | Time | Use Case |
|-----------|------|----------|
| No changes detected | ~0.1s | Check if re-index needed |
| Few files changed | ~1-2s | Normal development |
| Full rebuild (fast mode) | ~2-3s | After major refactoring |
| Full rebuild (full analysis) | ~10+ min | Before deep exploration |

### When to Re-index

| Scenario | Command |
|----------|---------|
| After editing code | `--incremental` |
| After adding new files | `--incremental` |
| After deleting files | `--incremental` |
| After major refactoring | `--force` |
| Before deep code exploration | `--full-analysis` |
| Search results seem stale | `--status` then decide |

---

## Search Capabilities

### Basic Search

```bash
# Find code related to a concept
python scripts/search_codebase.py "query expansion"

# Output shows file:line references
# cortical/query.py:55  [0.847]
#   def get_expanded_query_terms(...)
```

### Query Expansion

The search automatically expands queries with related terms:

```bash
python scripts/search_codebase.py "PageRank" --expand

# Shows: pagerank → importance, score, rank, algorithm, weight, ...
```

### Interactive Mode

For exploratory searching:

```bash
python scripts/search_codebase.py --interactive

# Commands in interactive mode:
# /expand <query>  - Show query expansion terms
# /concepts        - List concept clusters
# /stats           - Show corpus statistics
# /quit            - Exit
```

### Search Options

| Option | Description |
|--------|-------------|
| `--top N` | Return N results (default: 5) |
| `--verbose` | Show full passage text |
| `--expand` | Show query expansion terms |
| `--fast` | Document-level search only (faster) |
| `--interactive` | Interactive search mode |

---

## Claude Skills Integration

### codebase-search Skill

Use this skill to search the indexed codebase from Claude:

```
@claude: Use codebase-search to find how PageRank is implemented
```

The skill:
1. Loads the pre-built corpus (`corpus_dev.pkl`)
2. Executes semantic search
3. Returns file:line references with relevant passages

### corpus-indexer Skill

Use this skill to re-index after making changes:

```
@claude: Use corpus-indexer to update the index

# Or specifically:
@claude: Use corpus-indexer with --incremental flag
```

The skill runs `scripts/index_codebase.py` with appropriate options.

---

## Development Workflow

### Typical Development Cycle

```bash
# 1. Start by searching for relevant code
python scripts/search_codebase.py "feature I want to modify"

# 2. Make changes to the code
# ... edit files ...

# 3. Run tests
python -m unittest discover -s tests -v

# 4. Re-index to update search
python scripts/index_codebase.py --incremental

# 5. Verify changes are searchable
python scripts/search_codebase.py "my new function"
```

### Adding a New Feature

1. **Research existing code**
   ```bash
   python scripts/search_codebase.py "related functionality" --verbose
   ```

2. **Check the task list**
   ```bash
   python scripts/search_codebase.py "TASK_LIST feature name"
   ```

3. **Implement the feature**
   - Follow patterns found in search results
   - Add tests in `tests/`

4. **Update the index**
   ```bash
   python scripts/index_codebase.py --incremental --verbose
   ```

5. **Verify searchability**
   ```bash
   python scripts/search_codebase.py "new feature name"
   ```

### Debugging with Search

When debugging, use semantic search to find related code:

```bash
# Find error handling patterns
python scripts/search_codebase.py "handle error exception"

# Find similar implementations
python scripts/search_codebase.py "implementation pattern I'm looking at"

# Find test patterns
python scripts/search_codebase.py "test case for feature"
```

---

## Technical Details

### Fast Mode vs Full Analysis

**Fast Mode** (default):
- Skips `compute_bigram_connections()` - O(n²) on large corpora
- Computes: PageRank, TF-IDF, document connections
- Time: ~2-3 seconds
- Good for: Development, quick searches

**Full Analysis Mode** (`--full-analysis`):
- Runs complete `compute_all()` pipeline
- Includes: Bigram connections, concept clusters, semantic relations
- Time: ~10+ minutes for full codebase
- Good for: Deep exploration, research sessions

### Manifest File Format

```json
{
  "cortical/processor.py": 1702234567.89,
  "tests/test_processor.py": 1702234590.12,
  ...
}
```

Maps relative file paths to Unix modification timestamps.

### Index Contents

The `corpus_dev.pkl` file contains a serialized `CorticalTextProcessor` with:

- **Layer 0 (TOKENS)**: ~6,000+ unique terms from source code
- **Layer 1 (BIGRAMS)**: ~26,000+ word pairs
- **Layer 2 (CONCEPTS)**: Semantic clusters (if full analysis)
- **Layer 3 (DOCUMENTS)**: Each indexed file

---

## Troubleshooting

### Index Taking Too Long

**Symptom:** Indexer hangs at "Computing analysis"

**Cause:** `compute_bigram_connections()` has O(n²) complexity

**Solution:** Use fast mode (default) or add `--timeout`:
```bash
python scripts/index_codebase.py --timeout 60
```

### Search Results Seem Stale

**Check index status:**
```bash
python scripts/search_codebase.py --status
```

**Force rebuild:**
```bash
python scripts/index_codebase.py --force
```

### "No corpus found" Error

**Cause:** `corpus_dev.pkl` doesn't exist

**Solution:** Run initial indexing:
```bash
python scripts/index_codebase.py
```

### Memory Issues with Large Corpus

**Cause:** Full analysis mode creates many connections

**Solution:** Use fast mode or limit file count

### Index File Too Large

**Cause:** Full analysis mode creates extensive connection data

**Solution:** Use fast mode which produces smaller indices

---

## Best Practices

### 1. Index Frequently

Run `--incremental` after every significant code change:
```bash
python scripts/index_codebase.py --incremental
```

### 2. Use --status Before Decisions

Check what would change before rebuilding:
```bash
python scripts/index_codebase.py --status
```

### 3. Log for Debugging

When investigating issues, enable logging:
```bash
python scripts/index_codebase.py --verbose --log debug.log
```

### 4. Use Interactive Mode for Exploration

When researching unfamiliar code:
```bash
python scripts/search_codebase.py --interactive
```

### 5. Trust the Expansion

Let query expansion find related terms:
```bash
python scripts/search_codebase.py "authentication" --expand
# May find: auth, login, credential, token, session, ...
```

### 6. Combine with Git

Index before major refactoring to capture baseline:
```bash
git status
python scripts/index_codebase.py --force --log pre-refactor.log
```

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/index_codebase.py` | Codebase indexer with incremental support |
| `scripts/search_codebase.py` | Semantic search CLI |
| `corpus_dev.pkl` | Serialized index (generated) |
| `corpus_dev.manifest.json` | File modification times (generated) |
| `.claude/skills/codebase-search/` | Claude search skill |
| `.claude/skills/corpus-indexer/` | Claude indexer skill |

---

## Summary

Dog-fooding the Cortical Text Processor creates a virtuous cycle:

1. **The system searches itself** - Find relevant code by meaning
2. **Changes improve search** - Better algorithms help find code
3. **Incremental updates are fast** - Stay productive during development
4. **Claude integration automates** - Skills handle indexing and search

This self-referential capability accelerates development by making the codebase semantically searchable while actively improving it.

---

*Updated 2025-12-10*
