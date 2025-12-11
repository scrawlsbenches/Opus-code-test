---
name: corpus-indexer
description: Index or re-index the codebase for semantic search. Use after making significant code changes to keep the search corpus up-to-date.
allowed-tools: Bash
---
# Corpus Indexer Skill

This skill manages the codebase index used by the semantic search system and generates AI navigation metadata.

## When to Use

- After adding new files to the codebase
- After significant code changes
- When search results seem outdated
- To verify indexing statistics
- When AI metadata files (`.ai_meta`) are missing or stale

## Quick Commands

```bash
# Fast incremental update (only changed files, ~2s)
python scripts/index_codebase.py --incremental

# Check what would be indexed without doing it
python scripts/index_codebase.py --status

# Full rebuild (still fast, ~2s)
python scripts/index_codebase.py

# Force full rebuild even if no changes detected
python scripts/index_codebase.py --force

# RECOMMENDED: Index + generate AI metadata in one command
python scripts/index_codebase.py --incremental && python scripts/generate_ai_metadata.py --incremental
```

## AI Metadata Generation

Generate `.ai_meta` files that provide structured navigation for AI agents:

```bash
# Generate metadata for all modules
python scripts/generate_ai_metadata.py

# Incremental update (only changed files)
python scripts/generate_ai_metadata.py --incremental

# Generate for a single file
python scripts/generate_ai_metadata.py cortical/processor.py

# Clean and regenerate all
python scripts/generate_ai_metadata.py --clean && python scripts/generate_ai_metadata.py
```

**What metadata provides:**
- Module overview and docstring
- Function signatures with `see_also` cross-references
- Class structures with inheritance
- Logical section groupings
- Complexity hints for expensive operations

**For detailed usage, see the `ai-metadata` skill.**

## Options

| Option | Description |
|--------|-------------|
| `--incremental`, `-i` | Only re-index changed files (fastest) |
| `--status`, `-s` | Show what would change without indexing |
| `--force`, `-f` | Force full rebuild even if up-to-date |
| `--verbose`, `-v` | Show per-file progress |
| `--log FILE`, `-l FILE` | Write detailed log to file |
| `--timeout N`, `-t N` | Timeout in seconds (default: 300) |
| `--full-analysis` | Use complete semantic analysis (slower) |
| `--output FILE`, `-o FILE` | Custom output path (default: corpus_dev.pkl) |
| `--use-chunks` | Use git-compatible chunk-based storage |
| `--compact` | Compact old chunk files (use with `--use-chunks`) |
| `--before DATE` | Compact only chunks before this date (YYYY-MM-DD) | |

## Examples

```bash
# Standard incremental update
python scripts/index_codebase.py --incremental

# Verbose with logging (good for debugging)
python scripts/index_codebase.py --verbose --log index.log

# See what files changed
python scripts/index_codebase.py --status

# Full semantic analysis (takes longer, more accurate)
python scripts/index_codebase.py --full-analysis
```

## What Gets Indexed

The indexer processes:
- All Python files in `cortical/` (source code)
- All Python files in `tests/` (test code)
- Documentation: `CLAUDE.md`, `README.md`, `CONTRIBUTING.md`, `KNOWLEDGE_TRANSFER.md`
- Task management: `TASK_LIST.md`, `TASK_ARCHIVE.md`
- All docs in `docs/` directory (including `quickstart.md`)

## Output Statistics

After indexing, you'll see:
- Number of documents indexed
- Total lines of code
- Token count (unique terms)
- Bigram count (word pairs)
- Concept clusters (in full-analysis mode)
- Semantic relations (in full-analysis mode)

## Manifest File

The indexer creates `corpus_dev.manifest.json` to track file modification times.
This enables fast incremental updates by detecting only changed files.

## Performance

| Mode | Time | Use Case |
|------|------|----------|
| Incremental | ~1-2s | After small edits |
| Full rebuild (fast) | ~2-3s | Default mode |
| Full analysis | ~10+ min | Complete semantic analysis |

## Maintenance

- Use `--incremental` for quick updates during development
- Use `--status` to check if re-indexing is needed
- Use `--force` after major refactoring
- Use `--full-analysis` before deep exploration sessions
- After indexing, **test search quality** - see `docs/dogfooding-checklist.md`

## Git-Compatible Chunk Storage

For team collaboration, use `--use-chunks` to store changes as git-friendly JSON:

```bash
# Index with chunk storage
python scripts/index_codebase.py --incremental --use-chunks

# Check chunk status
python scripts/index_codebase.py --status --use-chunks
```

**Benefits:**
- No merge conflicts (unique timestamp filenames)
- Shared indexed state across branches
- Fast startup when cache is valid

**Files Created:**
- `corpus_chunks/*.json` - Tracked in git (append-only changes)
- `corpus_dev.pkl` - NOT tracked (local cache)
- `corpus_dev.pkl.hash` - NOT tracked (cache validation)

## Chunk Compaction

Over time, chunk files accumulate. Use compaction to consolidate them (like `git gc`):

**When to compact:**
- After 10+ chunk files accumulate
- When you see size warnings during save
- Before merging branches with chunk histories
- To clean up old deleted entries

**Commands:**
```bash
# Compact all chunks into one
python scripts/index_codebase.py --compact --use-chunks

# Compact chunks before a date
python scripts/index_codebase.py --compact --before 2025-12-01 --use-chunks
```

**What happens:**
1. All chunks are read in timestamp order
2. Operations are replayed (later timestamps win)
3. A single compacted chunk is created
4. Old chunk files are removed
5. Cache is preserved if valid

**Recommended frequency:**
- Weekly for active development
- Monthly for maintenance mode
- Before major releases
