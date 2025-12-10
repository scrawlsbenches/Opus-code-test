---
name: corpus-indexer
description: Index or re-index the codebase for semantic search. Use after making significant code changes to keep the search corpus up-to-date.
allowed-tools: Bash
---
# Corpus Indexer Skill

This skill manages the codebase index used by the semantic search system.

## When to Use

- After adding new files to the codebase
- After significant code changes
- When search results seem outdated
- To verify indexing statistics

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
```

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
- Documentation: `CLAUDE.md`, `TASK_LIST.md`, `README.md`, `KNOWLEDGE_TRANSFER.md`
- Intelligence docs in `docs/` directory

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
