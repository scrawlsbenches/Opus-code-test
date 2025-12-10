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

## Index the Codebase

```bash
python scripts/index_codebase.py
```

### Options

- `--output FILE` or `-o FILE`: Custom output path (default: corpus_dev.pkl)
- `--verbose` or `-v`: Show detailed indexing progress

### Example

```bash
# Standard indexing
python scripts/index_codebase.py

# Verbose output to see what's being indexed
python scripts/index_codebase.py --verbose

# Custom output location
python scripts/index_codebase.py --output my_corpus.pkl
```

## What Gets Indexed

The indexer processes:
- All Python files in `cortical/` (source code)
- All Python files in `tests/` (test code)
- Documentation: `CLAUDE.md`, `TASK_LIST.md`, `README.md`, `KNOWLEDGE_TRANSFER.md`

## Output Statistics

After indexing, you'll see:
- Number of documents indexed
- Total lines of code
- Token count (unique terms)
- Bigram count (word pairs)
- Concept clusters
- Semantic relations extracted

## Maintenance

Re-index periodically to keep search accurate:
- After adding new modules
- After major refactoring
- Before deep codebase exploration sessions
