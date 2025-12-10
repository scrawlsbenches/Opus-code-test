---
name: codebase-search
description: Search the Cortical Text Processor codebase using semantic search. Use when looking for code patterns, understanding how features work, or finding relevant implementations. This skill uses the system's own IR algorithms to search its own codebase (dog-fooding).
allowed-tools: Read, Bash, Glob
---
# Codebase Search Skill

This skill enables semantic search over the Cortical Text Processor codebase using the system's own IR algorithms.

## When to Use

- Finding implementations of specific features
- Understanding how algorithms work
- Locating relevant code for modifications
- Discovering related functions and classes
- Exploring the codebase structure

## Prerequisites

Before using search, ensure the corpus is indexed:

```bash
python scripts/index_codebase.py
```

This creates `corpus_dev.pkl` with the indexed codebase.

## Search Commands

### Basic Search

```bash
python scripts/search_codebase.py "your query here"
```

### Options

- `--top N` or `-n N`: Number of results (default: 5)
- `--verbose` or `-v`: Show full passage text
- `--expand` or `-e`: Show query expansion terms
- `--interactive` or `-i`: Interactive search mode

### Example Queries

```bash
# Find PageRank implementation
python scripts/search_codebase.py "PageRank algorithm implementation"

# Find bigram handling code
python scripts/search_codebase.py "bigram separator" --verbose

# Explore query expansion
python scripts/search_codebase.py "query expansion semantic" --expand

# Interactive exploration
python scripts/search_codebase.py --interactive
```

## Understanding Results

Results include:
- **File:Line** reference (e.g., `cortical/analysis.py:127`)
- **Score** indicating relevance (higher is better)
- **Passage** showing relevant code or text

## Tips

1. Use natural language queries - the system understands concepts
2. Check query expansion (`--expand`) to see related terms
3. Start with broad queries, then refine
4. Use interactive mode for exploration sessions
