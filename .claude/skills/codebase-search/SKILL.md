---
name: codebase-search
description: Search the Cortical Text Processor codebase using semantic search. Use when looking for code patterns, understanding how features work, or finding relevant implementations. This skill uses the system's own IR algorithms to search its own codebase (dog-fooding).
allowed-tools: Read, Bash, Glob
---
# Codebase Search Skill

This skill enables **meaning-based search** over the Cortical Text Processor codebase. It finds relevant code by understanding intent and concepts, not just exact keyword matching.

## Key Capabilities

- **Meaning-based retrieval**: Finds related code even when exact words don't match
- **Query expansion**: Automatically includes related terms via co-occurrence and semantic relations
- **Code concept groups**: Knows "fetch", "get", "load" are synonyms in code context
- **Intent understanding**: Parses "where do we handle X?" into location + action + subject
- **Semantic fingerprinting**: Compare and explain code similarity
- **Fast search mode**: ~2-3x faster for large codebases
- **No ML required**: Works through graph algorithms on corpus statistics

## When to Use

- Finding implementations: "how does PageRank work"
- Locating code by intent: "where do we handle errors"
- Understanding relationships: "what connects to the tokenizer"
- Exploring concepts: "authentication and validation"

## Prerequisites

Ensure the corpus is indexed:

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

| Option | Description |
|--------|-------------|
| `--top N` | Number of results (default: 5) |
| `--verbose` | Show full passage text |
| `--expand` | Show query expansion terms |
| `--fast` | Fast search mode (~2-3x faster, document-level) |
| `--interactive` | Interactive search mode |

### Example Queries

```bash
# Find by concept (not exact words)
python scripts/search_codebase.py "graph importance algorithm"

# Natural language intent (parses action + subject)
python scripts/search_codebase.py "where do we handle authentication"

# Code concept synonyms (fetch finds get/load/retrieve too)
python scripts/search_codebase.py "fetch user data"

# Fast mode for quick lookups
python scripts/search_codebase.py "PageRank" --fast

# See what terms the system associates
python scripts/search_codebase.py "lateral connections" --expand

# Interactive exploration
python scripts/search_codebase.py --interactive
```

## Understanding Results

Results include:
- **File:Line** reference (e.g., `cortical/analysis.py:127`)
- **Score** indicating relevance (higher is better)
- **Passage** showing relevant code or text

## Tips

1. **Use natural language** - ask questions as you would to a colleague
2. **Check expansion** (`--expand`) - see what related terms are being searched
3. **Broad then narrow** - start general, refine based on results
4. **Interactive mode** - use `/expand`, `/concepts`, `/stats` for exploration
