---
name: ai-metadata
description: View AI-friendly metadata for code modules. Use when exploring unfamiliar modules to quickly understand structure, functions, and relationships without reading entire files.
allowed-tools: Read, Bash, Glob
---
# AI Metadata Viewer Skill

This skill provides **rapid module understanding** through pre-generated metadata files. When you need to understand a module's structure quickly, use the `.ai_meta` files instead of reading entire source files.

## What AI Metadata Provides

Each `.ai_meta` file contains:

| Section | Description |
|---------|-------------|
| `module_doc` | Truncated docstring explaining module purpose |
| `sections` | Logical groupings of functions (Persistence, Query, Analysis, etc.) |
| `classes` | Class definitions with inheritance and method lists |
| `functions` | All functions with signatures, docs, and `see_also` cross-references |
| `imports` | Stdlib and local imports for understanding dependencies |
| `complexity_hints` | Warnings about expensive operations (PageRank, clustering, etc.) |

## Quick Start

### 1. Generate Metadata (if not present)

```bash
# Generate for all cortical modules
python scripts/generate_ai_metadata.py

# Or regenerate after code changes
python scripts/generate_ai_metadata.py --incremental
```

### 2. View Module Overview

```bash
# Read metadata for a specific module
cat cortical/analysis.py.ai_meta

# Or for packages, check the __init__.py metadata
cat cortical/processor/__init__.py.ai_meta

# Or use Read tool on the .ai_meta file
```

### 3. Find Available Metadata

```bash
# List all metadata files
ls cortical/*.ai_meta tests/*.ai_meta
```

## Usage Patterns

### Understanding a New Module

Instead of reading the entire source file, read the `.ai_meta` first:

```
# EFFICIENT: Read metadata first (structured overview)
Read cortical/processor.py.ai_meta

# THEN: Read specific functions as needed
Read cortical/processor.py (lines 127-150)
```

### Finding Related Functions

The `see_also` field connects related functions:

```yaml
functions:
  add_lateral_connection:
    see_also:
      - add_typed_connection
      - add_feedforward_connection
```

### Identifying Expensive Operations

The `complexity_hints` section warns about slow operations:

```yaml
complexity_hints:
  - compute_pagerank: iterative_algorithm
  - extract_corpus_semantics: corpus_wide_computation
```

### Understanding Code Sections

Functions are grouped into logical sections:

```yaml
sections:
  - name: Persistence
    functions: [save, load, to_dict, from_dict]
  - name: Query
    functions: [find_documents, search, expand_query]
```

## Metadata File Locations

| Source File | Metadata File |
|-------------|---------------|
| `cortical/processor/__init__.py` | `cortical/processor/__init__.py.ai_meta` |
| `cortical/query/__init__.py` | `cortical/query/__init__.py.ai_meta` |
| `cortical/analysis.py` | `cortical/analysis.py.ai_meta` |
| `tests/test_processor.py` | `tests/test_processor.py.ai_meta` |

> **Note:** `processor/` and `query/` are packages with multiple modules. Check `__init__.py.ai_meta` for the public API, or individual module `.ai_meta` files for implementation details.

## Commands Reference

```bash
# Generate all metadata (clean build)
python scripts/generate_ai_metadata.py

# Incremental update (only changed files)
python scripts/generate_ai_metadata.py --incremental

# Generate for a single file
python scripts/generate_ai_metadata.py cortical/processor.py

# Remove all metadata files
python scripts/generate_ai_metadata.py --clean

# Check metadata status
ls -la cortical/*.ai_meta
```

## Integration with Corpus Indexer

The corpus indexer can generate AI metadata automatically:

```bash
# Index codebase AND generate metadata
python scripts/index_codebase.py --incremental && python scripts/generate_ai_metadata.py --incremental
```

## Tips for AI Agents

1. **Start with metadata** - Read `.ai_meta` files before diving into source code
2. **Use `see_also`** - Follow cross-references to understand related functionality
3. **Check `complexity_hints`** - Be aware of expensive operations before calling them
4. **Trust the sections** - Functions are grouped by purpose, not just alphabetically
5. **Regenerate after changes** - Run `--incremental` after modifying code

## Example: Understanding the Processor Package

```bash
# Step 1: Read the package's public API metadata
cat cortical/processor/__init__.py.ai_meta | head -50

# Step 2: Find functions related to search
grep -A5 "find_documents" cortical/processor/query_api.py.ai_meta

# Step 3: Check complexity hints in compute module
grep -A10 "complexity_hints" cortical/processor/compute.py.ai_meta

# Step 4: Now read specific source code as needed
```

## Why Use This Instead of Reading Source?

| Approach | Time | Context Used | Understanding |
|----------|------|--------------|---------------|
| Read entire module | Slow | High | Complete but overwhelming |
| Read `.ai_meta` first | Fast | Low | Structured overview |
| Search with grep | Fast | Low | Fragmented results |

The metadata approach gives you **structured understanding with minimal context usage**.
