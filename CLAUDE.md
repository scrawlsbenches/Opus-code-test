# Claude.md - Project Guide for Claude Code

This file provides guidance for Claude Code when working with the Cortical Text Processor codebase.

## Project Overview

A neocortex-inspired text processing library with zero external dependencies that performs semantic analysis and document retrieval using a hierarchical biological computing model.

## Quick Start

```python
from cortical import CorticalTextProcessor

processor = CorticalTextProcessor()
processor.process_document("doc1", "Neural networks process information...")
processor.compute_all()
results = processor.find_documents_for_query("neural processing")
```

## Project Structure

```
cortical/
├── __init__.py          # Public API exports
├── processor.py         # Main orchestrator (CorticalTextProcessor)
├── minicolumn.py        # Core data structure
├── layers.py            # Hierarchical layer definitions
├── tokenizer.py         # Text tokenization
├── analysis.py          # PageRank, TF-IDF, clustering
├── semantics.py         # Semantic extraction, retrofitting
├── embeddings.py        # Graph-based embeddings
├── query.py             # Query expansion and search
├── gaps.py              # Knowledge gap detection
└── persistence.py       # Save/load functionality

tests/
├── test_tokenizer.py
├── test_processor.py
├── test_layers.py
├── test_analysis.py
├── test_embeddings.py
├── test_semantics.py
├── test_gaps.py
└── test_persistence.py
```

## Running Tests

```bash
python -m unittest discover -s tests -v
```

All 129 tests should pass.

## Running the Showcase

```bash
python showcase.py
```

## Key Classes

### CorticalTextProcessor
Main entry point. Coordinates document processing, computations, and queries.

### HierarchicalLayer
Manages minicolumns at a given hierarchy level. Has `get_by_id()` for O(1) lookups.

### Minicolumn
Represents a concept/feature. Tracks:
- `doc_occurrence_counts`: Per-document term frequencies
- `lateral_connections`: Associations with other terms
- `pagerank`, `tfidf`: Importance scores

## Recent Changes (2025-12-09)

### Bug Fixes Applied
1. **TF-IDF calculation** - Now uses actual per-document occurrence counts
2. **O(1) ID lookups** - Added `_id_index` and `get_by_id()` method
3. **Type annotations** - Fixed `any` → `Any` in semantics.py
4. **Unused imports** - Removed `Counter` from analysis.py
5. **Verbose parameter** - Added to `export_graph_json()`

### Performance Improvements
- Graph algorithms improved from O(n²) to O(n) via ID index

## Coding Conventions

- Use type hints for all function parameters and returns
- Follow Google-style docstrings
- Line length: 100 characters (per pyproject.toml)
- Run tests before committing changes

## Common Tasks

### Add a new document
```python
processor.process_document("doc_id", "document content")
processor.compute_all(verbose=False)
```

### Query the corpus
```python
results = processor.find_documents_for_query("search terms", top_n=5)
expanded = processor.expand_query("term", max_expansions=10)
```

### Analyze knowledge gaps
```python
gaps = processor.analyze_knowledge_gaps()
anomalies = processor.detect_anomalies(threshold=0.1)
```

### Compute embeddings
```python
stats = processor.compute_graph_embeddings(dimensions=32, method='adjacency')
similar = processor.find_similar_by_embedding("term", top_n=5)
```

### Save/Load state
```python
processor.save("model.pkl")
loaded = CorticalTextProcessor.load("model.pkl")
```
