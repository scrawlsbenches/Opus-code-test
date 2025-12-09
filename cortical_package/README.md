# Cortical Text Processor

A neocortex-inspired text processing library with **zero external dependencies**.

## Overview

This library provides a biologically-inspired approach to text processing, organizing information through a hierarchical structure similar to the visual cortex:

| Layer | Name | Analogy | Purpose |
|-------|------|---------|---------|
| 0 | Tokens | V1 (edges) | Individual words |
| 1 | Bigrams | V2 (patterns) | Word pairs |
| 2 | Concepts | V4 (shapes) | Semantic clusters |
| 3 | Documents | IT (objects) | Full documents |

## Key Features

- **Hierarchical Processing**: Feedforward and lateral connections like the neocortex
- **PageRank Importance**: Graph-based term importance scoring
- **TF-IDF Weighting**: Statistical term distinctiveness
- **Corpus-Derived Semantics**: No external knowledge bases needed
- **Graph Embeddings**: Multiple embedding methods with retrofitting
- **Gap Detection**: Find weak spots in your corpus
- **Query Expansion**: Smart retrieval with synonym handling
- **Zero Dependencies**: Pure Python, no pip installs required

## Installation

```bash
pip install cortical-text-processor
```

Or install from source:

```bash
git clone https://github.com/example/cortical-text-processor.git
cd cortical-text-processor
pip install -e .
```

## Quick Start

```python
from cortical import CorticalTextProcessor

# Create processor
processor = CorticalTextProcessor()

# Add documents
processor.process_document("doc1", "Neural networks process information hierarchically.")
processor.process_document("doc2", "The brain uses layers of neurons for processing.")
processor.process_document("doc3", "Machine learning enables pattern recognition.")

# Build the network
processor.propagate_activation()
processor.compute_importance()
processor.compute_tfidf()

# Query
results = processor.find_documents_for_query("neural processing")
print(results)  # [('doc1', 0.85), ('doc2', 0.72), ...]

# Analyze corpus health
health = processor.compute_corpus_health()
print(f"Corpus health: {health['overall_score']:.0%}")

# Save for later
processor.save("my_corpus.pkl")
```

## Core API

### Document Processing

```python
processor.process_document(doc_id, content)
processor.process_documents_from_directory(path)
```

### Network Building

```python
processor.propagate_activation()      # Spread activation
processor.compute_importance()        # PageRank scores
processor.compute_tfidf()            # TF-IDF weights
processor.build_concept_clusters()   # Cluster tokens
processor.compute_document_connections()  # Link documents
```

### Semantics & Embeddings

```python
processor.extract_corpus_semantics()  # Extract relations
processor.retrofit_connections()      # Blend with semantics
processor.compute_graph_embeddings()  # Term embeddings
processor.retrofit_embeddings()       # Improve embeddings
```

### Query & Retrieval

```python
processor.expand_query(text)              # Expand query
processor.find_documents_for_query(text)  # Search
processor.find_related_documents(doc_id)  # Related docs
processor.summarize_document(doc_id)      # Summarize
```

### Analysis

```python
processor.analyze_knowledge_gaps()  # Find gaps
processor.detect_anomalies()        # Find outliers
processor.compute_corpus_health()   # Health score
```

## Performance

Evaluation on a 37-document corpus:

| Category | Score |
|----------|-------|
| **Overall** | **90.1%** |
| Factual Retrieval | 91.7% |
| Cross-Document Synthesis | 93.3% |
| Gap Detection | 94.4% |
| Query Expansion | 93.3% |

## Package Structure

```
cortical/
├── __init__.py      # Public API
├── processor.py     # Main class
├── tokenizer.py     # Tokenization + stemming
├── minicolumn.py    # Core data structure
├── layers.py        # Hierarchical layers
├── analysis.py      # PageRank, TF-IDF
├── semantics.py     # Semantic extraction
├── embeddings.py    # Graph embeddings
├── query.py         # Search and retrieval
├── gaps.py          # Gap detection
└── persistence.py   # Save/load
```

## License

MIT License
