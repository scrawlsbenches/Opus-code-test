# Cortical Text Processor

A neocortex-inspired text processing library with **zero external dependencies** for semantic analysis, document retrieval, and knowledge gap detection.

---

> *"What if we built a text search engine the way evolution built a brain?"*

Your visual cortex doesn't grep through pixels looking for cats. It builds hierarchies—edges become patterns, patterns become shapes, shapes become objects. This library applies the same principle to text.

Feed it documents. It tokenizes them into "minicolumns" (Layer 0), connects co-occurring words through Hebbian learning ("neurons that fire together, wire together"), clusters them into concepts (Layer 2), and links documents by shared meaning (Layer 3). The result: a graph that understands your corpus well enough to expand queries, complete analogies, and tell you where your knowledge has gaps.

No PyTorch. No transformers. No API keys. Just 337 tests, ~3000 lines of pure Python, and a data structure that would make a neuroscientist squint approvingly.

---

## Overview

This library provides a biologically-inspired approach to text processing, organizing information through a hierarchical structure similar to the visual cortex:

| Layer | Name | Analogy | Purpose |
|-------|------|---------|---------|
| 0 | Tokens | V1 (edges) | Individual words |
| 1 | Bigrams | V2 (patterns) | Word pairs |
| 2 | Concepts | V4 (shapes) | Semantic clusters |
| 3 | Documents | IT (objects) | Full documents |

## Key Features

- **Hierarchical Processing**: Feedforward, feedback, and lateral connections like the neocortex
- **PageRank Importance**: Graph-based term importance with relation-weighted and cross-layer propagation
- **TF-IDF Weighting**: Statistical term distinctiveness with per-document occurrence tracking
- **Corpus-Derived Semantics**: Pattern-based commonsense relation extraction without external knowledge bases
- **Graph Embeddings**: Multiple embedding methods (adjacency, spectral, random walk) with semantic retrofitting
- **ConceptNet-Style Relations**: Typed edges (IsA, HasA, PartOf, etc.) with multi-hop inference
- **Concept Inheritance**: IsA hierarchy propagation for concept properties
- **Analogy Completion**: Relation matching and vector arithmetic for analogical reasoning
- **Gap Detection**: Find weak spots and isolated documents in your corpus
- **Query Expansion**: Smart retrieval with synonym handling and semantic relations
- **RAG System Support**: Chunk-level passage retrieval, document metadata, and multi-stage ranking
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

# Build the network (runs all computations)
processor.compute_all()

# Query
results = processor.find_documents_for_query("neural processing")
print(results)  # [('doc1', 0.85), ('doc2', 0.72), ...]

# Get corpus summary
summary = processor.get_corpus_summary()
print(f"Documents: {summary['documents']}, Connections: {summary['total_connections']}")

# Save for later
processor.save("my_corpus.pkl")
```

## Core API

### Document Processing

```python
processor.process_document(doc_id, content, metadata=None)
processor.add_document_incremental(doc_id, content)  # Incremental indexing
processor.add_documents_batch([(doc_id, content, metadata), ...])  # Batch processing
```

### Network Building

```python
# All-in-one computation with connection strategies
processor.compute_all(
    verbose=False,
    connection_strategy='hybrid',  # 'document_overlap', 'semantic', 'embedding', 'hybrid'
    cluster_strictness=0.5,        # 0.0-1.0, lower = fewer, larger clusters
    bridge_weight=0.3              # 0.0-1.0, cross-document bridging
)

# Individual computations
processor.propagate_activation()      # Spread activation
processor.compute_importance()        # PageRank scores
processor.compute_tfidf()             # TF-IDF weights
processor.build_concept_clusters()    # Cluster tokens
processor.compute_document_connections()  # Link documents
processor.compute_bigram_connections()    # Bigram lateral connections
```

### Semantics & Embeddings

```python
processor.extract_corpus_semantics()  # Extract relations
processor.retrofit_connections()      # Blend with semantics
processor.compute_graph_embeddings(dimensions=32, method='adjacency')
processor.retrofit_embeddings()       # Improve embeddings
processor.expand_query_multihop(query, max_hops=2)  # Multi-hop query expansion
processor.complete_analogy(a, b, c)   # Analogy completion (a:b :: c:?)
```

### Query & Retrieval

```python
processor.expand_query(text, max_expansions=10)  # Expand query
processor.find_documents_for_query(text, top_n=5)  # Search
processor.find_related_documents(doc_id)  # Related docs
processor.find_documents_batch(queries)  # Process multiple queries
processor.find_passages_for_query(query, top_n=5)  # Chunk-level RAG retrieval
```

### Analysis

```python
processor.analyze_knowledge_gaps()  # Find gaps
processor.detect_anomalies(threshold=0.1)  # Find outliers
processor.get_corpus_summary()      # Corpus statistics
processor.export_conceptnet_json(filepath)  # ConceptNet-style visualization export
```

## Connection Strategies

For documents with different topics or minimal overlap, use connection strategies:

```python
# Hybrid strategy combines all methods for maximum connectivity
processor.compute_all(
    connection_strategy='hybrid',
    cluster_strictness=0.5,
    bridge_weight=0.3
)
```

| Strategy | Description |
|----------|-------------|
| `document_overlap` | Traditional Jaccard similarity (default) |
| `semantic` | Connect via semantic relations between members |
| `embedding` | Connect via embedding centroid similarity |
| `hybrid` | Combine all three for maximum connectivity |

## Performance

Tested with 92 sample documents covering topics from neural networks to medieval falconry to sourdough breadmaking.

| Metric | Value |
|--------|-------|
| Documents processed | 92 |
| Token minicolumns | 6,506 |
| Bigram minicolumns | 20,114 |
| Lateral connections | 116,332 |
| Test coverage | 337 tests passing |
| Graph algorithms | O(1) ID lookups |

**What the processor discovers:**
- Most central concept: `data` (PageRank: 0.0046)
- Most distinctive terms: `gradient`, `pagerank`, `patent` (high TF-IDF, rare but meaningful)
- Most connected document: `comprehensive_machine_learning` (91 connections to other docs)
- Isolated outliers detected: `sumo_wrestling`, `medieval_falconry` (low similarity to corpus)

## Package Structure

```
cortical/
├── __init__.py      # Public API (v2.0.0)
├── processor.py     # Main orchestrator
├── tokenizer.py     # Tokenization + stemming
├── minicolumn.py    # Core data structure with typed edges
├── layers.py        # Hierarchical layers with O(1) lookups
├── analysis.py      # PageRank, TF-IDF, cross-layer propagation
├── semantics.py     # Semantic extraction, inference, analogy
├── embeddings.py    # Graph embeddings with retrofitting
├── query.py         # Search, retrieval, batch processing
├── gaps.py          # Gap detection and anomalies
└── persistence.py   # Save/load with full state

evaluation/
└── evaluator.py     # Evaluation framework

tests/               # 337 comprehensive tests
showcase.py          # Interactive demonstration (run it!)
samples/             # 92 documents: from quantum computing to cheese affinage
```

## Development History

This project evolved through systematic improvements:

1. **Initial Release**: Core hierarchical text processing
2. **Code Review & Fixes**: TF-IDF calculation, O(1) lookups, type annotations
3. **RAG Enhancements**: Chunk-level retrieval, metadata support, concept clustering
4. **ConceptNet Integration**: Typed edges, relation-weighted PageRank, multi-hop inference
5. **Connection Strategies**: Multiple strategies for Layer 2 concept connections
6. **Showcase & Polish**: Interactive demo with real corpus analysis

## Running the Showcase

```bash
python showcase.py
```

The showcase processes 92 diverse sample documents and demonstrates every major feature. Here's what you'll see:

### Concept Associations (Hebbian Learning)

The processor discovers that `neural` connects to `networks` (weight: 23), `artificial` (7), `knowledge` (7)—while `bread` meekly connects to `beer`, `wine`, and `pyruvate` (weight: 1 each). Neurons that fire together really do wire together.

### Query Expansion in Action

```
🔍 Query: 'neural networks'
   Expanded with: knowledge, data, graph, network, deep, artificial

   Top documents:
     • comprehensive_machine_learning (score: 26.384)
     • attention_mechanism_research (score: 19.178)
     • cortical_semantic_networks (score: 18.470)
```

### The Polysemy Problem

Search for "candle sticks" and you'll find both `candlestick_patterns` (trading charts) and `letterpress_printing` (composing sticks). The system correctly retrieves both meanings but can't read your mind about which one you wanted. Word sense disambiguation remains an open challenge.

### Knowledge Gap Detection

The analyzer flags `sumo_wrestling` and `medieval_falconry` as isolated documents—they don't fit well with the rest of the corpus. It also identifies weak topics: terms like `patent` appear in only 1 document. This is how you find holes in your knowledge base.

## Running Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT License
