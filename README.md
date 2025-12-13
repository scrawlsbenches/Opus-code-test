# Cortical Text Processor

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-1729%20passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)
![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-orange.svg)

A neocortex-inspired text processing library with **zero external dependencies** for semantic analysis, document retrieval, and knowledge gap detection.

---

> *"What if we built a text search engine the way evolution built a brain?"*

Your visual cortex doesn't grep through pixels looking for cats. It builds hierarchies—edges become patterns, patterns become shapes, shapes become objects. This library applies the same principle to text.

Feed it documents. It tokenizes them into "minicolumns" (Layer 0), connects co-occurring words through Hebbian learning ("neurons that fire together, wire together"), clusters them into concepts (Layer 2), and links documents by shared meaning (Layer 3). The result: a graph that understands your corpus well enough to expand queries, complete analogies, and tell you where your knowledge has gaps.

No PyTorch. No transformers. No API keys. Just 1,729 tests, 16,800 lines of pure Python, and a data structure that would make a neuroscientist squint approvingly.

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

Install from source:

```bash
git clone <repository-url>
cd cortical-text-processor
pip install -e .
```

Or simply copy the `cortical/` directory into your project—zero dependencies means no pip required.

## Quick Start

Run the showcase to see the processor analyze 161 documents covering everything from neural networks to medieval falconry:

```bash
python showcase.py
```

**Output:**
```
    ╔══════════════════════════════════════════════════════════════════════╗
    ║            🧠  CORTICAL TEXT PROCESSOR SHOWCASE  🧠                  ║
    ║     Mimicking how the neocortex processes and understands text       ║
    ╚══════════════════════════════════════════════════════════════════════╝

Loading documents from: samples
Processing through cortical hierarchy...
(Like visual information flowing V1 → V2 → V4 → IT)

  📄 comprehensive_machine_learning (2445 words)
  📄 attention_mechanism_research   (644 words)
  📄 neural_network_optimization    (648 words)
  ... 158 more documents ...

✓ Processed 161 documents
✓ Created 8,789 token minicolumns
✓ Created 46,374 bigram minicolumns
✓ Formed 2,483,316 total connections

══════════════════════════════════════════════════════════════════════
                       KEY CONCEPTS (PageRank)
══════════════════════════════════════════════════════════════════════

PageRank identifies central concepts - highly connected 'hub' words:

  Rank  Concept            PageRank
  ─────────────────────────────────────────────
    1.  market             ████████████████████ 0.0051
    2.  patterns           ████████████████░░░░ 0.0042
    3.  systems            ████████████░░░░░░░░ 0.0033
    ...

══════════════════════════════════════════════════════════════════════
                        QUERY DEMONSTRATION
══════════════════════════════════════════════════════════════════════

🔍 Query: 'neural networks'
   Expanded with: graph, knowledge, network, learn, deep, artificial

   Top documents:
     • graph_neural_networks_code_analysis (score: 86.397)
     • graph_neural_networks (score: 72.267)
     • neural_pagerank (score: 33.687)
```

### Programmatic Usage

```python
from cortical import CorticalTextProcessor

processor = CorticalTextProcessor()

# Add documents
processor.process_document("doc1", "Neural networks process information hierarchically.")
processor.process_document("doc2", "The brain uses layers of neurons for processing.")
processor.process_document("doc3", "Machine learning enables pattern recognition.")

# Build the network
processor.compute_all()

# Query
results = processor.find_documents_for_query("neural processing")
print(results)  # [('doc1', 0.877), ('doc2', 0.832)]

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

Tested with 161 sample documents covering topics from neural networks to medieval falconry to sourdough breadmaking.

| Metric | Value |
|--------|-------|
| Documents processed | 161 |
| Token minicolumns | 8,789 |
| Bigram minicolumns | 46,374 |
| Total connections | 2,483,316 |
| Concept clusters | 84 |
| Test coverage | 1,729 tests passing (85%) |
| Graph algorithms | O(1) ID lookups |

**What the processor discovers:**
- Most central concept: `market` (PageRank: 0.0051)
- Most distinctive terms: `fractal`, `delta`, `entropy` (high TF-IDF, rare but meaningful)
- Most connected document: `comprehensive_machine_learning` (160 connections to other docs)
- Isolated outliers detected: `sumo_wrestling`, `medieval_falconry`, `umami_taste` (low similarity to corpus)

## Package Structure

```
cortical/
├── __init__.py      # Public API (v2.0.0)
├── processor.py     # Main orchestrator (2,301 lines)
├── tokenizer.py     # Tokenization + stemming
├── minicolumn.py    # Core data structure with typed edges
├── layers.py        # Hierarchical layers with O(1) lookups
├── analysis.py      # PageRank, TF-IDF, cross-layer propagation
├── semantics.py     # Semantic extraction, inference, analogy
├── embeddings.py    # Graph embeddings with retrofitting
├── query/           # Search, retrieval, batch processing (8 modules)
│   ├── expansion.py # Query expansion
│   ├── search.py    # Document search
│   ├── passages.py  # RAG passage retrieval
│   └── ...          # + 5 more specialized modules
├── gaps.py          # Gap detection and anomalies
├── persistence.py   # Save/load with full state
├── mcp_server.py    # MCP Server for Claude Desktop (NEW)
├── validation.py    # Input validation decorators (NEW)
└── config.py        # Configuration with validation

evaluation/
└── evaluator.py     # Evaluation framework

tests/               # 1,729 comprehensive tests
showcase.py          # Interactive demonstration (run it!)
samples/             # 161 documents: from quantum computing to cheese affinage
```

## AI Agent Support

This project includes tools designed specifically for AI coding assistants:

### AI Metadata Files (`.ai_meta`)

Pre-generated metadata files provide structured navigation for AI agents:

```bash
# Generate metadata for rapid module understanding
python scripts/generate_ai_metadata.py

# View a module's structure without reading source
cat cortical/processor.py.ai_meta
```

**What metadata provides:**
- Function signatures with `see_also` cross-references
- Class structures with inheritance
- Complexity hints for expensive operations
- Logical section groupings

### Claude Skills

Three Claude Code skills are available in `.claude/skills/`:

| Skill | Purpose |
|-------|---------|
| `codebase-search` | Semantic search over the codebase |
| `corpus-indexer` | Index/re-index after code changes |
| `ai-metadata` | View and use module metadata |

### For AI Agents

See the **AI Agent Onboarding** section in [CLAUDE.md](CLAUDE.md) for:
- Step-by-step setup guide
- Navigation tips for efficient exploration
- Example workflow using metadata

## MCP Server (Claude Desktop Integration)

The Cortical Text Processor includes an MCP (Model Context Protocol) server for native integration with Claude Desktop and other AI assistants.

### Available Tools

| Tool | Description |
|------|-------------|
| `search` | Find relevant documents for a query |
| `passages` | Retrieve RAG-ready text passages |
| `expand_query` | Get query expansion terms |
| `corpus_stats` | Get corpus statistics |
| `add_document` | Index a new document |

### Quick Start

```bash
# Run the MCP server
python -m cortical.mcp_server

# Or programmatically
from cortical.mcp_server import create_mcp_server
server = create_mcp_server(corpus_path="my_corpus.pkl")
server.run()
```

### Claude Desktop Configuration

Add to your Claude Desktop MCP config:

```json
{
  "mcpServers": {
    "cortical": {
      "command": "python",
      "args": ["-m", "cortical.mcp_server"],
      "env": {
        "CORPUS_PATH": "/path/to/corpus.pkl"
      }
    }
  }
}
```

## Simplified Facade Methods

For common use cases, these one-call methods provide sensible defaults:

```python
# Quick search - just get document IDs
docs = processor.quick_search("neural networks")
# Returns: ['doc1', 'doc2', 'doc3']

# RAG retrieve - get passages ready for LLM context
passages = processor.rag_retrieve("how does PageRank work", top_n=3)
# Returns: [{'text': '...', 'doc_id': '...', 'score': 0.85}, ...]

# Explore - search with query expansion info
result = processor.explore("machine learning")
# Returns: {'results': [...], 'expansion': {...}, 'original_terms': [...]}
```

## Development History

This project evolved through systematic improvements:

1. **Initial Release**: Core hierarchical text processing
2. **Code Review & Fixes**: TF-IDF calculation, O(1) lookups, type annotations
3. **RAG Enhancements**: Chunk-level retrieval, metadata support, concept clustering
4. **ConceptNet Integration**: Typed edges, relation-weighted PageRank, multi-hop inference
5. **Connection Strategies**: Multiple strategies for Layer 2 concept connections
6. **Showcase & Polish**: Interactive demo with real corpus analysis
7. **MCP Server**: Claude Desktop integration with 5 tools for AI-native access
8. **Simplified API**: Facade methods (`quick_search`, `rag_retrieve`, `explore`)
9. **Code Search**: Intent detection, definition search, doc-type boosting
10. **Unit Test Initiative**: 1,729 tests, 85% coverage, 19 modules at 90%+

## Running the Showcase

```bash
python showcase.py
```

The showcase processes 161 diverse sample documents and demonstrates every major feature. Here's what you'll see:

### Concept Associations (Hebbian Learning)

The processor discovers that `neural` connects to `networks` (weight: 44), `graph` (22), `network` (11)—while `bread` meekly connects to `beer`, `wine`, and `pyruvate` (weight: 1 each). Neurons that fire together really do wire together.

### Query Expansion in Action

```
🔍 Query: 'neural networks'
   Expanded with: graph, knowledge, network, learn, deep, artificial

   Top documents:
     • graph_neural_networks_code_analysis (score: 86.397)
     • graph_neural_networks (score: 72.267)
     • neural_pagerank (score: 33.687)
```

### The Polysemy Problem

Search for "candle sticks" and you'll find `candlestick_patterns` (trading charts) at the top—but also `letterpress_printing` (composing sticks) and `wine_tasting_vocabulary`. The query tokenizes to `['candle', 'sticks']`: "candle" matches the trading document (which discusses "single candle patterns"), while "sticks" matches the printing document. Classic information retrieval challenge: compound words fragment, partial matches surface, and the system can't read your mind about intent.

### Knowledge Gap Detection

The analyzer flags `sumo_wrestling`, `medieval_falconry`, and `umami_taste` as isolated documents—they don't fit well with the rest of the corpus. It also identifies weak topics: terms like `fractal` and `delta` appear in only 1-2 documents. This is how you find holes in your knowledge base.

## Documentation

Detailed documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Documentation index with reading paths |
| [docs/quickstart.md](docs/quickstart.md) | 5-minute getting started guide |
| [docs/architecture.md](docs/architecture.md) | 4-layer system design |
| [docs/algorithms.md](docs/algorithms.md) | Core IR algorithms (PageRank, TF-IDF, Louvain) |
| [docs/query-guide.md](docs/query-guide.md) | Query formulation guide |
| [docs/cookbook.md](docs/cookbook.md) | Common patterns and recipes |
| [docs/glossary.md](docs/glossary.md) | Terminology definitions |

For AI agents, see also [docs/claude-usage.md](docs/claude-usage.md) and [CLAUDE.md](CLAUDE.md).

## Running Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT License
