# Cortical Text Processor

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-2941%20passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-%3E89%25-brightgreen.svg)
![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-orange.svg)

A neocortex-inspired text processing library with **zero external dependencies** for semantic analysis, document retrieval, and knowledge gap detection.

---

> *"What if we built a text search engine the way evolution built a brain?"*

Your visual cortex doesn't grep through pixels looking for cats. It builds hierarchies—edges become patterns, patterns become shapes, shapes become objects. This library applies the same principle to text.

Feed it documents. It tokenizes them into "minicolumns" (Layer 0), connects co-occurring words through Hebbian learning ("neurons that fire together, wire together"), clusters them into concepts (Layer 2), and links documents by shared meaning (Layer 3). The result: a graph that understands your corpus well enough to expand queries, complete analogies, and tell you where your knowledge has gaps.

No PyTorch. No transformers. No API keys. Just 2900+ tests, 19,000+ lines of pure Python, and a data structure that would make a neuroscientist squint approvingly.

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

## Use Cases & When to Use

### Ideal Use Cases

| Use Case | Why It's a Good Fit |
|----------|---------------------|
| **Internal Documentation Search** | Understands domain-specific terminology through corpus-derived semantics; no training data needed |
| **Knowledge Base Q&A** | Query expansion finds related documents even when exact keywords don't match |
| **Code Repository Search** | Built-in code tokenization splits `getUserName` → `get`, `user`, `name`; programming synonym expansion |
| **Research Paper Organization** | Concept clustering automatically groups related papers; gap detection finds missing coverage |
| **RAG/LLM Context Retrieval** | Chunk-level passage retrieval with relevance scoring; designed for retrieval-augmented generation |
| **Offline/Air-gapped Environments** | Zero dependencies, no API calls, works completely offline |
| **Privacy-Sensitive Applications** | All processing happens locally; no data leaves your machine |
| **Educational Projects** | Clean, well-documented codebase demonstrates IR algorithms (PageRank, TF-IDF, Louvain clustering) |

### Good Fit For Developers Who...

- **Need explainable search** - Every result can be traced through the graph; see exactly why documents matched
- **Want to avoid ML complexity** - No model training, GPU requirements, or hyperparameter tuning
- **Work with specialized domains** - Corpus-derived semantics adapts to your terminology automatically
- **Need lightweight deployment** - Single Python package, no Docker, no external services
- **Value reproducibility** - Deterministic algorithms produce consistent results
- **Build RAG pipelines** - First-class support for passage retrieval with configurable chunking

### When NOT to Use

| Scenario | Better Alternative |
|----------|-------------------|
| Need state-of-the-art semantic similarity | Use sentence transformers or OpenAI embeddings |
| Processing millions of documents | Use Elasticsearch, Meilisearch, or vector databases |
| Need real-time indexing at scale | Use purpose-built search infrastructure |
| Require cross-lingual search | Use multilingual embedding models |
| Need image/multimodal search | Use CLIP or similar multimodal models |

### Example: Building a Documentation Search

```python
from cortical import CorticalTextProcessor
import os

# Initialize processor
processor = CorticalTextProcessor()

# Index your documentation
for filename in os.listdir("docs/"):
    if filename.endswith(".md"):
        with open(f"docs/{filename}") as f:
            processor.process_document(filename, f.read())

# Build the semantic network
processor.compute_all(verbose=False)

# Search with query expansion
results = processor.find_documents_for_query("authentication setup")
# Finds docs about "auth", "login", "credentials" even if "authentication" isn't mentioned

# Get relevant passages for RAG
passages = processor.find_passages_for_query("how to configure OAuth", top_n=3)
for passage, score, doc_id in passages:
    print(f"[{doc_id}] {passage[:100]}...")
```

### Example: Code Search with Intent

```python
# Enable code-aware tokenization
processor = CorticalTextProcessor(split_identifiers=True)

# Index source files
for filepath in glob.glob("src/**/*.py", recursive=True):
    with open(filepath) as f:
        processor.process_document(filepath, f.read())

processor.compute_all()

# Intent-based search understands natural language questions
results = processor.search_by_intent("where do we handle user authentication?")
# Returns files dealing with auth, login, session management

# Code-specific query expansion
expanded = processor.expand_query_for_code("fetch data")
# Expands to include: get, load, retrieve, request, download
```

## Installation

Install from source:

```bash
git clone <repository-url>
cd cortical-text-processor
pip install -e .
```

Or simply copy the `cortical/` directory into your project—zero dependencies means no pip required.

## Quick Start

Run the showcase to see the processor analyze 92 documents covering everything from neural networks to medieval falconry:

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
  ... 89 more documents ...

✓ Processed 92 documents
✓ Created 6,506 token minicolumns
✓ Created 20,114 bigram minicolumns
✓ Formed 116,332 lateral connections

══════════════════════════════════════════════════════════════════════
                       KEY CONCEPTS (PageRank)
══════════════════════════════════════════════════════════════════════

PageRank identifies central concepts - highly connected 'hub' words:

  Rank  Concept            PageRank
  ─────────────────────────────────────────────
    1.  data               ████████████████████ 0.0046
    2.  model              ███████████████████░ 0.0044
    3.  learning           ██████████████████░░ 0.0041
    ...

══════════════════════════════════════════════════════════════════════
                        QUERY DEMONSTRATION
══════════════════════════════════════════════════════════════════════

🔍 Query: 'neural networks'
   Expanded with: knowledge, data, graph, network, deep, artificial

   Top documents:
     • comprehensive_machine_learning (score: 26.384)
     • attention_mechanism_research (score: 19.178)
     • cortical_semantic_networks (score: 18.470)
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

Tested with 92 sample documents covering topics from neural networks to medieval falconry to sourdough breadmaking.

| Metric | Value |
|--------|-------|
| Documents processed | 92 |
| Token minicolumns | 6,506 |
| Bigram minicolumns | 20,114 |
| Lateral connections | 116,332 |
| Test coverage | 2900+ tests passing |
| Graph algorithms | O(1) ID lookups |

**What the processor discovers:**
- Most central concept: `data` (PageRank: 0.0046)
- Most distinctive terms: `gradient`, `pagerank`, `patent` (high TF-IDF, rare but meaningful)
- Most connected document: `comprehensive_machine_learning` (91 connections to other docs)
- Isolated outliers detected: `sumo_wrestling`, `medieval_falconry` (low similarity to corpus)

## Package Structure

```
cortical/
├── __init__.py          # Public API exports
├── processor/           # Main orchestrator (mixin-based architecture)
│   ├── __init__.py      # CorticalTextProcessor class composition
│   ├── core.py          # Initialization, staleness tracking (169 lines)
│   ├── documents.py     # Document add/remove/batch (456 lines)
│   ├── compute.py       # PageRank, TF-IDF, clustering (1041 lines)
│   ├── query_api.py     # Search, expansion, retrieval (719 lines)
│   ├── introspection.py # State inspection, summaries (357 lines)
│   └── persistence_api.py # Save/load/export (245 lines)
├── query/               # Search & retrieval (8 focused modules)
│   ├── expansion.py     # Query expansion (459 lines)
│   ├── search.py        # Document search (422 lines)
│   ├── ranking.py       # Multi-stage ranking (472 lines)
│   ├── passages.py      # RAG passage retrieval (407 lines)
│   ├── chunking.py      # Text chunking (335 lines)
│   ├── intent.py        # Intent-based queries (220 lines)
│   ├── definitions.py   # Definition search (375 lines)
│   └── analogy.py       # Analogy completion (330 lines)
├── analysis.py          # Graph algorithms: PageRank, TF-IDF, Louvain
├── semantics.py         # Relation extraction, inference, retrofitting
├── minicolumn.py        # Core data structure with typed edges
├── layers.py            # Hierarchical layers with O(1) lookups
├── tokenizer.py         # Tokenization, stemming, code splitting
├── embeddings.py        # Graph embeddings with retrofitting
├── fingerprint.py       # Semantic fingerprinting
├── gaps.py              # Gap detection and anomalies
├── persistence.py       # Save/load with full state
├── config.py            # CorticalConfig with validation
├── observability.py     # Metrics, timing, tracing
└── code_concepts.py     # Programming synonym expansion

tests/                   # 2900+ tests (smoke, unit, integration, behavioral)
├── smoke/               # Quick sanity checks
├── unit/                # Fast isolated tests
├── integration/         # Component interaction tests
├── performance/         # Timing regression tests
└── behavioral/          # Search quality tests

showcase.py              # Interactive demonstration (run it!)
samples/                 # 92 documents: quantum computing to cheese affinage
scripts/                 # Developer tools (indexing, profiling, tasks)
```

## AI Agent Support

This project includes tools designed specifically for AI coding assistants:

### AI Metadata Files (`.ai_meta`)

Pre-generated metadata files provide structured navigation for AI agents:

```bash
# Generate metadata for rapid module understanding
python scripts/generate_ai_metadata.py

# View a module's structure without reading source
cat cortical/processor/__init__.py.ai_meta
cat cortical/query/search.py.ai_meta
```

**What metadata provides:**
- Function signatures with `see_also` cross-references
- Class structures with inheritance
- Complexity hints for expensive operations
- Logical section groupings

### Claude Skills

Four Claude Code skills are available in `.claude/skills/`:

| Skill | Purpose |
|-------|---------|
| `codebase-search` | Semantic search over the codebase |
| `corpus-indexer` | Index/re-index after code changes |
| `ai-metadata` | View and use module metadata |
| `task-manager` | Manage tasks with merge-friendly IDs |

### For AI Agents

See the **AI Agent Onboarding** section in [CLAUDE.md](CLAUDE.md) for:
- Step-by-step setup guide
- Navigation tips for efficient exploration
- Example workflow using metadata

## Text-as-Memories System

Capture and organize institutional knowledge alongside your code:

- **Daily Memories** (`samples/memories/YYYY-MM-DD-*.md`) - Learning entries
- **Decision Records** (`samples/decisions/adr-*.md`) - Architectural decisions
- **Concept Documents** - Consolidated knowledge on topics

```bash
# Create a memory entry
python scripts/new_memory.py "What I learned about validation"

# Create a decision record
python scripts/new_memory.py "Use JSON over pickle" --decision
```

See [docs/text-as-memories.md](docs/text-as-memories.md) for the complete guide.

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

Search for "candle sticks" and you'll find `candlestick_patterns` (trading charts) at the top—but also `letterpress_printing` (composing sticks) and `wine_tasting_vocabulary`. The query tokenizes to `['candle', 'sticks']`: "candle" matches the trading document (which discusses "single candle patterns"), while "sticks" matches the printing document. Classic information retrieval challenge: compound words fragment, partial matches surface, and the system can't read your mind about intent.

### Knowledge Gap Detection

The analyzer flags `sumo_wrestling` and `medieval_falconry` as isolated documents—they don't fit well with the rest of the corpus. It also identifies weak topics: terms like `patent` appear in only 1 document. This is how you find holes in your knowledge base.

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

## Security Considerations

### Pickle Deserialization Warning

⚠️ **The default `.pkl` format uses Python's `pickle` module, which can execute arbitrary code during deserialization.**

**Risk**: Loading a malicious `.pkl` file could result in remote code execution (RCE). Only load pickle files from sources you trust completely.

**Recommendations**:

1. **For untrusted sources**: Use the JSON state format instead:
   ```python
   from cortical.state_storage import StateLoader

   # Save as JSON (safe to share)
   StateLoader.save(processor, "corpus_state.json")

   # Load from JSON (safe from untrusted sources)
   processor = StateLoader.load("corpus_state.json")
   ```

2. **For trusted sources**: Continue using pickle for faster serialization:
   ```python
   processor.save("corpus.pkl")  # Fast, but only load files you trust
   processor = CorticalTextProcessor.load("corpus.pkl")
   ```

3. **For maximum security**: Never load pickle files from:
   - Downloaded files from the internet
   - User uploads
   - Shared network locations with untrusted access
   - Email attachments

See [Python's pickle documentation](https://docs.python.org/3/library/pickle.html) for more details on pickle security.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup and workflow
- Code style and testing requirements
- Pull request guidelines

Quality resources:
- [Definition of Done](docs/definition-of-done.md)
- [Code of Ethics](docs/code-of-ethics.md)

## License

MIT License
