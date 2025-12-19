# Deployment Guide

This guide covers deploying the Cortical Text Processor in various environments, from local installations to production servers and Docker containers.

---

## Overview

The Cortical Text Processor can be deployed in three primary modes:

| Deployment Mode | Use Case | Requirements |
|-----------------|----------|--------------|
| **Python Library** | Embed in applications | Python 3.9+ only |
| **MCP Server** | AI agent integration (Claude Desktop, etc.) | Python 3.11+ + MCP SDK |
| **Standalone Scripts** | CLI tools, batch processing | Python 3.9+ |

**Key Feature:** The library has **zero runtime dependencies**—only Python standard library modules. Optional dependencies (MCP, protobuf) are only needed for specific features.

---

## Library Installation

### Quick Install (Recommended)

```bash
# Install from repository
git clone https://github.com/scrawlsbenches/Opus-code-test.git
cd Opus-code-test
pip install -e .
```

That's it! No external dependencies required for core functionality.

### From Source (Manual)

If you prefer not to use pip, simply copy the `cortical/` directory into your project:

```bash
# Copy library files
cp -r cortical/ /path/to/your/project/

# Use directly
from cortical import CorticalTextProcessor
processor = CorticalTextProcessor()
```

### Development Installation

For development work (testing, MCP server, etc.):

```bash
# Install with all dev dependencies
pip install -e ".[dev]"

# Or install from requirements.txt
pip install -r requirements.txt
```

Development dependencies:
- `coverage>=7.0` - Test coverage reporting
- `pytest>=7.0` - Test framework
- `mcp>=1.0` - MCP server support
- `pyyaml>=6.0` - Workflow integration

### Verifying Installation

```python
# Test basic functionality
from cortical import CorticalTextProcessor

processor = CorticalTextProcessor()
processor.process_document("test", "Hello world")
processor.compute_all()
print("Installation successful!")
```

---

## MCP Server Deployment

The Model Context Protocol (MCP) server enables AI agents like Claude Desktop to use the Cortical Text Processor for semantic search and document retrieval.

### Prerequisites

```bash
# Install MCP SDK (Python 3.11+ required)
pip install mcp

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Running the MCP Server

#### Command Line

```bash
# Start with an empty corpus
python -m cortical.mcp_server

# Load a pre-indexed corpus
CORTICAL_CORPUS_PATH=/path/to/corpus.pkl python -m cortical.mcp_server

# With custom log level
CORTICAL_LOG_LEVEL=DEBUG python -m cortical.mcp_server
```

#### Programmatically

```python
from cortical.mcp_server import create_mcp_server

# Start with empty corpus
server = create_mcp_server()
server.run(transport="stdio")

# Or load existing corpus
server = create_mcp_server(corpus_path="/path/to/corpus.pkl")
server.run(transport="stdio")
```

### Claude Desktop Integration

Add the following to your Claude Desktop MCP configuration file:

**Configuration File Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Basic Configuration:**

```json
{
  "mcpServers": {
    "cortical-text-processor": {
      "command": "python",
      "args": ["-m", "cortical.mcp_server"],
      "env": {
        "CORTICAL_CORPUS_PATH": "/absolute/path/to/corpus.pkl",
        "CORTICAL_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**For Dynamic Document Addition (Empty Corpus):**

```json
{
  "mcpServers": {
    "cortical-text-processor": {
      "command": "python",
      "args": ["-m", "cortical.mcp_server"],
      "env": {
        "CORTICAL_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Available MCP Tools

The server exposes 5 tools to AI agents:

1. **search** - Find documents relevant to a query
2. **passages** - Retrieve RAG-ready text passages
3. **expand_query** - Get query expansion terms using semantic connections
4. **corpus_stats** - Get statistics about the current corpus
5. **add_document** - Index a new document incrementally

For detailed MCP server documentation, see [MCP_SERVER_README.md](../MCP_SERVER_README.md).

---

## Docker Deployment

### Basic Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy library files
COPY cortical/ /app/cortical/
COPY pyproject.toml /app/
COPY README.md /app/

# Install the library (zero dependencies for core)
RUN pip install --no-cache-dir -e .

# Optional: Install MCP server support
RUN pip install --no-cache-dir mcp>=1.0

# Expose any necessary ports (if running web service)
# EXPOSE 8000

# Copy pre-indexed corpus (optional)
COPY corpus.pkl /app/corpus.pkl

# Default command runs MCP server
ENV CORTICAL_CORPUS_PATH=/app/corpus.pkl
CMD ["python", "-m", "cortical.mcp_server"]
```

### Building and Running

```bash
# Build the Docker image
docker build -t cortical-text-processor:latest .

# Run with empty corpus
docker run -it cortical-text-processor:latest

# Run with mounted corpus volume
docker run -it \
  -v /path/to/corpus.pkl:/app/corpus.pkl \
  -e CORTICAL_CORPUS_PATH=/app/corpus.pkl \
  cortical-text-processor:latest

# Run with custom Python script
docker run -it \
  -v /path/to/script.py:/app/script.py \
  cortical-text-processor:latest \
  python script.py
```

### Docker Compose Example

```yaml
version: '3.8'

services:
  cortical-mcp:
    build: .
    image: cortical-text-processor:latest
    environment:
      - CORTICAL_CORPUS_PATH=/app/data/corpus.pkl
      - CORTICAL_LOG_LEVEL=INFO
    volumes:
      - ./corpus.pkl:/app/data/corpus.pkl:ro
      - ./logs:/app/logs
    restart: unless-stopped
```

### Minimal Alpine-based Image

For smaller container size (~50MB vs ~150MB):

```dockerfile
FROM python:3.11-alpine

WORKDIR /app

# Copy only necessary files
COPY cortical/ /app/cortical/

# No dependencies needed for core library!
ENV PYTHONUNBUFFERED=1

# Run your script
CMD ["python", "-m", "your_script"]
```

---

## Production Tuning

### Performance Configuration

For large corpora (10,000+ documents), adjust these configuration parameters:

```python
from cortical import CorticalTextProcessor, CorticalConfig

# Production-optimized configuration
config = CorticalConfig(
    # PageRank settings (reduce iterations for speed)
    pagerank_iterations=10,  # Default: 20
    pagerank_tolerance=1e-4,  # Default: 1e-6 (looser convergence)

    # Clustering settings (fewer, larger clusters)
    louvain_resolution=1.0,  # Default: 2.0 (lower = fewer clusters)
    min_cluster_size=5,  # Default: 3 (larger clusters)

    # Chunking for RAG (adjust based on LLM context)
    chunk_size=1024,  # Default: 512 (larger chunks)
    chunk_overlap=256,  # Default: 128 (more overlap)

    # Query expansion (reduce for speed)
    max_query_expansions=5,  # Default: 10

    # BM25 scoring (recommended for large corpora)
    scoring_algorithm='bm25',  # Default: 'bm25'
    bm25_k1=1.2,  # Term frequency saturation
    bm25_b=0.75,  # Length normalization
)

processor = CorticalTextProcessor(config=config)
```

### Memory Optimization

```python
# Use incremental indexing for large corpora
from cortical import CorticalTextProcessor

processor = CorticalTextProcessor()

# Add documents in batches
batch_size = 1000
for i, (doc_id, text) in enumerate(documents):
    processor.add_document_incremental(doc_id, text, recompute='tfidf')

    # Periodically save checkpoints
    if (i + 1) % batch_size == 0:
        processor.save(f"checkpoint_{i+1}.pkl")
        print(f"Saved checkpoint at {i+1} documents")

# Final computation of all metrics
processor.compute_all()
processor.save("final_corpus.pkl")
```

### Fast Search (Pre-built Indexes)

For production systems with frequent queries:

```python
# Build search index once
processor = CorticalTextProcessor.load("corpus.pkl")
search_index = processor.build_search_index()

# Use pre-built index for 2-3x faster searches
results = processor.search_with_index("query", search_index)

# Or use fast_find_documents (automatic caching)
results = processor.fast_find_documents("query", top_n=10)
```

### Observability and Monitoring

Enable metrics collection for production monitoring:

```python
# Enable metrics tracking
processor = CorticalTextProcessor(enable_metrics=True)

# After processing
processor.process_document("doc1", "...")
processor.compute_all()
processor.find_documents_for_query("query")

# Get metrics summary
print(processor.get_metrics_summary())

# Access specific metrics
metrics = processor.get_metrics()
if "find_documents_for_query" in metrics:
    query_stats = metrics["find_documents_for_query"]
    print(f"Average query time: {query_stats['avg_ms']:.2f}ms")
    print(f"Total queries: {query_stats['count']}")
```

### Resource Limits

Set resource constraints for production:

```python
import resource

# Limit memory usage (in bytes)
memory_limit = 4 * 1024 * 1024 * 1024  # 4GB
resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

# Set CPU time limits (in seconds)
cpu_limit = 3600  # 1 hour
resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
```

### Corpus Persistence Strategy

```python
# Strategy 1: Regular checkpoints
def process_with_checkpoints(documents, checkpoint_interval=1000):
    processor = CorticalTextProcessor()

    for i, (doc_id, text) in enumerate(documents):
        processor.process_document(doc_id, text)

        if (i + 1) % checkpoint_interval == 0:
            processor.compute_all()
            processor.save(f"checkpoint_{i+1}.pkl")

    processor.compute_all()
    processor.save("final.pkl")
    return processor

# Strategy 2: Git-friendly chunk storage
from cortical.chunk_index import ChunkBasedIndex

chunk_index = ChunkBasedIndex("corpus_chunks/")
processor = CorticalTextProcessor()

# Add documents
for doc_id, text in documents:
    processor.process_document(doc_id, text)

# Save using chunk storage (git-friendly)
processor.compute_all()
chunk_index.save_processor(processor)

# Load from chunks
processor = chunk_index.load_processor()
```

---

## Configuration Options

### Core Configuration Parameters

The `CorticalConfig` dataclass provides comprehensive configuration:

#### PageRank Settings

Control graph-based term importance:

```python
config = CorticalConfig(
    pagerank_damping=0.85,      # Damping factor (0-1). Higher = more weight to links
    pagerank_iterations=20,     # Max iterations before stopping
    pagerank_tolerance=1e-6,    # Convergence threshold
)
```

**Tuning Tips:**
- Increase `pagerank_damping` (0.90) for more authority-based ranking
- Decrease `pagerank_iterations` (10) for faster computation
- Increase `pagerank_tolerance` (1e-4) for looser convergence

#### Clustering Settings

Control concept cluster formation:

```python
config = CorticalConfig(
    min_cluster_size=3,         # Minimum nodes per cluster
    cluster_strictness=1.0,     # Aggressiveness (0.0-1.0)
    louvain_resolution=2.0,     # Higher = more, smaller clusters
)
```

**Tuning Tips:**
- For large corpora: `louvain_resolution=1.0` (fewer, larger clusters)
- For fine-grained concepts: `louvain_resolution=5.0` (more, smaller clusters)
- Typical range: 1.0-10.0

#### Query Expansion

Control semantic query expansion:

```python
config = CorticalConfig(
    max_query_expansions=10,            # Max expansion terms
    semantic_expansion_discount=0.7,    # Weight for semantic expansions
)
```

#### Scoring Algorithms

Choose between TF-IDF and BM25:

```python
# BM25 (recommended, better for code search)
config = CorticalConfig(
    scoring_algorithm='bm25',
    bm25_k1=1.2,    # Term frequency saturation (0.0-3.0)
    bm25_b=0.75,    # Length normalization (0.0-1.0)
)

# Traditional TF-IDF
config = CorticalConfig(
    scoring_algorithm='tfidf',
)
```

**BM25 Parameters:**
- `bm25_k1=1.2`: Default saturation. Higher (2.0) = more weight to term frequency
- `bm25_b=0.75`: Default normalization. Lower (0.4) = less length penalty
- `bm25_b=0.0`: Disable length normalization entirely

#### Chunking for RAG

Control text passage chunking:

```python
config = CorticalConfig(
    chunk_size=512,      # Characters per chunk
    chunk_overlap=128,   # Overlap between chunks
)
```

**Tuning Tips:**
- For LLMs with large context: `chunk_size=1024, chunk_overlap=256`
- For precise retrieval: `chunk_size=256, chunk_overlap=64`
- Ensure `chunk_overlap < chunk_size`

#### Gap Detection Thresholds

Control knowledge gap analysis:

```python
config = CorticalConfig(
    isolation_threshold=0.02,           # Below = isolated document
    well_connected_threshold=0.03,      # Above = well-integrated
    weak_topic_tfidf_threshold=0.005,   # Significant topic threshold
    bridge_similarity_min=0.005,        # Min similarity for bridges
    bridge_similarity_max=0.03,         # Max similarity for bridges
)
```

#### Semantic Relation Weights

Control influence of relation types:

```python
config = CorticalConfig(
    relation_weights={
        'IsA': 1.5,          # Hierarchical relations
        'PartOf': 1.2,       # Compositional relations
        'SimilarTo': 1.3,    # Similarity relations
        'HasA': 1.0,         # Possession relations
        'UsedFor': 0.8,      # Functional relations
        'CapableOf': 0.7,    # Capability relations
        'HasProperty': 1.1,  # Property relations
        'RelatedTo': 1.0,    # General relations
        'Causes': 1.0,       # Causal relations
        'Antonym': 0.3,      # Opposite relations (low weight)
    }
)
```

### Complete Example

Production configuration for a code search system:

```python
from cortical import CorticalTextProcessor, CorticalConfig

# Code search optimized configuration
config = CorticalConfig(
    # PageRank: Fast convergence
    pagerank_iterations=15,
    pagerank_tolerance=1e-4,

    # Clustering: Moderate granularity
    louvain_resolution=1.5,
    min_cluster_size=4,

    # Scoring: BM25 optimized for code
    scoring_algorithm='bm25',
    bm25_k1=1.5,  # Higher weight to term frequency
    bm25_b=0.5,   # Lower length penalty

    # Chunking: Medium chunks for code context
    chunk_size=768,
    chunk_overlap=192,

    # Query: Moderate expansion
    max_query_expansions=7,
    semantic_expansion_discount=0.8,

    # Semantic relations: Boost code-relevant relations
    relation_weights={
        'IsA': 1.8,
        'PartOf': 1.5,
        'HasA': 1.2,
        'UsedFor': 1.2,
        'CapableOf': 1.0,
        'SimilarTo': 1.4,
    }
)

processor = CorticalTextProcessor(config=config)
```

---

## Environment Variables

Control behavior via environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `CORTICAL_CORPUS_PATH` | Path to corpus pickle file | None (empty corpus) |
| `CORTICAL_LOG_LEVEL` | Logging level | `INFO` |
| `ML_COLLECTION_ENABLED` | Enable ML data collection | `1` (enabled) |

Example:

```bash
export CORTICAL_CORPUS_PATH=/data/corpus.pkl
export CORTICAL_LOG_LEVEL=DEBUG
python -m cortical.mcp_server
```

---

## Troubleshooting

### Common Issues

**1. Memory Issues with Large Corpora**

```python
# Use incremental indexing
processor.add_document_incremental(doc_id, text, recompute='tfidf')

# Instead of full compute
# processor.process_document(doc_id, text)
# processor.compute_all()
```

**2. Slow PageRank Convergence**

```python
# Reduce iterations or loosen tolerance
config = CorticalConfig(
    pagerank_iterations=10,
    pagerank_tolerance=1e-4,
)
```

**3. Too Many/Too Few Clusters**

```python
# Adjust louvain_resolution
config = CorticalConfig(
    louvain_resolution=1.0,  # Fewer clusters
    # or
    louvain_resolution=5.0,  # More clusters
)
```

**4. MCP Server Connection Issues**

```bash
# Check Python version (3.11+ required)
python --version

# Verify MCP installation
python -c "import mcp; print('MCP OK')"

# Check Claude Desktop logs
tail -f ~/Library/Logs/Claude/mcp*.log  # macOS
```

---

## Next Steps

- **Basic Usage**: See [docs/quickstart.md](quickstart.md) for quick examples
- **Advanced Features**: See [docs/cookbook.md](cookbook.md) for patterns and recipes
- **Query Guide**: See [docs/query-guide.md](query-guide.md) for search capabilities
- **Architecture**: See [docs/architecture.md](architecture.md) for system design
- **MCP Server**: See [MCP_SERVER_README.md](../MCP_SERVER_README.md) for detailed MCP setup

---

**Questions?** Open an issue at https://github.com/scrawlsbenches/Opus-code-test/issues
