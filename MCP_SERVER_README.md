# MCP Server for Cortical Text Processor

This document describes how to use the MCP (Model Context Protocol) server for the Cortical Text Processor with Claude Desktop and other AI agents.

## Overview

The Cortical MCP Server provides AI agents with direct access to the Cortical Text Processor's semantic search and text analysis capabilities through a standardized protocol.

### Available Tools

The server exposes 5 tools:

1. **search** - Find documents relevant to a query
2. **passages** - Retrieve RAG-ready text passages
3. **expand_query** - Get query expansion terms using semantic connections
4. **corpus_stats** - Get statistics about the current corpus
5. **add_document** - Index a new document incrementally

## Installation

### Prerequisites

```bash
# Install the MCP SDK
pip install mcp

# Or install from the repository with dev dependencies
pip install -e ".[dev]"
```

The MCP server requires Python 3.11+ and the following dependencies (installed automatically with `mcp`):
- anyio
- httpx
- pydantic
- jsonschema

## Usage

### Running the Server

#### Command Line

```bash
# Start with an empty corpus
python -m cortical.mcp_server

# Or load a pre-indexed corpus
CORTICAL_CORPUS_PATH=/path/to/corpus.pkl python -m cortical.mcp_server
```

#### Programmatically

```python
from cortical.mcp_server import create_mcp_server

# Create server with empty corpus
server = create_mcp_server()
server.run(transport="stdio")

# Or load existing corpus
server = create_mcp_server(corpus_path="corpus.pkl")
server.run(transport="stdio")
```

### Configuration for Claude Desktop

Add the following to your Claude Desktop MCP configuration file:

**Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Configuration:**

```json
{
  "mcpServers": {
    "cortical-text-processor": {
      "command": "python",
      "args": [
        "-m",
        "cortical.mcp_server"
      ],
      "env": {
        "CORTICAL_CORPUS_PATH": "/absolute/path/to/your/corpus.pkl",
        "CORTICAL_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**For empty corpus (dynamic document addition):**

```json
{
  "mcpServers": {
    "cortical-text-processor": {
      "command": "python",
      "args": [
        "-m",
        "cortical.mcp_server"
      ],
      "env": {
        "CORTICAL_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## Tool Reference

### 1. search

Find documents relevant to a query.

**Parameters:**
- `query` (string, required): Search query string
- `top_n` (integer, optional): Number of results to return (default: 5)

**Returns:**
```json
{
  "results": [
    {"doc_id": "doc1", "score": 0.95},
    {"doc_id": "doc2", "score": 0.82}
  ],
  "count": 2
}
```

**Example:**
```python
# In Claude or another AI agent using MCP:
# "Search for documents about neural networks"
# -> Uses search(query="neural networks", top_n=5)
```

### 2. passages

Retrieve relevant text passages for RAG systems.

**Parameters:**
- `query` (string, required): Search query string
- `top_n` (integer, optional): Number of passages to return (default: 5)
- `chunk_size` (integer, optional): Size of text chunks in characters

**Returns:**
```json
{
  "passages": [
    {
      "doc_id": "doc1",
      "text": "Neural networks are...",
      "start": 0,
      "end": 200,
      "score": 0.95
    }
  ],
  "count": 1
}
```

**Example:**
```python
# "Find passages explaining PageRank algorithm"
# -> Uses passages(query="PageRank algorithm", top_n=3)
```

### 3. expand_query

Expand a query with semantically related terms.

**Parameters:**
- `query` (string, required): Query string to expand
- `max_expansions` (integer, optional): Maximum expansion terms (default: 10)

**Returns:**
```json
{
  "expansions": {
    "neural": 1.0,
    "network": 0.85,
    "deep": 0.72,
    "learning": 0.68
  },
  "count": 4
}
```

**Example:**
```python
# "What terms are related to 'machine learning'?"
# -> Uses expand_query(query="machine learning")
```

### 4. corpus_stats

Get statistics about the corpus.

**Parameters:** None

**Returns:**
```json
{
  "document_count": 125,
  "layer_stats": {
    "TOKENS": {"count": 5420},
    "BIGRAMS": {"count": 8930},
    "CONCEPTS": {"count": 42},
    "DOCUMENTS": {"count": 125}
  }
}
```

**Example:**
```python
# "How many documents are indexed?"
# -> Uses corpus_stats()
```

### 5. add_document

Index a new document with incremental updates.

**Parameters:**
- `doc_id` (string, required): Unique identifier for the document
- `content` (string, required): Document text content
- `recompute` (string, optional): Recomputation level - 'none', 'tfidf', or 'full' (default: 'tfidf')

**Returns:**
```json
{
  "stats": {
    "tokens": 234,
    "bigrams": 189,
    "unique_tokens": 156
  },
  "doc_id": "new_document"
}
```

**Example:**
```python
# "Index this new research paper about transformers..."
# -> Uses add_document(doc_id="paper_123", content="...", recompute="tfidf")
```

## Environment Variables

- `CORTICAL_CORPUS_PATH`: Path to pre-indexed corpus file (optional)
- `CORTICAL_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)

## Error Handling

All tools return error information in the response if something goes wrong:

```json
{
  "error": "Query must be a non-empty string",
  "results": [],
  "count": 0
}
```

## Example Workflows

### Building a Knowledge Base

```bash
# 1. Start server with empty corpus
python -m cortical.mcp_server

# 2. In Claude Desktop, add documents:
# "Add this document about neural networks: [paste content]"
# "Add this document about deep learning: [paste content]"

# 3. Search the indexed content:
# "Find documents about backpropagation"
```

### RAG-Powered Q&A

```bash
# 1. Load pre-indexed corpus
CORTICAL_CORPUS_PATH=docs_corpus.pkl python -m cortical.mcp_server

# 2. In Claude Desktop, ask questions:
# "Find relevant passages about the PageRank algorithm"
# -> Receives passages tool results with actual text chunks
```

### Semantic Exploration

```bash
# "What terms are related to 'cortical processing'?"
# -> expand_query returns: cortical, processing, hierarchical, layers, neurons, etc.

# "Search for documents using those expanded terms"
# -> Uses expansion results for broader search
```

## Testing

Run the test suite to verify the MCP server:

```bash
# Run all MCP server tests
python -m pytest tests/test_mcp_server.py -v

# Test specific functionality
python -m pytest tests/test_mcp_server.py::TestSearchTool -v
```

## Troubleshooting

### Server won't start

- Verify MCP is installed: `pip install mcp`
- Check Python version: `python --version` (requires 3.11+)
- Verify corpus path exists if using `CORTICAL_CORPUS_PATH`

### Claude Desktop can't connect

- Check configuration file location and JSON syntax
- Use absolute paths, not relative paths
- Restart Claude Desktop after configuration changes
- Check logs: Claude Desktop shows connection status

### Tools return errors

- Empty query: Queries must be non-empty strings
- Invalid parameters: Check parameter types and ranges
- Missing corpus: Add documents first if starting with empty corpus

## Advanced Usage

### Custom Configuration

```python
from cortical import CorticalConfig
from cortical.mcp_server import create_mcp_server

config = CorticalConfig(
    max_query_expansions=20,
    chunk_size=300,
    chunk_overlap=75
)

server = create_mcp_server(config=config)
server.run()
```

### Different Transports

```python
# STDIO (default, for Claude Desktop)
server.run(transport="stdio")

# SSE (for web integrations)
server.run(transport="sse")

# HTTP (for REST-like access)
server.run(transport="streamable-http")
```

## See Also

- [Main README](README.md) - General Cortical Text Processor documentation
- [CLAUDE.md](CLAUDE.md) - Development guide and architecture
- [MCP Specification](https://modelcontextprotocol.io/) - Official MCP documentation
