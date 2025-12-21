# Cortical Projects - Optional Extensions

This directory contains **optional projects** that extend the core Cortical Text Processor library with additional features. Each project is self-contained and has its own dependencies.

## Quick Start

```bash
# Install core library (zero dependencies)
pip install cortical-text-processor

# Install with specific projects
pip install cortical-text-processor[mcp]      # MCP server
pip install cortical-text-processor[proto]    # Protobuf serialization
pip install cortical-text-processor[mcp,proto]  # Multiple projects
```

## Available Projects

### 🔌 MCP (Model Context Protocol)

**Location:** `cortical/projects/mcp/`
**Status:** ✅ Fully implemented
**Dependencies:** `mcp>=1.0`

Exposes the Cortical Text Processor as MCP tools for AI assistants like Claude Desktop.

```python
from cortical.projects.mcp import CorticalMCPServer

server = CorticalMCPServer(corpus_path="corpus_dev.json")
server.run()
```

**Provides:**
- `search` - Find relevant documents
- `passages` - Retrieve RAG-ready text passages
- `expand_query` - Get query expansion terms
- `corpus_stats` - Get corpus statistics
- `add_document` - Index a new document

**Installation:** `pip install cortical-text-processor[mcp]`

---

### 📦 Proto (Protobuf Serialization)

**Location:** `cortical/projects/proto/`
**Status:** ⚠️ Stub implementation (not yet functional)
**Dependencies:** `protobuf>=4.0`

Provides cross-language serialization using Protocol Buffers.

```python
from cortical.projects.proto import to_proto, from_proto

# Serialize processor state (not yet implemented)
proto_bytes = to_proto(processor)

# Deserialize (not yet implemented)
processor = from_proto(proto_bytes)
```

**Note:** This is currently a stub. Use JSON serialization instead:
```python
# Working alternative
processor.save("corpus.json")  # JSON format
processor = CorticalTextProcessor.load("corpus.json")
```

**Installation:** `pip install cortical-text-processor[proto]`

---

### 🖥️ CLI (Command-Line Interface)

**Location:** `cortical/projects/cli/`
**Status:** 📝 Placeholder only
**Dependencies:** None yet

Reserved for future CLI-specific features. Currently, CLI functionality is in the core library (`cortical/cli_wrapper.py`).

---

## Directory Structure

```
cortical/projects/
├── __init__.py                  # Projects registry
├── README.md                    # This file
├── mcp/                         # MCP server project
│   ├── __init__.py              # Graceful degradation
│   ├── server.py                # MCP server implementation
│   └── tests/
│       ├── __init__.py
│       └── test_server.py       # MCP tests
├── proto/                       # Protobuf project
│   ├── __init__.py              # Graceful degradation
│   ├── serialization.py         # Protobuf stub
│   └── tests/
│       ├── __init__.py
│       └── test_serialization.py
└── cli/                         # CLI project (placeholder)
    └── __init__.py
```

## Design Principles

1. **Core Stays Lean** - Zero runtime dependencies in core library
2. **Opt-In Installation** - Users install only what they need
3. **Graceful Degradation** - Clear errors when dependencies are missing
4. **Import Direction** - Projects import from core, never reverse
5. **Independent Testing** - Projects can fail without blocking core

## Adding a New Project

See [docs/projects-architecture.md](../../docs/projects-architecture.md) for detailed guide on adding new projects.

**Quick checklist:**
1. Create `cortical/projects/your_project/` directory
2. Add `__init__.py` with graceful degradation pattern
3. Implement your feature
4. Add tests marked with `@pytest.mark.optional`
5. Update `pyproject.toml` with dependencies
6. Register in `cortical/projects/__init__.py`

## Backward Compatibility

Old import paths are supported via shims:

```python
# Old (deprecated but still works)
from cortical.mcp_server import CorticalMCPServer

# New (recommended)
from cortical.projects.mcp import CorticalMCPServer
```

## Documentation

- **Architecture Guide:** [docs/projects-architecture.md](../../docs/projects-architecture.md)
- **Main README:** [README.md](../../README.md)
- **CLAUDE.md:** [CLAUDE.md](../../CLAUDE.md)

## Testing

Projects use optional test markers:

```bash
# Run all tests (including optional)
pytest tests/ -m ""

# Run only MCP tests
pytest tests/ -m "mcp"

# Run only proto tests
pytest tests/ -m "protobuf"

# Skip optional tests (default)
pytest tests/
```

## Support

For issues, questions, or contributions, see [CONTRIBUTING.md](../../CONTRIBUTING.md).
