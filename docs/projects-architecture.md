# Projects Architecture

## Overview

The **Projects architecture** is a design pattern for isolating optional features from the core library. Projects are self-contained extensions that:

- **Have zero impact on core** - The core library (`cortical/`) has zero runtime dependencies
- **Are opt-in via installation** - Users install only the projects they need
- **Can fail independently** - Project test failures don't block core releases
- **Follow a consistent pattern** - All projects use the same structure and conventions

**Key principle:** Keep the core library lean, fast, and dependency-free. Everything else is a project.

## Why Projects?

### Problem: Dependency Bloat

Without projects, adding features like MCP server support or Protobuf serialization would require:
- Adding dependencies to the core library
- Increasing installation size for all users
- Creating compatibility issues across Python versions
- Making the library harder to audit and maintain

### Solution: Isolated Projects

Projects solve this by:
1. **Isolation** - Each project has its own dependencies defined separately
2. **Optional Installation** - Users choose `pip install cortical-text-processor[mcp]` for MCP support
3. **Independent Testing** - MCP tests can fail without breaking core CI
4. **Clear Boundaries** - Projects import from core, never the reverse

## Directory Structure

```
cortical/
├── __init__.py                      # Core library (zero dependencies)
├── processor/                       # Main API
├── query/                           # Search algorithms
├── analysis.py                      # Graph algorithms
├── ...                              # Other core modules
└── projects/                        # Optional extensions
    ├── __init__.py                  # Projects registry
    ├── mcp/                         # Model Context Protocol project
    │   ├── __init__.py              # Graceful degradation on missing deps
    │   ├── server.py                # MCP server implementation
    │   └── tests/
    │       ├── __init__.py
    │       └── test_server.py       # Project-specific tests
    ├── proto/                       # Protobuf serialization project
    │   ├── __init__.py              # Graceful degradation
    │   ├── serialization.py         # Protobuf implementation
    │   └── tests/
    │       └── __init__.py
    └── cli/                         # CLI tools project (placeholder)
        └── __init__.py              # Future: CLI-specific features
```

## Available Projects

### MCP Project

**Purpose:** Expose the Cortical Text Processor as MCP (Model Context Protocol) tools for AI assistants like Claude Desktop.

**Location:** `cortical/projects/mcp/`

**Dependencies:**
- `mcp>=1.0` - MCP SDK
- `pydantic` - Data validation (transitive from mcp)
- `httpx` - HTTP client (transitive from mcp)
- `uvicorn` - ASGI server (transitive from mcp)
- `starlette` - Web framework (transitive from mcp)

**Installation:**
```bash
pip install cortical-text-processor[mcp]
```

**Usage:**
```python
from cortical.projects.mcp import CorticalMCPServer

server = CorticalMCPServer(corpus_path="corpus_dev.json")
server.run()
```

**Provides:**
- `search` - Find relevant documents for a query
- `passages` - Retrieve RAG-ready text passages
- `expand_query` - Get query expansion terms
- `corpus_stats` - Get corpus statistics
- `add_document` - Index a new document

**Tests:** `cortical/projects/mcp/tests/test_server.py`

### Proto Project

**Purpose:** Provide Protobuf serialization for cross-language interoperability.

**Location:** `cortical/projects/proto/`

**Dependencies:**
- `protobuf>=4.0` - Protocol Buffers runtime

**Installation:**
```bash
pip install cortical-text-processor[proto]
```

**Usage:**
```python
from cortical.projects.proto import to_proto, from_proto

# Serialize processor state
proto_bytes = to_proto(processor)

# Deserialize
processor = from_proto(proto_bytes)
```

**Provides:**
- `to_proto()` - Serialize CorticalTextProcessor to protobuf bytes
- `from_proto()` - Deserialize protobuf bytes to CorticalTextProcessor

**Tests:** `tests/unit/test_protobuf_serialization.py` (marked `@pytest.mark.optional`)

### CLI Project

**Purpose:** Command-line interface tools and wrappers (placeholder for future migration).

**Location:** `cortical/projects/cli/`

**Status:** Placeholder only. CLI wrapper is currently in core (`cortical/cli_wrapper.py`) and may be migrated here in a future sprint if it becomes problematic.

**Dependencies:** None yet (potential: `click` for enhanced CLI)

**Note:** This project exists to demonstrate the pattern and reserve namespace for future CLI-specific features.

## Adding a New Project

Follow these steps to add a new project:

### 1. Create Project Directory

```bash
mkdir -p cortical/projects/your_project
mkdir -p cortical/projects/your_project/tests
```

### 2. Create `__init__.py` with Graceful Degradation

```python
# cortical/projects/your_project/__init__.py
"""
YourProject - Brief description.

This project provides [feature description].

Installation:
    pip install cortical-text-processor[your_project]

Usage:
    from cortical.projects.your_project import YourFeature

Dependencies:
    - some-package>=1.0
    - another-package>=2.0
"""

try:
    from .implementation import YourFeature
    __all__ = ['YourFeature']
except ImportError as e:
    # Dependencies not installed - provide helpful error
    def _missing_deps(*args, **kwargs):
        raise ImportError(
            "YourProject dependencies not installed. "
            "Install with: pip install cortical-text-processor[your_project]"
        ) from e
    YourFeature = _missing_deps
    __all__ = ['YourFeature']
```

**Key pattern:** The project must gracefully degrade when dependencies are missing. Users should get a clear error message telling them how to install the dependencies.

### 3. Implement Your Feature

```python
# cortical/projects/your_project/implementation.py
from cortical.processor import CorticalTextProcessor

class YourFeature:
    """Your feature implementation."""

    def __init__(self, processor: CorticalTextProcessor):
        self.processor = processor

    def do_something(self):
        # Use the processor
        return self.processor.find_documents_for_query("example")
```

### 4. Add Tests

```python
# cortical/projects/your_project/tests/test_implementation.py
import pytest

@pytest.mark.optional
@pytest.mark.your_project  # Custom marker for your project
class TestYourFeature:
    def test_basic_functionality(self):
        # Your tests here
        pass
```

**Important:** Mark tests with `@pytest.mark.optional` so they're skipped during development if dependencies aren't installed.

### 5. Update `pyproject.toml`

Add your project's dependencies as an optional extra:

```toml
[project.optional-dependencies]
dev = [
    "coverage>=7.0",
    "pytest>=7.0",
    # ... existing dev deps
]

# Add your project
your_project = [
    "some-package>=1.0",
    "another-package>=2.0",
]
```

### 6. Update Test Markers

Add your project's marker to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "optional: Tests requiring optional dependencies (mcp, hypothesis, numpy)",
    "your_project: YourProject tests (requires some-package)",
    # ... other markers
]
```

### 7. Register in Projects Index

Update `cortical/projects/__init__.py`:

```python
"""
Cortical Projects - Optional extensions to the core library.

Available Projects:
- mcp: Model Context Protocol server integration
- proto: Protobuf serialization support
- cli: Command-line interface tools
- your_project: Your feature description
"""

__all__ = ['mcp', 'proto', 'cli', 'your_project']
```

### 8. Add CI Testing (Optional)

If your project needs dedicated CI testing, add a job to `.github/workflows/ci.yml`:

```yaml
your-project-tests:
  name: "🔌 YourProject Tests"
  runs-on: ubuntu-latest
  needs: smoke-tests
  steps:
  - uses: actions/checkout@v4

  - name: Set up Python 3.11
    uses: actions/setup-python@v5
    with:
      python-version: '3.11'

  - name: Install dependencies
    run: |
      python -m pip install --upgrade pip
      pip install -e ".[dev,your_project]"

  - name: Run YourProject tests
    run: |
      python -m pytest cortical/projects/your_project/tests/ -v --tb=short
```

## Installation

### Core Library Only (Zero Dependencies)

```bash
pip install cortical-text-processor
```

Installs only the core library with zero runtime dependencies.

### With Specific Projects

```bash
# Install with MCP support
pip install cortical-text-processor[mcp]

# Install with Protobuf support
pip install cortical-text-processor[proto]

# Install multiple projects
pip install cortical-text-processor[mcp,proto]
```

### Development Installation

```bash
# Install core + dev dependencies (includes mcp for testing)
pip install -e ".[dev]"

# Install with all projects
pip install -e ".[dev,mcp,proto]"
```

## CI/CD Integration

### Test Organization

Projects follow the same test organization as core:

| Test Location | When Run | Coverage |
|--------------|----------|----------|
| `cortical/projects/*/tests/` | CI only (marked `@pytest.mark.optional`) | No |
| `tests/unit/test_*_project.py` | CI only (marked `@pytest.mark.optional`) | Yes |
| `tests/integration/test_*_project.py` | CI only (marked `@pytest.mark.optional`) | Yes |

### CI Behavior

**Development (local):**
```bash
# Default: excludes optional tests (via pyproject.toml addopts)
pytest tests/

# Projects tests are skipped - dependencies may not be installed
```

**CI (GitHub Actions):**
```bash
# CI overrides addopts with -m "" to run ALL tests
pytest tests/ -m ""

# Projects tests run if dependencies are installed in CI
```

**Principle:** Core tests always run and must pass. Project tests are optional and can fail independently.

### Dependency Installation in CI

Projects that need CI testing must install their dependencies explicitly:

```yaml
# .github/workflows/ci.yml
- name: Install test dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -e ".[dev]"  # Includes mcp for testing
```

The `dev` extra includes MCP dependencies because MCP tests are part of integration testing. Other projects may need separate CI jobs if they have heavy dependencies.

## Design Principles

### 1. Core Stays Lean

**Rule:** The core library MUST have zero runtime dependencies.

**Why:**
- Fast installation
- Minimal security surface
- Easy to audit
- Works everywhere Python runs

**Enforcement:** Any new dependency must go in a project, not core.

### 2. Projects Are Opt-In

**Rule:** Projects MUST be installable independently via `pip install cortical-text-processor[project_name]`.

**Why:**
- Users only install what they need
- Smaller installation footprint
- Fewer compatibility issues

**Enforcement:** Dependencies must be in `[project.optional-dependencies]`, not `dependencies`.

### 3. Graceful Degradation

**Rule:** Projects MUST handle missing dependencies gracefully with helpful error messages.

**Why:**
- Users shouldn't see confusing import errors
- Clear path to resolution
- Better developer experience

**Enforcement:** All project `__init__.py` files must use the try/except pattern shown above.

### 4. Import Direction

**Rule:** Projects can import from core. Core MUST NEVER import from projects.

**Why:**
- Prevents circular dependencies
- Keeps core independent
- Makes dependency tree clear

**Enforcement:** Code review and CI checks.

### 5. Independent Testing

**Rule:** Project tests MUST be marked `@pytest.mark.optional` so they can be skipped if dependencies aren't installed.

**Why:**
- Developers can run tests without installing all projects
- Faster local testing
- Projects can fail in CI without blocking core

**Enforcement:** Test markers in `pyproject.toml` and CI configuration.

### 6. Consistent Structure

**Rule:** All projects MUST follow the same directory structure and patterns.

**Why:**
- Predictable for developers
- Easy to maintain
- Clear conventions

**Enforcement:** This documentation and code review.

## Common Patterns

### Pattern: Optional Dependency Error

All projects use this pattern to handle missing dependencies:

```python
try:
    from .implementation import Feature
    __all__ = ['Feature']
except ImportError as e:
    def _missing_deps(*args, **kwargs):
        raise ImportError(
            "Feature dependencies not installed. "
            "Install with: pip install cortical-text-processor[project_name]"
        ) from e
    Feature = _missing_deps
    __all__ = ['Feature']
```

This provides:
- Clear error message with installation command
- Preserves import chaining (`from e`)
- Keeps `__all__` consistent
- Maintains API surface even when unavailable

### Pattern: Project-Specific Tests

Mark all project tests with appropriate markers:

```python
import pytest

@pytest.mark.optional  # Skip if dependencies missing
@pytest.mark.mcp       # Project-specific marker
class TestMCPServer:
    def test_feature(self):
        pass
```

### Pattern: Transitive Dependencies

If your project depends on a package that has many transitive dependencies, list only the direct dependency:

```toml
[project.optional-dependencies]
mcp = [
    "mcp>=1.0",  # This brings in pydantic, httpx, uvicorn, starlette
]
```

Don't list transitive dependencies unless you need a specific version.

## Migration Guide: Core → Project

If a feature in core needs to become a project:

1. **Create project directory** following structure above
2. **Move implementation** to `cortical/projects/project_name/`
3. **Add graceful degradation** in project `__init__.py`
4. **Update core imports** to use project
5. **Move tests** to project tests directory or mark with `@pytest.mark.optional`
6. **Update pyproject.toml** with dependencies
7. **Test both scenarios:**
   - Core without project installed
   - Core with project installed
8. **Update documentation** to reflect new import path

**Example:** The CLI project is a placeholder for future migration of `cortical/cli_wrapper.py` if it accumulates dependencies.

## Troubleshooting

### "ImportError: Project dependencies not installed"

**Cause:** You're trying to use a project without installing its dependencies.

**Solution:** Install the project: `pip install cortical-text-processor[project_name]`

### Tests Failing in CI but Passing Locally

**Cause:** CI runs with `-m ""` which includes optional tests. Locally you may be skipping them.

**Solution:** Run locally with: `pytest tests/ -m ""`

### Project Tests Not Running in CI

**Cause:** Dependencies not installed in CI job.

**Solution:** Add project to CI installation: `pip install -e ".[dev,project_name]"`

### Circular Import Between Core and Project

**Cause:** Core is importing from project (forbidden).

**Solution:** Refactor so only project imports from core, never the reverse.

## Future Expansion

Potential future projects:

- **numpy**: NumPy-accelerated embeddings (`cortical/projects/numpy/`)
- **transformers**: Hugging Face model integration (`cortical/projects/transformers/`)
- **fastapi**: REST API server (`cortical/projects/fastapi/`)
- **cli**: Full CLI migration from core (`cortical/projects/cli/`)

Each would follow the same pattern established here.

## Summary

The Projects architecture provides:

✅ **Zero-dependency core** - Fast, auditable, works everywhere
✅ **Opt-in features** - Users install only what they need
✅ **Independent testing** - Projects can fail without blocking core
✅ **Graceful degradation** - Clear errors with solutions
✅ **Consistent patterns** - Predictable structure across all projects

**Remember:** When adding new functionality, ask: "Should this be core or a project?" If it adds dependencies, it's a project.
