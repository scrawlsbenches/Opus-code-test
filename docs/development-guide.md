# Development Guide

> **Gate**: Adding new features or modifying core structures? Read this first.

---

## Common Tasks

### Adding a New Analysis Function

1. Add function to `analysis.py` with proper signature:
   ```python
   def compute_your_analysis(
       layers: Dict[CorticalLayer, HierarchicalLayer],
       **kwargs
   ) -> Dict[str, Any]:
       """Your analysis description."""
       layer0 = layers[CorticalLayer.TOKENS]
       # Implementation
       return {'result': ..., 'stats': ...}
   ```

2. Add wrapper method to `CorticalTextProcessor` in the `processor/` package (appropriate mixin):
   ```python
   def compute_your_analysis(self, **kwargs) -> Dict[str, Any]:
       """Wrapper with docstring."""
       return compute_your_analysis(self.layers, **kwargs)
   ```

3. Add tests in `tests/test_analysis.py`

### Adding a New Query Function

1. Add to the `query/` package following existing patterns (e.g., `query/search.py`)
2. Use `get_expanded_query_terms()` helper for query expansion
3. Use `layer.get_by_id()` for O(1) lookups, not iteration
4. Add wrapper to the `processor/` package (likely `processor/query_api.py`)
5. Add tests in `tests/test_processor.py`

### Modifying Minicolumn Structure

1. Update `Minicolumn` class in `minicolumn.py`
2. Update `to_dict()` and `from_dict()` for persistence
3. Update `__slots__` if adding new fields
4. Increment state version in `persistence.py` if breaking change
5. Add migration logic for backward compatibility

---

## Code Style Guidelines

```python
# Imports: stdlib, then local
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .layers import CorticalLayer, HierarchicalLayer
from .minicolumn import Minicolumn

# Type hints on all public functions
def find_documents(
    query: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Find documents matching query.

    Args:
        query: Search query string
        layers: Dictionary of hierarchical layers
        top_n: Number of results to return

    Returns:
        List of (doc_id, score) tuples sorted by relevance
    """
    # Implementation
```

---

## API Exploration with inspect

When encountering unfamiliar APIs, **use Python's `inspect` module**:

```bash
# Check function signature
python3 -c "import inspect; from module import Class; print(inspect.signature(Class.__init__))"

# List public methods
python3 -c "from module import Class; print([m for m in dir(Class) if not m.startswith('_')])"

# Find source file
python3 -c "import inspect; from module import Class; print(inspect.getfile(Class))"

# Show class hierarchy
python3 -c "import inspect; from module import Class; print(inspect.getmro(Class))"
```

**Use inspect WHEN:**
- Calling a class/function you haven't used before
- Documentation is missing or unclear
- Verifying required vs optional parameters
- Understanding inheritance hierarchy

**Prefer inspect OVER:**
- Guessing parameter names or order
- Reading entire source files to find one signature
- Trial-and-error with function calls
