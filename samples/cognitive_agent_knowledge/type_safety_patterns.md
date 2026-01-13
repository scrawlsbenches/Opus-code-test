# Type Safety Patterns in Cortical

## The Core Problem

Python's dynamic typing means APIs can return different types:
- Sometimes a dict, sometimes an object
- Sometimes a string ID, sometimes the full entity
- Old code expects one type, new code returns another

This causes runtime errors like:
- `AttributeError: 'MyClass' object has no attribute 'get'`
- `TypeError: 'MyClass' object is not subscriptable`

## Pattern 1: Safe Attribute Access with getattr()

### Problem
```python
# BROKEN: Assumes object has .get() method
status = handoff.get("status", "?")  # AttributeError!
```

### Solution
```python
# SAFE: Works for any object
status = getattr(handoff, 'status', '?')
```

### When to Use
- Accessing attributes that might not exist
- Working with objects from APIs that might change
- Defensive programming in CLI handlers

## Pattern 2: Type-Aware Access

### Problem
```python
# BROKEN: Code path depends on hidden type
value = data.get("key")  # Works if dict, fails if object
```

### Solution
```python
# SAFE: Check type first
if isinstance(data, dict):
    value = data.get("key", default)
else:
    value = getattr(data, "key", default)
```

### Compact Version
```python
value = data.get("key", default) if isinstance(data, dict) else getattr(data, "key", default)
```

## Pattern 3: Extract ID from Result

### Problem
APIs that create entities might return:
- Just the ID string: `"H-20260112-..."`
- The full entity object: `Handoff(id='H-20260112-...', ...)`

```python
# BROKEN: Prints full object if entity returned
result = manager.create_something()
print(f"Created: {result}")  # Might print entire object!
```

### Solution
```python
# SAFE: Extract ID regardless of return type
result = manager.create_something()
result_id = result.id if hasattr(result, 'id') else str(result)
print(f"Created: {result_id}")
```

## Pattern 4: Python inspect API

Before using unfamiliar APIs, inspect them:

```python
import inspect

# Check function signature
sig = inspect.signature(manager.initiate_handoff)
print(sig)  # Shows parameters and defaults

# Check what a function returns
source = inspect.getsource(manager.initiate_handoff)
# Look for 'return' statements
```

### Quick CLI Check
```bash
python3 -c "
import inspect
from cortical.got.api import GoTManager
sig = inspect.signature(GoTManager.initiate_handoff)
print(sig)
"
```

## Pattern 5: Dataclass Field Access

Dataclasses use attribute access, not dict access:

```python
from dataclasses import dataclass

@dataclass
class Handoff:
    id: str
    status: str

h = Handoff(id="123", status="pending")

# WRONG
h.get("status")  # AttributeError!
h["status"]      # TypeError!

# RIGHT
h.status         # Works
getattr(h, "status")  # Also works
```

## Real Examples from This Codebase

### Example 1: Handoff CLI (Fixed)
```python
# BEFORE (cortical/got/cli/handoff.py)
status = handoff.get("status", "?")

# AFTER
status = getattr(handoff, 'status', '?')
```

### Example 2: KT List Command
```python
# Safe access pattern used in knowledge_transfer.py
kt_id = kt.get("id", "?") if isinstance(kt, dict) else getattr(kt, "id", "?")
```

### Example 3: Query Results
```python
# query.py handles both dict and object results
source_id = edge.source_id if hasattr(edge, 'source_id') else edge.get('source_id')
```

## Prevention Checklist

1. **New CLI handlers**: Use `getattr()` for all entity attribute access
2. **API changes**: When changing return types, grep for usages first
3. **Tests**: Add tests that verify attribute access works
4. **Type hints**: Add type hints to catch issues at lint time
5. **Defensive defaults**: Always provide defaults in `getattr(obj, 'attr', default)`

## Quick Reference

| Want to do | Dict | Object | Safe for both |
|------------|------|--------|---------------|
| Get value | `d.get("k")` | `o.k` | `getattr(o, "k", None)` |
| Check exists | `"k" in d` | `hasattr(o, "k")` | `hasattr(o, "k")` |
| Get with default | `d.get("k", v)` | `getattr(o, "k", v)` | `getattr(o, "k", v)` |
| Set value | `d["k"] = v` | `o.k = v` | `setattr(o, "k", v)` |
