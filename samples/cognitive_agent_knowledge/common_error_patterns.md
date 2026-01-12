# Common Error Patterns in Cortical Codebase

## Overview

This document catalogs common error patterns encountered during development,
their root causes, and proven fixes. Use this as a reference when debugging.

## 1. Type Confusion: Object vs Dict Access

### The Pattern
```python
# BROKEN: Treating object as dict
value = obj.get("key")  # AttributeError: 'MyClass' object has no attribute 'get'
value = obj["key"]      # TypeError: 'MyClass' object is not subscriptable

# FIXED: Use attribute access
value = getattr(obj, 'key', default)
value = obj.key
```

### Real Example: Handoff Show Bug
The `cmd_handoff_show` function was treating Handoff objects as dicts:
```python
# BROKEN (cortical/got/cli/handoff.py:178)
if h.get("id") == handoff_id:

# FIXED
if h.id == handoff_id:
```

### Prevention Pattern
```python
# Safe access that works for both objects and dicts
def safe_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
```

## 2. Missing Full Identifiers in Tokenization

### The Pattern
When tokenizing code, we need BOTH:
- Full identifier: `texttoatomsbridge` (for exact matches)
- Split parts: `text`, `to`, `atoms`, `bridge` (for semantic search)

### The Bug
Original `split_identifier` only returned split parts:
```python
split_identifier("TextToAtomsBridge")
# Returned: ['text', 'to', 'atoms', 'bridge']
# Missing: 'texttoatomsbridge'
```

### The Fix (text_bridge.py)
```python
def replace_identifier(match):
    identifier = match.group(0)
    if '_' in identifier or any(c.isupper() for c in identifier[1:]):
        parts = split_identifier(identifier)
        full_lower = identifier.lower().replace('_', '')
        # Include BOTH full identifier AND split parts
        return f"{full_lower} {' '.join(parts)}"
    return identifier.lower()
```

## 3. Generation Loop Detection

### The Pattern
Text generation falls into loops due to high-confidence paths:
```
self → dict → str → any → none → if → not → in → self (loops back!)
```

### Root Cause
Python type annotations (`Dict[str, Any]`, `Optional[str]`) are extremely
common in the training data, creating strong FOLLOWS links.

### The Fix: N-gram Detection
```python
seen_ngrams = set()
for i in range(max_tokens):
    # Check if adding next_word creates a repeating trigram
    if len(generated) >= 2:
        test_ngram = tuple(generated[-2:] + [next_word])
        if test_ngram in seen_ngrams:
            # Try alternative candidate or stop
            ...
    seen_ngrams.add(tuple(generated[-3:]))
```

## 4. Train/Reindex Order Matters

### The Pattern
```bash
# WRONG ORDER - reindex calculates IDF on stale data
python -m cortical.cognitive reindex
python -m cortical.cognitive train cortical/

# CORRECT ORDER - train first, then recalculate IDF
python -m cortical.cognitive train cortical/
python -m cortical.cognitive reindex
```

### Why It Matters
- `train` creates word atoms and raw similarity links
- `reindex` calculates IDF (Inverse Document Frequency) weights
- IDF weights determine which words are "rare and meaningful"
- If you reindex before training, you're weighting old vocabulary

## 5. WAL Commit Order

### The Critical Bug
Writing entities before WAL fsync can cause data loss on crash:
```python
# BROKEN (data loss on crash)
store.save(entity)           # Write entity
wal.log_tx_commit(tx_id)     # Log to WAL
wal.fsync()                  # Fsync WAL

# FIXED (crash-safe)
wal.log_tx_commit(tx_id)     # Log to WAL
wal.fsync()                  # Fsync WAL - THIS IS THE COMMIT POINT
store.save(entity)           # Write entity (can be recovered if crash here)
```

### The Rule
**WAL-first**: The transaction IS committed once the WAL record is durable.
Entity writes can be replayed from WAL during recovery.

## 6. Process Lock vs Thread Lock

### When to Use Each
| Lock Type | Use Case | Scope |
|-----------|----------|-------|
| `threading.Lock()` | Same process, multiple threads | In-memory |
| `ProcessLock` | Multiple processes | File-based (fcntl) |

### The Pattern: Use Both for Critical Sections
```python
# Storage layer uses both for full safety
with self._write_lock:           # Thread safety
    with self._write_process_lock:  # Process safety
        # Critical section
```

## 7. CLI Output: Full Object vs ID

### The Bug
```python
# BROKEN: Prints full object representation
handoff_id = manager.initiate_handoff(...)
print(f"Created: {handoff_id}")  # Prints entire Handoff(...) object!

# FIXED: Extract ID from object
handoff = manager.initiate_handoff(...)
handoff_id = handoff.id if hasattr(handoff, 'id') else str(handoff)
print(f"Created: {handoff_id}")  # Prints just "H-20260112-..."
```

## Quick Debugging Checklist

1. **AttributeError on .get()**: Object isn't a dict, use getattr()
2. **Search not finding code**: Check if full identifier is preserved
3. **Generation loops**: Check n-gram detection, try higher temperature
4. **Stale search results**: Run train then reindex in that order
5. **Data loss after crash**: Verify WAL-first commit pattern
6. **Race conditions**: Verify both thread and process locks used
7. **Weird CLI output**: Check if printing object vs ID
