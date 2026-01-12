# Session Knowledge: Manifest Invalid Entry Bug Fix

*Date: 2026-01-12 | Branch: claude/recover-idf-cognitive-agent-gnH6g*

---

## What Was Fixed

The training manifest (`models/cognitive_agent/training_manifest.json`) contained an invalid entry with path `.` (current directory). This caused warnings during `rebuild-links`:

```
Warning: Could not process .: [Errno 21] Is a directory: '.'
```

## How I Discovered It

1. Ran cold-start validation: `./scripts/bootstrap_cognitive.sh`
2. Noticed warning about processing "." as a file
3. Inspected manifest with:
   ```python
   import json
   m = json.load(open('models/cognitive_agent/training_manifest.json'))
   problem = [k for k in m['documents'] if k in ['.', '..']]
   # Found: ['.']
   ```

## Root Cause Analysis

Two issues contributed:

### Issue 1: No validation in `add_document()`
The `TrainingManifest.add_document()` method accepted any path without validation. If a training run somehow included ".", it would be recorded.

### Issue 2: Buggy condition in `_run_rebuild_links()`
```python
# BEFORE (buggy)
elif Path("samples") / path.name:  # Always truthy!
    alt_path = Path("samples") / path.name
    if alt_path.exists():
        docs_to_process.append(alt_path)
```

`Path("samples") / path.name` creates a Path object, which is **always truthy** in Python regardless of whether the file exists. This meant the `elif` always executed.

## The Fix

### Fix 1: Guard in `add_document()` (training.py:132-134)
```python
def add_document(self, path: str, ...):
    # Skip invalid paths (directories, empty paths)
    if not path or path in ('.', '..') or path.endswith('/'):
        return
    self.documents[path] = TrainedDocument(...)
```

### Fix 2: Guard in `_run_rebuild_links()` (training.py:1650-1660)
```python
for doc_path in trainer.manifest.documents:
    # Skip invalid manifest entries
    if not doc_path or doc_path in ('.', '..') or doc_path.endswith('/'):
        continue
    path = Path(doc_path)
    if path.exists() and path.is_file():  # Added is_file() check
        docs_to_process.append(path)
    else:
        alt_path = Path("samples") / path.name
        if alt_path.exists() and alt_path.is_file():
            docs_to_process.append(alt_path)
```

### Fix 3: Clean the manifest
```python
# Remove problematic entries
del manifest['documents']['.']
manifest['total_documents'] = len(manifest['documents'])
# Save
```

## How I Knew What To Do

1. **Read the error message** - "Is a directory: '.'" told me exactly what failed
2. **Used Python inspect pattern** - Quick one-liner to find the bad entry
3. **Read the code** - `training.py:1650-1660` showed the rebuild loop
4. **Recognized the Python truthiness bug** - Path objects are always truthy
5. **Applied defensive programming** - Added guards at entry points

## Verification

After fix:
```bash
rm -rf models/cognitive_agent/bridge/
python -m cortical.cognitive rebuild-links --metrics
# No warnings, completes in ~9s
```

## Key Files

| File | Lines | What Changed |
|------|-------|--------------|
| `cortical/cognitive/training.py` | 132-134 | Guard in add_document() |
| `cortical/cognitive/training.py` | 1650-1660 | Guard in rebuild-links |
| `models/cognitive_agent/training_manifest.json` | - | Removed "." entry |

## Lessons for Future Sessions

1. **Path objects are always truthy** - Check `.exists()` and `.is_file()` explicitly
2. **Validate input at entry points** - add_document() should reject bad paths
3. **Inspect manifest directly** - Quick Python one-liner finds issues fast
4. **Test cold-start** - `./scripts/bootstrap_cognitive.sh` reveals many issues

## Commands Used

```bash
# Check manifest
python3 -c "import json; m=json.load(open('models/cognitive_agent/training_manifest.json')); print([k for k in m['documents'] if k in ['.', '..']])"

# Validate rebuild
rm -rf models/cognitive_agent/bridge/
python -m cortical.cognitive rebuild-links --metrics

# Test ask
python -m cortical.cognitive ask "What is the cognitive agent?"
```

---

*This document helps future agents understand: the bug, the fix, the reasoning, and how to verify.*
