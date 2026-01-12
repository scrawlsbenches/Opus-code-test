# AST Integration Plan: Code Structure in Cognitive Graph

## Goal

Enable the cognitive agent to understand code structure, not just code-as-text. This allows queries like:
- "What functions call `compute_pagerank`?"
- "What classes inherit from `StorageBackend`?"
- "Show the dependency graph of the `got` module"

## Critical Design Principle: References, Not Code

**We store REFERENCES to code, not code itself.**

```
┌─────────────────────────────────────────────────────────────────┐
│                    WHAT WE STORE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Atom Type   │ Stored Data                │ NOT Stored          │
│  ───────────────────────────────────────────────────────────────│
│  FILE        │ path: "cortical/got/api.py"│ File contents       │
│  CLASS       │ name: "GoTAPI"             │ Class source code   │
│              │ file_path + lineno         │                     │
│  FUNCTION    │ name: "create_task"        │ Function body       │
│              │ file_path + lineno + args  │                     │
│  MODULE      │ name: "cortical.got"       │ Module contents     │
│                                                                  │
│  Atoms are POINTERS with metadata. Code stays in .py files.    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Metadata stored per atom type:**

```python
# FILE atom
{"name": "cortical/got/api.py", "type": "FILE"}

# CLASS atom
{"name": "GoTAPI", "type": "CLASS",
 "meta": {"file_path": "cortical/got/api.py", "lineno": 45,
          "docstring": "Graph of Thought API."}}

# FUNCTION atom
{"name": "GoTAPI.create_task", "type": "FUNCTION",
 "meta": {"file_path": "cortical/got/api.py", "lineno": 72,
          "args": ["title", "priority"], "docstring": "Create a new task."}}
```

## Edge Cases: Training Time & Storage

### Training Time Estimates

| Codebase Size | Files | Est. Atoms | Est. Time |
|---------------|-------|------------|-----------|
| Small (this repo) | ~200 | ~5,000 | <5 seconds |
| Medium (Django) | ~2,000 | ~50,000 | <30 seconds |
| Large (CPython) | ~5,000 | ~150,000 | <2 minutes |

**Why it's fast:**
- Python's `ast` module is C-implemented, very fast
- We only parse, not execute
- No network I/O
- Single-threaded is sufficient

**Mitigation if slow:**
- Progress callback for CLI feedback
- Incremental indexing (only changed files)
- File hash tracking to skip unchanged

### Storage Requirements

| Atom Type | Per-Atom Size | Typical Count | Total |
|-----------|---------------|---------------|-------|
| FILE | ~100 bytes | 200 | 20 KB |
| CLASS | ~200 bytes | 500 | 100 KB |
| FUNCTION | ~250 bytes | 2,000 | 500 KB |
| CALLS links | ~150 bytes | 5,000 | 750 KB |
| IMPORTS links | ~100 bytes | 1,000 | 100 KB |
| **Total** | | | **~1.5 MB** |

**Compare to existing:**
- Current graph: 470K atoms, ~130 MB (sharded)
- Code atoms: ~8K atoms, ~1.5 MB
- **Code is <2% of total storage**

**Conclusion:** Storage is not a concern. No sharding needed for code atoms.

## Current State

### Already Have
1. **ASTIndex** (`cortical/spark/ast_index.py`)
   - Parses Python files using `ast` module
   - Extracts: classes, functions, imports, call graph, inheritance
   - Serialization support (to_dict/from_dict)
   - 500+ lines, well-tested

2. **CODE AtomTypes** (already in `graph.py`)
   - `FILE`, `CLASS`, `FUNCTION`, `MODULE` - code entities
   - `CALLS`, `IMPORTS`, `DEFINES`, `CONTAINS` - code relationships
   - `REFERS_TO` - semantic bridge (WORD -> CODE entity)

3. **Sharded Storage** (`graph_storage.py`)
   - Git-friendly file sizes
   - Type-based sharding

### Need to Build
1. **Behavioral tests** (TDD: tests first)
2. **CodeBridge** - converts ASTIndex to cognitive atoms
3. **CLI command** - `index-code` to index directories
4. **Query support** - traverse code relationships

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CognitiveAgent                              │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ TextBridge  │    │ CodeBridge  │    │ QueryEngine │         │
│  │ (text→atoms)│    │ (AST→atoms) │    │ (traversal) │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                   │                   │                │
│         └───────────────────┴───────────────────┘                │
│                             │                                    │
│                    ┌────────┴────────┐                          │
│                    │ CognitiveGraph  │                          │
│                    │ (unified store) │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Steps (TDD Order)

### Step 1: Behavioral Tests (Write First)
**File:** `tests/behavioral/test_code_indexing_spec.py`

```python
class TestCodeAtomCreation:
    """Verify code entities become atoms correctly."""

    def test_file_creates_file_atom(self, code_bridge, temp_python_file):
        """Indexing a file should create a FILE atom."""
        code_bridge.index_file(temp_python_file)
        file_atom = code_bridge.graph.get_by_name(str(temp_python_file))
        assert file_atom is not None
        assert file_atom.atom_type == AtomType.FILE

    def test_class_creates_class_atom_with_metadata(self, code_bridge):
        """CLASS atoms should have file_path and lineno in metadata."""
        # ... atom should have meta.file_path, meta.lineno

    def test_function_creates_function_atom_with_args(self, code_bridge):
        """FUNCTION atoms should have args in metadata."""

class TestCodeLinks:
    """Verify relationships are captured as links."""

    def test_file_defines_class(self, code_bridge):
        """FILE --DEFINES--> CLASS link should exist."""

    def test_class_contains_method(self, code_bridge):
        """CLASS --CONTAINS--> FUNCTION link should exist."""

    def test_function_calls_function(self, code_bridge):
        """FUNCTION --CALLS--> FUNCTION link should exist."""

    def test_class_inherits_from_parent(self, code_bridge):
        """CLASS --INHERITANCE--> CLASS link should exist."""

class TestCodeQueries:
    """Verify code queries work correctly."""

    def test_find_callers_of_function(self, indexed_code):
        """query_code('callers_of', 'helper') returns caller functions."""

    def test_find_children_of_class(self, indexed_code):
        """query_code('subclasses_of', 'Base') returns child classes."""
```

### Step 2: CodeBridge Class
**File:** `cortical/cognitive/code_bridge.py`

```python
@dataclass
class IndexStats:
    """Statistics from indexing operation."""
    files: int = 0
    classes: int = 0
    functions: int = 0
    calls_links: int = 0
    inheritance_links: int = 0
    parse_errors: int = 0
    elapsed_seconds: float = 0.0

class CodeBridge:
    """Converts ASTIndex to CognitiveGraph atoms.

    Stores REFERENCES to code, not code itself.
    Atoms point to file paths and line numbers.
    """

    def __init__(self, graph: CognitiveGraph):
        self.graph = graph

    def index_file(self, path: Path) -> IndexStats:
        """Index a single Python file."""
        ast_index = ASTIndex()
        if ast_index.index_file(path):
            return self._convert_ast_to_atoms(ast_index)
        return IndexStats(parse_errors=1)

    def index_directory(self, path: Path,
                        exclude: List[str] = None,
                        progress_callback: Callable = None) -> IndexStats:
        """Index all Python files in directory."""
        ast_index = ASTIndex()
        ast_index.index_directory(path, exclude=exclude or [])
        return self._convert_ast_to_atoms(ast_index, progress_callback)

    def _convert_ast_to_atoms(self, ast_index: ASTIndex,
                               progress_callback: Callable = None) -> IndexStats:
        """Convert ASTIndex entries to atoms and links."""
        stats = IndexStats()

        # 1. Create FILE atoms (with deduplication)
        # 2. Create CLASS atoms with DEFINES links
        # 3. Create FUNCTION atoms with CONTAINS/DEFINES links
        # 4. Create CALLS links
        # 5. Create INHERITANCE links
        # 6. Create IMPORTS links

        return stats
```

**Atom ID Strategy:**
- FILE: `f"file:{relative_path}"`
- CLASS: `f"class:{file_path}:{class_name}"`
- FUNCTION: `f"func:{file_path}:{full_name}"`
- MODULE: `f"mod:{module_name}"`

This ensures uniqueness across the codebase.

### Step 3: CLI Command
**File:** `cortical/cognitive/__main__.py`

```python
# index-code command
index_code_parser = subparsers.add_parser(
    "index-code",
    help="Index Python code structure into cognitive graph"
)
index_code_parser.add_argument("path", help="Directory to index")
index_code_parser.add_argument(
    "--exclude", nargs="*",
    default=["__pycache__", ".git", "node_modules", "venv", ".venv"],
    help="Directories to exclude"
)
index_code_parser.add_argument(
    "--model-dir", default="models/cognitive_agent",
    help="Model directory"
)
```

### Step 4: Query Methods
**File:** `cortical/cognitive/graph.py` (add to CognitiveAgent)

```python
def query_code(self, query_type: str, target: str) -> List[Atom]:
    """Query code structure.

    Query types:
    - callers_of: functions that call target
    - calls: functions that target calls
    - subclasses_of: classes that inherit from target
    - parent_of: parent class of target
    - defines: entities defined in file
    - defined_in: file that defines entity
    """
```

### Step 5: Semantic Bridge (Optional, Lower Priority)
Connect WORD atoms to CODE atoms via REFERS_TO links.
This enables queries like "what code relates to 'pagerank'?"

## Resolved Design Decisions

1. **Name collisions**: Use qualified IDs (`file:path`, `class:path:name`)
   - Display name stays simple: "GoTAPI"
   - ID is unique: "class:cortical/got/api.py:GoTAPI"

2. **Incremental updates**: Track file modification times
   - On re-index, compare mtime
   - Delete old atoms for changed files, re-create

3. **Cross-file calls**: Best-effort resolution
   - If callee is in indexed files, create CALLS link
   - If callee is external (stdlib, deps), create MODULE reference

## File Changes Summary

| File | Change |
|------|--------|
| `tests/behavioral/test_code_indexing_spec.py` | NEW - behavioral tests (FIRST) |
| `cortical/cognitive/code_bridge.py` | NEW - CodeBridge class |
| `cortical/cognitive/__main__.py` | Add `index-code` command |
| `cortical/cognitive/graph.py` | Add `query_code()` to CognitiveAgent |

## Success Criteria

1. All behavioral tests pass
2. `python -m cortical.cognitive index-code cortical/` completes in <10s
3. Can query "callers_of" and "subclasses_of" with correct results
4. Storage overhead is <5 MB for our codebase
5. No regressions in existing tests

## Execution Order

1. **Write behavioral tests** (TDD: red first)
2. **Implement CodeBridge** (make tests green)
3. **Add CLI command** (usability)
4. **Add query methods** (utility)
5. **Test on our codebase** (validation)
6. **Semantic bridge** (if time permits, lower priority)
