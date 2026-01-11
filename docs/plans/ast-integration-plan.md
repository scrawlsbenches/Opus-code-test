# AST Integration Plan: Code Structure in Cognitive Graph

## Goal

Enable the cognitive agent to understand code structure, not just code-as-text. This allows queries like:
- "What functions call `compute_pagerank`?"
- "What classes inherit from `StorageBackend`?"
- "Show the dependency graph of the `got` module"

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
1. **CodeBridge** - converts ASTIndex to cognitive atoms
2. **CLI command** - `index-code` to index directories
3. **Query support** - traverse code relationships

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

## Implementation Steps

### Step 1: CodeBridge Class
**File:** `cortical/cognitive/code_bridge.py`

```python
class CodeBridge:
    """Converts ASTIndex to CognitiveGraph atoms."""

    def __init__(self, graph: CognitiveGraph):
        self.graph = graph

    def index_directory(self, path: Path) -> IndexStats:
        """Index all Python files in directory."""
        ast_index = ASTIndex()
        ast_index.index_directory(path)
        return self._convert_ast_to_atoms(ast_index)

    def _convert_ast_to_atoms(self, ast_index: ASTIndex) -> IndexStats:
        """Convert ASTIndex entries to atoms and links."""
        # 1. Create FILE atoms
        # 2. Create CLASS atoms with DEFINES links from FILE
        # 3. Create FUNCTION atoms with CONTAINS links from CLASS
        # 4. Create CALLS links between FUNCTIONs
        # 5. Create INHERITS links between CLASSes
        # 6. Create IMPORTS links between FILEs and MODULEs
```

**Atom creation patterns:**

| AST Entity | Atom Type | Name Format | Links Created |
|------------|-----------|-------------|---------------|
| File | FILE | `"cortical/got/api.py"` | - |
| Class | CLASS | `"GoTAPI"` | FILE --DEFINES--> CLASS |
| Method | FUNCTION | `"GoTAPI.create_task"` | CLASS --CONTAINS--> FUNCTION |
| Function | FUNCTION | `"standalone_func"` | FILE --DEFINES--> FUNCTION |
| Import | MODULE | `"cortical.got"` | FILE --IMPORTS--> MODULE |
| Call | - | - | FUNCTION --CALLS--> FUNCTION |
| Inheritance | - | - | CLASS --INHERITANCE--> CLASS |

### Step 2: CLI Command
**File:** `cortical/cognitive/__main__.py`

```python
# index-code command
index_code_parser = subparsers.add_parser(
    "index-code",
    help="Index Python code structure into cognitive graph"
)
index_code_parser.add_argument(
    "path",
    help="Directory to index"
)
index_code_parser.add_argument(
    "--exclude",
    nargs="*",
    default=["__pycache__", ".git", "node_modules"],
    help="Directories to exclude"
)
```

### Step 3: Query Support

Enable traversals like:

```python
# What calls function X?
agent.query_code("callers_of", "compute_pagerank")

# What does class X inherit from?
agent.query_code("parents_of", "GoTManager")

# What does file X define?
agent.query_code("defines", "cortical/got/api.py")
```

Implementation in CognitiveAgent:

```python
def query_code(self, query_type: str, target: str) -> List[Atom]:
    """Query code structure."""
    target_atom = self.graph.get_by_name(target)
    if not target_atom:
        return []

    if query_type == "callers_of":
        # Find CALLS links pointing TO target
        return self._find_incoming(target_atom, AtomType.CALLS)
    elif query_type == "calls":
        # Find CALLS links pointing FROM target
        return self._find_outgoing(target_atom, AtomType.CALLS)
    # ... etc
```

### Step 4: Semantic Bridge (REFERS_TO)

Connect text vocabulary to code entities:

```python
def _create_refers_to_links(self):
    """Link WORDs to CODE entities they reference."""
    # "pagerank" WORD --REFERS_TO--> "compute_pagerank" FUNCTION
    # "got" WORD --REFERS_TO--> "cortical.got" MODULE

    for word_atom in self.graph.get_by_type(AtomType.WORD):
        word = word_atom.name.lower()

        # Check if word matches any code entity
        for code_atom in self._get_code_atoms():
            if word in code_atom.name.lower():
                self.graph.link(
                    AtomType.REFERS_TO,
                    [word_atom, code_atom],
                    TruthValue(strength=0.8, confidence=0.6)
                )
```

### Step 5: Update Sharded Storage

Add CODE types to large type handling if needed:

```python
# In graph_storage.py
LARGE_TYPES = {'SIMILARITY', 'FOLLOWS', 'CALLS'}
```

## Behavioral Tests

**File:** `tests/behavioral/test_code_indexing_spec.py`

```python
class TestCodeIndexing:
    """Verify code structure is captured correctly."""

    def test_file_defines_class(self, code_bridge):
        """FILE atoms should have DEFINES links to CLASS atoms."""

    def test_class_contains_methods(self, code_bridge):
        """CLASS atoms should have CONTAINS links to FUNCTION atoms."""

    def test_call_graph_captured(self, code_bridge):
        """CALLS links should reflect actual function calls."""

    def test_inheritance_captured(self, code_bridge):
        """INHERITANCE links should reflect class hierarchy."""

class TestCodeQueries:
    """Verify code queries work correctly."""

    def test_find_callers(self, indexed_codebase):
        """Should find all functions that call target."""

    def test_find_inheritance_chain(self, indexed_codebase):
        """Should traverse inheritance hierarchy."""
```

## File Changes Summary

| File | Change |
|------|--------|
| `cortical/cognitive/code_bridge.py` | NEW - CodeBridge class |
| `cortical/cognitive/__main__.py` | Add `index-code` command |
| `cortical/cognitive/graph.py` | Add `query_code()` to CognitiveAgent |
| `cortical/cognitive/graph_storage.py` | Add CALLS to LARGE_TYPES if needed |
| `tests/behavioral/test_code_indexing_spec.py` | NEW - behavioral tests |

## Success Criteria

1. `python -m cortical.cognitive index-code cortical/` completes successfully
2. Can query "what calls X" and get correct results
3. Can query "what inherits from X" and get correct results
4. Code atoms integrate with existing WORD atoms via REFERS_TO
5. All tests pass, no regressions

## Open Questions

1. **Name collisions**: Multiple classes named `Config` in different files. Use qualified names?
   - Recommendation: Yes, use `"file/path:ClassName"` format for uniqueness

2. **Incremental updates**: How to handle file changes?
   - Recommendation: Delete old atoms for file, re-index. Track file hashes.

3. **Cross-file references**: Method calls to external modules?
   - Recommendation: Create MODULE atoms, link via IMPORTS. Resolve where possible.

## Timeline

Not providing time estimates (per CLAUDE.md). Steps are ordered by dependency:
1. CodeBridge (foundation)
2. CLI command (usability)
3. Behavioral tests (confidence)
4. Query support (utility)
5. Semantic bridge (integration)
6. Test on our codebase (validation)
