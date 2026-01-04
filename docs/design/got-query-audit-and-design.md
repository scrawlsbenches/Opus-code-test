# Graph of Thought Query System: Forensic Audit and Design for Complex Query Expressions

**Auditor:** Senior Principal Computer Scientist / Software Engineer
**Date:** 2026-01-04
**Status:** DRAFT - Pending Approval
**Version:** 2.4

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-04 | Auditor | Initial audit and design |
| 2.0 | 2026-01-04 | Auditor | Generalized architecture, GoT workflow, agent protocols |
| 2.1 | 2026-01-04 | Auditor | Added validated internal API structure from Python testing |
| 2.2 | 2026-01-04 | Auditor | Added API Discovery Protocol for agent assumption validation |
| 2.3 | 2026-01-04 | Auditor | Added Part 7: Operational Considerations & Risk Mitigations |
| 2.4 | 2026-01-04 | Auditor | Replaced hardcoded schema with existing schema introspection |

**Approval Required Before:**
- Creating any GoT entities (Epic, Sprint, Task)
- Writing any implementation code
- Creating any test files

---

## Executive Summary

This document provides:
1. **Forensic Audit** of the existing GoT query infrastructure
2. **Generalized Design** for complex query expressions using extensible patterns
3. **Project Management Structure** using GoT to manage GoT development (dog-fooding)
4. **Agent Workflow Protocol** for context-limited AI agents
5. **Quality Gates** with BDD/TDD requirements

### Critical Design Principles

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DESIGN PRINCIPLES                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. NO HARDCODED ENTITIES                                               │
│     All graph functions use a registry pattern. No "inSprint" or        │
│     "blocks" literals in core code. Everything is configurable.         │
│                                                                          │
│  2. DOG-FOOD OUR OWN SYSTEM                                             │
│     This project is managed using GoT (got_utils.py). We prove          │
│     the system works by using it to build itself.                       │
│                                                                          │
│  3. ASSUME CONTEXT LOSS                                                 │
│     Every task is designed for an agent with no prior context.          │
│     Knowledge transfers and handoffs bridge sessions.                   │
│                                                                          │
│  4. TEST BEFORE CODE                                                    │
│     Behavioral tests define requirements. Unit tests verify             │
│     implementation. No code without failing tests first.                │
│                                                                          │
│  5. CLEANUP REQUIRES APPROVAL                                           │
│     All cleanup tasks are blocked by an approval task.                  │
│     No automatic cleanup without human review.                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Forensic Audit Results

### 1.1 System Health Check

```
GoT VALIDATION REPORT
============================================================
📊 STATISTICS
   Tasks: 336
   Edges: 425
   Edge density: 1.08 edges/node
   Orphan nodes: 63 (16.1%)

📁 ENTITY FILES
   Task files: 336
   Edge files: 434
   Decision files: 56
   Handoff files: 41

📈 TASKS BY STATUS
   blocked: 1
   completed: 238
   pending: 97

✅ HEALTHY - No issues detected
```

### 1.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                             │
│                    (got_utils.py, cli/*.py)                  │
├─────────────────────────────────────────────────────────────┤
│                       API Layer                              │
│              (api.py, query_api.py, query_builder.py)        │
├─────────────────────────────────────────────────────────────┤
│                    Graph Operations                          │
│         (graph_walker.py, path_finder.py, pattern_matcher.py)│
├─────────────────────────────────────────────────────────────┤
│                    Index & Storage                           │
│           (indexer.py, tx_manager.py, versioned_store.py)    │
├─────────────────────────────────────────────────────────────┤
│                     Data Types                               │
│                (types.py, entity_schemas.py)                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Current Query Capabilities

| Component | File | Purpose | Assessment |
|-----------|------|---------|------------|
| Query Builder | `query_builder.py` | Fluent SQL-like API | ★★★★★ |
| Query API | `query_api.py` | Read-only operations | ★★★★★ |
| Pattern Matcher | `pattern_matcher.py` | Subgraph isomorphism | ★★★★★ |
| Graph Walker | `graph_walker.py` | BFS/DFS traversal | ★★★★★ |
| Path Finder | `path_finder.py` | Shortest/all paths | ★★★★★ |
| Indexer | `indexer.py` | Status/priority indexes | ★★★★☆ |
| CLI Query | `got_utils.py:query()` | Natural language | ★★★☆☆ |

### 1.4 Validated Internal API Structure

**Verified through direct Python API testing on 2026-01-04:**

#### TransactionalGoTAdapter (scripts/got_utils.py)
High-level facade used by CLI. Returns `ThoughtNode` objects.

```python
# Instantiation
manager = TransactionalGoTAdapter()  # defaults to Path('.got')

# Entity creation (returns entity ID string)
task_id = manager.create_task(title, priority='medium', category='feature',
                               description='', sprint_id=None,
                               depends_on=None, blocks=None)
sprint_id = manager.create_sprint(name, number=None, epic_id=None, description=None)
epic_id = manager.create_epic(name, epic_id=None, properties=None)
decision_id = manager.create_decision(content, rationale='', task_id=None, alternatives=None)
kt_id = manager.create_knowledge_transfer(title, session_id='', summary='', ...)

# Relationships
manager.add_dependency(task_id, depends_on_id)  # T-A depends on T-B
manager.add_blocks(task_id, blocks_id)          # T-A blocks T-B
manager.link_task_to_sprint(sprint_id, task_id)
manager.add_edge(source_id, target_id, edge_type, weight=1.0, reason='')
```

#### Query Builder (cortical/got/query_builder.py)
**Fluent SQL-like API that already exists and is powerful:**

```python
from cortical.got.api import GoTManager
from cortical.got.query_builder import Query

manager = GoTManager(Path('.got'))

# Validated working examples:
Query(manager).tasks().where(status='pending').limit(3).execute()
Query(manager).tasks().where(priority='high').or_where(priority='critical').limit(5).execute()
Query(manager).tasks().where(status='pending').order_by('priority', desc=True).limit(5).execute()
Query(manager).tasks().where(status='completed').count()  # Returns int: 238
Query(manager).tasks().where(status='blocked').exists()   # Returns bool: True
Query(manager).tasks().connected_to('S-019', via='CONTAINS').execute()  # 11 tasks
Query(manager).sprints().limit(3).execute()
Query(manager).decisions().limit(3).execute()
Query(manager).edges().limit(5).execute()
Query(manager).tasks().where(status='pending').explain()  # Returns QueryPlan
```

#### Current CLI query() Method (got_utils.py:2388-2685)
**Uses hardcoded pattern matching, does NOT use Query builder:**

```python
# Current implementation uses string matching:
if query_str.startswith("what blocks "):
    task_id = original_query[12:].strip()
    # ...
elif query_str == "blocked tasks":
    # ...
elif query_str.startswith("tasks in sprint "):
    # ...
```

### 1.5 Key Insight: The Gap

**The Query builder already provides all the power we need. The gap is an expression parser that compiles DSL expressions to Query builder calls.**

```
Current State:
  CLI query string → Hardcoded pattern matching → Direct API calls
  (Limited, not extensible, duplicates Query builder logic)

Target State:
  CLI query string → Expression Parser → Query Builder → Execute
  (Extensible, reuses existing infrastructure, DRY)
```

### 1.6 Gap Analysis

| Missing Feature | Impact | Priority |
|-----------------|--------|----------|
| Boolean expression parsing | Cannot combine AND/OR in CLI | Critical |
| Comparison operators | No `priority > medium` | High |
| Extensible function registry | Functions are hardcoded | Critical |
| Field projections | No `SELECT field1, field2` | Medium |
| Aggregation in CLI | No `COUNT BY status` | High |
| Date range queries | Limited time filtering | High |
| **Expression→QueryBuilder bridge** | **CLI doesn't use existing power** | **Critical** |

---

## Part 2: Generalized Architecture

### 2.1 Function Registry Pattern

**Problem:** The original design hardcoded functions like `inSprint`, `blocks`, `dependsOn`.

**Solution:** Use a **Function Registry** that allows runtime registration of query functions.

```python
# cortical/got/expression/registry.py

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass

@dataclass
class FunctionSignature:
    """Describes a registered function's interface."""
    name: str
    description: str
    required_args: List[str]
    optional_args: Dict[str, Any]  # name -> default
    returns: str  # description of return type

class QueryFunction(ABC):
    """Base class for all query functions."""

    @classmethod
    @abstractmethod
    def signature(cls) -> FunctionSignature:
        """Return function signature for validation and help."""
        pass

    @abstractmethod
    def execute(self, manager: "GoTManager", args: List[Any], kwargs: Dict[str, Any]) -> Any:
        """Execute the function and return results."""
        pass

class FunctionRegistry:
    """
    Registry for query functions.

    Functions are registered by name and can be looked up at runtime.
    This allows new functions to be added without modifying core code.
    """

    _instance: Optional["FunctionRegistry"] = None
    _functions: Dict[str, Type[QueryFunction]] = {}

    @classmethod
    def instance(cls) -> "FunctionRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def register(cls, name: str) -> Callable[[Type[QueryFunction]], Type[QueryFunction]]:
        """Decorator to register a function."""
        def decorator(func_class: Type[QueryFunction]) -> Type[QueryFunction]:
            cls._functions[name.lower()] = func_class
            return func_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[QueryFunction]]:
        """Look up a function by name."""
        return cls._functions.get(name.lower())

    @classmethod
    def list_functions(cls) -> List[FunctionSignature]:
        """List all registered functions."""
        return [f.signature() for f in cls._functions.values()]
```

### 2.2 Example Function Implementations

Functions are defined separately and registered - NOT hardcoded in the executor:

```python
# cortical/got/expression/functions/graph_functions.py

from ..registry import QueryFunction, FunctionRegistry, FunctionSignature

@FunctionRegistry.register("connected_to")
class ConnectedToFunction(QueryFunction):
    """Find entities connected to a given entity."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="connected_to",
            description="Find entities connected to a given entity",
            required_args=["entity_id"],
            optional_args={"via": None, "direction": "both", "depth": 1},
            returns="List of connected entities"
        )

    def execute(self, manager, args, kwargs):
        entity_id = args[0]
        edge_type = kwargs.get("via")
        direction = kwargs.get("direction", "both")
        depth = kwargs.get("depth", 1)

        # Use existing GraphWalker infrastructure
        from cortical.got.graph_walker import GraphWalker

        walker = GraphWalker(manager).starting_from(entity_id).max_depth(depth)

        if edge_type:
            walker = walker.follow(edge_type)

        if direction == "outgoing":
            walker = walker.outgoing()
        elif direction == "incoming":
            walker = walker.incoming()
        # else: both (default)

        return list(walker.bfs().iter())


@FunctionRegistry.register("path")
class PathFunction(QueryFunction):
    """Find path between two entities."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="path",
            description="Find shortest path between two entities",
            required_args=["from_id", "to_id"],
            optional_args={"via": None, "max_length": 10},
            returns="List of entity IDs in path, or empty if no path"
        )

    def execute(self, manager, args, kwargs):
        from_id = args[0]
        to_id = args[1]
        edge_type = kwargs.get("via")
        max_length = kwargs.get("max_length", 10)

        from cortical.got.path_finder import PathFinder

        finder = PathFinder(manager).max_length(max_length)

        if edge_type:
            finder = finder.via_edges(edge_type)

        return finder.shortest_path(from_id, to_id) or []


@FunctionRegistry.register("aggregate")
class AggregateFunction(QueryFunction):
    """Aggregate entities by a field."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="aggregate",
            description="Count or group entities by a field",
            required_args=["field"],
            optional_args={"operation": "count"},
            returns="Dict mapping field values to counts/results"
        )

    def execute(self, manager, args, kwargs):
        field = args[0]
        operation = kwargs.get("operation", "count")

        from cortical.got.query_builder import Query

        if operation == "count":
            return Query(manager).tasks().group_by(field).count().execute()
        # Add more operations as needed
        return {}
```

### 2.3 Executor Using Registry

The executor uses the registry - no hardcoded function names:

```python
# cortical/got/expression/executor.py

from .registry import FunctionRegistry
from .ast import FunctionCall, Expression

class QueryExecutor:
    """Execute AST against GoT storage using registered functions."""

    def __init__(self, manager: "GoTManager"):
        self.manager = manager
        self.registry = FunctionRegistry.instance()

    def _apply_function(self, func: FunctionCall):
        """Execute a function call using the registry."""
        func_class = self.registry.get(func.name)

        if func_class is None:
            available = [f.name for f in self.registry.list_functions()]
            raise ValueError(
                f"Unknown function '{func.name}'. "
                f"Available functions: {', '.join(available)}"
            )

        # Extract arg values from AST
        arg_values = [self._evaluate(arg) for arg in func.args]
        kwarg_values = {k: self._evaluate(v) for k, v in func.kwargs.items()}

        # Instantiate and execute
        instance = func_class()
        return instance.execute(self.manager, arg_values, kwarg_values)
```

### 2.4 Module Structure (Revised)

```
cortical/got/expression/
├── __init__.py              # Public API: parse(), execute()
├── lexer.py                 # Tokenization
├── parser.py                # Recursive descent parser
├── ast.py                   # AST node types
├── optimizer.py             # Query optimization
├── executor.py              # AST execution (uses registry)
├── registry.py              # Function registry pattern
├── errors.py                # Custom exceptions
└── functions/               # Pluggable functions
    ├── __init__.py          # Auto-registers all functions
    ├── graph_functions.py   # connected_to, path, etc.
    ├── filter_functions.py  # where, having, etc.
    └── aggregate_functions.py # count, sum, avg, etc.
```

---

## Part 3: Project Management Structure (GoT)

This section documents the GoT entities that will be created AFTER this document is approved.

### 3.1 Epic

```yaml
Epic:
  title: "Complex Query Expression System"
  description: |
    Implement a generalized, extensible query expression parser for GoT
    that compiles to existing Query builder infrastructure. Uses function
    registry pattern for extensibility. Follows sovereignty principle.
  priority: critical
  success_criteria:
    - All behavioral tests pass
    - Unit test coverage >= 90%
    - No hardcoded entity references in core code
    - Function registry supports runtime extension
    - CLI integration complete
    - Performance within 2x of direct Query builder
```

### 3.2 Sprints

```yaml
Sprint-1:
  title: "Foundation: Lexer, AST, and Test Infrastructure"
  goal: "Establish parsing foundation with comprehensive test coverage"
  tasks: [T-001 through T-006]

Sprint-2:
  title: "Parser and Basic Execution"
  goal: "Parse expressions and execute simple queries"
  tasks: [T-007 through T-012]

Sprint-3:
  title: "Function Registry and Graph Functions"
  goal: "Extensible function system with graph operations"
  tasks: [T-013 through T-018]

Sprint-4:
  title: "Optimization and CLI Integration"
  goal: "Query optimization and CLI integration"
  tasks: [T-019 through T-024]

Sprint-5:
  title: "Documentation and Cleanup"
  goal: "Complete documentation and cleanup"
  tasks: [T-025 through T-028]
```

### 3.3 Detailed Task Specifications

Each task includes:
- **Behavioral Test Requirement**: The BDD scenario to implement FIRST
- **Affected Files**: Files the agent should examine/modify
- **Validation Steps**: How to verify completion
- **Cleanup Tasks**: What cleanup is needed (blocked by approval)

---

#### Sprint 1: Foundation

##### T-001: Create Expression Module Skeleton

```yaml
Task: T-001
Title: "Create expression module skeleton with __init__.py"
Priority: critical
Category: feature
Sprint: Sprint-1

Behavioral_Test_First: |
  # tests/behavioral/test_expression_module.py

  Feature: Expression Module Structure

    Scenario: Module is importable
      Given the expression module exists
      When I import cortical.got.expression
      Then no ImportError is raised
      And parse function is available
      And execute function is available

    Scenario: Submodules are importable
      Given the expression module exists
      When I import cortical.got.expression.lexer
      And I import cortical.got.expression.parser
      And I import cortical.got.expression.ast
      Then no ImportError is raised

Affected_Files:
  - cortical/got/expression/__init__.py (create)
  - cortical/got/expression/lexer.py (create stub)
  - cortical/got/expression/parser.py (create stub)
  - cortical/got/expression/ast.py (create stub)
  - cortical/got/expression/registry.py (create stub)
  - cortical/got/expression/executor.py (create stub)
  - cortical/got/expression/errors.py (create)
  - tests/behavioral/test_expression_module.py (create)

Validation_Steps:
  1. Run: python -c "from cortical.got.expression import parse, execute"
  2. Run: python -m pytest tests/behavioral/test_expression_module.py -v
  3. Verify all imports succeed

Agent_Instructions: |
  Before implementing:
  1. Read cortical/got/__init__.py to understand existing module patterns
  2. Read cortical/got/query_builder.py for API style reference
  3. Challenge: Are there existing expression patterns in the codebase?
  4. Write the behavioral test FIRST
  5. Run test to see it fail
  6. Implement minimal code to pass

Cleanup_Tasks:
  - Remove any debugging print statements
  - Ensure all files have module docstrings
  - Blocked by: T-CLEANUP-APPROVAL
```

##### T-002: Implement AST Node Types

```yaml
Task: T-002
Title: "Implement AST node types with dataclasses"
Priority: critical
Category: feature
Sprint: Sprint-1
Depends_On: [T-001]

Behavioral_Test_First: |
  # tests/behavioral/test_ast_nodes.py

  Feature: AST Node Types

    Scenario: Create a literal node
      Given I want to represent the value "pending"
      When I create a Literal node with value "pending"
      Then the node.value equals "pending"
      And the node is an instance of Expression

    Scenario: Create a comparison node
      Given I want to represent "status = 'pending'"
      When I create a Comparison with field "status", op EQ, value "pending"
      Then the node.field.name equals "status"
      And the node.op equals Op.EQ
      And the node.value.value equals "pending"

    Scenario: Create a boolean AND expression
      Given I have two comparison nodes
      When I create an AndExpr with both as children
      Then the node has 2 children
      And both children are Comparison nodes

    Scenario: Create a function call node
      Given I want to represent "connected_to('T-123', via='DEPENDS_ON')"
      When I create a FunctionCall with name "connected_to"
      And I add positional arg "T-123"
      And I add keyword arg via="DEPENDS_ON"
      Then the node.name equals "connected_to"
      And the node has 1 positional arg
      And the node has 1 keyword arg

Affected_Files:
  - cortical/got/expression/ast.py (implement)
  - tests/behavioral/test_ast_nodes.py (create)
  - tests/unit/test_ast.py (create)

Unit_Test_Requirements: |
  # tests/unit/test_ast.py

  - test_literal_string_value
  - test_literal_number_value
  - test_literal_list_value
  - test_field_name_validation
  - test_comparison_all_operators
  - test_and_expr_flattening
  - test_or_expr_flattening
  - test_function_call_args
  - test_function_call_kwargs
  - test_query_with_order_by
  - test_query_with_limit_offset
  - test_expression_equality
  - test_expression_repr

Validation_Steps:
  1. Run: python -m pytest tests/behavioral/test_ast_nodes.py -v
  2. Run: python -m pytest tests/unit/test_ast.py -v
  3. Run: python -m coverage run -m pytest tests/unit/test_ast.py
  4. Verify coverage >= 95% for ast.py

Agent_Instructions: |
  Before implementing:
  1. Read cortical/got/types.py for dataclass patterns used in this project
  2. Read cortical/got/pattern_matcher.py for NodeConstraint/EdgeConstraint patterns
  3. Challenge: Should Expression be a Protocol or ABC?
  4. Challenge: Should nodes be frozen dataclasses?
  5. Write behavioral test FIRST
  6. Write unit tests
  7. Implement to make tests pass

Cleanup_Tasks:
  - Ensure all dataclasses have __repr__ and __eq__
  - Add type hints to all fields
  - Blocked by: T-CLEANUP-APPROVAL
```

##### T-003: Implement Lexer (Tokenization)

```yaml
Task: T-003
Title: "Implement lexer for tokenizing query expressions"
Priority: critical
Category: feature
Sprint: Sprint-1
Depends_On: [T-001]

Behavioral_Test_First: |
  # tests/behavioral/test_lexer.py

  Feature: Query Expression Tokenization

    Scenario: Tokenize simple comparison
      Given the query string "status = 'pending'"
      When I tokenize the string
      Then I get tokens: IDENTIFIER("status"), EQ, STRING("pending"), EOF

    Scenario: Tokenize boolean expression
      Given the query string "status = 'pending' AND priority = 'high'"
      When I tokenize the string
      Then I get tokens including AND keyword
      And I get 2 IDENTIFIER tokens
      And I get 2 STRING tokens

    Scenario: Tokenize comparison operators
      Given the query string "count > 5 AND count <= 10"
      When I tokenize the string
      Then I get GT and LTE operator tokens

    Scenario: Tokenize function call
      Given the query string "connected_to('T-123', via='DEPENDS_ON')"
      When I tokenize the string
      Then I get IDENTIFIER("connected_to")
      And I get LPAREN and RPAREN
      And I get STRING tokens for arguments

    Scenario: Handle whitespace correctly
      Given the query string "  status   =   'pending'  "
      When I tokenize the string
      Then whitespace is ignored
      And I get the same tokens as without extra whitespace

    Scenario: Error on invalid character
      Given the query string "status @ 'pending'"
      When I tokenize the string
      Then a LexerError is raised
      And the error includes position information

Affected_Files:
  - cortical/got/expression/lexer.py (implement)
  - cortical/got/expression/errors.py (add LexerError)
  - tests/behavioral/test_lexer.py (create)
  - tests/unit/test_lexer.py (create)

Unit_Test_Requirements: |
  - test_tokenize_string_single_quotes
  - test_tokenize_string_double_quotes
  - test_tokenize_string_with_escapes
  - test_tokenize_integer
  - test_tokenize_float
  - test_tokenize_date_iso_format
  - test_tokenize_identifier_simple
  - test_tokenize_identifier_with_underscore
  - test_tokenize_identifier_with_hyphen (for entity IDs like T-123)
  - test_tokenize_all_operators
  - test_tokenize_all_keywords
  - test_tokenize_list_literal
  - test_tokenize_nested_parens
  - test_error_unclosed_string
  - test_error_invalid_character
  - test_error_includes_position
  - test_token_position_tracking

Validation_Steps:
  1. Run: python -m pytest tests/behavioral/test_lexer.py -v
  2. Run: python -m pytest tests/unit/test_lexer.py -v
  3. Verify coverage >= 95% for lexer.py

Agent_Instructions: |
  Before implementing:
  1. Research: Are there existing tokenizers in this codebase? Check cortical/utils/
  2. Read: Python's tokenize module for patterns (but don't use it - sovereignty)
  3. Challenge: How to handle entity IDs with hyphens (T-123) vs subtraction?
  4. Challenge: Should keywords be case-sensitive?
  5. Write behavioral tests FIRST
  6. Implement incrementally, one token type at a time

Cleanup_Tasks:
  - Ensure error messages are user-friendly
  - Add position tracking for error reporting
  - Blocked by: T-CLEANUP-APPROVAL
```

##### T-004: Implement Error Types

```yaml
Task: T-004
Title: "Implement custom exception types with position tracking"
Priority: high
Category: feature
Sprint: Sprint-1
Depends_On: [T-001]

Behavioral_Test_First: |
  Feature: Expression Error Handling

    Scenario: LexerError includes position
      Given an invalid query "status @ pending"
      When lexer encounters '@' at position 7
      Then LexerError is raised
      And error.position equals 7
      And error.message includes "Unexpected character '@'"

    Scenario: ParseError includes context
      Given a malformed query "status = AND"
      When parser encounters unexpected AND
      Then ParseError is raised
      And error includes expected token types
      And error includes what was found

    Scenario: ExecutionError for unknown function
      Given a query "unknown_func('T-123')"
      When executor cannot find function
      Then ExecutionError is raised
      And error lists available functions

Affected_Files:
  - cortical/got/expression/errors.py (implement fully)
  - tests/unit/test_errors.py (create)

Validation_Steps:
  1. Run: python -m pytest tests/unit/test_errors.py -v
  2. Verify all errors have helpful messages
```

##### T-005: Implement Function Registry

```yaml
Task: T-005
Title: "Implement function registry with decorator-based registration"
Priority: critical
Category: feature
Sprint: Sprint-1
Depends_On: [T-001, T-002]

Behavioral_Test_First: |
  Feature: Function Registry

    Scenario: Register a function
      Given I have a QueryFunction subclass
      When I decorate it with @FunctionRegistry.register("my_func")
      Then the function is retrievable via FunctionRegistry.get("my_func")

    Scenario: List all registered functions
      Given multiple functions are registered
      When I call FunctionRegistry.list_functions()
      Then I get signatures for all registered functions

    Scenario: Function signature describes interface
      Given a registered function with required and optional args
      When I get its signature
      Then signature.required_args lists required parameters
      And signature.optional_args lists optional parameters with defaults

    Scenario: Unknown function returns None
      Given no function named "nonexistent"
      When I call FunctionRegistry.get("nonexistent")
      Then None is returned

Affected_Files:
  - cortical/got/expression/registry.py (implement)
  - cortical/got/expression/functions/__init__.py (create)
  - tests/behavioral/test_registry.py (create)
  - tests/unit/test_registry.py (create)

Unit_Test_Requirements: |
  - test_register_function_decorator
  - test_get_registered_function
  - test_get_unregistered_returns_none
  - test_list_functions_returns_signatures
  - test_function_signature_validation
  - test_case_insensitive_lookup
  - test_registry_singleton_pattern
  - test_cannot_register_same_name_twice (or can with warning?)

Agent_Instructions: |
  Before implementing:
  1. Read: cortical/query/query_builder.py for existing patterns
  2. Challenge: Should registry be a singleton or dependency-injected?
  3. Challenge: What happens if same function registered twice?
  4. Write behavioral test FIRST
```

##### T-006: Create Sprint-1 Validation Gate

```yaml
Task: T-006
Title: "Sprint-1 validation gate: all foundation tests pass"
Priority: critical
Category: test
Sprint: Sprint-1
Depends_On: [T-001, T-002, T-003, T-004, T-005]

Validation_Script: |
  #!/bin/bash
  # scripts/validate_sprint1.sh

  set -e

  echo "=== Sprint 1 Validation Gate ==="

  # 1. Module imports
  echo "Checking module imports..."
  python -c "from cortical.got.expression import parse, execute" || exit 1

  # 2. All behavioral tests
  echo "Running behavioral tests..."
  python -m pytest tests/behavioral/test_expression_module.py -v || exit 1
  python -m pytest tests/behavioral/test_ast_nodes.py -v || exit 1
  python -m pytest tests/behavioral/test_lexer.py -v || exit 1
  python -m pytest tests/behavioral/test_registry.py -v || exit 1

  # 3. All unit tests with coverage
  echo "Running unit tests with coverage..."
  python -m coverage run -m pytest tests/unit/test_ast.py tests/unit/test_lexer.py tests/unit/test_registry.py -v || exit 1

  # 4. Coverage check
  echo "Checking coverage..."
  python -m coverage report --include="cortical/got/expression/*" --fail-under=90 || exit 1

  # 5. Smoke tests still pass
  echo "Verifying smoke tests..."
  python -m pytest tests/smoke/ -v || exit 1

  echo "=== Sprint 1 PASSED ==="

Affected_Files:
  - scripts/validate_sprint1.sh (create)

Validation_Steps:
  1. Run: bash scripts/validate_sprint1.sh
  2. All checks must pass
  3. Create knowledge transfer document
```

---

#### Sprint 2: Parser and Basic Execution

##### T-007: Implement Recursive Descent Parser

```yaml
Task: T-007
Title: "Implement recursive descent parser for expressions"
Priority: critical
Category: feature
Sprint: Sprint-2
Depends_On: [T-003, T-002]

Behavioral_Test_First: |
  Feature: Expression Parsing

    Scenario: Parse simple comparison
      Given the query "status = 'pending'"
      When I parse the query
      Then I get a Query AST
      And the expression is a Comparison
      And field is "status" and value is "pending"

    Scenario: Parse AND expression
      Given the query "status = 'pending' AND priority = 'high'"
      When I parse the query
      Then the expression is an AndExpr
      And it has 2 children

    Scenario: Parse OR expression
      Given the query "status = 'blocked' OR status = 'failed'"
      When I parse the query
      Then the expression is an OrExpr

    Scenario: Parse mixed AND/OR with correct precedence
      Given the query "a = 1 AND b = 2 OR c = 3"
      When I parse the query
      Then OR is the root (lower precedence)
      And left child is AndExpr

    Scenario: Parse parenthesized expression
      Given the query "(a = 1 OR b = 2) AND c = 3"
      When I parse the query
      Then AND is the root
      And left child is OrExpr (parentheses respected)

    Scenario: Parse function call
      Given the query "connected_to('T-123', via='DEPENDS_ON')"
      When I parse the query
      Then the expression is a FunctionCall
      And function name is "connected_to"

    Scenario: Parse ORDER BY clause
      Given the query "status = 'pending' ORDER BY created_at DESC"
      When I parse the query
      Then query.order_by equals ("created_at", "DESC")

    Scenario: Parse LIMIT and OFFSET
      Given the query "status = 'pending' LIMIT 10 OFFSET 5"
      When I parse the query
      Then query.limit equals 10
      And query.offset equals 5

Affected_Files:
  - cortical/got/expression/parser.py (implement)
  - tests/behavioral/test_parser.py (create)
  - tests/unit/test_parser.py (create)

Unit_Test_Requirements: |
  - test_parse_comparison_eq
  - test_parse_comparison_ne
  - test_parse_comparison_gt_lt_gte_lte
  - test_parse_comparison_in_list
  - test_parse_comparison_like
  - test_parse_and_two_terms
  - test_parse_and_three_terms
  - test_parse_or_two_terms
  - test_parse_mixed_precedence
  - test_parse_parentheses_override
  - test_parse_nested_parentheses
  - test_parse_function_no_args
  - test_parse_function_positional_args
  - test_parse_function_keyword_args
  - test_parse_function_mixed_args
  - test_parse_order_by_asc
  - test_parse_order_by_desc
  - test_parse_order_by_default_asc
  - test_parse_limit_only
  - test_parse_limit_and_offset
  - test_parse_error_unexpected_token
  - test_parse_error_unclosed_paren
  - test_parse_error_missing_value

Agent_Instructions: |
  Before implementing:
  1. Read: parser.py patterns in this codebase (search for "parse" functions)
  2. Study: The grammar in section 2.2 of this document
  3. Challenge: Is recursive descent the right choice? Consider Pratt parsing.
  4. Challenge: How to handle error recovery for better UX?
  5. Write behavioral tests FIRST
  6. Implement one grammar rule at a time
```

##### T-008 through T-012: [Additional Sprint 2 Tasks]

*(Detailed specifications for executor, integration with Query builder, etc.)*

---

#### Sprint 3: Function Registry and Graph Functions

##### T-013: Implement Core Graph Functions

```yaml
Task: T-013
Title: "Implement core graph functions using registry"
Priority: high
Category: feature
Sprint: Sprint-3
Depends_On: [T-005, T-008]

Behavioral_Test_First: |
  Feature: Graph Query Functions

    Scenario: connected_to finds connected entities
      Given entity T-001 is connected to T-002 via DEPENDS_ON
      When I execute "connected_to('T-001', via='DEPENDS_ON')"
      Then T-002 is in the results

    Scenario: path finds shortest path
      Given T-001 -> T-002 -> T-003 path exists
      When I execute "path('T-001', 'T-003')"
      Then result is ['T-001', 'T-002', 'T-003']

    Scenario: aggregate counts by field
      Given tasks with various statuses
      When I execute "aggregate('status')"
      Then result is a dict with status counts

Functions_To_Implement:
  - connected_to(entity_id, via=None, direction="both", depth=1)
  - path(from_id, to_id, via=None, max_length=10)
  - aggregate(field, operation="count")
  - exists(entity_id)
  - type_of(entity_id)

Affected_Files:
  - cortical/got/expression/functions/graph_functions.py (create)
  - cortical/got/expression/functions/aggregate_functions.py (create)
  - cortical/got/expression/functions/__init__.py (update)
  - tests/behavioral/test_graph_functions.py (create)
  - tests/unit/test_graph_functions.py (create)

Agent_Instructions: |
  Before implementing:
  1. Read: cortical/got/graph_walker.py for traversal patterns
  2. Read: cortical/got/path_finder.py for path algorithms
  3. Read: cortical/got/query_builder.py for aggregation patterns
  4. Challenge: What edge types exist? Don't hardcode them.
  5. Challenge: What entity types exist? Use introspection.
  6. Write behavioral tests FIRST with real GoT data
```

---

### 3.4 Cleanup Approval Task

```yaml
Task: T-CLEANUP-APPROVAL
Title: "Cleanup Approval Gate"
Priority: critical
Category: governance
Sprint: Sprint-5

Description: |
  This task BLOCKS all cleanup tasks. Cleanup tasks include:
  - Removing debug code
  - Removing TODO comments
  - Removing unused imports
  - Reformatting code
  - Deleting temporary files

  Cleanup tasks may only proceed AFTER this task is marked complete
  by a human reviewer who has verified:
  1. All implementation is complete
  2. All tests pass
  3. Coverage requirements met
  4. No functionality will be lost

Approval_Checklist:
  - [ ] All Sprint validation gates passed
  - [ ] Code review completed
  - [ ] No pending implementation tasks
  - [ ] Documentation complete
  - [ ] Human approval granted

Blocks:
  - T-CLEANUP-001: Remove debug statements
  - T-CLEANUP-002: Clean up TODOs
  - T-CLEANUP-003: Format code
  - T-CLEANUP-004: Remove unused imports
```

---

## Part 4: Agent Workflow Protocol

### 4.1 Task Execution Workflow

Every agent MUST follow this workflow when assigned a task:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENT TASK EXECUTION WORKFLOW                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 1: RESEARCH (Before writing ANY code)                            │
│  ───────────────────────────────────────────                            │
│  1. Read the task specification completely                              │
│  2. Read ALL files listed in Affected_Files                             │
│  3. Search codebase for related patterns                                │
│  4. Check for existing implementations that might conflict              │
│  5. Verify dependencies are complete (check Depends_On tasks)           │
│                                                                          │
│  PHASE 2: VERIFY (Challenge assumptions)                                │
│  ───────────────────────────────────────                                │
│  1. Question: Is the task specification complete?                       │
│  2. Question: Are there edge cases not mentioned?                       │
│  3. Question: Does this conflict with existing code?                    │
│  4. Question: Are the affected files correct?                           │
│  5. If answers unclear: Create clarification request, DO NOT PROCEED   │
│                                                                          │
│  PHASE 2.5: API DISCOVERY (Validate through execution)                  │
│  ─────────────────────────────────────────────────────                  │
│  1. Don't trust your assumptions from reading code alone                │
│  2. Execute Python directly to validate API behavior                    │
│  3. Test actual method signatures, return types, and behavior           │
│  4. Document discoveries - they inform your implementation              │
│  5. Update your understanding before writing any code                   │
│                                                                          │
│  PHASE 3: TEST FIRST (BDD/TDD)                                          │
│  ─────────────────────────────                                          │
│  1. Create behavioral test file from Behavioral_Test_First spec         │
│  2. Run behavioral tests - they MUST FAIL                               │
│  3. Create unit test file from Unit_Test_Requirements                   │
│  4. Run unit tests - they MUST FAIL                                     │
│  5. If tests pass before implementation: STOP - something is wrong      │
│                                                                          │
│  PHASE 4: IMPLEMENT (Minimal code to pass tests)                        │
│  ─────────────────────────────────────────────                          │
│  1. Write minimal code to pass ONE test                                 │
│  2. Run that test to verify it passes                                   │
│  3. Repeat for next test                                                │
│  4. Do NOT add code beyond what tests require                           │
│                                                                          │
│  PHASE 5: VALIDATE (Verify all requirements met)                        │
│  ─────────────────────────────────────────────                          │
│  1. Run ALL tests in Validation_Steps                                   │
│  2. Check coverage meets requirements                                   │
│  3. Run smoke tests to verify no regressions                            │
│  4. Verify GoT system still healthy: python scripts/got_utils.py validate│
│                                                                          │
│  PHASE 6: CLEANUP TASK CREATION (If needed)                             │
│  ──────────────────────────────────────────                             │
│  1. Identify any cleanup needed (debug code, TODOs, etc.)               │
│  2. Create cleanup task in GoT                                          │
│  3. Add edge: cleanup_task BLOCKED_BY T-CLEANUP-APPROVAL                │
│  4. DO NOT perform cleanup - only document it                           │
│                                                                          │
│  PHASE 7: KNOWLEDGE TRANSFER (Preserve context)                         │
│  ─────────────────────────────────────────────                          │
│  1. Create KT document with discoveries                                 │
│  2. Document any decisions made                                         │
│  3. Document any challenges encountered                                 │
│  4. Mark task complete with retrospective                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Research Checklist

Before ANY implementation, agent MUST verify:

```yaml
Research_Checklist:
  Task_Understanding:
    - [ ] I can state the task goal in one sentence
    - [ ] I understand what "done" looks like
    - [ ] I know which files I will modify
    - [ ] I know which tests I will create

  Codebase_Research:
    - [ ] I have read all Affected_Files
    - [ ] I have searched for similar patterns
    - [ ] I understand the existing architecture
    - [ ] I know where new code should go

  Dependency_Verification:
    - [ ] All Depends_On tasks are complete
    - [ ] Required modules exist and are importable
    - [ ] No conflicting changes in progress

  Challenge_Questions:
    - [ ] Is the specification complete?
    - [ ] Are there unstated requirements?
    - [ ] Could this break existing functionality?
    - [ ] Is this the right approach?
```

### 4.3 API Discovery Protocol

**Why This Matters:**
Reading code and documentation gives you a mental model. Executing Python directly
validates that model and often reveals unexpected capabilities, gaps, or behaviors
that change your approach. This is how understanding evolves from assumed to verified.

**The Discovery Process:**

```python
# Step 1: Inspect class signatures and available methods
python3 -c "
import inspect
from module import ClassName

# Get __init__ signature
sig = inspect.signature(ClassName.__init__)
print(f'__init__{sig}')

# List all public methods with signatures
for name in dir(ClassName):
    if not name.startswith('_'):
        method = getattr(ClassName, name)
        if callable(method):
            try:
                msig = inspect.signature(method)
                print(f'{name}{msig}')
            except (ValueError, TypeError):
                print(f'{name}(...)')
"

# Step 2: Instantiate and test actual behavior
python3 -c "
from module import ClassName

# Create instance with real data
instance = ClassName(actual_path)

# Test methods and observe return types
result = instance.some_method(args)
print(f'Type: {type(result)}')
print(f'Value: {result}')

# Check object attributes
if hasattr(result, '__dict__'):
    print(f'Attributes: {result.__dict__.keys()}')
"

# Step 3: Validate assumptions with edge cases
python3 -c "
# Test what happens with empty input
# Test what happens with invalid input
# Test boundary conditions
# Document actual behavior vs expected
"
```

**Example: How This Document's API Validation Was Performed**

```bash
# 1. Discovered TransactionalGoTAdapter signature
python3 -c "
from scripts.got_utils import TransactionalGoTAdapter
import inspect
sig = inspect.signature(TransactionalGoTAdapter.__init__)
print(f'__init__{sig}')
# Output: (self, got_dir: pathlib.Path = PosixPath('.got'))
"

# 2. Listed all public methods (80+ methods discovered)
python3 -c "
from scripts.got_utils import TransactionalGoTAdapter
methods = [m for m in dir(TransactionalGoTAdapter) if not m.startswith('_')]
for m in sorted(methods):
    print(m)
"

# 3. Tested Query builder (discovered it was more powerful than expected!)
python3 -c "
from cortical.got.api import GoTManager
from cortical.got.query_builder import Query
from pathlib import Path

manager = GoTManager(Path('.got'))

# This worked! The Query builder already supports chained conditions
results = Query(manager).tasks().where(status='pending').or_where(status='blocked').execute()
print(f'Found {len(results)} tasks')

# This worked! Graph traversal already exists
results = Query(manager).tasks().connected_to('S-019', via='CONTAINS').execute()
print(f'Found {len(results)} connected tasks')
"

# KEY INSIGHT: The Query builder already had the power we needed.
# The gap wasn't in the Query builder - it was in connecting
# the CLI's string parsing to the Query builder.
```

**What To Document From Discovery:**

| Discovery Type | Example | Action |
|----------------|---------|--------|
| Method exists that wasn't documented | `Query.connected_to()` | Add to validated API section |
| Method behaves differently than expected | Returns `Task` not `ThoughtNode` | Note the difference, adjust code |
| Method is missing that was expected | No `Query.between_dates()` | Add to gap analysis |
| Error condition not documented | Raises `ValueError` on empty | Add to edge case tests |
| Performance characteristic discovered | O(n) scan on each query | Note in performance section |

**When To Re-Run Discovery:**

- Before starting any new task
- After reading code that you'll modify
- When tests fail unexpectedly
- When assumptions prove wrong
- After pulling changes from others

### 4.4 Knowledge Transfer Template

```yaml
# Created via: python scripts/got_utils.py kt create "Session: [TOPIC]"

Knowledge_Transfer:
  session_id: KT-XXXX
  created: [TIMESTAMP]
  author: [AGENT_ID]

  Context:
    task_completed: [T-XXXX]
    sprint: [Sprint-N]
    epic: "Complex Query Expression System"

  Summary: |
    [2-3 sentence summary of what was accomplished]

  Key_Decisions:
    - decision: "[What was decided]"
      rationale: "[Why this choice]"
      alternatives_considered: "[What else was considered]"

  Discoveries:
    - "[Something learned about the codebase]"
    - "[A pattern that should be followed]"
    - "[A gotcha to watch out for]"

  Challenges_Encountered:
    - challenge: "[What was difficult]"
      resolution: "[How it was resolved]"

  Files_Modified:
    - path: "[file path]"
      changes: "[summary of changes]"

  Tests_Created:
    - path: "[test file path]"
      coverage: "[what it tests]"

  Next_Steps:
    - "[What the next agent should do]"
    - "[What to watch out for]"

  Handoff_Notes: |
    [Anything the next agent needs to know that doesn't fit above]
```

### 4.4 Handoff Protocol

When context must be preserved across sessions:

```yaml
# Created via: python scripts/got_utils.py handoff initiate [TASK_ID]

Handoff:
  id: H-XXXX
  from_session: [SESSION_ID]
  status: pending

  Current_State:
    task: [T-XXXX]
    progress: "[What has been done]"
    blockers: "[What is blocking progress]"

  Files_In_Progress:
    - path: "[file path]"
      state: "[complete|partial|not_started]"
      notes: "[current state of changes]"

  Tests_Status:
    behavioral: "[passing|failing|not_created]"
    unit: "[passing|failing|not_created]"
    coverage: "[percentage]"

  Instructions_For_Next_Agent: |
    1. [First thing to do]
    2. [Second thing to do]
    3. [Third thing to do]

  Warnings: |
    - [Things to be careful of]
    - [Mistakes to avoid]

  Required_Reading:
    - [File or doc that must be read first]
    - [Another important reference]
```

---

## Part 5: Validation Gates

### 5.1 Per-Task Validation

Every task must pass:

```bash
# Minimum validation for any task
python scripts/got_utils.py validate                    # GoT healthy
python -m pytest tests/smoke/ -v                        # No regressions
python -m pytest tests/behavioral/test_[FEATURE].py -v  # Behavioral pass
python -m pytest tests/unit/test_[FEATURE].py -v        # Unit pass
python -m coverage report --include="[FILES]" --fail-under=90
```

### 5.2 Sprint Validation Gates

Each sprint has a validation script:

| Sprint | Script | Checks |
|--------|--------|--------|
| Sprint-1 | `scripts/validate_sprint1.sh` | Module structure, AST, Lexer, Registry |
| Sprint-2 | `scripts/validate_sprint2.sh` | Parser, Basic execution |
| Sprint-3 | `scripts/validate_sprint3.sh` | Functions, Graph operations |
| Sprint-4 | `scripts/validate_sprint4.sh` | Optimization, CLI integration |
| Sprint-5 | `scripts/validate_sprint5.sh` | Full integration, Documentation |

### 5.3 Final Validation Gate

Before epic completion:

```bash
#!/bin/bash
# scripts/validate_epic.sh

set -e

echo "=== Epic Validation Gate ==="

# All sprint gates
for i in 1 2 3 4 5; do
    bash scripts/validate_sprint${i}.sh
done

# Full test suite
python -m pytest tests/ -v

# Coverage
python -m coverage run -m pytest tests/
python -m coverage report --include="cortical/got/expression/*" --fail-under=90

# Performance benchmark
python scripts/benchmark_expression.py

# GoT health
python scripts/got_utils.py validate

echo "=== Epic COMPLETE ==="
```

---

## Part 6: GoT Commands Reference

### 6.1 Creating Entities

```bash
# Epic
python scripts/got_utils.py epic create "Title" --description "..."

# Sprint
python scripts/got_utils.py sprint create "Title" --goal "..."

# Task
python scripts/got_utils.py task create "Title" \
    --priority [critical|high|medium|low] \
    --category [feature|bugfix|refactor|docs|test]

# Edge (dependency)
python scripts/got_utils.py edge add T-001 T-002 DEPENDS_ON
python scripts/got_utils.py edge add T-003 T-CLEANUP-APPROVAL BLOCKED_BY

# Decision
python scripts/got_utils.py decision log "Decision title" \
    --rationale "Why this decision was made"
```

### 6.2 Managing Work

```bash
# Start task
python scripts/got_utils.py task start T-XXXX

# Complete task
python scripts/got_utils.py task complete T-XXXX \
    --retrospective "What worked, what didn't, what was learned"

# Create knowledge transfer
python scripts/got_utils.py kt create "Session: Topic" \
    --summary "Key outcomes..."

# Initiate handoff
python scripts/got_utils.py handoff initiate T-XXXX \
    --target agent \
    --instructions "What to do next..."

# Accept handoff
python scripts/got_utils.py handoff accept H-XXXX
```

### 6.3 Querying State

```bash
# Current state
python scripts/got_utils.py task list --status in_progress
python scripts/got_utils.py blocked
python scripts/got_utils.py validate

# Pending handoffs
python scripts/got_utils.py handoff list --status pending

# Knowledge transfers
python scripts/got_utils.py kt list --status draft
```

---

## Part 7: Operational Considerations & Risk Mitigations

### 7.1 API Status and Freedom to Change

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    API STATUS: ALPHA (Internal Only)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  got_utils.py query is ALPHA and used only by the development team.    │
│                                                                          │
│  This means:                                                            │
│  ✓ We can change the interface freely                                  │
│  ✓ We can break backwards compatibility                                │
│  ✓ We don't need deprecation periods                                   │
│                                                                          │
│  BUT we must:                                                           │
│  ✗ NOT break working functionality carelessly                          │
│  ✗ NOT make changes that block ongoing work                            │
│  ✗ Always maintain a path toward the working solution                  │
│                                                                          │
│  The existing CLI commands (task list, sprint list, etc.) are STABLE   │
│  and use separate code paths that won't be affected by query changes.  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Debugging-Style Error Messages

Query errors must be helpful like a debugger, not cryptic like a compiler:

```python
# BAD: Unhelpful error
ParseError: Unexpected token at position 15

# GOOD: Debugging-style error
ParseError: Unexpected token 'AND' at position 15

  status = pending AND
                   ^^^

  Expected: value (string, number, or identifier)

  Did you mean:
    status = 'pending' AND ...   (quote the string)
    status = pending_status AND ...   (use a field name)

  Hint: String values must be quoted: 'pending' or "pending"
```

**Error Message Requirements:**

| Error Type | Must Include |
|------------|--------------|
| Lexer error | Position, invalid character, suggestion |
| Parser error | Position, expected tokens, what was found, did-you-mean |
| Executor error | Function name, available functions, argument mismatch |
| Schema error | Field name, valid fields for entity type |
| Type error | Expected type, actual type, conversion hint |

**Implementation Pattern:**

```python
class QueryError(Exception):
    """Base class with debugging context."""

    def __init__(self, message: str, position: int = None,
                 source: str = None, suggestions: List[str] = None):
        self.message = message
        self.position = position
        self.source = source
        self.suggestions = suggestions or []

    def format_with_context(self) -> str:
        """Format error with source context and suggestions."""
        lines = [f"QueryError: {self.message}"]

        if self.source and self.position is not None:
            lines.append(f"\n  {self.source}")
            lines.append(f"  {' ' * self.position}^^^")

        if self.suggestions:
            lines.append("\n  Did you mean:")
            for s in self.suggestions:
                lines.append(f"    {s}")

        return "\n".join(lines)
```

### 7.3 Testing Strategy: Use Existing GoT Data

**Do NOT create synthetic test data. Use the real GoT repository.**

```
Available Test Data (verified 2026-01-04):
├── Tasks: 336
│   ├── completed: 238
│   ├── pending: 97
│   └── blocked: 1
├── Edges: 434
├── Sprints: 46
└── Epics: 14

Sample entities for testing:
├── Task: T-20251231-230615-b1e918b6 (pending)
├── Task: T-20251222-204134-140ee2d3 (pending)
├── Sprint: S-sprint-020-forensic-remediation
└── Sprint: S-sprint-003
```

**Why Use Real Data:**
1. Tests reflect actual usage patterns
2. No risk of divergence between test and production schemas
3. Validates against real edge cases we've encountered
4. Proves the system works on itself (dog-fooding)

**How to Test Queries:**

```python
# In tests, use the real .got directory
def test_query_pending_tasks():
    manager = TransactionalGoTAdapter()

    # Use real data, don't create test fixtures
    results = manager.query("status = 'pending'")

    # Assert on structure, not specific counts (data changes)
    assert isinstance(results, list)
    for r in results:
        assert 'id' in r
        assert 'title' in r

# For edge cases, find existing examples
def test_query_blocked_task():
    manager = TransactionalGoTAdapter()

    # We know there's 1 blocked task from our survey
    results = manager.query("status = 'blocked'")
    assert len(results) >= 1
```

### 7.4 Task Sizing and Sub-Task Guidelines

**Problem:** Complex tasks may exceed agent context windows.

**Solution:** Use schema limits and explicit sub-task decomposition.

```yaml
Task_Size_Limits:
  Maximum_Lines_Changed: 300  # Per task
  Maximum_Files_Modified: 5   # Per task
  Maximum_Test_Cases: 20      # Per task

  If_Exceeded:
    1. Stop and decompose into sub-tasks
    2. Create sub-tasks in GoT with DEPENDS_ON edges
    3. Each sub-task must be completable in one session
    4. Parent task becomes a coordination task only
```

**Example Decomposition:**

```
T-007: Implement recursive descent parser
  ├── T-007-A: Implement expression parsing (primary, and_expr, or_expr)
  ├── T-007-B: Implement comparison parsing (operators, values)
  ├── T-007-C: Implement function call parsing
  ├── T-007-D: Implement clause parsing (ORDER BY, LIMIT)
  └── T-007-E: Integration and edge case handling

Edges:
  T-007-B DEPENDS_ON T-007-A
  T-007-C DEPENDS_ON T-007-A
  T-007-D DEPENDS_ON T-007-A
  T-007-E DEPENDS_ON T-007-B, T-007-C, T-007-D
```

**When to Decompose:**

| Signal | Action |
|--------|--------|
| Task description > 100 lines | Decompose |
| Affected_Files > 5 | Decompose |
| Multiple distinct features | Decompose |
| "AND" in task title | Probably needs decomposition |
| Estimated > 2 hours | Decompose |

### 7.5 Schema Discovery and Help System

**Users need to know the schema before querying, like a database.**

**IMPORTANT: We already have a schema system. DO NOT create a new one.**

The existing schema infrastructure is in:
- `cortical/got/schema.py` - `SchemaRegistry`, `BaseSchema`, `Field`, `FieldType`
- `cortical/got/entity_schemas.py` - All entity schemas registered
- `cortical/got/types.py` - `VALID_EDGE_TYPES` (22 edge types)

```bash
# Future CLI support (to be implemented)
python scripts/got_utils.py query --help-syntax
python scripts/got_utils.py query --list-fields tasks
python scripts/got_utils.py query --list-functions
python scripts/got_utils.py query --explain "status = 'pending'"
```

**How to Introspect the Schema Dynamically:**

```python
from cortical.got.entity_schemas import ensure_schemas_registered
from cortical.got.schema import get_registry
from cortical.got.types import VALID_EDGE_TYPES

# Ensure schemas are loaded
ensure_schemas_registered()
registry = get_registry()

# Get registered entity types dynamically
entity_types = list(registry._schemas.keys())
# Returns: ['task', 'decision', 'sprint', 'epic', 'edge', 'handoff', ...]

# Get schema for an entity type
task_schema = registry.get_schema('task')

# Introspect fields
for field_name, field in task_schema.fields.items():
    print(f'{field_name}: {field.field_type.name}')
    if field.choices:  # ENUM fields have valid values
        print(f'  Valid values: {field.choices}')
    if field.description:
        print(f'  Description: {field.description}')

# Get valid edge types
edge_types = sorted(VALID_EDGE_TYPES)
# Returns: ['BLOCKS', 'CAUSED_BY', 'CHILD_OF', 'CONTAINS', ...]
```

**What the Schema Provides (verified 2026-01-04):**

```
Registered Entity Types: 12
  claudemd_layer, claudemd_version, decision, document, edge, epic,
  handoff, knowledge_transfer, persona_profile, sprint, task, team

Task Schema Fields:
  id: STRING - Unique entity identifier
  title: STRING - Task title
  status: ENUM (choices: ['pending', 'in_progress', 'completed', 'blocked'])
  priority: ENUM (choices: ['low', 'medium', 'high', 'critical'])
  description: STRING - Detailed task description
  properties: DICT - Arbitrary key-value properties
  metadata: DICT - System metadata

Valid Edge Types: 22
  BLOCKS, CAUSED_BY, CHILD_OF, CONTAINS, CONTINUES, CONTRADICTS,
  DEPENDS_ON, DERIVED_FROM, DOCUMENTED_BY, DOCUMENTS, FAILED_ATTEMPT,
  IMPLEMENTS, JUSTIFIES, MOTIVATES, PARENT_OF, PART_OF, PRODUCES,
  REFERENCES, RELATES_TO, REQUIRES, SUPERSEDES, TRANSFERS
```

**Benefits for Query Expressions:**

| Benefit | How It Helps |
|---------|--------------|
| Dynamic field discovery | Query validator can check field names at parse time |
| ENUM choices | Can validate values and suggest corrections |
| Field descriptions | Can include in --help-syntax output |
| Type information | Can validate comparison operators (no `status > 5`) |
| Extensibility | New entity types/fields automatically available |

**Error Messages Use Introspected Schema:**

```python
# When user queries unknown field, introspect to suggest
from cortical.got.schema import get_registry
import difflib

def suggest_field(entity_type: str, typo: str) -> str:
    schema = get_registry().get_schema(entity_type)
    valid_fields = list(schema.fields.keys())
    matches = difflib.get_close_matches(typo, valid_fields, n=1)
    return matches[0] if matches else None

# Usage in error message:
# QueryError: Unknown field 'statsu' for entity type 'task'
#   Did you mean: status
#   Valid fields: id, title, status, priority, description, ...
```

### 7.6 Query Optimizer Requirements

The query optimizer must understand both the query language AND the schema:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUERY OPTIMIZER REQUIREMENTS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. SCHEMA-AWARE OPTIMIZATION                                           │
│     - Know which fields are indexed (status, priority)                  │
│     - Know cardinality of enum fields                                   │
│     - Know relationship types and their frequencies                     │
│                                                                          │
│  2. QUERY PLAN GENERATION                                               │
│     - Choose index scan vs full scan based on selectivity               │
│     - Order joins by estimated result size                              │
│     - Push filters down before traversal                                │
│                                                                          │
│  3. COST ESTIMATION                                                     │
│     - Estimate rows examined                                            │
│     - Estimate memory usage                                             │
│     - Warn on expensive queries                                         │
│                                                                          │
│  4. EXPLAIN SUPPORT                                                     │
│     - Show query plan before execution                                  │
│     - Show actual vs estimated costs after execution                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Integration with Query Builder:**

```python
# The optimizer compiles expressions to Query builder calls
# and chooses the optimal execution strategy

expression = parse("status = 'pending' AND priority = 'high'")

# Optimizer sees:
#   - status is indexed → use index
#   - priority is indexed → use index
#   - AND → intersect results

optimized = optimizer.optimize(expression)
# Result: use_index('status', 'pending').use_index('priority', 'high')

# vs naive:
# Result: full_scan().filter(status='pending').filter(priority='high')
```

### 7.7 Security Review Protocol

**Security concerns will be discovered during implementation.** Create review tasks as they're found:

```yaml
Security_Review_Protocol:
  When_To_Create_Review_Task:
    - Parsing user input (injection risks)
    - Graph traversal (DoS via deep recursion)
    - Large result sets (memory exhaustion)
    - Error messages (information leakage)
    - Any code handling external data

  Task_Template:
    title: "SECURITY: Review [component] for [risk type]"
    priority: high
    category: security
    description: |
      ## Risk Identified
      [Description of potential security issue]

      ## Location
      [File and line numbers]

      ## Review Checklist
      - [ ] Input validation sufficient
      - [ ] No injection vectors
      - [ ] Resource limits in place
      - [ ] Error messages don't leak info
      - [ ] Tested with malicious input

  Process:
    1. Developer notices potential security issue
    2. Create security review task immediately
    3. Link to implementation task via RELATED_TO edge
    4. Security task blocks deployment, not development
    5. Separate security-focused review session
```

**Known Security Considerations (Pre-Implementation):**

| Area | Risk | Mitigation |
|------|------|------------|
| Expression parsing | ReDoS in regex | Use non-backtracking patterns |
| Graph traversal | Infinite loops | Max depth limits |
| Large results | Memory exhaustion | Default LIMIT, max LIMIT |
| Error messages | Schema leakage | Sanitize in production mode |
| Function registry | Code injection | No eval/exec, whitelist only |

## Appendix A: Original Audit Findings

*(Preserved from version 1.0 - see sections 1.1-1.4 of original document)*

---

## Appendix B: Grammar Specification

```
<query>           ::= <expression> [<order_clause>] [<limit_clause>]

<expression>      ::= <and_expr> ( 'OR' <and_expr> )*
<and_expr>        ::= <primary> ( 'AND' <primary> )*
<primary>         ::= <comparison> | <function_call> | '(' <expression> ')'

<comparison>      ::= <field> <operator> <value>
<operator>        ::= '=' | '!=' | '>' | '<' | '>=' | '<=' | 'IN' | 'LIKE'

<function_call>   ::= <identifier> '(' [<arg_list>] ')'
<arg_list>        ::= <arg> ( ',' <arg> )*
<arg>             ::= <value> | <identifier> '=' <value>

<field>           ::= <identifier>
<value>           ::= <string> | <number> | <date> | <list>
<list>            ::= '[' [<value> (',' <value>)*] ']'

<order_clause>    ::= 'ORDER' 'BY' <field> ['ASC'|'DESC']
<limit_clause>    ::= 'LIMIT' <number> ['OFFSET' <number>]
```

---

## Document Approval

**This document must be approved before ANY GoT entities are created or code is written.**

Approval signifies agreement with:
- [ ] Generalized architecture (function registry pattern)
- [ ] Epic/Sprint/Task structure
- [ ] Agent workflow protocol
- [ ] Cleanup governance model
- [ ] Validation gates
- [ ] Knowledge transfer requirements
- [ ] API Discovery Protocol (Phase 2.5)
- [ ] Operational considerations (Part 7)
  - [ ] Debugging-style error messages
  - [ ] Testing with existing GoT data
  - [ ] Task sizing and sub-task guidelines
  - [ ] Schema discovery system
  - [ ] Query optimizer requirements
  - [ ] Security review protocol

---

*Document Version 2.4 - Awaiting Approval*

*Key additions in this version:*
- *Section 7.5 now uses existing schema system (`cortical/got/schema.py`, `entity_schemas.py`) instead of proposing hardcoded schemas*
- *Dynamic schema introspection enables future-proof query validation*
- *No hardcoded field names, status values, or edge types - all discovered at runtime*
