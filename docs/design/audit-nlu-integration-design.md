# Audit NLU Integration Design Document

**Version:** 1.0
**Date:** 2026-01-08
**Status:** Draft
**Author:** Claude (Opus 4.5)

---

## Executive Summary

This document proposes integrating Natural Language Understanding (NLU) capabilities into the audit reasoning tool (`scripts/audit_reasoning.py`) by leveraging existing GoT query infrastructure and PRISM/Spark components. The goal is to enable users to "talk to their codebase" with minimal training time while providing value to anyone who uses the tool.

### Key Insight

The GoT codebase already has a robust query infrastructure:
- **Expression Parser** (`cortical/got/expression/parser.py`) - Recursive descent parser
- **Query Builder** (`cortical/got/query_builder.py`) - Fluent SQL-like API
- **Translator** (`cortical/got/expression/translator.py`) - Natural language → DSL

This design **generalizes the existing translator pattern** rather than building new NLU from scratch.

---

## 1. Problem Statement

### Current State

The audit tool requires explicit CLI flags:
```bash
python scripts/audit_reasoning.py cortical/ --with-git --aggregate revision --show-related
```

### Desired State

Natural language queries:
```bash
python scripts/audit_reasoning.py "risky files in reasoning/ not tests"
python scripts/audit_reasoning.py "why is prism_pln.py flagged"
python scripts/audit_reasoning.py "files with high churn and TODOs"
```

### Success Criteria

1. **Zero Training Time** - Pattern-based parsing, no ML required
2. **Explainable Results** - Users understand WHY files are flagged
3. **Backward Compatible** - Existing CLI flags still work
4. **Value to Anyone** - Clear, actionable output

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      AUDIT NLU ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Input: "risky files in reasoning/ not tests"                      │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────┐                         │
│  │     AuditQueryTranslator (NEW)             │                         │
│  │     Extends: got/expression/translator.py  │                         │
│  │                                            │                         │
│  │     Extracts:                              │                         │
│  │     - scope: "reasoning/"                  │                         │
│  │     - negations: ["tests"]                 │                         │
│  │     - intent: "list_risky"                 │                         │
│  └──────────────────┬─────────────────────────┘                         │
│                     │                                                    │
│                     ▼                                                    │
│  ┌────────────────────────────────────────────┐                         │
│  │     AuditQueryBuilder (NEW)                │                         │
│  │     Wraps: got/query_builder.py patterns   │                         │
│  │                                            │                         │
│  │     Builds:                                │                         │
│  │     - File filters (scope, negations)      │                         │
│  │     - Risk thresholds                      │                         │
│  │     - Output format                        │                         │
│  └──────────────────┬─────────────────────────┘                         │
│                     │                                                    │
│                     ▼                                                    │
│  ┌────────────────────────────────────────────┐                         │
│  │     AuditReasoner (EXISTING)               │                         │
│  │     + ExplainedResult (NEW)                │                         │
│  │                                            │                         │
│  │     Executes:                              │                         │
│  │     - PLN reasoning                        │                         │
│  │     - Importance weighting                 │                         │
│  │     - Result explanation                   │                         │
│  └──────────────────┬─────────────────────────┘                         │
│                     │                                                    │
│                     ▼                                                    │
│  ┌────────────────────────────────────────────┐                         │
│  │     ExplainedAuditResult (NEW)             │                         │
│  │                                            │                         │
│  │     Output:                                │                         │
│  │     - File path                            │                         │
│  │     - Risk score with breakdown            │                         │
│  │     - Triggered rules                      │                         │
│  │     - Suggested actions                    │                         │
│  └────────────────────────────────────────────┘                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Design

### 3.1 AuditQueryTranslator

**Location:** `scripts/audit_query_translator.py` (or integrate into `audit_reasoning.py`)

**Purpose:** Translate natural language audit queries to structured parameters.

**Design Pattern:** Extends the GoT translator pattern from `cortical/got/expression/translator.py`

```python
@dataclass
class AuditQuery:
    """Structured representation of an audit query."""
    # Scope
    directory: Optional[str] = None
    file_patterns: List[str] = field(default_factory=list)

    # Filters
    negations: List[str] = field(default_factory=list)  # Exclude these
    include_patterns: List[str] = field(default_factory=list)

    # Intent
    intent: str = "list"  # list, explain, compare, trace
    target_file: Optional[str] = None  # For "why is X flagged"

    # Thresholds
    min_risk: float = 0.0
    max_results: Optional[int] = None

    # Output
    explain: bool = False
    format: str = "table"  # table, json, detail


def translate_audit_query(query: str) -> AuditQuery:
    """
    Translate natural language to AuditQuery.

    Pattern matching approach (no ML required):

    Examples:
        "risky files in reasoning/"
        → AuditQuery(directory="reasoning/", min_risk=0.5)

        "why is prism_pln.py flagged"
        → AuditQuery(intent="explain", target_file="prism_pln.py")

        "files not tests with high churn"
        → AuditQuery(negations=["tests"], include_patterns=["high_churn"])
    """
```

**Supported Patterns:**

| Pattern | Translation |
|---------|-------------|
| `risky files in <dir>` | `directory=<dir>, min_risk=0.5` |
| `why is <file> flagged` | `intent="explain", target_file=<file>` |
| `<query> not <term>` | `negations.append(<term>)` |
| `files with <trait>` | `include_patterns.append(<trait>)` |
| `high priority` | `min_risk=0.7` |
| `top N` | `max_results=N` |
| `explain` | `explain=True` |

### 3.2 ExplainedAuditResult

**Purpose:** Provide transparent, understandable audit results.

**Inspired by:** `nlu_showcase.py:ExplainedResult`

```python
@dataclass
class ExplainedAuditResult:
    """Audit result with full explanation."""
    file_path: str
    risk_score: float
    importance_score: float

    # Explanation components
    triggered_rules: List[TriggeredRule]
    trait_contributions: Dict[str, float]
    pattern_matches: List[PatternMatch]

    # Context
    related_files: List[str]
    suggested_actions: List[str]

    def explain(self) -> str:
        """Human-readable explanation."""
        lines = [
            f"{self.file_path}",
            f"  Risk: {self.risk_score:.0%} | Importance: {self.importance_score:.0%}",
            "",
            "  Why flagged:"
        ]

        for rule in self.triggered_rules[:3]:
            lines.append(f"    ├─ {rule.name}: {rule.contribution:.0%}")
            lines.append(f"    │  {rule.explanation}")

        if self.suggested_actions:
            lines.append("")
            lines.append("  Suggested actions:")
            for action in self.suggested_actions[:2]:
                lines.append(f"    → {action}")

        return "\n".join(lines)


@dataclass
class TriggeredRule:
    """A PLN rule that contributed to the risk score."""
    name: str
    contribution: float  # How much this rule added to score
    explanation: str     # Human-readable explanation
    truth_value: float   # PLN confidence
```

### 3.3 Integration Points

#### 3.3.1 With Existing GoT Infrastructure

```python
# Reuse the expression parser for structured queries
from cortical.got.expression import parse, execute

# Example: Allow DSL expressions directly
def execute_audit_query(query: str, reasoner: AuditReasoner):
    """Execute audit query - supports both NL and DSL."""

    # Try translating natural language first
    audit_query = translate_audit_query(query)

    # If translation didn't change it, try DSL parse
    if audit_query.directory is None and audit_query.intent == "list":
        try:
            # Try parsing as DSL expression
            ast = parse(query)
            # Execute against file data
            return execute_dsl_audit(ast, reasoner)
        except ParseError:
            pass

    # Execute translated query
    return execute_audit(audit_query, reasoner)
```

#### 3.3.2 With GitHistoryTrainer (Phase 3)

```python
from cortical.spark.git_trainer import GitHistoryTrainer

class EnhancedAuditReasoner(AuditReasoner):
    """Audit reasoner with enhanced git analysis."""

    def __init__(self, ...):
        super().__init__(...)
        self.git_trainer = GitHistoryTrainer()

    def analyze_git_enhanced(self, directory: Path) -> Dict[str, float]:
        """Use weighted commit analysis instead of simple churn count."""

        # Train on git history
        self.git_trainer.train_on_directory(directory)

        # Get weighted importance per file
        weights = {}
        for file_path in self.files:
            # Uses branch weights, quality signals, temporal decay
            weights[file_path] = self.git_trainer.get_file_weight(file_path)

        return weights
```

#### 3.3.3 With SparkCodeIntelligence (Phase 4)

```python
from cortical.spark.intelligence import SparkCodeIntelligence

class StructuralAuditReasoner(EnhancedAuditReasoner):
    """Audit reasoner with code structure analysis."""

    def trace_risk_propagation(self, risky_file: str) -> List[str]:
        """Find files that might be affected by a risky file."""

        intel = SparkCodeIntelligence()
        intel.train()  # Index codebase

        affected = set()

        # Find functions in risky file
        functions = intel.ast_index.functions_by_file.get(risky_file, [])

        for func in functions:
            # Find all callers
            callers = intel.find_callers(func)
            for caller in callers:
                affected.add(caller['file'])

        return list(affected)
```

---

## 4. Implementation Phases

### Phase 1: Query Translation (Immediate)

**Scope:** Add `AuditQueryTranslator` and basic natural language support.

**Files to modify:**
- `scripts/audit_reasoning.py` - Add query parsing to CLI

**New files:**
- None required - can be inline in audit_reasoning.py (~80 lines)

**Effort:** ~2 hours

**Deliverables:**
```bash
# These should work after Phase 1:
python scripts/audit_reasoning.py "cortical/ not tests"
python scripts/audit_reasoning.py "risky files in reasoning/"
python scripts/audit_reasoning.py "files with high_churn"
```

### Phase 2: Explainable Results (This Week)

**Scope:** Add `ExplainedAuditResult` and transparent output.

**Files to modify:**
- `scripts/audit_reasoning.py` - Add explanation generation

**Effort:** ~4 hours

**Deliverables:**
```bash
# Output shows WHY files are flagged:
python scripts/audit_reasoning.py "explain prism_pln.py"

# Output:
# prism_pln.py
#   Risk: 75% | Importance: 85%
#
#   Why flagged:
#     ├─ high_churn: 30%
#     │  Modified 47 times in 30 days
#     ├─ has_todo: 25%
#     │  Contains 3 TODO comments
#     ├─ complexity: 20%
#     │  930 lines, high cyclomatic complexity
#
#   Suggested actions:
#     → Review and resolve TODO comments
#     → Consider splitting into smaller modules
```

### Phase 3: Enhanced Git Analysis (Next Week)

**Scope:** Replace basic git analysis with `GitHistoryTrainer`.

**Files to modify:**
- `scripts/audit_reasoning.py` - Replace git analysis section

**Integration:**
```python
# Before (current):
churn_count = count_commits(file)
is_high_churn = churn_count > threshold

# After (Phase 3):
weighted_importance = git_trainer.get_file_weight(file)
# Includes: branch weights, quality signals, temporal decay
```

**Effort:** ~3 hours

### Phase 4: Structural Insights (Future)

**Scope:** Add call graph tracing and related file discovery.

**New capabilities:**
```bash
python scripts/audit_reasoning.py "what calls prism_pln.py"
python scripts/audit_reasoning.py "files related to woven_mind.py"
```

**Effort:** ~6 hours

---

## 5. Query Pattern Reference

### 5.1 Scope Patterns

| Natural Language | Structured Query |
|------------------|------------------|
| `<dir>` | `directory=<dir>` |
| `in <dir>` | `directory=<dir>` |
| `files in <dir>` | `directory=<dir>` |
| `<dir>/**` | `directory=<dir>` |

### 5.2 Filter Patterns

| Natural Language | Structured Query |
|------------------|------------------|
| `not <term>` | `negations.append(<term>)` |
| `without <term>` | `negations.append(<term>)` |
| `exclude <term>` | `negations.append(<term>)` |
| `only <term>` | `include_patterns=[<term>]` |
| `with <trait>` | `include_patterns.append(<trait>)` |

### 5.3 Intent Patterns

| Natural Language | Structured Query |
|------------------|------------------|
| `why is <file> flagged` | `intent="explain", target_file=<file>` |
| `explain <file>` | `intent="explain", target_file=<file>` |
| `what calls <file>` | `intent="trace", target_file=<file>` |
| `files related to <file>` | `intent="related", target_file=<file>` |
| `compare <file1> <file2>` | `intent="compare", targets=[<file1>, <file2>]` |

### 5.4 Threshold Patterns

| Natural Language | Structured Query |
|------------------|------------------|
| `risky` | `min_risk=0.5` |
| `high risk` | `min_risk=0.7` |
| `critical` | `min_risk=0.9` |
| `top N` | `max_results=N` |
| `first N` | `max_results=N` |

### 5.5 Trait Patterns

| Natural Language | Maps to Trait |
|------------------|---------------|
| `high churn` | `high_churn` |
| `recently modified` | `recently_modified` |
| `has todos` | `has_todo` |
| `complex` | `high_complexity` |
| `large` | `large_file` |
| `test files` | `is_test` |

---

## 6. Backward Compatibility

### Existing CLI Preserved

All existing CLI flags continue to work:

```bash
# These still work exactly as before:
python scripts/audit_reasoning.py cortical/
python scripts/audit_reasoning.py cortical/ --with-git
python scripts/audit_reasoning.py cortical/ --aggregate revision
python scripts/audit_reasoning.py cortical/ --show-state
python scripts/audit_reasoning.py cortical/ --no-persist
```

### Detection Logic

```python
def parse_cli_input(args):
    """Determine if input is path, flags, or natural language query."""

    # If first arg is a valid path, use traditional mode
    if Path(args[0]).exists():
        return "traditional", args

    # If first arg starts with --, it's a flag
    if args[0].startswith("--"):
        return "traditional", args

    # Otherwise, treat as natural language query
    query = " ".join(args)
    return "natural_language", query
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/unit/test_audit_query_translator.py

def test_scope_extraction():
    query = translate_audit_query("files in reasoning/")
    assert query.directory == "reasoning/"

def test_negation_extraction():
    query = translate_audit_query("cortical/ not tests")
    assert "tests" in query.negations

def test_intent_explain():
    query = translate_audit_query("why is prism_pln.py flagged")
    assert query.intent == "explain"
    assert query.target_file == "prism_pln.py"

def test_combined_patterns():
    query = translate_audit_query("risky files in reasoning/ not tests top 10")
    assert query.directory == "reasoning/"
    assert "tests" in query.negations
    assert query.min_risk == 0.5
    assert query.max_results == 10
```

### 7.2 Behavioral Tests

```python
# tests/behavioral/test_audit_nlu_stories.py

def test_user_asks_why_file_flagged():
    """
    GIVEN a codebase with prism_pln.py having high churn
    WHEN user asks "why is prism_pln.py flagged"
    THEN output explains the risk factors
    """
    result = run_audit("why is prism_pln.py flagged")
    assert "high_churn" in result.explanation
    assert result.triggered_rules is not None

def test_user_excludes_test_files():
    """
    GIVEN a codebase with test files
    WHEN user queries "cortical/ not tests"
    THEN no test files appear in results
    """
    results = run_audit("cortical/ not tests")
    for r in results:
        assert "test" not in r.file_path.lower()
```

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pattern ambiguity | Medium | Low | Fall back to literal interpretation |
| Performance with large codebases | Low | Medium | Lazy evaluation, limit defaults |
| User expectation mismatch | Medium | Medium | Clear documentation of supported patterns |
| Breaking existing workflows | Low | High | Preserve all existing CLI flags |

---

## 9. Success Metrics

### Usability Metrics

1. **Zero Training Time** - Tool works immediately on any codebase
2. **Pattern Coverage** - 80%+ of common queries supported
3. **Explanation Clarity** - Users understand why files are flagged

### Technical Metrics

1. **Backward Compatibility** - 100% of existing CLI commands work
2. **Response Time** - <2s for typical queries on medium codebases
3. **Test Coverage** - 90%+ for new code

---

## 10. Future Considerations

### 10.1 PRISM Integration (Deferred)

Spreading activation for related file discovery could be added later:

```python
# Future: Use PRISM for "files related to X"
from cortical.reasoning.prism_slm import TransitionGraph

def find_related_files(target: str, file_graph: TransitionGraph) -> List[str]:
    """Use spreading activation to find related files."""
    activations = file_graph.spreading_activation({target: 1.0})
    return sorted(activations.keys(), key=lambda f: activations[f], reverse=True)
```

### 10.2 Interactive Mode (Deferred)

Q&A mode for exploring audit results:

```bash
python scripts/audit_reasoning.py --interactive

🔍 Audit> risky files in cortical/
[Shows results]

🔍 Audit> explain first result
[Shows detailed explanation]

🔍 Audit> what calls it
[Shows callers]
```

---

## 11. Appendix: Existing Infrastructure Reference

### GoT Expression Parser

**Location:** `cortical/got/expression/parser.py`

**Grammar:**
```
query       ::= expression [order_clause] [limit_clause]
expression  ::= and_expr ('OR' and_expr)*
and_expr    ::= not_expr ('AND' not_expr)*
not_expr    ::= 'NOT' not_expr | primary
primary     ::= comparison | function_call | '(' expression ')'
```

**Operators:** `=`, `!=`, `>`, `<`, `>=`, `<=`, `IN`, `NOT IN`, `LIKE`, `NOT LIKE`

### GoT Translator

**Location:** `cortical/got/expression/translator.py`

**Pattern:** Natural language → DSL string

**Examples:**
- `"blocked tasks"` → `"blocked()"`
- `"what blocks T-001"` → `"blockers('T-001')"`
- `"high priority pending"` → `"priority = 'high' AND status = 'pending'"`

### NLU Intent Parser

**Location:** `cortical/query/intent.py`

**Capabilities:**
- Intent classification (where/how/what/why)
- Action verb extraction
- Subject identification
- Term expansion

---

## 12. Approval

### Review Checklist

- [ ] Design aligns with sovereignty principle (no new dependencies)
- [ ] Leverages existing GoT infrastructure
- [ ] Backward compatible with existing CLI
- [ ] Clear phase boundaries
- [ ] Testable requirements

### Approvals

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | Claude (Opus 4.5) | 2026-01-08 | Draft |
| Reviewer | | | Pending |
| Approver | | | Pending |
