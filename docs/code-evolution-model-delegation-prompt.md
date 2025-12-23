# Code Evolution Model: Delegation Prompt for New Session

**Created:** 2025-12-23
**Purpose:** Continue Code Evolution Model implementation in fresh context with parallel sub-agents
**Branch:** `claude/setup-pytest-sparksml-71W05`

---

## Quick Start (Copy This to New Thread)

```
I want to continue implementing the Code Evolution Model for SparkSLM.

## Context
Research phase is COMPLETE. Four comprehensive research documents exist:
- `docs/diff-tokenization-research.md` (847 lines) - DiffTokenizer design
- `docs/commit-intent-parsing-research.md` (1325 lines) - IntentParser design
- `docs/code-evolution-co-change-research.md` (762 lines) - Co-Change Model design
- `docs/code-evolution-model-architecture.md` (816 lines) - System architecture
- `docs/code-evolution-model-research-paper-outline.md` - Unified paper outline

## Existing Implementation (Already Tested)
- `cortical/spark/tokenizer.py` - CodeTokenizer (32 tests passing)
- `cortical/spark/ast_index.py` - ASTIndex (29 tests passing)
- `cortical/spark/intelligence.py` - SparkCodeIntelligence (57 tests passing)
- `cortical/spark/ngram.py` - NGramModel (existing tests)

## What Needs Implementation
Three new components based on research docs:

1. **IntentParser** (`cortical/spark/intent_parser.py`)
   - Parse conventional commits: `feat(scope): description`
   - Extract: type, scope, action, entities, references
   - Fallback to keyword extraction for free-form messages
   - See: `docs/commit-intent-parsing-research.md` Section 5

2. **DiffTokenizer** (`cortical/spark/diff_tokenizer.py`)
   - Tokenize git diffs with special tokens: [FILE], [HUNK], [ADD], [DEL], [CTX]
   - Adaptive context sizing based on diff size
   - Pattern detection (guard, cache, error handling)
   - See: `docs/diff-tokenization-research.md` Section 4

3. **CoChangeModel** (`cortical/spark/co_change.py`)
   - Learn file co-occurrence from git history
   - Temporal weighting with exponential decay
   - Predict related files given seed files
   - See: `docs/code-evolution-co-change-research.md` Section 4

## Approach
Use parallel sub-agents:
- Agent 1: Implement IntentParser + unit tests
- Agent 2: Implement DiffTokenizer + unit tests
- Agent 3: Implement CoChangeModel + unit tests
- Main agent: Review, integrate, add integration tests

## Test Requirements
Each component needs:
- Unit tests (30+ tests per component)
- Edge case coverage
- Integration with existing SparkCodeIntelligence
- Performance tests for large inputs

## Key Technical Details
- All classes need `to_dict()`/`from_dict()` for JSON serialization
- Follow existing patterns in `cortical/spark/` package
- Update `cortical/spark/__init__.py` to export new classes
- No external dependencies (zero-dependency library)

Please read the research documents first, then spawn sub-agents to implement in parallel.
```

---

## Detailed Context

### What Was Accomplished (Previous Session)

1. **SparkCodeIntelligence Module Split**
   - Split 1800-line script into modular packages
   - Created: `tokenizer.py`, `ast_index.py`, `intelligence.py`
   - 118 tests all passing (unit + integration + performance)

2. **Code Evolution Model Research**
   - Spawned 4 sub-agents for parallel research
   - Created comprehensive research documents (~3,750 lines total)
   - Consolidated into paper outline
   - Committed and pushed to branch

### Research Document Summaries

#### 1. IntentParser (`docs/commit-intent-parsing-research.md`)
```python
# Key data structure
@dataclass
class IntentResult:
    type: str           # feat, fix, refactor, docs, test, chore
    scope: Optional[str]  # Module scope from (scope)
    action: str         # add, fix, update, remove
    entities: List[str] # Extracted keywords
    description: str    # Full description
    breaking: bool      # Breaking change flag
    priority: str       # critical, high, medium, low
    references: List[str]  # Issue/PR/Task IDs
    confidence: float   # 0.0-1.0
    method: str         # Classification method used

# Key patterns
CONVENTIONAL_COMMIT = r'^(feat|fix|refactor|docs|test|chore)(\([^)]+\))?!?:\s*(.+)'
```

#### 2. DiffTokenizer (`docs/diff-tokenization-research.md`)
```python
# Special tokens
SPECIAL_TOKENS = {
    '[FILE]', '[FILE_NEW]', '[FILE_DEL]', '[FILE_REN]',
    '[HUNK]', '[FUNC]', '[CLASS]',
    '[ADD]', '[DEL]', '[MOD]', '[CTX]',
    '[PATTERN:guard]', '[PATTERN:cache]', '[PATTERN:refactor]',
    '[LANG:python]', '[TYPE:feat]', '[IMPACT:high]'
}

# Adaptive context sizing
def adaptive_context_size(total_changes: int) -> int:
    if total_changes < 50: return 10
    elif total_changes < 200: return 5
    else: return 2
```

#### 3. CoChangeModel (`docs/code-evolution-co-change-research.md`)
```python
# Key data structures
@dataclass
class CoChangeEdge:
    source_file: str
    target_file: str
    co_change_count: int
    weighted_score: float  # Temporal decay applied
    confidence: float      # Normalized probability
    last_co_change: datetime
    commits: List[str]     # Commit SHAs

# Temporal weighting
weight = exp(-decay_lambda * age_days)  # λ=0.01, half-life ~69 days
```

### Parallel Sub-Agent Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN AGENT (Coordinator)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: Spawn Implementation Agents (Parallel)                 │
│  ├── Sub-Agent A: IntentParser implementation                   │
│  │   └── Read commit-intent-parsing-research.md                 │
│  │   └── Implement cortical/spark/intent_parser.py              │
│  │   └── Create tests/unit/test_intent_parser.py (30+ tests)    │
│  │                                                               │
│  ├── Sub-Agent B: DiffTokenizer implementation                  │
│  │   └── Read diff-tokenization-research.md                     │
│  │   └── Implement cortical/spark/diff_tokenizer.py             │
│  │   └── Create tests/unit/test_diff_tokenizer.py (30+ tests)   │
│  │                                                               │
│  └── Sub-Agent C: CoChangeModel implementation                  │
│      └── Read code-evolution-co-change-research.md              │
│      └── Implement cortical/spark/co_change.py                  │
│      └── Create tests/unit/test_co_change.py (30+ tests)        │
│                                                                   │
│  Phase 2: Review & Integrate (Main Agent)                        │
│  └── Review all implementations                                  │
│  └── Update cortical/spark/__init__.py                          │
│  └── Create integration tests                                    │
│  └── Run full test suite                                         │
│                                                                   │
│  Phase 3: Hybrid Fusion (After Phase 2)                          │
│  └── Integrate components into SparkCodeIntelligence             │
│  └── Add predict_related_changes() method                        │
│  └── Performance testing                                         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure After Implementation

```
cortical/spark/
├── __init__.py           # Update exports
├── ngram.py              # ✅ Existing
├── alignment.py          # ✅ Existing
├── predictor.py          # ✅ Existing
├── anomaly.py            # ✅ Existing
├── quality.py            # ✅ Existing
├── tokenizer.py          # ✅ Existing (CodeTokenizer)
├── ast_index.py          # ✅ Existing (ASTIndex)
├── intelligence.py       # ✅ Existing (SparkCodeIntelligence)
├── intent_parser.py      # 🆕 NEW (IntentParser)
├── diff_tokenizer.py     # 🆕 NEW (DiffTokenizer)
└── co_change.py          # 🆕 NEW (CoChangeModel)

tests/unit/
├── test_intent_parser.py     # 🆕 NEW (30+ tests)
├── test_diff_tokenizer.py    # 🆕 NEW (30+ tests)
└── test_co_change.py         # 🆕 NEW (30+ tests)

tests/integration/
└── test_code_evolution_integration.py  # 🆕 NEW
```

### Success Criteria

1. **All unit tests pass** (90+ new tests across 3 components)
2. **Integration tests pass** (components work together)
3. **JSON serialization works** (`to_dict()`/`from_dict()`)
4. **No external dependencies** (stdlib + existing cortical only)
5. **Performance acceptable** (<100ms for typical operations)
6. **Coverage maintained** (don't drop below 89%)

### Git Information

- **Branch:** `claude/setup-pytest-sparksml-71W05`
- **Latest commit:** `2584e6f0 docs(research): Add Code Evolution Model research`
- **Remote:** Push when implementation complete

---

## Alternative: Simpler Single-Component Approach

If parallel implementation seems too complex, start with just IntentParser:

```
I want to implement the IntentParser component for the Code Evolution Model.

Read `docs/commit-intent-parsing-research.md` and implement:
1. `cortical/spark/intent_parser.py` - IntentParser class with IntentResult dataclass
2. `tests/unit/test_intent_parser.py` - 30+ unit tests

Follow the existing patterns in `cortical/spark/tokenizer.py` for code style.
Key features: conventional commit parsing, keyword extraction, reference extraction.
```

---

**This document serves as a complete handoff for continuing the Code Evolution Model implementation in a fresh context window.**
