# Unified Cognitive Knowledge System

**Architecture Document**
**Date:** 2026-01-11
**Status:** Design Phase

---

## Executive Summary

This document describes the architecture for a unified knowledge system that enables intelligent code navigation through natural language queries. The system bridges semantic understanding (word associations) with structural knowledge (code entities) to answer questions like "where is authentication handled?" with confidence-aware, explainable responses.

The core insight: **the graph should be an extension of the assistant's working memory**, persisting what it learns across sessions and enabling intelligent retrieval instead of repeated grep/glob searches.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Vision](#vision)
3. [Current State Analysis](#current-state-analysis)
4. [Architectural Design](#architectural-design)
5. [Data Model](#data-model)
6. [Query Flow](#query-flow)
7. [Uncertainty Handling](#uncertainty-handling)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Integration Points](#integration-points)
10. [Design Decisions](#design-decisions)

---

## Problem Statement

### The Current Pain

When working on coding tasks, the assistant repeatedly:

1. **Greps for patterns** → Gets 50+ files, most irrelevant
2. **Reads wrong files** → Wastes context on false positives
3. **Forgets navigation paths** → Rediscovers the same routes each session
4. **Can't answer semantic questions** → "Where is auth handled?" requires manual search

### The Root Cause

Two knowledge systems exist but don't communicate:

```
┌─────────────────────┐          ┌─────────────────────┐
│   SEMANTIC LAYER    │    ╳     │   STRUCTURAL LAYER  │
│                     │          │                     │
│  CognitiveAgent     │  NO      │  ASTIndex           │
│  - WORD atoms       │  BRIDGE  │  - Classes          │
│  - SIMILARITY links │          │  - Functions        │
│  - Associations     │          │  - Imports          │
│                     │          │  - Call graph       │
└─────────────────────┘          └─────────────────────┘
```

The semantic layer knows "authentication" relates to "login", "session", "jwt".
The structural layer knows `auth/handler.py` defines `AuthHandler` class.
But there's no connection between them.

### What We Need

```
Query: "where is authentication handled?"

Response: {
  understood: ["authentication", "auth", "login", "session"],
  code_entities: [
    {file: "auth/handler.py", confidence: 0.9, why: "defines AuthHandler, docstring mentions authentication"},
    {file: "middleware/session.py", confidence: 0.7, why: "imports from auth/, handles sessions"},
  ],
  uncertain_about: ["handler" - could mean HTTP handler or auth handler],
  suggestion: "Did you mean AuthHandler in auth/handler.py?"
}
```

---

## Vision

### The Graph as Extended Working Memory

The cognitive graph should function as an extension of the assistant's mind:

- **When context fills up**: Graph remembers what was learned
- **When starting fresh**: Query instead of rediscover
- **When navigating code**: Follow learned paths, not grep
- **When uncertain**: Honestly say "I don't know"

### Core Capabilities

1. **Natural Language Understanding (NLU)**
   - Parse intent: "where", "how", "what", "why"
   - Extract subjects: "authentication handler"
   - Expand semantically: "auth" → "login", "session", "jwt"

2. **Code Entity Resolution**
   - Map words to code: "authentication" → `auth/handler.py`
   - Understand structure: classes, functions, imports
   - Track relationships: who calls what, who imports what

3. **Natural Language Generation (NLG)**
   - Predict next word from context (FOLLOWS links)
   - Generate explanations for findings
   - Produce coherent responses

4. **Honest Uncertainty**
   - Unknown words: "I don't recognize this term"
   - No code mapping: "I know this concept but not where it's implemented"
   - Ambiguous matches: "I found several possibilities"
   - Confident answers: Clear, direct responses when warranted

---

## Current State Analysis

### What Exists

#### Cognitive Module (`cortical/cognitive/`)

| Component | Purpose | Status |
|-----------|---------|--------|
| `CognitiveGraph` | Hypergraph with atoms and links | ✅ Working |
| `CognitiveAgent` | Associations, prediction, goals | ✅ Working |
| `TextToAtomsBridge` | Text → WORD atoms + SIMILARITY | ✅ Working |
| `IncrementalTrainer` | Train on documents | ✅ Working |
| `AssociativePredictor` | Predict from co-occurrences | ⚠️ Uses separate dict, not graph |

**AtomTypes available:**
- Nodes: CONCEPT, PERSON, PREDICATE, VARIABLE, NUMBER, WORD
- Links: INHERITANCE, SIMILARITY, EVALUATION, MEMBER, LIST, BELIEVES, DOUBTS, IMPLIES, EVIDENCE_FOR, CONTEXT, STRONGER_THAN

#### Spark Module (`cortical/spark/`)

| Component | Purpose | Status |
|-----------|---------|--------|
| `ASTIndex` | Parse Python → classes, functions, imports | ✅ Working |
| `NGramModel` | Statistical language model | ✅ Working |
| `SparkPredictor` | Unified prediction facade | ✅ Working |
| `IntentParser` | Parse commit messages | ✅ Working (but for commits, not queries) |

**ASTIndex capabilities:**
- `index_file()`, `index_directory()` - Parse Python
- `find_callers()` - Who calls a function
- `find_class()`, `find_function()` - Find definitions
- `get_inheritance_tree()` - Class hierarchy
- `find_imports_of()` - Who imports a module

#### Query Module (`cortical/query/`)

| Component | Purpose | Status |
|-----------|---------|--------|
| `parse_intent_query()` | Parse "where is X?" → structured intent | ✅ Working |
| `expand_query()` | Semantic term expansion | ✅ Working |
| `find_definition_passages()` | Find "class X" or "def Y" | ✅ Working |
| Intent types | location, implementation, definition, rationale | ✅ Defined |

#### Audit Module (`cortical/cli/audit/`, `cortical/audits/`)

| Component | Purpose | Status |
|-----------|---------|--------|
| `AuditReasoner` | PLN-based risk reasoning | ✅ Working |
| `WovenMindDiscovery` | Pattern discovery, Hebbian learning | ✅ Working |
| `scan` command | Find suspicious comments | ✅ Working |
| `health` command | Codebase metrics | ✅ Working |

### What's Missing

#### 1. CODE Entity Atoms
```python
# Need to add:
FILE = auto()       # Source file
CLASS = auto()      # Class definition
FUNCTION = auto()   # Function/method
MODULE = auto()     # Package/module
```

#### 2. REFERS_TO Links (Semantic ↔ Structural Bridge)
```python
# "authentication" REFERS_TO auth/handler.py
REFERS_TO = auto()  # WORD → CODE_ENTITY with weight
```

#### 3. FOLLOWS Links (Directional Prediction)
```python
# "neural" FOLLOWS "network" (B follows A)
FOLLOWS = auto()    # Directional, for next-word prediction
```

#### 4. Unified Query Handler
```python
class UnifiedQueryHandler:
    def query(self, query_text: str) -> QueryResult:
        # 1. Parse intent
        # 2. Expand terms (SIMILARITY)
        # 3. Resolve code entities (REFERS_TO)
        # 4. Rank and explain
        # 5. Assess confidence
        pass
```

#### 5. Code Entity Bridge
```python
class CodeEntityBridge:
    def index_file(self, path: str, content: str) -> None:
        # 1. Parse with ASTIndex
        # 2. Create FILE, CLASS, FUNCTION atoms
        # 3. Create DEFINES, CONTAINS links
        # 4. Create REFERS_TO from identifiers
        pass
```

---

## Architectural Design

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         QUERY LAYER                                  │
│                                                                      │
│  UnifiedQueryHandler                                                │
│  ├── parse_intent_query() → Intent                                  │
│  ├── expand_terms() → Expanded words (via SIMILARITY)               │
│  ├── resolve_entities() → Code entities (via REFERS_TO)             │
│  ├── rank_and_explain() → Scored results with reasons               │
│  └── assess_confidence() → Overall confidence level                 │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                         BRIDGE LAYER                                 │
│                                                                      │
│  CodeEntityBridge                    TextToAtomsBridge              │
│  ├── ASTIndex parsing                ├── BPE tokenization           │
│  ├── FILE/CLASS/FUNCTION atoms       ├── WORD atoms                 │
│  ├── DEFINES/CONTAINS links          ├── SIMILARITY links           │
│  └── REFERS_TO links                 └── FOLLOWS links (new)        │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                         GRAPH LAYER                                  │
│                                                                      │
│  CognitiveGraph                                                     │
│  ├── Unified storage for all atoms                                  │
│  ├── Truth values (strength, confidence)                            │
│  ├── Attention (STI/LTI)                                            │
│  └── Persistence (sharded storage)                                  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                         REASONING LAYER                              │
│                                                                      │
│  CognitiveAgent                      AuditReasoner                  │
│  ├── get_associations()              ├── PLN inference              │
│  ├── predict_next() (new)            ├── Risk assessment            │
│  ├── Goal tracking                   ├── Rule aggregation           │
│  └── Attention focus                 └── Explainability             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query: "where is authentication handled?"
                    │
                    ▼
┌─────────────────────────────────────────┐
│         1. PARSE INTENT                  │
│                                          │
│  parse_intent_query()                    │
│  → intent: LOCATION                      │
│  → subject: "authentication"             │
│  → action: "handled"                     │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         2. EXPAND TERMS                  │
│                                          │
│  get_associations("authentication")      │
│  → ["auth", "login", "session", "jwt"]   │
│                                          │
│  get_associations("handled")             │
│  → ["handler", "process", "manage"]      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         3. RESOLVE CODE ENTITIES         │
│                                          │
│  For each expanded term:                 │
│    Follow REFERS_TO links → CODE atoms   │
│                                          │
│  "auth" REFERS_TO auth/handler.py (0.9) │
│  "session" REFERS_TO session.py (0.7)   │
│  "handler" REFERS_TO AuthHandler (0.8)  │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         4. RANK AND EXPLAIN              │
│                                          │
│  Aggregate scores per file               │
│  Generate explanations                   │
│                                          │
│  auth/handler.py: 0.9                    │
│    "defines AuthHandler, matches 'auth'" │
│                                          │
│  middleware/session.py: 0.7              │
│    "imports auth, handles sessions"      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         5. ASSESS CONFIDENCE             │
│                                          │
│  Top result >> others? → HIGH            │
│  Multiple similar? → MEDIUM + uncertain  │
│  No results? → NONE + unknown terms      │
└─────────────────────────────────────────┘
                    │
                    ▼
QueryResult {
  understood_terms: ["authentication", "auth", "login", ...],
  intent: LOCATION,
  entities: [(auth/handler.py, 0.9, "defines AuthHandler"), ...],
  overall_confidence: HIGH,
  uncertain_about: [],
  suggestions: []
}
```

---

## Data Model

### Atom Types

```python
class AtomType(Enum):
    # === EXISTING NODE TYPES ===
    CONCEPT = auto()      # General concept
    WORD = auto()         # Lexical item (from text training)

    # === NEW CODE ENTITY TYPES ===
    FILE = auto()         # Source file: "auth/handler.py"
    CLASS = auto()        # Class definition: "AuthHandler"
    FUNCTION = auto()     # Function/method: "authenticate"
    MODULE = auto()       # Package/module: "cortical.cognitive"

    # === EXISTING LINK TYPES ===
    SIMILARITY = auto()   # Bidirectional semantic similarity
    INHERITANCE = auto()  # IS-A relationship

    # === NEW LINK TYPES ===
    FOLLOWS = auto()      # Directional: B follows A (for prediction)
    REFERS_TO = auto()    # WORD refers to CODE_ENTITY (weighted)
    DEFINES = auto()      # FILE defines CLASS/FUNCTION
    CONTAINS = auto()     # CLASS contains METHOD
    IMPORTS = auto()      # FILE imports MODULE
    CALLS = auto()        # FUNCTION calls FUNCTION
```

### Link Weights

#### REFERS_TO Weight Hierarchy

| Source | Weight | Rationale |
|--------|--------|-----------|
| Filename match | 1.0 | "auth" in "auth.py" → very strong |
| Class name match | 0.9 | "Auth" in "AuthHandler" → strong |
| Function name match | 0.8 | "authenticate" in function name |
| Docstring match | 0.5 | Mentioned in documentation |
| Code body match | 0.3 | Variable/identifier in code |
| Comment match | 0.2 | Mentioned in comments |

#### FOLLOWS Transition Storage

```python
# FOLLOWS links store:
# - source atom (word A)
# - target atom (word B that follows A)
# - count (how many times B followed A)
# - Probability computed as: count(A→B) / sum(count(A→*))
```

### Response Model

```python
@dataclass
class QueryResult:
    # What we understood
    understood_terms: List[str]
    intent: QueryIntent  # LOCATION, IMPLEMENTATION, DEFINITION, etc.

    # What we found
    entities: List[Tuple[CodeEntity, float, str]]  # (entity, confidence, reason)

    # Our confidence
    overall_confidence: ConfidenceLevel  # HIGH, MEDIUM, LOW, NONE

    # What we're uncertain about
    uncertain_about: List[str]      # Ambiguous terms
    suggestions: List[str]          # Clarifying questions
    unknown_terms: List[str]        # Words not in vocabulary
    no_code_mapping: List[str]      # Words without REFERS_TO links
```

---

## Query Flow

### Intent Types and Handling

| Intent | Question Words | Response Focus |
|--------|---------------|----------------|
| LOCATION | "where", "find" | File paths, line numbers |
| IMPLEMENTATION | "how" | Code details, logic flow |
| DEFINITION | "what" | Class/function definitions, docstrings |
| RELATIONSHIP | "what uses", "what calls" | Dependency relationships |
| RATIONALE | "why" | Comments, documentation, git history |

### Expansion Strategy

1. **Direct match**: "authentication" → atom if exists
2. **SIMILARITY links**: "authentication" ~ "auth" ~ "login"
3. **Stemming fallback**: "authenticating" → "authentic" → "auth"
4. **No expansion if unknown**: Preserve original for uncertainty reporting

### Confidence Calculation

```python
def calculate_confidence(candidates: List[Tuple[Entity, float]]) -> ConfidenceLevel:
    if not candidates:
        return ConfidenceLevel.NONE

    top_score = candidates[0][1]

    if len(candidates) == 1:
        if top_score > 0.7:
            return ConfidenceLevel.HIGH
        elif top_score > 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    # Multiple candidates - check gap
    second_score = candidates[1][1]
    gap = top_score - second_score

    if gap > 0.3 and top_score > 0.7:
        return ConfidenceLevel.HIGH
    elif gap > 0.1 or top_score > 0.5:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW
```

---

## Uncertainty Handling

### Types of Uncertainty

| Type | Detection | Response |
|------|-----------|----------|
| Unknown word | Not in vocabulary | "I don't recognize '{term}'" |
| No code mapping | Word exists, no REFERS_TO | "I know '{term}' but not where it's implemented" |
| Ambiguous match | Multiple similar scores | "'{term}' could refer to: [options]" |
| Low confidence | Top score < 0.4 | "I'm not certain, but possibly: [options]" |

### Response Strategies

```python
# Unknown word
if term in result.unknown_terms:
    return f"I don't recognize '{term}'. Could you rephrase or provide more context?"

# No code mapping
if term in result.no_code_mapping:
    return f"I understand '{term}' conceptually but haven't mapped it to code yet."

# Ambiguous
if result.overall_confidence == ConfidenceLevel.LOW:
    options = [e[0].name for e in result.entities[:3]]
    return f"'{query}' could refer to several things: {options}. Which did you mean?"

# Confident
if result.overall_confidence == ConfidenceLevel.HIGH:
    top = result.entities[0]
    return f"{top[0].file_path}:{top[0].line_number} - {top[2]}"
```

---

## Implementation Roadmap

### Phase 1: Foundation (FOLLOWS Links)

**Goal:** Enable directional prediction from the graph

1. Add `AtomType.FOLLOWS` to graph.py
2. Modify `TextToAtomsBridge.feed_text()` to create FOLLOWS links for adjacent words
3. Add `CognitiveAgent.predict_next()` method
4. Implement `Prediction` dataclass with confidence and boundary detection

**Behavioral spec:** `test_directional_prediction_spec.py` ✅

### Phase 2: Code Entity Bridge

**Goal:** Parse code structure into the graph

1. Add `AtomType.FILE`, `CLASS`, `FUNCTION`, `MODULE`
2. Create `CodeEntityBridge` class
3. Integrate with existing `ASTIndex` from Spark
4. Create `DEFINES`, `CONTAINS`, `IMPORTS`, `CALLS` links

**Files to create:**
- `cortical/cognitive/code_bridge.py`

### Phase 3: REFERS_TO Links

**Goal:** Connect semantic layer to structural layer

1. Add `AtomType.REFERS_TO`
2. Implement weight calculation based on match location
3. Create links during code indexing
4. Index identifiers, docstrings, comments

### Phase 4: Unified Query Handler

**Goal:** End-to-end natural language queries

1. Create `UnifiedQueryHandler` class
2. Integrate `parse_intent_query()` from query module
3. Implement term expansion via SIMILARITY
4. Implement entity resolution via REFERS_TO
5. Implement confidence assessment
6. Create `QueryResult` response structure

**Behavioral spec:** `test_unified_knowledge_query_spec.py` ✅

**Files to create:**
- `cortical/cognitive/query_handler.py`

### Phase 5: Audit Integration (Optional)

**Goal:** Add risk reasoning to unified queries

1. Bridge audit findings to CODE atoms
2. Allow queries like "which files are risky?"
3. Integrate PLN reasoning for explanations

---

## Integration Points

### With Existing Systems

| System | Integration Point | Benefit |
|--------|------------------|---------|
| `ASTIndex` | CodeEntityBridge uses it for parsing | Reuse proven AST parsing |
| `parse_intent_query()` | QueryHandler uses it for NLU | Reuse intent detection |
| `expand_query()` | Alternative to SIMILARITY traversal | Comparison/fallback |
| `AuditReasoner` | Optional risk layer | Answer "is X risky?" |
| `WovenMindDiscovery` | Share learning patterns | Unified pattern storage |

### CLI Integration

```bash
# Potential new command
python -m cortical.cognitive query "where is authentication handled?"

# Or integrate with existing
python -m cortical.got query "where is authentication handled?"
```

### API Integration

```python
from cortical.cognitive import CognitiveAgent, UnifiedQueryHandler

agent = CognitiveAgent()
handler = UnifiedQueryHandler(agent)

result = handler.query("where is authentication handled?")
print(result.top_entity())  # auth/handler.py:15
print(result.overall_confidence)  # HIGH
```

---

## Design Decisions

### Decision 1: Unified Graph vs Separate Stores

**Choice:** Unified graph with all atom types together

**Rationale:**
- Single source of truth
- Cross-layer queries possible (word → code → relationships)
- Shared attention mechanisms
- Simpler persistence

**Trade-off:** Larger graph, but sharding addresses this

### Decision 2: FOLLOWS Separate from SIMILARITY

**Choice:** Keep FOLLOWS (directional) separate from SIMILARITY (bidirectional)

**Rationale:**
- Different purposes: prediction vs association
- Different semantics: "A then B" vs "A related to B"
- Different storage: FOLLOWS is sparser (adjacent only)

### Decision 3: Confidence as First-Class Concept

**Choice:** Return confidence level with every response

**Rationale:**
- Enables honest uncertainty
- Consumer decides threshold
- Distinguishes "I don't know" from "I found nothing"
- Builds trust through calibration

### Decision 4: REFERS_TO Weight Hierarchy

**Choice:** Weight by match location (filename > classname > docstring > code)

**Rationale:**
- Filename matches are strongest signals
- Class/function names indicate purpose
- Docstrings are explicit documentation
- Code body matches are noisy but provide coverage

### Decision 5: Small Corpus First

**Choice:** Design for sparse data, graceful degradation

**Rationale:**
- Claude Code Web has limited context
- Sparse data → honest uncertainty
- Works with minimal training
- Scales up naturally

---

## Appendix: File Locations

### Existing Files (to modify)

- `cortical/cognitive/graph.py` - Add new AtomTypes
- `cortical/cognitive/text_bridge.py` - Add FOLLOWS link creation
- `cortical/cognitive/training.py` - Integrate code indexing

### New Files (to create)

- `cortical/cognitive/code_bridge.py` - CODE entity creation
- `cortical/cognitive/query_handler.py` - Unified query handling
- `cortical/cognitive/prediction.py` - Prediction dataclass and logic

### Test Specs (created)

- `tests/behavioral/test_directional_prediction_spec.py` ✅
- `tests/behavioral/test_unified_knowledge_query_spec.py` ✅

---

## Conclusion

This architecture enables the assistant to:

1. **Understand queries** through intent parsing and semantic expansion
2. **Find code** through REFERS_TO links bridging words to entities
3. **Predict text** through FOLLOWS links for directional transitions
4. **Express uncertainty** honestly at every step
5. **Learn incrementally** from each coding session

The graph becomes an extension of working memory—persisting knowledge across sessions and enabling intelligent retrieval instead of repeated search.

The key innovation is the **REFERS_TO bridge** connecting the semantic layer (WORD atoms from training) to the structural layer (CODE atoms from AST parsing), enabling natural language queries to resolve to concrete code locations with explainable confidence.
