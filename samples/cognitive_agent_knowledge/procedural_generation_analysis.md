# Procedural Generation Analysis for Cognitive Agent Training

*Deep analysis of sample generation strategies to overcome code abundance*

---

## Executive Summary

The cognitive agent's training corpus consists of 683 files (68,790 lines) with a **97.7% prose to 2.3% code** ratio. However, when training on the actual codebase (`cortical/`), this ratio inverts dramatically - the model ingests thousands of lines of Python code as linear text. This creates vocabulary pollution, semantic structure loss, and IDF dilution that degrades query quality.

This analysis explores procedural generation as a strategy to rebalance the corpus while preserving semantic relationships.

---

## Part 1: Current State Analysis

### 1.1 Corpus Composition

| Category | Files | Lines | % of Total |
|----------|-------|-------|------------|
| Plain text (.txt) | 531 | 35,412 | 51.4% |
| Markdown (.md) | 147 | 31,824 | 46.2% |
| Python code (.py) | 5 | 1,554 | 2.3% |
| **Total** | **683** | **68,790** | **100%** |

### 1.2 Training Architecture

The cognitive agent uses a three-stage pipeline:

```
                    ┌─────────────────────────────────────────────┐
                    │           TRAINING PIPELINE                  │
                    ├─────────────────────────────────────────────┤
                    │                                             │
     Raw Text ─────►│  BPETokenizer                               │
                    │  └─► Word-level tokenization                │
                    │  └─► Regex: \b[a-z][a-z]+\b                 │
                    │  └─► IDF tracking per document              │
                    │                                             │
                    ├─────────────────────────────────────────────┤
                    │                                             │
    Vocabulary ────►│  TextToAtomsBridge                          │
                    │  └─► Create WORD atoms                      │
                    │  └─► Create SIMILARITY links (co-occurrence)│
                    │  └─► Create FOLLOWS links (transitions)     │
                    │  └─► Apply IDF weighting                    │
                    │                                             │
                    ├─────────────────────────────────────────────┤
                    │                                             │
   Atom Graph ─────►│  CognitiveGraph Storage                     │
                    │  └─► Sharded JSON persistence               │
                    │  └─► ~23,000 atoms, ~150MB bridge           │
                    │                                             │
                    └─────────────────────────────────────────────┘
```

### 1.3 Code Processing Path (Separate from Text)

```python
# CodeBridge: AST-based indexing (not text-based)
# Creates structural atoms: FILE, CLASS, FUNCTION
# Links: DEFINES, CONTAINS, CALLS, INHERITANCE
# REFERS_TO bridges WORD atoms to CODE atoms
```

**Critical insight**: CodeBridge operates via AST parsing, NOT tokenization. It captures structure but doesn't contribute to word associations.

---

## Part 2: The Code Abundance Problem

### 2.1 When Training on cortical/

If we train on the Python codebase (`cortical/` with ~200 files, ~50,000 lines):

| Token Type | Frequency | IDF Impact |
|------------|-----------|------------|
| `self` | ~5,000+ | Very low (appears everywhere) |
| `def` | ~1,500+ | Very low |
| `return` | ~1,200+ | Very low |
| `import` | ~800+ | Very low |
| `class` | ~500+ | Low |
| Domain terms | ~50-200 | High (good) |

**Problem**: Code syntax tokens dominate vocabulary, drowning semantic terms.

### 2.2 Semantic Structure Loss

```python
# What the code looks like:
def compute_pagerank(graph, damping=0.85, iterations=100):
    """Compute PageRank scores for graph nodes."""
    scores = {node: 1.0 / len(graph) for node in graph}
    ...
```

**What the tokenizer sees**:
```
['compute', 'pagerank', 'graph', 'damping', 'iterations',
 'compute', 'pagerank', 'scores', 'graph', 'node', 'len',
 'graph', 'node', 'graph']
```

**What is lost**:
- Function signature structure
- Parameter relationships
- Type hints
- Call hierarchy
- Control flow

### 2.3 IDF Dilution Effect

When `def` appears in 80% of Python files:
```
IDF(def) = log((N+1)/(0.8*N+1)) ≈ 0.22  (very low weight)
```

When `pagerank` appears in 2 files:
```
IDF(pagerank) = log((N+1)/(2+1)) ≈ 4.5  (high weight)
```

**Result**: Links involving syntax tokens get weak weights, but they still consume vocabulary space and co-occurrence slots.

### 2.4 Context Mixing Problem

The model conflates:
- `graph` (data structure) vs `graph` (chart/plot)
- `node` (graph vertex) vs `node` (DOM element) vs `node` (Node.js)
- `class` (Python keyword) vs `class` (category/type)

Single vocabulary space lacks disambiguation.

---

## Part 3: Procedural Generation Strategies

### Strategy 1: Code-to-Prose Translation

**Concept**: Transform code into natural language descriptions that preserve semantics.

```python
# Input: Python function
def compute_pagerank(graph, damping=0.85, iterations=100):
    """Compute PageRank scores for graph nodes."""
    ...

# Generated prose:
"""
The compute_pagerank function calculates PageRank scores for all nodes
in a graph. It accepts a graph parameter (required), an optional damping
factor defaulting to 0.85, and an optional iteration count defaulting to 100.

PageRank is an algorithm that assigns importance scores to nodes based on
the structure of incoming links. Higher damping values preserve more link
structure. More iterations increase accuracy but require more computation.

This function is defined in cortical/analysis/pagerank.py at line 45.
It is called by compute_all() in the processor module.
"""
```

**Implementation approach**:
```python
def generate_function_description(func_info: FunctionInfo) -> str:
    """Generate natural language description from AST info."""
    parts = []

    # Function identity
    parts.append(f"The {func_info.name} function")

    # Purpose from docstring
    if func_info.docstring:
        parts.append(func_info.docstring)

    # Parameters
    if func_info.args:
        param_desc = describe_parameters(func_info.args)
        parts.append(param_desc)

    # Location
    parts.append(f"Located in {func_info.file_path}:{func_info.lineno}")

    # Relationships
    if func_info.calls:
        parts.append(f"Calls: {', '.join(func_info.calls)}")

    return " ".join(parts)
```

**Ramifications**:
- (+) Preserves semantics in trainable form
- (+) Vocabulary stays in natural language space
- (+) Co-occurrence reflects actual relationships
- (-) Requires robust AST parsing
- (-) Generated text may be repetitive/formulaic
- (-) Loses precise code details

---

### Strategy 2: Pattern-Based Synthetic Generation

**Concept**: Generate synthetic documents describing code patterns, not specific code.

```
# Template:
The {pattern_name} pattern is used when {use_case}.
It involves {components} working together to {purpose}.
In this codebase, examples include {examples}.
Common variations: {variations}.
Related concepts: {related}.

# Generated instance:
The observer pattern is used when objects need to be notified of
state changes in another object. It involves a subject and multiple
observers working together to decouple event producers from consumers.
In this codebase, examples include ProgressReporter and event callbacks.
Common variations: event emitters, pub/sub, reactive streams.
Related concepts: callback, event, subscription, notification, broadcast.
```

**Implementation**:
```python
PATTERN_TEMPLATES = {
    "dependency_injection": {
        "description": "...",
        "components": ["container", "factory", "interface"],
        "examples_query": "class.*Container|register|resolve",
        "related": ["inversion of control", "factory", "service locator"],
    },
    # ... more patterns
}

def generate_pattern_documents(codebase_patterns: Dict) -> List[str]:
    """Generate documents describing patterns found in codebase."""
    ...
```

**Ramifications**:
- (+) Creates rich semantic vocabulary
- (+) Bridges code structure to concepts
- (+) Can generate many variations
- (-) Requires pattern identification first
- (-) May not cover all code accurately
- (-) Generic descriptions may not help with specific queries

---

### Strategy 3: Docstring/Comment Extraction & Amplification

**Concept**: Extract prose already in code, amplify with generated context.

```python
# From code:
class CognitiveGraph:
    """
    Hypergraph storage for atoms with weighted links.

    Atoms can be nodes (WORD, CLASS, FUNCTION) or links (SIMILARITY, FOLLOWS).
    Links connect multiple atoms with truth values.
    """
```

**Generated amplification**:
```
## CognitiveGraph

CognitiveGraph is a hypergraph storage system for atoms with weighted links.

### Node Types
- WORD: Vocabulary terms from text
- CLASS: Python class definitions
- FUNCTION: Python function definitions

### Link Types
- SIMILARITY: Bidirectional co-occurrence relationships
- FOLLOWS: Directional word transitions

### Key Concepts
A hypergraph differs from a regular graph in that edges (links) can
connect more than two nodes. This enables representation of complex
relationships like "function A calls functions B, C, and D".

Truth values on links represent confidence: strength (0-1) indicates
how strongly we believe the relationship exists, confidence (0-1)
indicates how much evidence supports that belief.

### Related Components
- BPETokenizer: Builds vocabulary for WORD atoms
- TextToAtomsBridge: Creates atoms from text
- CodeBridge: Creates atoms from Python AST
```

**Ramifications**:
- (+) Preserves author intent (from docstrings)
- (+) Natural vocabulary (already prose)
- (+) Can enrich with cross-references
- (-) Depends on docstring quality/coverage
- (-) Amplification may introduce errors
- (-) Coverage limited to documented code

---

### Strategy 4: Multi-Modal Training with Separate Vocabularies

**Concept**: Maintain separate vocabulary spaces for code and prose, with bridging links.

```
┌────────────────────────────────────────────────────────────────────┐
│                    DUAL-VOCABULARY ARCHITECTURE                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  PROSE VOCABULARY              BRIDGE              CODE VOCABULARY │
│  ────────────────              ──────              ─────────────── │
│                                                                    │
│  "pagerank"      ◄────── REFERS_TO ──────►    compute_pagerank()  │
│  "algorithm"     ◄────── REFERS_TO ──────►    Algorithm (class)   │
│  "graph"         ◄────── REFERS_TO ──────►    CognitiveGraph      │
│  "importance"    ◄────── DESCRIBES ─────►     PageRank concept    │
│                                                                    │
│  ┌─────────────────┐    ┌─────────────┐    ┌──────────────────┐   │
│  │ Text Training   │    │   Bridge    │    │  Code Training   │   │
│  │ - samples/*.txt │    │   Links     │    │  - AST indexing  │   │
│  │ - docs/*.md     │◄───┤   PRISM     ├───►│  - Call graphs   │   │
│  │ - memories/     │    │   routing   │    │  - Inheritance   │   │
│  └─────────────────┘    └─────────────┘    └──────────────────┘   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**Implementation changes**:
```python
class DualBridge:
    prose_tokenizer: BPETokenizer  # For natural language
    code_tokenizer: CodeTokenizer   # Preserves structure
    bridge_links: List[Atom]        # REFERS_TO connections

    def feed_text(self, text: str) -> List[Atom]:
        # Use prose_tokenizer
        ...

    def feed_code(self, code: str, path: str) -> List[Atom]:
        # Use code_tokenizer + AST parsing
        ...

    def create_bridges(self) -> None:
        # Connect prose terms to code entities
        ...
```

**Ramifications**:
- (+) Clean separation prevents vocabulary pollution
- (+) Each domain uses appropriate tokenization
- (+) Bridge links preserve relationships
- (-) Significant architecture change
- (-) Query routing becomes complex
- (-) May fragment related concepts

---

### Strategy 5: Template-Based Variation Generation

**Concept**: Generate many variations of the same concept to reinforce patterns.

```python
CONCEPT_TEMPLATES = {
    "function_purpose": [
        "The {func} function {action}.",
        "{func} is responsible for {action}.",
        "To {action}, use the {func} function.",
        "When you need to {action}, call {func}.",
    ],
    "class_relationship": [
        "{child} inherits from {parent}.",
        "{child} extends {parent}.",
        "{child} is a specialized {parent}.",
        "The {child} class derives from {parent}.",
    ],
}

def generate_variations(entity: str, template_type: str, **kwargs) -> List[str]:
    """Generate multiple phrasings of the same relationship."""
    templates = CONCEPT_TEMPLATES[template_type]
    return [t.format(func=entity, **kwargs) for t in templates]
```

**Generated output**:
```
The compute_pagerank function calculates node importance scores.
compute_pagerank is responsible for calculating node importance scores.
To calculate node importance scores, use the compute_pagerank function.
When you need to calculate node importance scores, call compute_pagerank.
```

**Ramifications**:
- (+) Reinforces key relationships through repetition
- (+) Improves query recall (multiple phrasings match)
- (+) Simple to implement
- (-) Increases corpus size significantly
- (-) May create artificial co-occurrence patterns
- (-) Risk of overfitting to templates

---

## Part 4: Recommended Hybrid Approach

### 4.1 Three-Tier Generation Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HYBRID GENERATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TIER 1: Extraction (High Quality, Low Volume)                         │
│  ─────────────────────────────────────────────                         │
│  • Extract existing docstrings and comments                            │
│  • Parse README and documentation                                       │
│  • Preserve session memories and KT documents                          │
│  • ~20% of generated content                                           │
│                                                                         │
│  TIER 2: Translation (Medium Quality, Medium Volume)                   │
│  ────────────────────────────────────────────────────                  │
│  • Convert AST to prose descriptions                                   │
│  • Generate function/class summaries                                   │
│  • Create relationship narratives                                       │
│  • ~50% of generated content                                           │
│                                                                         │
│  TIER 3: Synthesis (Variable Quality, High Volume)                     │
│  ───────────────────────────────────────────────                       │
│  • Template-based variations                                           │
│  • Pattern-based concept documents                                     │
│  • Cross-reference expansions                                          │
│  • ~30% of generated content                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Implementation Phases

**Phase 1: Enhanced Extraction** (Low risk, immediate benefit)
```python
class EnhancedExtractor:
    def extract_docstrings(self, path: Path) -> List[Document]:
        """Extract all docstrings with context."""

    def extract_comments(self, path: Path) -> List[Document]:
        """Extract meaningful comments (ignore noise)."""

    def amplify_with_context(self, doc: Document) -> Document:
        """Add file path, related functions, etc."""
```

**Phase 2: AST-to-Prose Translation** (Medium risk, high benefit)
```python
class CodeTranslator:
    def translate_function(self, func: FunctionInfo) -> str:
        """Convert function AST to prose description."""

    def translate_class(self, cls: ClassInfo) -> str:
        """Convert class AST to prose description."""

    def translate_relationships(self, call_graph: Dict) -> List[str]:
        """Convert call relationships to narratives."""
```

**Phase 3: Synthetic Generation** (Higher risk, scalable)
```python
class SyntheticGenerator:
    def generate_pattern_docs(self) -> List[str]:
        """Create pattern description documents."""

    def generate_qa_pairs(self, entities: List[Entity]) -> List[str]:
        """Create Q&A training pairs."""

    def generate_variations(self, facts: List[str]) -> List[str]:
        """Create multiple phrasings of facts."""
```

### 4.3 Quality Control

```python
class GenerationValidator:
    def validate_accuracy(self, generated: str, source: Entity) -> float:
        """Check generated content matches source."""

    def validate_coherence(self, generated: str) -> float:
        """Check grammatical and logical coherence."""

    def validate_novelty(self, generated: str, corpus: List[str]) -> float:
        """Ensure not too similar to existing content."""

    def filter_low_quality(self, docs: List[str]) -> List[str]:
        """Remove documents below quality threshold."""
```

---

## Part 5: Ramifications Analysis

### 5.1 Positive Outcomes

| Outcome | Impact | Confidence |
|---------|--------|------------|
| Improved query precision | High | High |
| Better concept coverage | High | Medium |
| Reduced vocabulary noise | Medium | High |
| Scalable corpus growth | High | Medium |
| Preserved code semantics | High | Medium |

### 5.2 Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Generated content inaccuracy | High | Validation pipeline, human review |
| Template overfitting | Medium | Diverse templates, randomization |
| Corpus bloat | Low | Quality filtering, deduplication |
| Loss of code detail | Medium | Preserve raw code links |
| Increased complexity | Medium | Phased rollout, testing |

### 5.3 Trade-Off Matrix

```
                      HIGH BENEFIT
                           │
     Template Variations ──┼── AST Translation
              │            │           │
              │            │           │
  LOW RISK ───┼────────────┼───────────┼─── HIGH RISK
              │            │           │
              │            │           │
     Docstring Extraction ─┼── Multi-Modal Vocab
                           │
                      LOW BENEFIT
```

**Recommended priority**:
1. Docstring Extraction (low risk, medium benefit, immediate)
2. AST Translation (medium risk, high benefit, core)
3. Template Variations (low risk, medium benefit, enhancement)
4. Multi-Modal Vocab (high risk, high benefit, future)

---

## Part 6: Implementation Recommendations

### 6.1 Immediate Actions

1. **Create `cortical/cognitive/generators/` module**
   - `extractors.py` - Docstring/comment extraction
   - `translators.py` - AST to prose conversion
   - `synthesizers.py` - Template-based generation
   - `validators.py` - Quality control

2. **Add generation CLI commands**
   ```bash
   python -m cortical.cognitive generate-prose cortical/ --output samples/generated/
   python -m cortical.cognitive validate-corpus samples/
   ```

3. **Establish quality metrics**
   - Track vocabulary overlap between generated and existing
   - Monitor query accuracy on test questions
   - Measure co-occurrence pattern quality

### 6.2 Corpus Management

```python
# samples/ structure with generated content
samples/
├── curated/              # Human-written, high quality
│   ├── knowledge-base/
│   └── cognitive_agent_knowledge/
├── extracted/            # Docstrings, comments (auto-extracted)
│   ├── cortical_docstrings/
│   └── cortical_comments/
├── translated/           # AST-to-prose (auto-generated)
│   ├── function_descriptions/
│   ├── class_descriptions/
│   └── relationship_narratives/
├── synthesized/          # Template-based (auto-generated)
│   ├── pattern_documents/
│   ├── qa_variations/
│   └── concept_expansions/
└── domain/               # Domain knowledge (existing)
    ├── cognitive_science/
    ├── philosophy/
    └── ...
```

### 6.3 Training Configuration

```python
CORPUS_WEIGHTS = {
    "curated": 1.5,        # Boost human-written
    "extracted": 1.2,      # Slight boost for docstrings
    "translated": 1.0,     # Standard weight
    "synthesized": 0.8,    # Lower weight to prevent overfitting
    "domain": 1.0,         # Standard weight
}
```

---

## Conclusion

Procedural generation offers a viable path to overcome the code abundance problem while preserving semantic relationships. The recommended hybrid approach:

1. **Extracts** high-quality prose from existing code
2. **Translates** code structure into trainable natural language
3. **Synthesizes** variations to reinforce key concepts

This strategy shifts the corpus from raw code (which loses structure) to semantic descriptions (which preserve meaning). The result is a cognitive agent that understands both code and concepts, bridging the gap between what code does and what queries ask.

**Key insight**: The goal isn't to eliminate code from training, but to represent code in a form that the word-level semantic model can learn from effectively.

---

*Generated by Claude Opus 4.5 for the Cortical Text Processor project*
*Analysis date: 2026-01-12*
