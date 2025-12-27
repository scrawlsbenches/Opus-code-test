# Handoff: Cognitive NLU/NLG Exploration

**Date:** 2025-12-27
**From Branch:** `claude/run-dashboard-stats-benchmarks-3oyzp`
**Status:** Ready for continuation
**Focus:** Natural Language Understanding, Generation, and Meta-Learning

---

## Context: What Was Built

This session created a unified cognitive integration demo (`scripts/cognitive_integration_demo.py`) that combines:

| System | Role | Current State |
|--------|------|---------------|
| **WovenMind** | Dual-process cognition (System 1/2) | ✅ Working with persistence |
| **SparkSLM** | N-gram statistical predictions | ✅ Fixed, needs codebase audit |
| **PRISM-SLM** | Hebbian synaptic learning | ✅ Working with persistence |
| **PRISM-PLN** | Probabilistic logic networks | ✅ Working with persistence |
| **AnomalyDetector** | Input safety/injection detection | ✅ Calibrated |

**Run the demo:**
```bash
python scripts/cognitive_integration_demo.py --save ./state --verbose
python scripts/cognitive_integration_demo.py --load ./state --query "your query"
```

---

## The Vision: Learning to Learn and Do

The goal is building systems that exhibit **meta-learning** - the ability to improve their own learning process through experience. This goes beyond pattern matching to genuine understanding and generation.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LEARNING TO LEARN AND DO                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  LEVEL 0: Pattern Matching (current)                                 │
│  └── Recognize patterns seen before                                  │
│                                                                       │
│  LEVEL 1: Pattern Generalization                                     │
│  └── Apply patterns to novel situations                              │
│                                                                       │
│  LEVEL 2: Strategy Learning                                          │
│  └── Learn WHEN to apply which patterns                              │
│                                                                       │
│  LEVEL 3: Meta-Learning                                              │
│  └── Improve the learning process itself                             │
│                                                                       │
│  LEVEL 4: Self-Directed Learning                                     │
│  └── Identify gaps, seek knowledge, verify understanding            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Ideas to Explore

### 1. Compositional Semantics Engine

**Goal:** Understand meaning through composition, not just pattern matching.

**Concept:** Build meanings from parts, like humans do.

```python
# Proposed: cortical/reasoning/compositional.py

class CompositionalSemantics:
    """
    Build complex meanings from atomic concepts.

    "The cat sat on the mat"
    → Entity(cat) + Action(sit) + Relation(on) + Entity(mat)
    → Proposition(agent=cat, action=sit, location=mat)
    """

    def parse_to_structure(self, text: str) -> SemanticStructure:
        """Parse text into compositional semantic structure."""
        pass

    def compose(self, *parts: SemanticPart) -> SemanticStructure:
        """Compose atomic meanings into complex structures."""
        pass

    def decompose(self, structure: SemanticStructure) -> List[SemanticPart]:
        """Break down complex meaning into atomic parts."""
        pass
```

**Why it matters:** Current systems learn correlations. Compositional semantics learns *structure* that generalizes infinitely.

---

### 2. Analogical Reasoning Engine

**Goal:** Transfer knowledge between domains through structural similarity.

**Concept:** "Learning is recognizing the same pattern in different clothes."

```python
# Proposed: cortical/reasoning/analogy.py

class AnalogicalReasoner:
    """
    Find and exploit structural similarities between domains.

    Domain A: "Water flows downhill"
    Domain B: "Electricity flows through path of least resistance"

    Mapping: water→electricity, downhill→least_resistance, flows→flows
    Transfer: Properties of water flow → predictions about electricity
    """

    def find_mapping(self, source: Domain, target: Domain) -> StructuralMapping:
        """Find structural correspondence between domains."""
        pass

    def transfer_inference(self,
                          source_fact: Fact,
                          mapping: StructuralMapping) -> Fact:
        """Transfer knowledge from source to target domain."""
        pass

    def evaluate_analogy_quality(self, mapping: StructuralMapping) -> float:
        """Score analogy by structural depth and systematicity."""
        pass
```

**Integration with existing code:**
- Use `cortical/query/analogy.py` as foundation (already has analogy completion)
- Extend with Structure Mapping Engine (SME) principles
- Connect to PLN for probabilistic analogy evaluation

---

### 3. Explanation-Based Learning

**Goal:** Learn generalizable rules from single examples by explaining *why* they work.

**Concept:** Don't just memorize; understand the causal structure.

```python
# Proposed: cortical/reasoning/explanation.py

class ExplanationLearner:
    """
    Learn from explanations, not just examples.

    Example: "This code failed because the list was empty"

    Explanation trace:
    1. Function assumes non-empty list
    2. Empty list passed
    3. Index error raised

    Generalized rule: "Check for empty collections before indexing"
    """

    def explain_outcome(self,
                       situation: Situation,
                       outcome: Outcome) -> ExplanationChain:
        """Generate causal explanation for outcome."""
        pass

    def generalize_explanation(self,
                              explanation: ExplanationChain) -> Rule:
        """Extract generalizable rule from specific explanation."""
        pass

    def apply_rule(self, rule: Rule, new_situation: Situation) -> Prediction:
        """Apply learned rule to predict outcomes."""
        pass
```

**Integration:**
- Use PLN's inference chains as explanation structures
- Store rules in GoT as learned knowledge
- Connect to WovenMind's System 2 for deliberate rule application

---

### 4. Self-Monitoring Metacognition

**Goal:** Know what you know and don't know.

**Concept:** Uncertainty awareness enables targeted learning.

```python
# Proposed: cortical/reasoning/metacognition.py

class MetacognitiveMonitor:
    """
    Track confidence, detect knowledge gaps, direct learning.

    "I know about Python syntax (high confidence)
     I'm uncertain about Rust lifetimes (low confidence)
     I should learn more about Rust before answering"
    """

    def assess_confidence(self, query: str) -> ConfidenceAssessment:
        """Estimate confidence in answering a query."""
        return ConfidenceAssessment(
            knowledge_coverage=0.8,  # How much relevant knowledge exists
            inference_reliability=0.6,  # How reliable are the inference paths
            uncertainty_sources=["limited examples", "conflicting patterns"]
        )

    def identify_knowledge_gaps(self, domain: str) -> List[KnowledgeGap]:
        """Find what's missing in a knowledge domain."""
        pass

    def suggest_learning_priorities(self) -> List[LearningGoal]:
        """Recommend what to learn next based on gaps and goals."""
        pass
```

**Integration:**
- Use WovenMind's surprise detection as uncertainty signal
- Use PRISM-SLM connection strengths as confidence proxy
- Store learning goals in GoT as meta-tasks

---

### 5. Generative Understanding Loop

**Goal:** Verify understanding by generating and checking.

**Concept:** "If you can't explain it simply, you don't understand it."

```python
# Proposed: cortical/reasoning/generative_loop.py

class GenerativeUnderstandingLoop:
    """
    Verify understanding through generation-verification cycles.

    1. Receive input: "Python uses indentation for blocks"
    2. Generate implication: "Mixing tabs and spaces causes errors"
    3. Verify: Check if generated implication is correct
    4. If wrong: Refine understanding, repeat
    5. If right: Strengthen understanding, store
    """

    def comprehend(self, input_text: str) -> Understanding:
        """Build initial understanding of input."""
        pass

    def generate_implications(self,
                             understanding: Understanding) -> List[Implication]:
        """Generate testable implications of understanding."""
        pass

    def verify_implication(self,
                          implication: Implication) -> VerificationResult:
        """Check if generated implication holds."""
        pass

    def refine_understanding(self,
                            understanding: Understanding,
                            feedback: VerificationResult) -> Understanding:
        """Update understanding based on verification feedback."""
        pass
```

**Integration:**
- Use PLN for generating logical implications
- Use SparkSLM for generating likely continuations
- Use existing search to verify against corpus

---

### 6. Curriculum Learning Controller

**Goal:** Learn in the right order - simple to complex.

**Concept:** Build foundations before tackling advanced topics.

```python
# Proposed: cortical/reasoning/curriculum.py

class CurriculumController:
    """
    Organize learning from simple to complex.

    Knowledge dependency graph:
    variables → expressions → statements → functions → classes → patterns

    Don't try to learn "design patterns" before "functions"
    """

    def build_dependency_graph(self, domain: str) -> KnowledgeGraph:
        """Map prerequisite relationships between concepts."""
        pass

    def assess_current_level(self) -> LearnerState:
        """Determine what's already learned."""
        pass

    def recommend_next_topic(self,
                            state: LearnerState,
                            goal: LearningGoal) -> Topic:
        """Suggest optimal next learning topic."""
        pass

    def generate_exercises(self, topic: Topic, difficulty: float) -> List[Exercise]:
        """Create practice problems at appropriate difficulty."""
        pass
```

**Integration:**
- Use GoT edges (DEPENDS_ON, PREREQUISITE) for dependency tracking
- Use PLN confidence scores to assess mastery
- Use WovenMind mode switches to detect struggle (SLOW = struggling)

---

### 7. Multi-Modal Concept Grounding

**Goal:** Ground abstract concepts in concrete examples.

**Concept:** Meaning emerges from connection to experience.

```python
# Proposed: cortical/reasoning/grounding.py

class ConceptGrounder:
    """
    Connect abstract concepts to concrete instances.

    Abstract: "recursion"
    Grounded examples:
    - Factorial function calling itself
    - Directory tree traversal
    - Fractal patterns
    - Russian nesting dolls

    Grounding enables:
    - Richer understanding
    - Analogy finding
    - Novel application
    """

    def ground_concept(self, concept: str) -> List[GroundedExample]:
        """Find concrete instances of abstract concept."""
        pass

    def abstract_from_examples(self,
                              examples: List[Example]) -> AbstractConcept:
        """Induce abstract concept from concrete examples."""
        pass

    def verify_grounding(self,
                        concept: AbstractConcept,
                        example: Example) -> bool:
        """Check if example is valid instance of concept."""
        pass
```

---

## Implementation Roadmap

### Phase 1: Foundation (Start Here)

1. **Extend PLN with explanation chains**
   - Add `explain()` method to PLNReasoner
   - Track inference provenance
   - Enable "why did you conclude X?" queries

2. **Add confidence tracking to WovenMind**
   - Track per-pattern confidence scores
   - Expose uncertainty in process() results
   - Enable "how sure are you?" queries

3. **Build analogical mapper**
   - Start with simple structural alignment
   - Use existing `cortical/query/analogy.py` as base
   - Add domain transfer capabilities

### Phase 2: Meta-Learning Loop

4. **Implement metacognitive monitor**
   - Aggregate confidence signals from all systems
   - Identify knowledge gaps automatically
   - Generate learning recommendations

5. **Build generative verification loop**
   - Generate implications from understanding
   - Verify against corpus
   - Refine on mismatch

### Phase 3: Self-Directed Learning

6. **Implement curriculum controller**
   - Build knowledge dependency graphs
   - Track mastery levels
   - Generate appropriate challenges

7. **Add compositional semantics**
   - Parse to semantic structures
   - Enable compositional generation
   - Support infinite generalization

---

## Code Starting Points

### Existing Modules to Extend

| Module | Extension Ideas |
|--------|-----------------|
| `cortical/reasoning/prism_pln.py` | Add explanation chains, uncertainty propagation |
| `cortical/reasoning/woven_mind.py` | Add confidence tracking, learning rate adaptation |
| `cortical/reasoning/loom.py` | Expose surprise as uncertainty signal |
| `cortical/query/analogy.py` | Full structural mapping, domain transfer |
| `cortical/got/api.py` | Store learned rules, knowledge gaps |

### New Modules to Create

```
cortical/reasoning/
├── metacognition.py      # Self-monitoring, confidence tracking
├── explanation.py        # Explanation-based learning
├── curriculum.py         # Learning order optimization
├── compositional.py      # Compositional semantics
├── grounding.py          # Concept grounding
└── generative_loop.py    # Generation-verification cycles
```

---

## Key Questions to Explore

1. **How does understanding differ from pattern matching?**
   - Can we define "understanding" operationally?
   - What tests distinguish genuine understanding from mimicry?

2. **How do we represent "why" knowledge?**
   - Causal models? Explanation structures? Inference chains?
   - How to make "why" queryable?

3. **How does learning improve learning?**
   - What meta-parameters should adapt?
   - How to detect that current learning strategy isn't working?

4. **How to balance exploration vs exploitation in learning?**
   - When to consolidate existing knowledge?
   - When to seek new domains?

5. **How to ground symbols in experience?**
   - What counts as "grounding" in a text-only system?
   - Can we use code execution as grounding?

---

## Pending Task

**T-20251227-152115-79b83232:** Investigate SparkSLM train() API pattern bug
- Search codebase for similar issues
- Add warning for single-string input
- Update documentation

---

## Quick Resume Commands

```bash
# Load session state
python scripts/cognitive_integration_demo.py --load ./cognitive_state --verbose

# Check current GoT status
python scripts/got_utils.py dashboard

# View this handoff
cat samples/memories/2025-12-27-handoff-cognitive-nlu-nlg-exploration.md

# Start exploring PLN extensions
cat cortical/reasoning/prism_pln.py | head -100
```

---

## Closing Thought

> "The key to artificial general intelligence isn't more data or bigger models—it's the architecture of learning itself. A system that learns how to learn can bootstrap its way to any capability. Our cognitive integration work is a step toward that architecture."

Ready for the next agent to continue the journey.

---

*Tags: `meta-learning`, `NLU`, `NLG`, `cognitive-architecture`, `PLN`, `WovenMind`, `handoff`*
