# Through the Looking Glass: A Comparative Study of PRISM and Woven Mind

## Cognitive Architectures for Knowledge Graph Learning

---

> *"The only way to do great work is to love what you do. If you haven't found it yet, keep looking. Don't settle."*
> — Steve Jobs

> *"We shall not cease from exploration, and the end of all our exploring will be to arrive where we started and know the place for the first time."*
> — T.S. Eliot

---

## Prelude: The Journey That Brought Us Here

Dear reader, you hold in your hands the fruits of a peculiar journey—one that began with a simple question and wandered through gardens of synapses, forests of dreams, and oceans of inference. Along the way, two distinct yet harmonious visions emerged for how machines might learn to think.

The first, **PRISM** (Predictive Reasoning through Incremental Synaptic Memory), was born from the practical need to build systems that remember, predict, and reason. It lives in code—3,000 lines of working Python that you can run today.

The second, **Woven Mind**, emerged from imagination—a thought experiment asking: "What if we designed a brain from first principles?" It exists as architecture, as possibility, as a map to territories not yet explored.

This paper compares them. Not to declare a winner, but to understand what each reveals about the nature of machine thought. For in their similarities we find convergent wisdom, and in their differences we find complementary gifts.

Whether you are a seasoned researcher or a curious newcomer, welcome. We will walk slowly, explain generously, and pause to wonder at the view.

---

## Chapter 1: The Foundations (For Those New to These Ideas)

### 1.1 What Is a Knowledge Graph?

Before we compare architectures, let us establish common ground.

Imagine a web of ideas. Each idea is a **node**—a point in space holding some piece of knowledge. Between nodes run **edges**—connections indicating that two ideas relate somehow.

```
       [coffee] ──wakes up→ [programmer]
           │                     │
        contains             creates
           ▼                     ▼
       [caffeine]              [code]
```

This is a knowledge graph. Simple in concept, profound in application. Every thought you have, every word you speak, exists within such a web of connections inside your mind.

**Key insight**: The magic isn't in the nodes—it's in the connections. How strong are they? How did they form? Can they change?

### 1.2 What Is Hebbian Learning?

In 1949, psychologist Donald Hebb proposed a simple rule:

> *"Neurons that fire together, wire together."*

When two brain cells activate at the same time, the connection between them strengthens. This explains how we learn associations—why the smell of cinnamon might remind you of grandmother's kitchen, or why hearing "Once upon a time" primes you for a story.

```
Before learning:
  [bell] ──weak──→ [salivation]
  [food] ──strong──→ [salivation]

After repeated pairing:
  [bell] ──strong──→ [salivation]  ← Pavlov's dog learned!
```

Both PRISM and Woven Mind use this principle. But they use it differently.

### 1.3 What Is Predictive Processing?

Your brain doesn't just react to the world—it **predicts** it. When you walk down familiar stairs, your brain predicts where each step will be. When the prediction matches reality, you descend effortlessly. When it doesn't (a step is higher than expected), you stumble—and learn.

This is **predictive processing**:

1. **Predict** what will happen next
2. **Observe** what actually happens
3. **Compare** prediction to reality
4. **Learn** from the difference (prediction error)

The brain is, in this view, a prediction machine. It builds models of the world and continuously refines them through surprise.

### 1.4 Fast and Slow Thinking

Psychologist Daniel Kahneman described two modes of thought:

| System 1 | System 2 |
|----------|----------|
| Fast | Slow |
| Automatic | Deliberate |
| Effortless | Effortful |
| "I know this!" | "Let me think..." |
| Pattern matching | Logical reasoning |

When you recognize a friend's face, that's System 1. When you solve 17 × 24, that's System 2.

Both systems are valuable. The art is knowing when to use each—and how to let them collaborate.

---

## Chapter 2: PRISM — The Garden of Synapses

### 2.1 Philosophy

PRISM began with a biological metaphor: the **synapse**. In your brain, synapses are the connection points between neurons. They can strengthen (potentiation) or weaken (depression) based on experience.

PRISM asks: *What if every edge in our knowledge graph behaved like a synapse?*

### 2.2 Core Components

PRISM consists of four interlocking systems:

```
┌─────────────────────────────────────────────────────────────────────┐
│                           PRISM Framework                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐                        │
│  │   PRISM-GoT     │◄──►│   PRISM-SLM     │                        │
│  │ Synaptic Memory │    │ Language Model  │                        │
│  │ Graph of Thought│    │ Word Transitions│                        │
│  └────────┬────────┘    └────────┬────────┘                        │
│           │                      │                                  │
│           ▼                      ▼                                  │
│  ┌─────────────────┐    ┌─────────────────┐                        │
│  │   PRISM-PLN     │◄──►│PRISM-Attention  │                        │
│  │ Probabilistic   │    │ Selective Focus │                        │
│  │ Logic Networks  │    │ Multi-Head      │                        │
│  └─────────────────┘    └─────────────────┘                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### PRISM-GoT: The Memory Garden

At PRISM's heart lies the **Synaptic Memory Graph**. Every edge remembers:

```python
@dataclass
class SynapticEdge:
    """An edge that learns from experience."""
    source_id: str
    target_id: str
    weight: float = 1.0              # How strong is this connection?

    # Memory
    last_activation_time: datetime   # When was I last used?
    activation_count: int = 0        # How often have I been used?

    # Plasticity
    decay_factor: float = 0.99       # How fast do I fade if unused?

    # Prediction tracking
    prediction_accuracy: float = 0.5 # How reliable am I?
    prediction_correct: int = 0
    prediction_total: int = 0
```

**What this means for a novice**: Each connection in PRISM has a memory. It knows when it was last used, how often it's been used, and how accurate it's been. Unused connections slowly fade (decay), while frequently-used accurate ones grow stronger.

#### PRISM-SLM: The Language of Thought

PRISM-SLM treats words as synapses too:

```python
class SynapticTransition:
    """A connection between words that strengthens with use."""
    from_token: str
    to_token: str
    weight: float = 1.0

    def observe(self, amount: float = 0.1):
        """Hebbian strengthening: I was used, so I get stronger."""
        self.count += 1
        self.weight += amount

    def apply_decay(self):
        """If I'm not used, I slowly fade."""
        self.weight *= 0.99
```

This creates a model of language where common patterns become highways and rare ones become footpaths.

#### PRISM-PLN: Reasoning Under Uncertainty

Real knowledge is uncertain. "Birds usually fly" is true, but penguins exist. PRISM-PLN handles this with **truth values** that have both strength and confidence:

```python
@dataclass
class TruthValue:
    strength: float = 0.5    # How likely is this true? (0 to 1)
    confidence: float = 0.0  # How sure are we? (0 to 1)

    def revise(self, new_evidence: "TruthValue") -> "TruthValue":
        """Update beliefs when new evidence arrives."""
        # Bayesian-style update combining old and new
        ...
```

This lets PRISM reason: "I'm 85% confident that birds fly, based on seeing 100 birds, 85 of which could fly."

#### PRISM-Attention: The Spotlight

Not all information deserves equal focus. PRISM-Attention directs the system's "gaze":

```python
class AttentionLayer:
    """The Caterpillar's focused gaze - only what matters gets attention."""

    def attend(self, query: str) -> Dict[str, float]:
        """Given a query, weight all nodes by relevance."""
        # Like a spotlight, bright on relevant nodes, dim on others
        ...
```

### 2.3 How PRISM Learns

Let's trace a learning episode:

```python
# A question arises
q1 = reasoner.process_thought("How should we handle auth?", NodeType.QUESTION)

# A hypothesis emerges
h1 = reasoner.process_thought("Use JWT tokens", NodeType.HYPOTHESIS)
# PRISM creates an edge: q1 → h1

# Time passes... the JWT approach works!
reasoner.mark_outcome_success(path=[q1.id, h1.id])
# The edge q1 → h1 is strengthened
# The edge gains prediction accuracy

# Next time a similar question arises...
predictions = reasoner.predict_next(q1.id)
# h1 ranks higher because its edge is stronger and more accurate
```

**The beautiful insight**: PRISM learns not just *what* is connected, but *which connections lead to success*. Over time, paths that work become highways; paths that fail become overgrown and forgotten.

---

## Chapter 3: Woven Mind — The Dual-Process Dream

### 3.1 Philosophy

Woven Mind asks a different question: *What if we took Kahneman's two systems seriously?*

Instead of one unified system, Woven Mind proposes two distinct but collaborating systems—like the left and right hemispheres of a brain, each with its own gifts.

### 3.2 Core Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                          THE WOVEN MIND                             │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    CULTURED CORTEX                          │   │
│   │               "The Deliberate Dreamer"                      │   │
│   │                                                             │   │
│   │    • Predicts outcomes       • Learns from success/failure  │   │
│   │    • Builds abstractions     • Top-down guidance            │   │
│   │    • Slow, effortful         • Goal-directed                │   │
│   └───────────────────────┬─────────────────────────────────────┘   │
│                           │                                         │
│                    ═══════╪═══════  THE LOOM  ═══════╪═══════       │
│                           │                          │              │
│   ┌───────────────────────▼──────────────────────────▼──────────┐   │
│   │                      HEBBIAN HIVE                           │   │
│   │               "The Pattern Whisperer"                       │   │
│   │                                                             │   │
│   │    • Detects co-occurrence   • Strengthens what fires together │
│   │    • Forms associations      • Bottom-up discovery          │   │
│   │    • Fast, automatic         • Pattern-matching             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### The Hebbian Hive: System 1

The Hive discovers patterns through pure observation. It doesn't judge—it just notices.

```python
class HebbianHive:
    """The Pattern Whisperer. Fast, automatic, associative."""

    def observe(self, activated_nodes: Set[str]):
        """See what fires together, strengthen those connections."""

        # Apply lateral inhibition (competition for activation)
        winners = self._compete(activated_nodes)

        # Hebbian update: strengthen co-active connections
        for src in winners:
            for tgt in winners:
                if src != tgt:
                    self._strengthen_connection(src, tgt)

    def propagate(self, seed_nodes: Set[str], steps: int = 3):
        """Spread activation like ripples in a pond."""
        active = seed_nodes.copy()
        for _ in range(steps):
            newly_active = set()
            for node_id in active:
                for neighbor, weight in self.edges[node_id].items():
                    if weight > self.threshold:
                        newly_active.add(neighbor)
            active |= newly_active
        return active
```

**Key features**:
- **Lateral inhibition**: Not everything can be active at once. Nodes compete, and only the strongest survive. This creates sparse, distinct patterns.
- **Spreading activation**: When one idea activates, related ideas light up—just like how thinking of "apple" might bring up "pie," "orchard," "Newton."

#### The Cultured Cortex: System 2

The Cortex predicts and reflects. It has goals and learns from failure.

```python
class CulturedCortex:
    """The Deliberate Dreamer. Slow, predictive, goal-directed."""

    def __init__(self):
        self.predictions: List[PredictionRecord] = []
        self.abstractions: Dict[str, Abstraction] = {}
        self.goals: List[Goal] = []

    def predict(self, active_nodes: Set[str]) -> Dict[str, float]:
        """Given current state, what will happen next?"""
        predictions = {}
        for node_id in active_nodes:
            for pattern in self.learned_patterns:
                if pattern.matches(node_id):
                    predictions[pattern.predicts] = pattern.confidence
        return predictions

    def observe_outcome(self, predictions, actual, outcome_type):
        """Compare predictions to reality. Learn from surprise."""
        for node_id, predicted_prob in predictions.items():
            was_correct = node_id in actual
            error = abs((1.0 if was_correct else 0.0) - predicted_prob)

            if error > self.surprise_threshold:
                # Surprise! Time to learn.
                self._update_model(node_id, was_correct, error)

        return error  # Return total prediction error
```

**Key features**:
- **Prediction tracking**: The Cortex remembers what it predicted and whether it was right.
- **Abstraction formation**: When patterns repeat, the Cortex may create a higher-level concept—like realizing that "barking," "furry," and "loyal" often co-occur, so "dog" might be a useful abstraction.

#### The Loom: Where They Weave

The magic happens when Hive and Cortex collaborate:

```python
class WovenMind:
    """The integration layer that weaves fast and slow thinking."""

    def process(self, input_nodes: Set[str]) -> Dict[str, Any]:
        # 1. FAST PATH: Hebbian Hive activates patterns
        hive_activations = self.hive.propagate(input_nodes)

        # 2. CHECK: Is this surprising?
        surprise = self._compute_surprise(input_nodes, hive_activations)

        # 3. ROUTE: High surprise → engage Cortex
        if surprise > self.surprise_threshold:
            # Slow, deliberate processing
            cortex_predictions = self.cortex.predict(hive_activations)
            mode = "deliberate"
        else:
            # Fast, automatic response
            cortex_predictions = {}
            mode = "automatic"

        return {
            "activations": hive_activations,
            "predictions": cortex_predictions,
            "surprise": surprise,
            "mode": mode
        }
```

**The beautiful insight**: Most of the time, System 1 handles things automatically. But when something surprising happens—when the world doesn't match expectations—System 2 wakes up to investigate. This is efficient (why think hard when you don't need to?) and robust (but think hard when you must).

---

## Chapter 4: The Comparison — Two Paths Through Wonderland

### 4.1 Structural Mapping

| Concept | PRISM | Woven Mind | Notes |
|---------|-------|------------|-------|
| **Core metaphor** | Single synaptic system | Dual-process (fast/slow) | Different architectural bets |
| **Hebbian learning** | `SynapticEdge.strengthen()` | `HebbianHive.observe()` | Both implement "fire together, wire together" |
| **Decay/forgetting** | `apply_decay()` | `weight_decay = 0.999` | Both forget unused connections |
| **Prediction** | `IncrementalReasoner.predict_next()` | `CulturedCortex.predict()` | Both predict; PRISM is more integrated |
| **Learning from outcomes** | `mark_outcome_success()` | `observe_outcome()` | Both learn from success/failure |
| **Uncertainty** | `TruthValue(strength, confidence)` | Implicit in edge weights | PRISM is more explicit |
| **Attention** | `AttentionLayer`, `MultiHeadAttention` | Lateral inhibition, competition | PRISM has dedicated attention module |
| **Abstraction** | Emergent from graph structure | Explicit `Abstraction` class | Woven Mind makes it first-class |
| **Mode switching** | Implicit (all one system) | Explicit (surprise → engage System 2) | Woven Mind's key innovation |

### 4.2 What PRISM Does Better

#### 1. **Working Implementation**

PRISM exists as 3,000+ lines of tested Python code. You can:

```bash
# Run a demo right now
python examples/prism_got_demo.py
```

Woven Mind is a design document—valuable, but not executable.

#### 2. **Probabilistic Reasoning**

PRISM-PLN provides sophisticated uncertain reasoning:

```python
# Assert facts with uncertainty
reasoner.assert_fact("bird(tweety)", strength=0.99, confidence=0.95)
reasoner.assert_rule("bird(X)", "canfly(X)", strength=0.85, confidence=0.7)

# Query with propagated uncertainty
result = reasoner.query("canfly(tweety)")
# Returns TruthValue with computed strength and confidence
```

Woven Mind doesn't have an equivalent—it would need to be added.

#### 3. **Attention Mechanisms**

PRISM's attention is sophisticated:

```python
class UnifiedAttention:
    """Cross-system attention integrating GoT + SLM + PLN."""

    def attend(self, query: str) -> AttentionResult:
        # Combines multiple signals:
        # - Semantic relevance (TF-IDF)
        # - Synaptic strength
        # - PLN confidence
        # - Recent activation
```

This multi-faceted attention is powerful for real-world queries.

#### 4. **Language Model Integration**

PRISM-SLM provides a complete statistical language model:

```python
model = PRISMLanguageModel(context_size=3)
model.train(corpus_text)
generated = model.generate("The quick", max_tokens=10)
```

Woven Mind focuses on concept relations, not language generation.

### 4.3 What Woven Mind Does Better

#### 1. **Explicit Dual-Process Architecture**

Woven Mind's core insight is making System 1 and System 2 explicit:

```python
# Woven Mind knows when it's thinking fast vs. slow
result = woven_mind.process(input_nodes)
if result["mode"] == "deliberate":
    # System 2 is engaged—something surprised us
```

PRISM doesn't distinguish; everything flows through one synaptic system. This works, but loses the ability to reason about *how* the system is thinking.

#### 2. **Surprise-Driven Mode Switching**

Woven Mind uses surprise as a signal:

```python
def _compute_surprise(self, input_nodes, hive_activations) -> float:
    """How unexpected was this activation pattern?"""
    expected = self.cortex.predict(input_nodes)
    actual = hive_activations

    # Measure divergence
    surprise = self._kl_divergence(expected, actual)
    return surprise
```

High surprise triggers slow thinking. Low surprise allows fast responses. This is cognitively plausible and computationally efficient.

#### 3. **Explicit Abstraction Formation**

Woven Mind makes abstraction a first-class operation:

```python
def maybe_abstract(self):
    """If we see the same pattern often, create an abstraction."""
    for pattern in self.discovered_patterns:
        if pattern.frequency > self.abstraction_threshold:
            abstraction = Abstraction(
                id=generate_id(),
                components=pattern.nodes,
                level=pattern.max_level + 1
            )
            self.abstractions[abstraction.id] = abstraction
```

This creates hierarchical structure explicitly, rather than hoping it emerges.

#### 4. **Homeostatic Regulation**

Woven Mind includes biological realism like homeostasis:

```python
@dataclass
class HiveNode:
    target_activation: float = 0.05  # Target 5% average activation
    excitability: float = 1.0        # Adjusted to maintain target
```

This prevents runaway activation and keeps the system balanced—a problem biological brains solved that many AI systems ignore.

### 4.4 Where They Converge

Despite different architectures, PRISM and Woven Mind converge on key principles:

1. **Learning through strengthening**: Both strengthen connections that are used together.

2. **Forgetting through decay**: Both allow unused connections to fade.

3. **Prediction matters**: Both track predictions and learn from outcomes.

4. **Context is key**: Both maintain context (PRISM's ActivationTrace, Woven Mind's HiveNode.activation_history).

5. **Sparse is good**: Both prefer sparse activations (PRISM's attention selection, Woven Mind's lateral inhibition).

This convergence suggests these aren't arbitrary design choices—they're fundamental principles for learning systems.

---

## Chapter 5: The Synthesis — What We Learn From Both

### 5.1 A Unified Vision

Imagine combining PRISM's implementation strength with Woven Mind's architectural clarity:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRISM-WOVEN SYNTHESIS                           │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    CULTURED CORTEX                          │   │
│   │            (Extended from PRISM-GoT + PLN)                  │   │
│   │                                                             │   │
│   │    • PRISM-PLN for probabilistic predictions               │   │
│   │    • Explicit abstraction formation                        │   │
│   │    • Goal-directed reasoning                               │   │
│   └───────────────────────┬─────────────────────────────────────┘   │
│                           │                                         │
│              ┌────────────┴────────────┐                            │
│              │    PRISM-ATTENTION      │                            │
│              │  (Surprise Detection)   │                            │
│              │  (Mode Switching)       │                            │
│              └────────────┬────────────┘                            │
│                           │                                         │
│   ┌───────────────────────▼─────────────────────────────────────┐   │
│   │                      HEBBIAN HIVE                           │   │
│   │               (Extended from PRISM-SLM)                     │   │
│   │                                                             │   │
│   │    • Synaptic transitions with Hebbian learning            │   │
│   │    • Lateral inhibition for sparsity                       │   │
│   │    • Spreading activation for retrieval                    │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Design Principles Validated

Both systems validate these principles:

#### Principle 1: Learn from Experience, Not Programming

Neither PRISM nor Woven Mind is programmed with facts. They learn from observation:

```python
# PRISM: Learn from usage
edge.record_activation()
edge.strengthen(amount=0.1)

# Woven Mind: Learn from co-occurrence
hive.observe(activated_nodes)
```

**Implication**: The knowledge comes from the data, not the developer.

#### Principle 2: Forget Gracefully

Both systems decay:

```python
# PRISM
edge.weight *= edge.decay_factor  # 0.99

# Woven Mind
self.weight *= self.weight_decay  # 0.999
```

**Implication**: Memory is not storage but continuous recreation. What's not reinforced fades, making room for new learning.

#### Principle 3: Predict to Understand

Both are fundamentally predictive:

```python
# PRISM
predictions = reasoner.predict_next(current_node)

# Woven Mind
predictions = cortex.predict(active_nodes)
```

**Implication**: Understanding is the ability to predict. A system that predicts well has, in some sense, understood.

#### Principle 4: Balance Exploration and Exploitation

Both allow for both:

```python
# PRISM: prediction_accuracy guides exploitation
if edge.prediction_accuracy > 0.8:
    # Trust this path (exploit)
else:
    # Try alternatives (explore)

# Woven Mind: surprise triggers exploration
if surprise > threshold:
    # Something unexpected—explore with System 2
else:
    # Business as usual—exploit with System 1
```

**Implication**: Neither pure conservatism nor pure adventurism works. Systems need both.

### 5.3 Open Questions

This comparison raises questions worth exploring:

1. **When does dual-process help?** Woven Mind bets that separating fast/slow thinking is valuable. Is it? Under what conditions?

2. **How explicit should uncertainty be?** PRISM-PLN makes probability explicit. Woven Mind leaves it in edge weights. Which leads to better reasoning?

3. **What's the right decay rate?** PRISM uses 0.99, Woven Mind uses 0.999. Does it matter? Should it adapt?

4. **How do abstractions form?** Woven Mind proposes explicit abstraction. PRISM lets it emerge. Which produces better hierarchies?

5. **Can these systems scale?** Both are demonstrated on small corpora (~100 files). What happens at 1 million nodes?

---

## Chapter 6: For the Practitioner — Choosing Your Path

### 6.1 When to Use PRISM

Choose PRISM when you need:

- **A working system today**: PRISM is implemented and tested
- **Probabilistic reasoning**: PLN provides sophisticated uncertainty handling
- **Language modeling**: PRISM-SLM generates text
- **Multi-modal attention**: Unified attention across graph types

```bash
# Get started with PRISM
python examples/prism_got_demo.py
python examples/prism_slm_demo.py
```

### 6.2 When to Use Woven Mind

Choose Woven Mind when you want:

- **Architectural clarity**: The dual-process separation is easier to reason about
- **Cognitive plausibility**: If modeling human-like thought matters
- **Research foundation**: Clear components to study in isolation
- **Extension point**: A framework to build upon

### 6.3 When to Combine Them

The synthesis is most powerful:

- **For hybrid systems**: Use PRISM's PLN for the Cortex, PRISM-SLM for the Hive
- **For research**: Implement Woven Mind's mode switching on top of PRISM's graph
- **For production**: Start with PRISM, add dual-process when you need it

---

## Epilogue: The Road Goes Ever On

We began with a question about how machines might learn to think. We found not one answer but two—each illuminating the other.

PRISM shows us that synaptic learning *works*. Edges that remember, strengthen, decay, and predict can build meaningful knowledge graphs. The code runs. The tests pass. The demos impress.

Woven Mind shows us that architecture *matters*. Separating fast and slow thinking isn't just philosophically appealing—it suggests new capabilities: knowing *how* you're thinking, adapting your strategy to the situation, forming explicit abstractions.

Together, they point toward a future where:

- **Learning is continuous**: Systems that grow with experience
- **Prediction is central**: Understanding as anticipation
- **Dual-process is natural**: Fast when you can, slow when you must
- **Abstraction is automatic**: Hierarchies that form through use

The garden of synapses is planted. The forest of dreams awaits. We have maps and tools and companions for the journey.

Where shall we explore next?

---

> *"Would you tell me, please, which way I ought to go from here?"*
> *"That depends a good deal on where you want to get to," said the Cat.*
> *"I don't much care where—" said Alice.*
> *"Then it doesn't matter which way you go," said the Cat.*
> *"—so long as I get somewhere," Alice added as an explanation.*
> *"Oh, you're sure to do that," said the Cat, "if you only walk long enough."*

---

## References

### Primary Sources

1. PRISM Framework Implementation: `cortical/reasoning/prism_*.py`
2. Woven Mind Architecture: `docs/woven-mind-architecture.md`
3. PRISM Wonderland Roadmap: `docs/prism-wonderland-roadmap.md`

### Foundational Works

4. Hebb, D.O. (1949). *The Organization of Behavior*. Wiley.
5. Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
6. Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*.
7. Carroll, L. (1865). *Alice's Adventures in Wonderland*. Macmillan.

### Technical Background

8. PageRank: Page, L., et al. (1999). "The PageRank Citation Ranking."
9. PLN: Goertzel, B., et al. (2008). "Probabilistic Logic Networks."
10. Attention: Vaswani, A., et al. (2017). "Attention Is All You Need."

---

*Document generated on the winter solstice, when the longest night gives way to returning light.*

*May your synapses strengthen on useful paths, your predictions prove accurate, and your surprises lead to growth.*
