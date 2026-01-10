"""
Behavioral Specifications: Cognitive Architecture - The Factory of Mind

This specification addresses the construction of thought itself:
- How thoughts are built (Builder pattern with DI)
- How desires drive attention (Desire as first-class atoms)
- How we traverse the graph of knowing (Visitor/Strategy patterns)
- How meta-cognition emerges (thoughts about thoughts about thoughts)
- How infinite depth meets finite resources (Gradient-guided recursion)

Central Metaphor: THE FACTORY OF MIND
    Knowledge workers (strategies, visitors, builders) collaborate
    to manufacture thoughts from raw materials (atoms, desires, attention).

    The factory floor is the cognitive graph.
    The assembly lines are thought chains.
    Quality control is truth value assessment.
    The foreman is meta-cognition.
    Customer demand is desire.

Key Insight:
    Thoughts are not just DATA to be processed.
    Thoughts are CONSTRUCTED through collaborative synergy of:
        - Desire (what we want to know)
        - Attention (what we're focusing on)
        - Strategy (how we explore)
        - Construction (how we build understanding)

    This is not metaphor. This is mechanism.

Architectural Patterns:
    - Builder: Fluent thought construction with DI
    - Visitor: Strategic graph traversal
    - Strategy: Interchangeable exploration algorithms
    - Factory: Thought type creation
    - Observer: Desire satisfaction monitoring
    - Inversion of Control: All dependencies injected

Refinements (v2.0):
    This version addresses gaps in v1.0:

    1. DESIRE DYNAMICS - Desires conflict, compound, decay, and can intensify
       when partially satisfied (the "more you know, more you want to know" effect)

    2. ATTENTION DYNAMICS - Focus vs diffuse modes, fatigue, recovery, momentum,
       and the cost of context switching

    3. STRATEGY BLENDING - Continuous interpolation rather than discrete selection
       (70% depth-first + 30% associative)

    4. FORGETTING AND CONSOLIDATION - Memory management through decay, pruning,
       and importance-weighted retention during consolidation cycles

    5. SURPRISE AS SIGNAL - Prediction error drives learning; the unexpected
       captures attention and updates beliefs

    6. WORKING MEMORY LIMITS - The ~4 chunk constraint shapes how we think;
       chunking as compression; load management

    7. ANALOGICAL REASONING - Structure mapping between domains; metaphor as
       the carrier of understanding across contexts

    8. AFFECT AND VALENCE - The emotional coloring of cognition; curiosity,
       anxiety, satisfaction as distinct states that shape processing

    These aren't additions - they're recognition of what cognition actually is.
"""

import pytest
from typing import (
    Protocol, List, Dict, Any, Optional, Callable, Iterator,
    TypeVar, Generic, Set, Tuple, Union
)
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import uuid
from datetime import datetime


# =============================================================================
# CORE TYPES: The Vocabulary of Mind
# =============================================================================


class DesireState(Enum):
    """The lifecycle of wanting."""
    LATENT = auto()      # Want exists but not active
    ACTIVE = auto()      # Currently driving attention
    PURSUING = auto()    # Thoughts spawned to satisfy
    PARTIALLY_MET = auto()  # Some satisfaction achieved
    SATISFIED = auto()   # Desire fulfilled
    ABANDONED = auto()   # Gave up (resource exhaustion)
    CONFLICTED = auto()  # In tension with another desire


class AttentionMode(Enum):
    """
    Modes of attention - the rhythm of cognition.

    FOCUSED: Narrow, deep, analytical - good for known problem spaces
    DIFFUSE: Wide, shallow, associative - good for creativity and insight

    Real cognition oscillates between these. Neither is superior.
    """
    FOCUSED = auto()     # Narrow beam, deep processing
    DIFFUSE = auto()     # Wide net, loose associations
    TRANSITIONING = auto()  # Switching between modes (costly)


class AffectState(Enum):
    """
    The emotional coloring of cognition.

    Affect isn't separate from thinking - it IS part of thinking.
    Curiosity feels different from anxiety, and that difference matters.
    """
    CURIOUS = auto()      # Drawn toward novelty, approach
    ANXIOUS = auto()      # Threat-focused, narrow attention
    SATISFIED = auto()    # Goal achieved, reduced drive
    FRUSTRATED = auto()   # Goal blocked, may escalate or disengage
    BORED = auto()        # Insufficient stimulation, seek novelty
    FLOW = auto()         # Challenge matches skill, effortless focus


class MemoryStrength(Enum):
    """Memory retention levels after consolidation."""
    VIVID = auto()        # Recent, highly activated
    STABLE = auto()       # Consolidated, reliably accessible
    FADING = auto()       # Decaying, needs reactivation
    DORMANT = auto()      # Nearly forgotten, may be recoverable
    FORGOTTEN = auto()    # Gone (or are they?)


class TraversalStrategy(Enum):
    """Ways to walk the graph of knowing."""
    DEPTH_FIRST = auto()      # Go deep before wide
    BREADTH_FIRST = auto()    # Go wide before deep
    BEST_FIRST = auto()       # Follow highest value
    DESIRE_GRADIENT = auto()  # Follow steepest path to satisfaction
    RANDOM_WALK = auto()      # Explore stochastically
    ASSOCIATIVE = auto()      # Follow strongest connections


@dataclass
class Desire:
    """
    A want that drives cognition.

    Desires are first-class atoms in the meta-graph.
    They compete for attention, spawn thoughts, and measure satisfaction.

    Unlike tasks (which are work items), desires are persistent wants
    that may never be fully satisfied (curiosity, understanding, mastery).

    v2.0 Enhancements:
        - Desires can INTENSIFY when partially satisfied (curiosity begets curiosity)
        - Desires can CONFLICT with each other (approach-avoidance)
        - Desires have TEMPORAL DECAY (urgency fades without reinforcement)
        - Desires form HIERARCHIES (terminal vs instrumental)
    """
    id: str = field(default_factory=lambda: f"D-{uuid.uuid4().hex[:6]}")
    description: str = ""
    intensity: float = 0.5          # How much we want this (0-1)
    satisfaction: float = 0.0       # How satisfied we are (0-1)
    state: DesireState = DesireState.LATENT
    spawned_thoughts: List[str] = field(default_factory=list)
    parent_desire: Optional[str] = None  # Desires can derive from desires

    # v2.0: Enhanced dynamics
    intensification_rate: float = 0.0  # How much satisfaction increases intensity
    decay_rate: float = 0.01           # How fast urgency fades without action
    conflicts_with: List[str] = field(default_factory=list)  # Conflicting desire IDs
    is_terminal: bool = False          # Terminal = wanted for itself, not as means
    last_activated: Optional[datetime] = None

    @property
    def urgency(self) -> float:
        """
        Urgency = intensity * (1 - satisfaction) * temporal_factor.

        v2.0: Urgency decays over time without reinforcement.
        """
        base_urgency = self.intensity * (1.0 - self.satisfaction)

        # Temporal decay
        if self.last_activated:
            hours_since = (datetime.now() - self.last_activated).total_seconds() / 3600
            decay_factor = max(0.1, 1.0 - (hours_since * self.decay_rate))
            return base_urgency * decay_factor

        return base_urgency

    def partially_satisfy(self, amount: float) -> None:
        """
        Increase satisfaction (with diminishing returns).

        v2.0: Can also INCREASE intensity (the "more you know" effect).
        """
        remaining = 1.0 - self.satisfaction
        self.satisfaction += remaining * amount

        # v2.0: Intensification - partial satisfaction can increase wanting
        # Like curiosity: learning something makes you want to learn more
        if self.intensification_rate > 0:
            self.intensity = min(1.0, self.intensity + (amount * self.intensification_rate))

        if self.satisfaction > 0.95:
            self.state = DesireState.SATISFIED

    def check_conflict(self, other: 'Desire') -> float:
        """
        Calculate conflict tension with another desire.

        Returns 0.0 (no conflict) to 1.0 (maximum tension).
        High tension when both desires are urgent but incompatible.
        """
        if other.id not in self.conflicts_with:
            return 0.0

        # Tension = product of urgencies (both must be urgent to conflict)
        return self.urgency * other.urgency


@dataclass
class AttentionState:
    """
    The current state of the attention system.

    v2.0: Attention is not just allocation - it has:
        - Mode (focused vs diffuse)
        - Fatigue level
        - Momentum (inertia toward current focus)
        - Switching cost
    """
    mode: AttentionMode = AttentionMode.FOCUSED
    fatigue: float = 0.0            # 0.0 = fresh, 1.0 = exhausted
    current_focus: Optional[str] = None  # What we're attending to
    focus_duration: float = 0.0     # How long on current focus
    momentum: float = 0.0           # Resistance to switching

    # Costs and thresholds
    switch_cost: float = 0.1        # Resource cost to switch focus
    fatigue_recovery_rate: float = 0.05  # Recovery per rest unit
    max_focus_duration: float = 25.0     # Before forced break (Pomodoro-ish)

    def can_switch(self) -> bool:
        """Can we afford to switch focus?"""
        return self.fatigue < 0.9 and self.focus_duration > 2.0

    def get_effective_capacity(self) -> float:
        """Capacity reduced by fatigue."""
        return max(0.1, 1.0 - self.fatigue)

    def switch_to(self, new_focus: str) -> float:
        """
        Switch focus. Returns actual cost paid.

        Higher momentum = higher cost.
        """
        if new_focus == self.current_focus:
            return 0.0

        cost = self.switch_cost * (1.0 + self.momentum)
        self.fatigue = min(1.0, self.fatigue + cost)
        self.current_focus = new_focus
        self.focus_duration = 0.0
        self.momentum = 0.0
        self.mode = AttentionMode.TRANSITIONING

        return cost


@dataclass
class Surprise:
    """
    Prediction error as a cognitive signal.

    Surprise = |expected - observed|

    High surprise:
        - Captures attention involuntarily
        - Drives belief updating
        - Creates memorable experiences
        - Can indicate important learning opportunities
    """
    source_atom: str
    expected_value: float
    observed_value: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def magnitude(self) -> float:
        """How surprising was this? (0-1)"""
        return min(1.0, abs(self.expected_value - self.observed_value))

    @property
    def valence(self) -> float:
        """
        Positive surprise (better than expected) vs negative.

        Returns -1.0 to 1.0
        """
        return self.observed_value - self.expected_value

    def should_capture_attention(self, threshold: float = 0.3) -> bool:
        """Is this surprising enough to involuntarily capture attention?"""
        return self.magnitude > threshold


@dataclass
class WorkingMemorySlot:
    """
    A slot in working memory - the ~4 chunk limit.

    Each slot holds a "chunk" - a compressed representation.
    Chunks can be atoms, links, or compressed thought fragments.
    """
    chunk_id: str
    chunk_content: Any
    activation: float = 1.0      # Decays over time
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 1

    def decay(self, amount: float = 0.1) -> None:
        """Reduce activation (forgetting curve)."""
        self.activation = max(0.0, self.activation - amount)

    def refresh(self) -> None:
        """Re-access this chunk, boosting activation."""
        self.activation = min(1.0, self.activation + 0.3)
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class WorkingMemory:
    """
    The limited capacity workspace of cognition.

    Constraint: ~4 chunks maximum.
    This isn't a bug - it's a feature that forces chunking and abstraction.
    """
    slots: List[WorkingMemorySlot] = field(default_factory=list)
    max_capacity: int = 4

    def is_full(self) -> bool:
        return len(self.slots) >= self.max_capacity

    def get_weakest_slot(self) -> Optional[WorkingMemorySlot]:
        """Get the slot with lowest activation (candidate for eviction)."""
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: s.activation)

    def load(self, chunk_id: str, content: Any) -> bool:
        """
        Load a chunk into working memory.

        If full, evicts the weakest slot.
        Returns True if loaded, False if failed.
        """
        # Already present? Refresh it.
        for slot in self.slots:
            if slot.chunk_id == chunk_id:
                slot.refresh()
                return True

        # Need to evict?
        if self.is_full():
            weakest = self.get_weakest_slot()
            if weakest:
                self.slots.remove(weakest)

        self.slots.append(WorkingMemorySlot(chunk_id=chunk_id, chunk_content=content))
        return True

    def decay_all(self, amount: float = 0.1) -> None:
        """Apply decay to all slots."""
        for slot in self.slots:
            slot.decay(amount)

        # Remove forgotten slots
        self.slots = [s for s in self.slots if s.activation > 0.0]


@dataclass
class AnalogicalMapping:
    """
    A structural correspondence between two domains.

    Analogy is how we understand new things in terms of known things.
    "Electricity is like water flowing through pipes."

    The mapping preserves STRUCTURE, not surface features.
    """
    source_domain: str           # The known domain
    target_domain: str           # The new domain to understand
    correspondences: Dict[str, str] = field(default_factory=dict)  # source -> target
    structural_score: float = 0.0  # How well structure is preserved
    surface_score: float = 0.0     # Surface similarity (often misleading)

    def map(self, source_concept: str) -> Optional[str]:
        """Map a concept from source to target domain."""
        return self.correspondences.get(source_concept)

    def is_deep_analogy(self) -> bool:
        """Deep analogies have high structural score, may have low surface."""
        return self.structural_score > 0.7

    def transfer_inference(self, source_inference: str) -> Optional[str]:
        """
        Transfer an inference from source to target domain.

        This is the power of analogy - learning in one domain
        transfers to another.
        """
        # Would apply mapping to transform the inference
        # e.g., "high pressure causes fast flow" ->
        #       "high voltage causes high current"
        pass  # Implementation would use the correspondences


@dataclass
class StrategyBlend:
    """
    A weighted combination of traversal strategies.

    v2.0: Strategies aren't discrete choices - they BLEND.

    Example: 70% depth-first + 20% associative + 10% random
    This allows nuanced exploration that adapts to context.
    """
    weights: Dict[TraversalStrategy, float] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def sample_strategy(self) -> TraversalStrategy:
        """Sample a strategy according to weights (for stochastic blend)."""
        import random
        r = random.random()
        cumulative = 0.0
        for strategy, weight in self.weights.items():
            cumulative += weight
            if r <= cumulative:
                return strategy
        return list(self.weights.keys())[-1]

    def blend_decisions(self, candidates: List[str], scores_per_strategy: Dict[TraversalStrategy, Dict[str, float]]) -> str:
        """
        Blend multiple strategy scores to pick next node.

        Each strategy scores each candidate.
        Final score = weighted sum across strategies.
        """
        final_scores = {c: 0.0 for c in candidates}

        for strategy, weight in self.weights.items():
            if strategy in scores_per_strategy:
                for candidate, score in scores_per_strategy[strategy].items():
                    if candidate in final_scores:
                        final_scores[candidate] += weight * score

        return max(final_scores.items(), key=lambda x: x[1])[0]

    @classmethod
    def focused_exploration(cls) -> 'StrategyBlend':
        """Preset: Focused, deep exploration."""
        return cls(weights={
            TraversalStrategy.DEPTH_FIRST: 0.7,
            TraversalStrategy.BEST_FIRST: 0.2,
            TraversalStrategy.ASSOCIATIVE: 0.1,
        })

    @classmethod
    def creative_exploration(cls) -> 'StrategyBlend':
        """Preset: Diffuse, creative exploration."""
        return cls(weights={
            TraversalStrategy.ASSOCIATIVE: 0.5,
            TraversalStrategy.RANDOM_WALK: 0.3,
            TraversalStrategy.BREADTH_FIRST: 0.2,
        })


@dataclass
class VisitResult:
    """What a visitor learned at a node."""
    atom_id: str
    insights: List[str] = field(default_factory=list)
    satisfaction_delta: float = 0.0  # Did this help satisfy desire?
    should_continue: bool = True
    suggested_next: List[str] = field(default_factory=list)
    resources_consumed: float = 0.0


@dataclass
class ConstructionBlueprint:
    """
    A plan for building a thought.

    Created by ThoughtBuilder, executed by ThoughtFactory.
    Contains all dependencies and configuration.
    """
    thought_function: Callable
    dependencies: Dict[str, type] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    desires: List[str] = field(default_factory=list)  # What desires this serves
    strategies: List[TraversalStrategy] = field(default_factory=list)
    depth_budget: int = 3
    resource_budget: float = 100.0


# =============================================================================
# PROTOCOLS: The Contracts
# =============================================================================


class CognitiveVisitor(Protocol):
    """
    A visitor that walks the cognitive graph with purpose.

    Visitors are strategies made executable.
    They carry desire, consume resources, and report findings.
    """

    def visit(self, atom_id: str, context: 'VisitContext') -> VisitResult:
        """Visit an atom and learn from it."""
        ...

    def should_visit(self, atom_id: str, context: 'VisitContext') -> bool:
        """Decide whether to visit an atom."""
        ...

    def on_complete(self, context: 'VisitContext') -> Dict[str, Any]:
        """Called when traversal completes."""
        ...


class ThoughtFactory(Protocol):
    """
    Factory that constructs thoughts from blueprints.

    Uses IoC container to resolve all dependencies.
    Ensures proper wiring before thought execution.
    """

    def create(self, blueprint: ConstructionBlueprint) -> 'ConstructedThought':
        """Build a thought from blueprint."""
        ...

    def can_create(self, blueprint: ConstructionBlueprint) -> bool:
        """Check if all dependencies can be satisfied."""
        ...


class DesireManager(Protocol):
    """
    Manages the economy of wanting.

    Tracks desires, allocates attention, measures satisfaction.
    The invisible hand of the cognitive market.
    """

    def register_desire(self, desire: Desire) -> None:
        """Add a desire to the system."""
        ...

    def get_active_desires(self) -> List[Desire]:
        """Get desires currently driving attention."""
        ...

    def allocate_attention(self, total_attention: float) -> Dict[str, float]:
        """Allocate attention across desires by urgency."""
        ...

    def report_satisfaction(self, desire_id: str, amount: float) -> None:
        """Report progress toward satisfying a desire."""
        ...


# =============================================================================
# STORY: Desire-Driven Attention
# =============================================================================


class TestDesireDrivenAttention:
    """
    Epic: Desires Drive the Cognitive Economy

    As a cognitive system,
    I have DESIRES (persistent wants) not just TASKS (work items).
    Desires compete for attention based on urgency.
    Thoughts exist to satisfy desires.
    The system seeks equilibrium of satisfaction.
    """

    def test_scenario_desires_are_first_class_atoms(self):
        """
        Scenario: Desires exist in the cognitive graph

        Given a cognitive graph with hypergraph semantics
        When I create a desire
        Then it becomes an atom in the graph
        And can have relationships like any atom
        Because desires are knowledge about what we want
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType, TruthValue
        from cortical.cognitive.architecture import DesireAtom

        graph = CognitiveGraph()

        # Desire as atom
        understand_cats = graph.node(
            "desire:understand_cats",
            atom_type=AtomType.CONCEPT
        )

        # Can link desires to concepts
        cat = graph.node("cat")

        # The desire is ABOUT the concept
        about_link = graph.link(
            AtomType.EVALUATION,
            [understand_cats, cat],
            tv=TruthValue(1.0, 0.9)
        )

        # Desires can have sub-desires
        understand_behavior = graph.node("desire:understand_cat_behavior")
        derives_from = graph.link(
            AtomType.INHERITANCE,
            [understand_behavior, understand_cats]
        )

        assert graph.get_node("desire:understand_cats") is not None
        assert len(graph.get_incoming(cat.id)) >= 1

    def test_scenario_desires_compete_for_attention(self):
        """
        Scenario: Multiple desires share limited attention

        Given several active desires with different urgencies
        When attention is allocated
        Then higher urgency desires get more attention
        And all desires get some attention (no starvation)
        Because the cognitive market is efficient but fair
        """
        from cortical.cognitive.architecture import DesireManager, Desire

        manager = DesireManager()

        # Three competing desires
        manager.register_desire(Desire(
            description="understand quantum mechanics",
            intensity=0.9,
            satisfaction=0.1,  # Urgency = 0.9 * 0.9 = 0.81
        ))

        manager.register_desire(Desire(
            description="learn to cook",
            intensity=0.6,
            satisfaction=0.5,  # Urgency = 0.6 * 0.5 = 0.30
        ))

        manager.register_desire(Desire(
            description="organize files",
            intensity=0.3,
            satisfaction=0.8,  # Urgency = 0.3 * 0.2 = 0.06
        ))

        # Allocate 100 units of attention
        allocation = manager.allocate_attention(100.0)

        # Higher urgency gets more
        quantum_attention = allocation["understand quantum mechanics"]
        cook_attention = allocation["learn to cook"]
        files_attention = allocation["organize files"]

        assert quantum_attention > cook_attention > files_attention

        # But none are starved
        assert files_attention > 0

    def test_scenario_satisfaction_reduces_urgency(self):
        """
        Scenario: Meeting desires reduces their urgency

        Given a high-urgency desire
        When thoughts partially satisfy it
        Then urgency decreases
        And attention shifts to other desires
        Because satisfied wants demand less
        """
        from cortical.cognitive.architecture import DesireManager, Desire

        manager = DesireManager()

        desire = Desire(
            id="D-001",
            description="understand recursion",
            intensity=0.9,
            satisfaction=0.0,  # Initial urgency = 0.9
        )
        manager.register_desire(desire)

        initial_urgency = desire.urgency
        assert initial_urgency == 0.9

        # Thought provides partial understanding
        manager.report_satisfaction("D-001", 0.5)

        # Urgency reduced
        assert desire.satisfaction > 0
        assert desire.urgency < initial_urgency

    def test_scenario_desire_spawns_thoughts(self):
        """
        Scenario: Active desires spawn thoughts to satisfy them

        Given an active desire
        When the system allocates attention to it
        Then thoughts are spawned to pursue satisfaction
        And those thoughts know what desire they serve
        Because thoughts are the workers of desire
        """
        from cortical.cognitive.architecture import (
            DesireManager, Desire, ThoughtSpawner
        )

        manager = DesireManager()
        spawner = ThoughtSpawner(desire_manager=manager)

        desire = Desire(
            id="D-002",
            description="find patterns in data",
            intensity=0.8,
            satisfaction=0.0,
            state=DesireState.ACTIVE,
        )
        manager.register_desire(desire)

        # Desire spawns thoughts
        thoughts = spawner.spawn_for_desire("D-002", count=3)

        assert len(thoughts) == 3
        for thought in thoughts:
            assert thought.serving_desire == "D-002"
            assert "find patterns" in thought.purpose


# =============================================================================
# STORY: Strategic Graph Traversal (Visitor Pattern)
# =============================================================================


class TestStrategicGraphTraversal:
    """
    Epic: Walk the Graph with Purpose

    As a cognitive system exploring knowledge,
    I traverse the graph using strategies.
    Different strategies suit different goals.
    Visitors carry purpose and return insights.
    """

    def test_scenario_visitor_walks_graph_with_strategy(self):
        """
        Scenario: Visitor traverses graph using chosen strategy

        Given a cognitive graph with interconnected concepts
        When I apply a depth-first visitor
        Then it explores deeply before widely
        And reports what it found at each step
        Because strategy determines exploration pattern
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType
        from cortical.cognitive.architecture import (
            DepthFirstVisitor, VisitContext
        )

        graph = CognitiveGraph()

        # Build a small knowledge structure
        animal = graph.node("animal")
        mammal = graph.node("mammal")
        cat = graph.node("cat")
        persian = graph.node("persian")

        graph.link(AtomType.INHERITANCE, [mammal, animal])
        graph.link(AtomType.INHERITANCE, [cat, mammal])
        graph.link(AtomType.INHERITANCE, [persian, cat])

        # Create visitor with strategy
        visit_log = []

        class LoggingVisitor(DepthFirstVisitor):
            def visit(self, atom_id: str, context: VisitContext) -> VisitResult:
                atom = context.graph.get_atom(atom_id)
                visit_log.append(atom.name if atom else atom_id)
                return VisitResult(atom_id=atom_id, should_continue=True)

        visitor = LoggingVisitor()
        context = VisitContext(graph=graph, start_atom=persian.id)

        # Execute traversal
        visitor.traverse(context)

        # Depth-first: persian -> cat -> mammal -> animal
        assert visit_log[0] == "persian"
        assert "cat" in visit_log
        assert "animal" in visit_log

    def test_scenario_desire_gradient_strategy(self):
        """
        Scenario: Follow the gradient toward desire satisfaction

        Given a desire and a cognitive graph
        When I use desire-gradient traversal
        Then each step moves toward satisfaction
        And I stop when gradient becomes flat
        Because we follow the path of greatest progress
        """
        from cortical.cognitive.graph import CognitiveGraph
        from cortical.cognitive.architecture import (
            DesireGradientVisitor, Desire, VisitContext
        )

        graph = CognitiveGraph()

        # Build knowledge with varying relevance to desire
        target = graph.node("answer_to_question")
        close = graph.node("relevant_concept")
        medium = graph.node("somewhat_related")
        far = graph.node("barely_related")

        desire = Desire(
            description="find the answer",
            intensity=1.0,
            satisfaction=0.0
        )

        # Visitor that follows gradient
        visitor = DesireGradientVisitor(
            desire=desire,
            relevance_scorer=lambda atom, d: {
                "answer_to_question": 1.0,
                "relevant_concept": 0.7,
                "somewhat_related": 0.4,
                "barely_related": 0.1,
            }.get(atom.name, 0.0)
        )

        context = VisitContext(graph=graph, start_atom=far.id)
        path = visitor.find_path(context)

        # Path should move toward higher relevance
        relevances = [visitor.get_relevance(p) for p in path]
        assert relevances == sorted(relevances)  # Monotonically increasing

    def test_scenario_visitor_respects_resource_budget(self):
        """
        Scenario: Traversal stops when resources exhausted

        Given a visitor with limited resource budget
        When it traverses a large graph
        Then it stops when budget is exhausted
        And reports what it found so far
        Because infinite exploration needs finite bounds
        """
        from cortical.cognitive.graph import CognitiveGraph
        from cortical.cognitive.architecture import (
            ResourceBoundedVisitor, VisitContext
        )

        graph = CognitiveGraph()

        # Large graph
        for i in range(100):
            graph.node(f"concept_{i}")

        visitor = ResourceBoundedVisitor(
            resource_budget=10.0,
            cost_per_visit=1.0
        )

        context = VisitContext(graph=graph, start_atom="concept_0")
        result = visitor.traverse(context)

        # Stopped after 10 visits
        assert result.visits_completed == 10
        assert result.resources_remaining == 0.0
        assert result.exhausted_budget is True

    def test_scenario_visitor_can_switch_strategy(self):
        """
        Scenario: Switch strategy when current one stalls

        Given a visitor that isn't making progress
        When it detects stalling (flat gradient)
        Then it can switch to a different strategy
        And continue exploring from new angle
        Because adaptability beats rigidity
        """
        from cortical.cognitive.architecture import (
            AdaptiveVisitor, TraversalStrategy, VisitContext
        )

        visitor = AdaptiveVisitor(
            initial_strategy=TraversalStrategy.DEPTH_FIRST,
            fallback_strategies=[
                TraversalStrategy.BREADTH_FIRST,
                TraversalStrategy.RANDOM_WALK,
            ],
            stall_threshold=3  # Switch after 3 unproductive visits
        )

        # Simulate stalling
        for _ in range(3):
            visitor.record_unproductive_visit()

        # Should have switched
        assert visitor.current_strategy != TraversalStrategy.DEPTH_FIRST


# =============================================================================
# STORY: Thought Construction (Builder Pattern with DI)
# =============================================================================


class TestThoughtConstruction:
    """
    Epic: Build Thoughts with Precision

    As a cognitive architect,
    I construct thoughts using the Builder pattern.
    All dependencies are injected through IoC.
    Construction is explicit, testable, composable.
    """

    def test_scenario_fluent_thought_builder(self):
        """
        Scenario: Build thoughts with fluent API

        Given a ThoughtBuilder
        When I chain configuration methods
        Then I get a fully configured thought
        And all dependencies are explicit
        Because construction should be readable and deliberate
        """
        from cortical.cognitive.architecture import ThoughtBuilder
        from cortical.cognitive.graph import CognitiveGraph

        graph = CognitiveGraph()

        # Fluent construction
        thought = (
            ThoughtBuilder()
            .with_purpose("find patterns in concepts")
            .with_graph(graph)
            .with_strategy(TraversalStrategy.BEST_FIRST)
            .with_depth_budget(5)
            .with_resource_budget(100.0)
            .with_permissions(["read_graph", "create_nodes"])
            .serving_desire("D-001")
            .build()
        )

        assert thought.purpose == "find patterns in concepts"
        assert thought.depth_budget == 5
        assert thought.resource_budget == 100.0
        assert "read_graph" in thought.permissions
        assert thought.serving_desire == "D-001"

    def test_scenario_builder_validates_dependencies(self):
        """
        Scenario: Builder validates required dependencies

        Given an incomplete thought configuration
        When I try to build
        Then I get clear error about missing dependencies
        Because fail-fast beats fail-mysterious
        """
        from cortical.cognitive.architecture import (
            ThoughtBuilder, MissingDependencyError
        )

        builder = ThoughtBuilder()
        # Missing required: graph, purpose

        with pytest.raises(MissingDependencyError) as exc_info:
            builder.build()

        assert "graph" in str(exc_info.value) or "purpose" in str(exc_info.value)

    def test_scenario_builder_with_ioc_container(self):
        """
        Scenario: Builder resolves dependencies from IoC container

        Given an IoC container with registered services
        When I build a thought
        Then dependencies are injected automatically
        And I can override specific dependencies
        Because IoC enables flexible, testable construction
        """
        from cortical.common import Container
        from cortical.cognitive.graph import CognitiveGraph, InMemoryStorage
        from cortical.cognitive.architecture import ThoughtBuilder, Answerer

        # Set up container
        container = Container()
        container.register(CognitiveGraph, CognitiveGraph)

        class MockAnswerer(Answerer):
            def answer(self, query: str) -> str:
                return f"mock answer to: {query}"

        container.register(Answerer, MockAnswerer)

        # Builder uses container
        thought = (
            ThoughtBuilder(container=container)
            .with_purpose("test IoC integration")
            .auto_wire()  # Resolve from container
            .build()
        )

        # Dependencies were injected
        assert thought.graph is not None
        assert isinstance(thought.answerer, MockAnswerer)

    def test_scenario_composite_thought_construction(self):
        """
        Scenario: Build composite thoughts from sub-thoughts

        Given multiple thought blueprints
        When I compose them
        Then I get a meta-thought that orchestrates them
        And execution follows the composition
        Because complex thoughts are built from simpler ones
        """
        from cortical.cognitive.architecture import (
            ThoughtBuilder, CompositeThought, CompositionType
        )

        # Build sub-thoughts
        research_thought = (
            ThoughtBuilder()
            .with_purpose("gather information")
            .with_strategy(TraversalStrategy.BREADTH_FIRST)
            .as_blueprint()
        )

        analyze_thought = (
            ThoughtBuilder()
            .with_purpose("analyze findings")
            .with_strategy(TraversalStrategy.DEPTH_FIRST)
            .as_blueprint()
        )

        synthesize_thought = (
            ThoughtBuilder()
            .with_purpose("synthesize conclusions")
            .as_blueprint()
        )

        # Compose into meta-thought
        composite = (
            CompositeThought()
            .add(research_thought)
            .then(analyze_thought)  # Sequential after research
            .then(synthesize_thought)  # Sequential after analyze
            .build()
        )

        # Execution plan reflects composition
        plan = composite.get_execution_plan()
        assert plan[0].purpose == "gather information"
        assert plan[1].purpose == "analyze findings"
        assert plan[2].purpose == "synthesize conclusions"


# =============================================================================
# STORY: Meta-Cognitive Operations
# =============================================================================


class TestMetaCognitiveOperations:
    """
    Epic: Think About Thinking

    As a cognitive system with self-awareness,
    I can reason about my own reasoning.
    Meta-thoughts observe and modify thought processes.
    The graph contains knowledge about cognition itself.
    """

    def test_scenario_thought_observes_other_thoughts(self):
        """
        Scenario: Meta-thought watches other thoughts

        Given a running thought
        When a meta-thought observes it
        Then it sees the thought's progress and state
        And can reason about that observation
        Because thinking about thinking is thinking
        """
        from cortical.cognitive.architecture import (
            Thought, MetaThought, ThoughtObservation
        )

        # Primary thought
        primary = Thought(
            purpose="solve problem X",
            strategy=TraversalStrategy.DEPTH_FIRST
        )

        # Meta-thought observes
        observations = []

        def observe_and_reason(obs: ThoughtObservation) -> str:
            observations.append(obs)
            if obs.progress < 0.5 and obs.resources_used > 50:
                return "primary thought is inefficient, suggest strategy change"
            return "primary thought proceeding normally"

        meta = MetaThought(
            target=primary,
            observation_handler=observe_and_reason
        )

        # Run primary with meta observing
        primary.run(observer=meta)

        assert len(observations) > 0
        assert all(obs.thought_id == primary.id for obs in observations)

    def test_scenario_meta_thought_modifies_strategy(self):
        """
        Scenario: Meta-thought can modify observed thought's strategy

        Given a meta-thought observing inefficient exploration
        When it detects poor progress
        Then it can suggest or force strategy change
        And the observed thought adapts
        Because meta-cognition enables adaptation
        """
        from cortical.cognitive.architecture import (
            Thought, MetaThought, StrategyModification
        )

        primary = Thought(
            purpose="explore large space",
            strategy=TraversalStrategy.DEPTH_FIRST
        )

        modifications_applied = []

        class AdaptiveMetaThought(MetaThought):
            def on_stall_detected(self, thought: Thought) -> StrategyModification:
                mod = StrategyModification(
                    new_strategy=TraversalStrategy.BREADTH_FIRST,
                    reason="depth-first stalled, trying breadth-first"
                )
                modifications_applied.append(mod)
                return mod

        meta = AdaptiveMetaThought(target=primary)

        # Simulate stall
        primary.simulate_stall()
        meta.check_and_adapt()

        assert len(modifications_applied) == 1
        assert primary.current_strategy == TraversalStrategy.BREADTH_FIRST

    def test_scenario_cognitive_graph_contains_meta_knowledge(self):
        """
        Scenario: Graph stores knowledge about cognition

        Given a cognitive graph
        When I add meta-knowledge (knowledge about knowing)
        Then it coexists with object-level knowledge
        And can be queried and reasoned about
        Because the mind knows itself
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType, TruthValue

        graph = CognitiveGraph()

        # Object-level knowledge
        cat = graph.node("cat")
        animal = graph.node("animal")
        cat_is_animal = graph.link(AtomType.INHERITANCE, [cat, animal])

        # Meta-level knowledge (about the knowledge itself)
        inheritance_concept = graph.node("inheritance_relation")
        cat_is_animal_link = graph.get_atom(cat_is_animal.id)

        # "The cat-animal relationship is an inheritance" (meta)
        instance_of = graph.link(
            AtomType.MEMBER,
            [cat_is_animal.id, inheritance_concept.id]
        )

        # "Inheritance relations are transitive" (meta-meta)
        transitivity = graph.node("transitivity_property")
        inheritance_is_transitive = graph.link(
            AtomType.EVALUATION,
            [inheritance_concept, transitivity],
            tv=TruthValue(1.0, 0.99)
        )

        # Can query meta-knowledge
        assert graph.get_node("inheritance_relation") is not None
        assert graph.get_node("transitivity_property") is not None

    def test_scenario_recursive_meta_cognition(self):
        """
        Scenario: Meta-thoughts can have meta-meta-thoughts

        Given a meta-thought observing a thought
        When I create a meta-meta-thought to observe the meta
        Then we get proper recursion with depth limits
        Because infinite meta-regression needs bounds
        """
        from cortical.cognitive.architecture import (
            Thought, MetaThought, MetaLevel
        )

        # Level 0: Primary thought
        primary = Thought(purpose="solve X", meta_level=MetaLevel.OBJECT)

        # Level 1: Meta-thought about primary
        meta1 = MetaThought(
            target=primary,
            purpose="observe solving of X",
            meta_level=MetaLevel.META
        )

        # Level 2: Meta-meta-thought about meta1
        meta2 = MetaThought(
            target=meta1,
            purpose="observe the observation",
            meta_level=MetaLevel.META_META
        )

        # Level 3: Would exceed typical limit
        with pytest.raises(ValueError, match="meta.*depth"):
            meta3 = MetaThought(
                target=meta2,
                purpose="too deep",
                meta_level=MetaLevel.META_META_META  # Typically max depth
            )


# =============================================================================
# STORY: Resource-Bounded Infinite Recursion
# =============================================================================


class TestResourceBoundedInfiniteRecursion:
    """
    Epic: Infinite Depth with Finite Resources

    As a cognitive system facing infinite knowledge,
    I must explore without bound but with budget.
    Gradient-guided descent finds value efficiently.
    Partial results are valuable even when incomplete.
    """

    def test_scenario_gradient_guides_depth(self):
        """
        Scenario: Gradient determines when to go deeper

        Given a thought exploring recursively
        When each step has a gradient (progress toward desire)
        Then positive gradient continues, flat gradient stops
        And resources are spent where progress is made
        Because gradient descent is efficient exploration
        """
        from cortical.cognitive.architecture import (
            GradientGuidedThought, DesireGradient
        )

        gradients_observed = []

        def gradient_function(depth: int, state: Dict) -> float:
            """Gradient decreases with depth (diminishing returns)."""
            gradient = 1.0 / (depth + 1)
            gradients_observed.append((depth, gradient))
            return gradient

        thought = GradientGuidedThought(
            gradient_fn=gradient_function,
            min_gradient=0.1,  # Stop when gradient below this
            max_depth=100,     # Hard limit just in case
        )

        result = thought.descend()

        # Stopped when gradient dropped below threshold
        final_gradient = gradients_observed[-1][1]
        assert final_gradient <= 0.1

        # Didn't hit max depth (gradient stopped us earlier)
        assert result.depth_reached < 100

    def test_scenario_resource_budget_across_branches(self):
        """
        Scenario: Budget shared across exploration branches

        Given a thought that branches into multiple paths
        When each branch consumes resources
        Then total consumption respects budget
        And high-value branches get more resources
        Because resources flow to value
        """
        from cortical.cognitive.architecture import (
            BranchingThought, ResourceAllocator
        )

        allocator = ResourceAllocator(total_budget=100.0)

        branches = [
            {"id": "A", "estimated_value": 0.9},
            {"id": "B", "estimated_value": 0.5},
            {"id": "C", "estimated_value": 0.2},
        ]

        allocation = allocator.allocate_to_branches(branches)

        # Higher value gets more
        assert allocation["A"] > allocation["B"] > allocation["C"]

        # Total respects budget
        assert sum(allocation.values()) <= 100.0

    def test_scenario_partial_results_from_incomplete_exploration(self):
        """
        Scenario: Return useful partial results when budget exhausted

        Given exploration that ran out of resources
        When it returns
        Then it provides what it found so far
        And indicates what remains unexplored
        Because partial knowledge has value
        """
        from cortical.cognitive.architecture import (
            ResourceBoundedThought, PartialExplorationResult
        )

        thought = ResourceBoundedThought(
            purpose="explore vast space",
            resource_budget=10.0,
            cost_per_step=1.0
        )

        result: PartialExplorationResult = thought.explore()

        # Got partial results
        assert len(result.findings) > 0

        # Knows what's unexplored
        assert result.unexplored_frontier is not None
        assert len(result.unexplored_frontier) > 0

        # Can estimate value of continuing
        assert result.estimated_remaining_value >= 0

    def test_scenario_anytime_algorithm_behavior(self):
        """
        Scenario: Thought behaves as anytime algorithm

        Given a thought that can be interrupted
        When interrupted at any point
        Then it returns best answer so far
        And quality improves with more time
        Because anytime behavior enables flexible resource use
        """
        from cortical.cognitive.architecture import AnytimeThought

        thought = AnytimeThought(purpose="optimize something")

        # Run for different amounts of time
        results = []
        for budget in [1, 5, 10, 50, 100]:
            result = thought.run_with_budget(budget)
            results.append((budget, result.quality))

        # Quality improves with budget (monotonic or near-monotonic)
        qualities = [q for _, q in results]
        for i in range(1, len(qualities)):
            assert qualities[i] >= qualities[i-1] * 0.95  # Allow small variance


# =============================================================================
# STORY: Knowledge Worker Orchestration
# =============================================================================


class TestKnowledgeWorkerOrchestration:
    """
    Epic: Knowledge Workers Attend to Cognition

    As a cognitive system with internal workers,
    Workers are strategies made incarnate.
    They attend to different aspects of cognition.
    Orchestration coordinates their efforts.
    """

    def test_scenario_workers_specialize_in_strategies(self):
        """
        Scenario: Different workers for different strategies

        Given a pool of knowledge workers
        When each specializes in a traversal strategy
        Then they can be dispatched appropriately
        And their expertise is leveraged
        Because specialization enables mastery
        """
        from cortical.cognitive.architecture import (
            KnowledgeWorker, WorkerPool, TraversalStrategy
        )

        pool = WorkerPool()

        # Create specialized workers
        pool.add_worker(KnowledgeWorker(
            specialty=TraversalStrategy.DEPTH_FIRST,
            capacity=10
        ))
        pool.add_worker(KnowledgeWorker(
            specialty=TraversalStrategy.BREADTH_FIRST,
            capacity=10
        ))
        pool.add_worker(KnowledgeWorker(
            specialty=TraversalStrategy.DESIRE_GRADIENT,
            capacity=5
        ))

        # Dispatch by strategy need
        worker = pool.get_worker_for(TraversalStrategy.DESIRE_GRADIENT)

        assert worker.specialty == TraversalStrategy.DESIRE_GRADIENT

    def test_scenario_workers_share_cognitive_graph(self):
        """
        Scenario: All workers share the same cognitive graph

        Given multiple workers exploring
        When one worker adds knowledge
        Then others can see it
        And we build collective understanding
        Because knowledge is shared, not siloed
        """
        from cortical.cognitive.graph import CognitiveGraph
        from cortical.cognitive.architecture import (
            KnowledgeWorker, WorkerPool
        )

        graph = CognitiveGraph()
        pool = WorkerPool(shared_graph=graph)

        worker_a = pool.spawn_worker("explorer")
        worker_b = pool.spawn_worker("synthesizer")

        # Worker A adds knowledge
        worker_a.execute(lambda g: g.node("discovery_a"))

        # Worker B can see it
        can_see = worker_b.execute(lambda g: g.get_node("discovery_a") is not None)

        assert can_see

    def test_scenario_orchestrator_coordinates_workers(self):
        """
        Scenario: Orchestrator assigns work and gathers results

        Given an orchestrator managing workers
        When complex work arrives
        Then orchestrator decomposes and assigns
        And gathers and synthesizes results
        Because coordination multiplies capability
        """
        from cortical.cognitive.architecture import (
            Orchestrator, WorkerPool, Task
        )

        pool = WorkerPool()
        for i in range(5):
            pool.spawn_worker(f"worker_{i}")

        orchestrator = Orchestrator(worker_pool=pool)

        # Complex task
        task = Task(
            description="understand complex system",
            subtasks=[
                Task(description="map components"),
                Task(description="trace dependencies"),
                Task(description="identify patterns"),
                Task(description="synthesize understanding"),
            ]
        )

        # Orchestrator decomposes and assigns
        plan = orchestrator.plan(task)

        assert len(plan.assignments) == 4
        assert plan.has_synthesis_step

        # Execute and gather
        result = orchestrator.execute(plan)

        assert result.is_complete
        assert "understanding" in result.synthesis

    def test_scenario_workers_signal_attention_needs(self):
        """
        Scenario: Workers signal when they need more attention

        Given a worker pursuing a promising lead
        When it needs more resources to continue
        Then it signals the attention allocation system
        And may receive more if value justifies
        Because workers advocate for their work
        """
        from cortical.cognitive.architecture import (
            KnowledgeWorker, AttentionRequest, AttentionAllocator
        )

        allocator = AttentionAllocator(total_capacity=100)

        worker = KnowledgeWorker(
            specialty=TraversalStrategy.DEPTH_FIRST,
            capacity=10
        )
        allocator.assign_initial(worker, 10)

        # Worker finds promising lead
        request = AttentionRequest(
            from_worker=worker.id,
            additional_needed=20,
            justification="found high-value path, need more depth",
            estimated_value=0.9
        )

        # Submit request
        granted = allocator.request_more(request)

        # High value justified more resources
        assert granted > 0
        assert worker.current_capacity > 10


# =============================================================================
# STORY: Synergistic Construction
# =============================================================================


class TestSynergisticConstruction:
    """
    Epic: Construction Through Synergy

    As a cognitive system building understanding,
    Components synergize to create more than their sum.
    Desires, strategies, workers, and visitors collaborate.
    The whole emerges from coordinated parts.
    """

    def test_scenario_full_cognitive_cycle(self):
        """
        Scenario: Complete cycle from desire to satisfaction

        Given a desire to understand something
        When the full cognitive machinery engages
        Then desires spawn thoughts, thoughts use strategies,
             strategies guide visitors, visitors find knowledge,
             knowledge satisfies desires
        And the cycle completes with understanding
        Because this is how cognition works
        """
        from cortical.cognitive.graph import CognitiveGraph
        from cortical.cognitive.architecture import (
            Desire, DesireManager, ThoughtBuilder,
            KnowledgeWorker, WorkerPool, Orchestrator,
            CognitiveCycle
        )

        # Initialize the cognitive system
        graph = CognitiveGraph()
        desire_manager = DesireManager()
        worker_pool = WorkerPool(shared_graph=graph)
        orchestrator = Orchestrator(worker_pool=worker_pool)

        # Seed some knowledge
        graph.node("cat")
        graph.node("mammal")
        graph.node("animal")

        # Create a desire
        desire = Desire(
            description="understand cat taxonomy",
            intensity=0.8
        )
        desire_manager.register_desire(desire)

        # Run the cognitive cycle
        cycle = CognitiveCycle(
            graph=graph,
            desire_manager=desire_manager,
            orchestrator=orchestrator
        )

        result = cycle.run_until_satisfied(desire.id, max_iterations=10)

        # Desire should be at least partially satisfied
        assert desire_manager.get_desire(desire.id).satisfaction > 0

        # Knowledge should have grown
        assert len(graph._storage.all_atoms()) > 3

    def test_scenario_emergent_understanding(self):
        """
        Scenario: Understanding emerges from component interaction

        Given separate pieces of knowledge
        When cognitive processes connect them
        Then new understanding emerges
        That wasn't explicitly programmed
        Because emergence is the goal of cognition
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType
        from cortical.cognitive.architecture import (
            InferenceEngine, EmergentInsight
        )

        graph = CognitiveGraph()

        # Add separate facts
        graph.node("socrates")
        graph.node("human")
        graph.node("mortal")

        socrates_is_human = graph.link(
            AtomType.INHERITANCE,
            [graph.node("socrates"), graph.node("human")]
        )

        humans_are_mortal = graph.link(
            AtomType.INHERITANCE,
            [graph.node("human"), graph.node("mortal")]
        )

        # Run inference
        engine = InferenceEngine(graph)
        insights = engine.infer()

        # Should discover emergent conclusion
        emergent = [i for i in insights if isinstance(i, EmergentInsight)]

        # "Socrates is mortal" should emerge
        assert any(
            "socrates" in str(i).lower() and "mortal" in str(i).lower()
            for i in emergent
        )


# =============================================================================
# STORY: v2.0 - Attention Dynamics
# =============================================================================


class TestAttentionDynamics:
    """
    Epic: The Rhythm of Focus

    As a cognitive system with limited attention,
    I oscillate between focused and diffuse modes.
    Focus has momentum and fatigue.
    Switching has costs.
    Recovery requires rest.
    """

    def test_scenario_focused_vs_diffuse_modes(self):
        """
        Scenario: Different attention modes for different tasks

        Given a cognitive system processing information
        When the task requires deep analysis
        Then focused mode is appropriate
        But when seeking creative connections
        Then diffuse mode is better
        Because different tasks need different attention styles
        """
        from cortical.cognitive.architecture import (
            AttentionController, AttentionMode, Task
        )

        controller = AttentionController()

        # Analytical task
        analysis_task = Task(
            description="prove theorem",
            requires_depth=True,
            requires_creativity=False
        )

        recommended = controller.recommend_mode(analysis_task)
        assert recommended == AttentionMode.FOCUSED

        # Creative task
        creative_task = Task(
            description="brainstorm solutions",
            requires_depth=False,
            requires_creativity=True
        )

        recommended = controller.recommend_mode(creative_task)
        assert recommended == AttentionMode.DIFFUSE

    def test_scenario_attention_fatigue_accumulates(self):
        """
        Scenario: Sustained focus causes fatigue

        Given extended focused attention
        When fatigue accumulates
        Then effective capacity decreases
        And rest is needed to recover
        Because attention is a depletable resource
        """
        from cortical.cognitive.architecture import AttentionState

        state = AttentionState(mode=AttentionMode.FOCUSED)

        assert state.fatigue == 0.0
        assert state.get_effective_capacity() == 1.0

        # Simulate extended focus
        for _ in range(10):
            state.fatigue = min(1.0, state.fatigue + 0.1)

        assert state.fatigue == 1.0
        assert state.get_effective_capacity() == 0.1  # Severely degraded

    def test_scenario_switching_has_cost(self):
        """
        Scenario: Context switching costs resources

        Given focus on task A
        When switching to task B
        Then a switching cost is paid
        And the cost increases with momentum
        Because context switching is expensive
        """
        from cortical.cognitive.architecture import AttentionState

        state = AttentionState(
            current_focus="task_A",
            momentum=0.5  # Some inertia built up
        )

        initial_fatigue = state.fatigue
        cost = state.switch_to("task_B")

        assert cost > 0
        assert state.fatigue > initial_fatigue
        assert state.current_focus == "task_B"
        assert state.momentum == 0.0  # Reset after switch

    def test_scenario_diffuse_mode_enables_connections(self):
        """
        Scenario: Diffuse mode finds distant associations

        Given a cognitive graph with loosely connected concepts
        When in diffuse mode
        Then distant associations become accessible
        That focused mode would miss
        Because diffuse mode casts a wider net
        """
        from cortical.cognitive.graph import CognitiveGraph
        from cortical.cognitive.architecture import (
            DiffuseModeExplorer, AttentionMode
        )

        graph = CognitiveGraph()

        # Create two loosely connected domains
        graph.node("music")
        graph.node("rhythm")
        graph.node("mathematics")
        graph.node("pattern")

        # Weak connection
        graph.node("periodicity")  # Bridges both domains

        explorer = DiffuseModeExplorer(graph, mode=AttentionMode.DIFFUSE)

        # Should find the bridge
        connections = explorer.find_distant_connections("music", "mathematics")

        assert "periodicity" in [c.name for c in connections]


# =============================================================================
# STORY: v2.0 - Surprise and Learning
# =============================================================================


class TestSurpriseAndLearning:
    """
    Epic: The Unexpected Drives Learning

    As a cognitive system that predicts,
    Prediction errors (surprise) are information.
    High surprise captures attention.
    Surprise drives belief updating.
    Without surprise, no learning.
    """

    def test_scenario_surprise_captures_attention(self):
        """
        Scenario: Unexpected events capture attention involuntarily

        Given expectations about the world
        When observation violates expectations
        Then attention is involuntarily captured
        Regardless of current focus
        Because the unexpected is potentially important
        """
        from cortical.cognitive.architecture import (
            Surprise, AttentionController
        )

        controller = AttentionController()
        controller.focus_on("routine task")

        # Something unexpected happens
        surprise = Surprise(
            source_atom="unexpected_event",
            expected_value=0.1,  # Expected to be rare
            observed_value=0.9   # But it happened strongly
        )

        assert surprise.magnitude > 0.7  # Very surprising
        assert surprise.should_capture_attention()

        # Attention shifts despite focus
        if surprise.should_capture_attention():
            controller.interrupt_with(surprise)

        assert controller.current_focus == "unexpected_event"

    def test_scenario_surprise_updates_beliefs(self):
        """
        Scenario: Prediction error updates truth values

        Given a belief with certain confidence
        When observation contradicts it
        Then belief strength is updated
        Proportional to surprise magnitude
        Because beliefs should track reality
        """
        from cortical.cognitive.graph import CognitiveGraph, TruthValue
        from cortical.cognitive.architecture import (
            Surprise, BeliefUpdater
        )

        graph = CognitiveGraph()

        # Initial belief
        belief = graph.node("birds_fly")
        belief.tv = TruthValue(strength=0.95, confidence=0.8)

        # Observe exception
        surprise = Surprise(
            source_atom=belief.id,
            expected_value=0.95,  # Expected birds fly
            observed_value=0.0    # But this one doesn't (penguin)
        )

        updater = BeliefUpdater(graph)
        updater.update_from_surprise(belief.id, surprise)

        # Belief should be weakened
        assert belief.tv.strength < 0.95

    def test_scenario_no_surprise_no_learning(self):
        """
        Scenario: Predicted outcomes don't update beliefs

        Given accurate predictions
        When observations match expectations
        Then beliefs remain stable
        Because correct predictions need no update
        """
        from cortical.cognitive.architecture import Surprise, BeliefUpdater

        surprise = Surprise(
            source_atom="predicted_event",
            expected_value=0.8,
            observed_value=0.8  # Exactly as expected
        )

        assert surprise.magnitude == 0.0
        assert not surprise.should_capture_attention()

        # No update needed
        # Learning only from prediction ERROR

    def test_scenario_positive_vs_negative_surprise(self):
        """
        Scenario: Distinguish better-than-expected from worse

        Given an outcome that differs from expectation
        When outcome is better than expected
        Then surprise has positive valence
        When outcome is worse than expected
        Then surprise has negative valence
        Because the direction of surprise matters
        """
        from cortical.cognitive.architecture import Surprise

        # Better than expected
        positive_surprise = Surprise(
            source_atom="lottery",
            expected_value=0.0,
            observed_value=1.0  # Won!
        )
        assert positive_surprise.valence > 0

        # Worse than expected
        negative_surprise = Surprise(
            source_atom="exam",
            expected_value=0.9,
            observed_value=0.3  # Failed!
        )
        assert negative_surprise.valence < 0


# =============================================================================
# STORY: v2.0 - Working Memory Constraints
# =============================================================================


class TestWorkingMemoryConstraints:
    """
    Epic: The ~4 Chunk Limit

    As a cognitive system with limited working memory,
    Only ~4 chunks can be active simultaneously.
    This constraint forces chunking and abstraction.
    Items decay without rehearsal.
    Full memory requires eviction.
    """

    def test_scenario_four_chunk_limit(self):
        """
        Scenario: Cannot hold more than ~4 chunks

        Given an empty working memory
        When loading more than 4 chunks
        Then the oldest/weakest is evicted
        To make room for the new
        Because capacity is fundamentally limited
        """
        from cortical.cognitive.architecture import WorkingMemory

        wm = WorkingMemory(max_capacity=4)

        # Load 4 chunks
        for i in range(4):
            wm.load(f"chunk_{i}", f"content_{i}")

        assert len(wm.slots) == 4

        # Load 5th - must evict one
        wm.load("chunk_4", "content_4")

        assert len(wm.slots) == 4  # Still 4
        assert any(s.chunk_id == "chunk_4" for s in wm.slots)

    def test_scenario_decay_without_rehearsal(self):
        """
        Scenario: Unrehearsed items decay

        Given items in working memory
        When time passes without access
        Then activation decays
        And eventually items are forgotten
        Because memory requires maintenance
        """
        from cortical.cognitive.architecture import WorkingMemory

        wm = WorkingMemory()
        wm.load("item", "content")

        initial_activation = wm.slots[0].activation
        assert initial_activation == 1.0

        # Time passes without rehearsal
        for _ in range(15):
            wm.decay_all(amount=0.1)

        # Should be forgotten (removed)
        assert len(wm.slots) == 0

    def test_scenario_rehearsal_refreshes_items(self):
        """
        Scenario: Accessing items refreshes them

        Given a decaying item in working memory
        When the item is accessed again
        Then its activation is boosted
        And it resists forgetting
        Because rehearsal maintains memory
        """
        from cortical.cognitive.architecture import WorkingMemory

        wm = WorkingMemory()
        wm.load("item", "content")

        # Some decay
        wm.decay_all(amount=0.3)
        decayed_activation = wm.slots[0].activation

        # Rehearse (re-access)
        wm.load("item", "content")  # Same item

        # Activation boosted
        assert wm.slots[0].activation > decayed_activation

    def test_scenario_chunking_overcomes_limit(self):
        """
        Scenario: Chunking compresses information

        Given more than 4 pieces of information
        When chunked into meaningful groups
        Then all information fits in 4 slots
        Because chunking is compression
        """
        from cortical.cognitive.architecture import (
            WorkingMemory, ChunkingStrategy
        )

        # 12 items - too many
        items = [f"item_{i}" for i in range(12)]

        chunker = ChunkingStrategy()

        # Chunk into groups of 3
        chunks = chunker.chunk(items, chunk_size=3)

        assert len(chunks) == 4  # Now fits

        # Load chunked representation
        wm = WorkingMemory()
        for i, chunk in enumerate(chunks):
            wm.load(f"chunk_{i}", chunk)

        assert len(wm.slots) == 4  # All info represented


# =============================================================================
# STORY: v2.0 - Analogical Reasoning
# =============================================================================


class TestAnalogicalReasoning:
    """
    Epic: Understanding Through Comparison

    As a cognitive system that learns,
    Analogy is core to understanding.
    We understand new things in terms of known things.
    Structure matters more than surface.
    Inferences transfer across domains.
    """

    def test_scenario_structure_mapping(self):
        """
        Scenario: Map structure between domains

        Given a known domain (water flow)
        And a new domain (electricity)
        When I find structural correspondences
        Then I can understand electricity through water
        Because structure is transferable
        """
        from cortical.cognitive.architecture import (
            AnalogicalMapping, AnalogicalMatcher
        )

        # Source domain: water flow
        water_concepts = {
            "water": "substance",
            "pipe": "conduit",
            "pressure": "driving_force",
            "flow_rate": "movement",
        }

        # Target domain: electricity
        electricity_concepts = {
            "electrons": "substance",
            "wire": "conduit",
            "voltage": "driving_force",
            "current": "movement",
        }

        matcher = AnalogicalMatcher()
        mapping = matcher.find_mapping(water_concepts, electricity_concepts)

        assert mapping.map("water") == "electrons"
        assert mapping.map("pressure") == "voltage"
        assert mapping.map("flow_rate") == "current"
        assert mapping.structural_score > 0.8

    def test_scenario_inference_transfer(self):
        """
        Scenario: Transfer inferences via analogy

        Given a known inference in source domain
        When I have a structural mapping
        Then I can transfer the inference to target
        Because if structure matches, inference may hold
        """
        from cortical.cognitive.architecture import (
            AnalogicalMapping, InferenceTransfer
        )

        mapping = AnalogicalMapping(
            source_domain="water",
            target_domain="electricity",
            correspondences={
                "high_pressure": "high_voltage",
                "fast_flow": "high_current",
                "causes": "causes",
            },
            structural_score=0.9
        )

        # Known in water domain
        source_inference = "high_pressure causes fast_flow"

        transfer = InferenceTransfer(mapping)
        target_inference = transfer.transfer(source_inference)

        assert target_inference == "high_voltage causes high_current"

    def test_scenario_surface_vs_structural_similarity(self):
        """
        Scenario: Deep analogies may have low surface similarity

        Given two domains
        When surface features differ but structure matches
        Then the analogy is deep and valuable
        Because surface similarity can mislead
        """
        from cortical.cognitive.architecture import AnalogicalMapping

        # High structural, low surface (good analogy)
        deep_analogy = AnalogicalMapping(
            source_domain="solar_system",
            target_domain="atom",
            structural_score=0.85,  # Similar structure
            surface_score=0.1       # Completely different size/appearance
        )

        assert deep_analogy.is_deep_analogy()

        # Low structural, high surface (misleading)
        shallow_analogy = AnalogicalMapping(
            source_domain="kidney_bean",
            target_domain="kidney",
            structural_score=0.1,  # Very different function
            surface_score=0.8      # Shape is similar
        )

        assert not shallow_analogy.is_deep_analogy()


# =============================================================================
# STORY: v2.0 - Strategy Blending
# =============================================================================


class TestStrategyBlending:
    """
    Epic: Beyond Discrete Strategies

    As a cognitive system exploring knowledge,
    Strategies are not discrete choices.
    They blend in continuous proportions.
    The blend adapts to context.
    Creativity emerges from the mix.
    """

    def test_scenario_blend_multiple_strategies(self):
        """
        Scenario: Combine strategies in weighted blend

        Given multiple traversal strategies
        When I blend them with weights
        Then exploration uses all proportionally
        And the blend can be tuned
        Because rigid strategy selection is suboptimal
        """
        from cortical.cognitive.architecture import (
            StrategyBlend, TraversalStrategy
        )

        blend = StrategyBlend(weights={
            TraversalStrategy.DEPTH_FIRST: 0.6,
            TraversalStrategy.ASSOCIATIVE: 0.3,
            TraversalStrategy.RANDOM_WALK: 0.1,
        })

        # Weights should normalize
        total = sum(blend.weights.values())
        assert abs(total - 1.0) < 0.01

        # Can sample according to weights
        samples = [blend.sample_strategy() for _ in range(1000)]

        depth_ratio = sum(1 for s in samples if s == TraversalStrategy.DEPTH_FIRST) / 1000
        assert 0.5 < depth_ratio < 0.7  # Should be around 0.6

    def test_scenario_blend_for_focused_vs_creative(self):
        """
        Scenario: Different blends for different cognitive modes

        Given focused attention mode
        When I select a blend
        Then it favors depth and best-first
        But in diffuse mode
        Then it favors associative and random
        Because mode should influence strategy
        """
        from cortical.cognitive.architecture import (
            StrategyBlend, TraversalStrategy
        )

        focused = StrategyBlend.focused_exploration()
        creative = StrategyBlend.creative_exploration()

        # Focused emphasizes depth
        assert focused.weights.get(TraversalStrategy.DEPTH_FIRST, 0) > 0.5

        # Creative emphasizes association
        assert creative.weights.get(TraversalStrategy.ASSOCIATIVE, 0) > 0.3

    def test_scenario_blend_adapts_to_context(self):
        """
        Scenario: Blend adjusts based on exploration results

        Given a strategy blend
        When one strategy produces better results
        Then its weight should increase
        Over time optimizing the blend
        Because exploration should learn what works
        """
        from cortical.cognitive.architecture import (
            AdaptiveStrategyBlend, TraversalStrategy, ExplorationFeedback
        )

        blend = AdaptiveStrategyBlend(initial_weights={
            TraversalStrategy.DEPTH_FIRST: 0.5,
            TraversalStrategy.BREADTH_FIRST: 0.5,
        })

        # Depth-first produces good results
        feedback = ExplorationFeedback(
            strategy=TraversalStrategy.DEPTH_FIRST,
            value_found=0.9
        )
        blend.update_from_feedback(feedback)

        # Its weight should increase
        assert blend.weights[TraversalStrategy.DEPTH_FIRST] > 0.5


# =============================================================================
# STORY: v2.0 - Forgetting and Consolidation
# =============================================================================


class TestForgettingAndConsolidation:
    """
    Epic: Memory Management

    As a cognitive system with finite storage,
    Forgetting is not failure - it's curation.
    Consolidation strengthens important memories.
    Pruning removes noise.
    The system must forget to remain coherent.
    """

    def test_scenario_unused_knowledge_decays(self):
        """
        Scenario: Unaccessed knowledge weakens over time

        Given knowledge in the cognitive graph
        When it is not accessed for extended time
        Then its activation decays
        And it may be pruned
        Because storage is finite
        """
        from cortical.cognitive.graph import CognitiveGraph, TruthValue
        from cortical.cognitive.architecture import MemoryConsolidator

        graph = CognitiveGraph()
        node = graph.node("obscure_fact")
        node.lti = 0.1  # Low long-term importance

        consolidator = MemoryConsolidator(graph)

        # Simulate time passing without access
        for _ in range(10):
            consolidator.decay_cycle()

        # Node should be marked for pruning
        prunable = consolidator.get_prunable_atoms(threshold=0.05)
        assert node.id in [a.id for a in prunable]

    def test_scenario_frequently_accessed_strengthens(self):
        """
        Scenario: Repeated access consolidates memory

        Given knowledge that is frequently accessed
        When consolidation runs
        Then its long-term importance increases
        And it becomes resistant to decay
        Because use indicates value
        """
        from cortical.cognitive.graph import CognitiveGraph
        from cortical.cognitive.architecture import MemoryConsolidator

        graph = CognitiveGraph()
        node = graph.node("important_fact")
        node.lti = 0.3
        node.sti = 0.8  # Recently accessed frequently

        consolidator = MemoryConsolidator(graph)

        # High STI should boost LTI during consolidation
        consolidator.consolidation_cycle()

        assert node.lti > 0.3  # LTI increased

    def test_scenario_consolidation_prunes_contradictions(self):
        """
        Scenario: Consolidation resolves contradictions

        Given contradictory beliefs in the graph
        When consolidation runs
        Then weaker belief is weakened further
        And coherence increases
        Because the mind seeks consistency
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType, TruthValue
        from cortical.cognitive.architecture import MemoryConsolidator

        graph = CognitiveGraph()

        # Contradictory beliefs
        a = graph.node("A")
        not_a = graph.node("not_A")

        belief1 = graph.link(AtomType.EVALUATION, [a], tv=TruthValue(0.8, 0.6))
        belief2 = graph.link(AtomType.EVALUATION, [not_a], tv=TruthValue(0.3, 0.4))

        consolidator = MemoryConsolidator(graph)
        consolidator.resolve_contradictions()

        # Weaker belief should be weakened
        assert belief2.tv.strength < 0.3 or belief2.tv.confidence < 0.4


# =============================================================================
# STORY: v2.0 - Desire Dynamics (Enhanced)
# =============================================================================


class TestDesireDynamicsEnhanced:
    """
    Epic: The Complexity of Wanting

    As a cognitive system driven by desires,
    Desires are not simple scalars.
    They intensify, decay, conflict, and compound.
    The landscape of desire shapes all cognition.
    """

    def test_scenario_curiosity_intensifies_with_knowledge(self):
        """
        Scenario: Learning increases desire to learn more

        Given a desire with intensification rate
        When partially satisfied
        Then intensity INCREASES (not just satisfaction)
        Because the more you know, the more you want to know
        """
        from cortical.cognitive.architecture import Desire

        curiosity = Desire(
            description="understand quantum mechanics",
            intensity=0.5,
            satisfaction=0.0,
            intensification_rate=0.3  # Learning intensifies curiosity
        )

        initial_intensity = curiosity.intensity

        # Learn something (partial satisfaction)
        curiosity.partially_satisfy(0.2)

        # Intensity increased, not decreased
        assert curiosity.intensity > initial_intensity
        assert curiosity.satisfaction > 0

    def test_scenario_desires_conflict(self):
        """
        Scenario: Conflicting desires create tension

        Given two incompatible desires
        When both are urgent
        Then high tension exists
        And resolution is needed
        Because you can't satisfy both
        """
        from cortical.cognitive.architecture import Desire, DesireConflictResolver

        # Want to work AND want to rest
        work = Desire(
            id="work",
            description="finish project",
            intensity=0.8,
            satisfaction=0.0,
            conflicts_with=["rest"]
        )

        rest = Desire(
            id="rest",
            description="take a break",
            intensity=0.7,
            satisfaction=0.0,
            conflicts_with=["work"]
        )

        tension = work.check_conflict(rest)

        assert tension > 0.5  # High tension

        # Need to resolve
        resolver = DesireConflictResolver()
        resolution = resolver.resolve(work, rest)

        # One should be temporarily suppressed
        assert resolution.suppressed_desire in ["work", "rest"]

    def test_scenario_desire_decay_without_reinforcement(self):
        """
        Scenario: Desires fade without reinforcement

        Given a desire that hasn't been pursued
        When time passes
        Then urgency decays
        Because abandoned wants fade
        """
        from cortical.cognitive.architecture import Desire
        from datetime import datetime, timedelta

        desire = Desire(
            description="learn guitar",
            intensity=0.8,
            satisfaction=0.1,
            decay_rate=0.1,
            last_activated=datetime.now() - timedelta(hours=24)
        )

        # Urgency should be reduced by decay
        # Base urgency = 0.8 * (1 - 0.1) = 0.72
        # With 24 hours decay at 0.1/hour...
        assert desire.urgency < 0.72

    def test_scenario_terminal_vs_instrumental_desires(self):
        """
        Scenario: Terminal desires are ends, instrumental are means

        Given a terminal desire (wanted for itself)
        And instrumental desires (wanted as means)
        When the terminal is satisfied
        Then instrumentals become obsolete
        Because means only matter for their ends
        """
        from cortical.cognitive.architecture import (
            Desire, DesireHierarchy
        )

        # Terminal: Be healthy (wanted for itself)
        health = Desire(
            id="health",
            description="be healthy",
            is_terminal=True,
            intensity=0.9
        )

        # Instrumental: Exercise (means to health)
        exercise = Desire(
            id="exercise",
            description="exercise regularly",
            is_terminal=False,
            parent_desire="health",
            intensity=0.7
        )

        hierarchy = DesireHierarchy()
        hierarchy.add(health)
        hierarchy.add(exercise)

        # If health is satisfied by other means
        health.satisfaction = 0.95
        health.state = DesireState.SATISFIED

        # Exercise becomes less urgent
        hierarchy.propagate_satisfaction()

        # Instrumental desire should be reduced
        assert exercise.urgency < 0.7 * 0.5


# =============================================================================
# STORY: v2.0 - Affect and Cognition
# =============================================================================


class TestAffectAndCognition:
    """
    Epic: The Emotional Coloring of Thought

    As a cognitive system with affect,
    Emotions are not separate from cognition.
    Affect shapes attention, memory, and strategy.
    Different affective states process differently.
    Ignoring affect is ignoring computation.
    """

    def test_scenario_curiosity_broadens_attention(self):
        """
        Scenario: Curiosity widens the attentional spotlight

        Given a curious affective state
        When exploring the cognitive graph
        Then more nodes are considered
        And unusual connections are followed
        Because curiosity is approach-oriented
        """
        from cortical.cognitive.architecture import (
            AffectState, AttentionController, CognitiveExplorer
        )

        controller = AttentionController()
        controller.set_affect(AffectState.CURIOUS)

        explorer = CognitiveExplorer(attention=controller)

        # Curious exploration is broader
        curious_scope = explorer.get_attention_scope()

        controller.set_affect(AffectState.ANXIOUS)
        anxious_scope = explorer.get_attention_scope()

        assert curious_scope > anxious_scope

    def test_scenario_anxiety_narrows_attention(self):
        """
        Scenario: Anxiety focuses attention on threats

        Given an anxious affective state
        When processing information
        Then attention narrows to threat-relevant items
        And peripheral information is ignored
        Because anxiety is about survival
        """
        from cortical.cognitive.architecture import (
            AffectState, AttentionController
        )

        controller = AttentionController()
        controller.set_affect(AffectState.ANXIOUS)

        # Anxiety biases toward threats
        threat_weight = controller.get_category_weight("threat")
        neutral_weight = controller.get_category_weight("neutral")

        assert threat_weight > neutral_weight

    def test_scenario_flow_state_optimizes_processing(self):
        """
        Scenario: Flow state maximizes cognitive efficiency

        Given flow state (challenge matches skill)
        When processing tasks
        Then efficiency is maximized
        And fatigue accumulates slowly
        Because flow is the optimal state
        """
        from cortical.cognitive.architecture import (
            AffectState, CognitiveEfficiency
        )

        efficiency = CognitiveEfficiency()

        flow_efficiency = efficiency.get_rate(AffectState.FLOW)
        bored_efficiency = efficiency.get_rate(AffectState.BORED)
        anxious_efficiency = efficiency.get_rate(AffectState.ANXIOUS)

        assert flow_efficiency > bored_efficiency
        assert flow_efficiency > anxious_efficiency

    def test_scenario_frustration_triggers_strategy_change(self):
        """
        Scenario: Frustration signals need for new approach

        Given repeated failures on a task
        When frustration builds
        Then strategy change is triggered
        Because what we're doing isn't working
        """
        from cortical.cognitive.architecture import (
            AffectState, AffectMonitor, StrategyRecommender
        )

        monitor = AffectMonitor()

        # Simulate repeated failures
        for _ in range(5):
            monitor.record_failure()

        assert monitor.current_affect == AffectState.FRUSTRATED

        recommender = StrategyRecommender(affect_monitor=monitor)
        should_change = recommender.should_change_strategy()

        assert should_change


# =============================================================================
# SUMMARY: The Architecture of Mind (v2.0)
# =============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  THE FACTORY OF MIND v2.0: A Complete Cognitive Architecture               │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  NEW IN v2.0:                                                              │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ATTENTION DYNAMICS                                                  │   │
│  │                                                                      │   │
│  │   FOCUSED ◄──────────────────────────────────► DIFFUSE              │   │
│  │     │                                              │                 │   │
│  │     │  Deep, narrow                    Wide, associative            │   │
│  │     │  Analytical                      Creative                     │   │
│  │     │  Fatiguing                       Restorative                  │   │
│  │     │                                                               │   │
│  │     └──────────── SWITCHING COST ─────────────────┘                 │   │
│  │                   (momentum, fatigue)                               │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SURPRISE-DRIVEN LEARNING                                           │   │
│  │                                                                      │   │
│  │   Prediction ──compare──► Observation                               │   │
│  │       │                        │                                    │   │
│  │       └───────┬────────────────┘                                    │   │
│  │               │                                                      │   │
│  │               ▼                                                      │   │
│  │   ┌─────────────────────┐                                           │   │
│  │   │     SURPRISE        │  magnitude = |expected - observed|       │   │
│  │   │                     │  valence = observed - expected           │   │
│  │   └─────────────────────┘                                           │   │
│  │               │                                                      │   │
│  │       ┌───────┴───────┐                                             │   │
│  │       │               │                                              │   │
│  │       ▼               ▼                                              │   │
│  │   CAPTURE          UPDATE                                           │   │
│  │   ATTENTION        BELIEFS                                          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  WORKING MEMORY (~4 CHUNKS)                                         │   │
│  │                                                                      │   │
│  │   ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐                       │   │
│  │   │ Chunk │  │ Chunk │  │ Chunk │  │ Chunk │  ← MAX CAPACITY       │   │
│  │   │   1   │  │   2   │  │   3   │  │   4   │                       │   │
│  │   │ ████░ │  │ ███░░ │  │ ██░░░ │  │ █░░░░ │  ← Activation decay  │   │
│  │   └───────┘  └───────┘  └───────┘  └───────┘                       │   │
│  │                                                                      │   │
│  │   • Decay without rehearsal                                         │   │
│  │   • Eviction when full                                              │   │
│  │   • Chunking compresses information                                 │   │
│  │   • Constraint forces abstraction                                   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ANALOGICAL REASONING                                                │   │
│  │                                                                      │   │
│  │   SOURCE DOMAIN              TARGET DOMAIN                          │   │
│  │   ┌─────────────┐            ┌─────────────┐                        │   │
│  │   │   water     │──────────►│   electrons │                        │   │
│  │   │   pipe      │──────────►│   wire      │                        │   │
│  │   │   pressure  │──────────►│   voltage   │                        │   │
│  │   │   flow      │──────────►│   current   │                        │   │
│  │   └─────────────┘            └─────────────┘                        │   │
│  │         │                          │                                 │   │
│  │         │ STRUCTURAL MAPPING       │                                 │   │
│  │         └──────────────────────────┘                                 │   │
│  │                                                                      │   │
│  │   Inference Transfer:                                               │   │
│  │   "high pressure → fast flow" ══► "high voltage → high current"    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STRATEGY BLENDING                                                   │   │
│  │                                                                      │   │
│  │   Not: "Use depth-first OR breadth-first"                           │   │
│  │   But: "Use 60% depth + 30% associative + 10% random"               │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │░░░░░░░░░░░░░░░░░░░░░░│▓▓▓▓▓▓▓▓▓▓▓│████│                     │   │   │
│  │   │      DEPTH           │ASSOCIATIVE│RAND│                     │   │   │
│  │   │      60%             │   30%     │10% │                     │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │   Blends adapt: feedback reinforces successful strategies          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DESIRE DYNAMICS                                                     │   │
│  │                                                                      │   │
│  │   Intensification: Learning can INCREASE desire (curiosity)        │   │
│  │                    satisfy(0.2) → intensity += 0.2 × rate          │   │
│  │                                                                      │   │
│  │   Conflict: Incompatible desires create tension                     │   │
│  │             tension = urgency₁ × urgency₂                           │   │
│  │                                                                      │   │
│  │   Decay: Desires fade without reinforcement                         │   │
│  │          urgency *= (1 - hours × decay_rate)                        │   │
│  │                                                                      │   │
│  │   Hierarchy: Terminal (ends) vs Instrumental (means)               │   │
│  │              Instrumentals obsolete when terminal satisfied        │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  AFFECT STATES                                                       │   │
│  │                                                                      │   │
│  │   CURIOUS ───► Broad attention, exploration                         │   │
│  │   ANXIOUS ───► Narrow attention, threat focus                       │   │
│  │   FLOW    ───► Optimal efficiency, low fatigue                      │   │
│  │   BORED   ───► Seek novelty, disengage                              │   │
│  │   FRUSTRATED ► Change strategy, escalate                            │   │
│  │                                                                      │   │
│  │   Affect is not separate from cognition - it IS cognition          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FORGETTING AND CONSOLIDATION                                        │   │
│  │                                                                      │   │
│  │   Decay:         Unused knowledge weakens                           │   │
│  │   Consolidation: Frequent access strengthens (STI → LTI)           │   │
│  │   Pruning:       Remove low-importance atoms                        │   │
│  │   Resolution:    Weaken contradictory beliefs                       │   │
│  │                                                                      │   │
│  │   Forgetting is not failure - it's curation                         │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  THE COMPLETE COGNITIVE CYCLE (v2.0):                                      │
│                                                                             │
│                    ┌─────────────────┐                                     │
│                    │     DESIRE      │                                     │
│                    │  (intensifies,  │                                     │
│                    │   decays,       │                                     │
│                    │   conflicts)    │                                     │
│                    └────────┬────────┘                                     │
│                             │                                              │
│                             ▼                                              │
│   ┌─────────────┐   ┌─────────────────┐   ┌─────────────┐                 │
│   │   AFFECT    │◄──│    ATTENTION    │──►│  SURPRISE   │                 │
│   │ (colors all │   │ (focused/diffuse│   │ (captures,  │                 │
│   │  processing)│   │  fatigues,      │   │  updates)   │                 │
│   └─────────────┘   │  switches)      │   └──────┬──────┘                 │
│                     └────────┬────────┘          │                        │
│                              │                   │                        │
│                              ▼                   ▼                        │
│                     ┌─────────────────┐   ┌─────────────┐                 │
│                     │    THOUGHT      │   │  LEARNING   │                 │
│                     │ (built, blended │   │  (beliefs   │                 │
│                     │  strategies)    │   │   updated)  │                 │
│                     └────────┬────────┘   └─────────────┘                 │
│                              │                                            │
│                              ▼                                            │
│                     ┌─────────────────┐                                   │
│                     │ WORKING MEMORY  │                                   │
│                     │  (~4 chunks,    │                                   │
│                     │   decay, evict) │                                   │
│                     └────────┬────────┘                                   │
│                              │                                            │
│                              ▼                                            │
│   ┌─────────────┐   ┌─────────────────┐   ┌─────────────┐                 │
│   │   ANALOGY   │◄──│   KNOWLEDGE     │──►│   FORGET/   │                 │
│   │ (structure  │   │   (graph,       │   │ CONSOLIDATE │                 │
│   │  mapping)   │   │    links)       │   │  (prune)    │                 │
│   └─────────────┘   └────────┬────────┘   └─────────────┘                 │
│                              │                                            │
│                              ▼                                            │
│                    ┌─────────────────┐                                    │
│                    │  SATISFACTION   │───────────────────┐                │
│                    │  (may intensify │                   │                │
│                    │   desire!)      │                   │                │
│                    └─────────────────┘                   │                │
│                              │                           │                │
│                              └───────────────────────────┘                │
│                                                                           │
│  ═══════════════════════════════════════════════════════════════════════  │
│                                                                           │
│  "Cognition is not computation on static data.                           │
│   It is a dynamic, affective, resource-bounded process                   │
│   of desire-driven exploration, surprise-guided learning,                │
│   and analogical understanding - with forgetting as feature."            │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
"""


# =============================================================================
# ORIGINAL SUMMARY (v1.0) - PRESERVED FOR REFERENCE
# =============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  THE FACTORY OF MIND: A Complete Cognitive Architecture                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         DESIRE LAYER                                 │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                        │   │
│  │  │ Desire A  │  │ Desire B  │  │ Desire C  │  ← What we WANT        │   │
│  │  │ urgency:  │  │ urgency:  │  │ urgency:  │                        │   │
│  │  │   0.81    │  │   0.30    │  │   0.06    │                        │   │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                        │   │
│  │        │              │              │                               │   │
│  │        └──────────────┼──────────────┘                               │   │
│  │                       ▼                                              │   │
│  │              ┌─────────────────┐                                     │   │
│  │              │ ATTENTION       │  ← Economic allocation              │   │
│  │              │ ALLOCATOR       │    (urgency determines share)       │   │
│  │              └────────┬────────┘                                     │   │
│  └───────────────────────┼─────────────────────────────────────────────┘   │
│                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       THOUGHT LAYER                                  │   │
│  │                                                                      │   │
│  │   ┌────────────────────────────────────────────────────────────┐    │   │
│  │   │                   THOUGHT BUILDER                          │    │   │
│  │   │   .with_purpose()                                          │    │   │
│  │   │   .with_graph()         ← Fluent construction              │    │   │
│  │   │   .with_strategy()        with DI                          │    │   │
│  │   │   .serving_desire()                                        │    │   │
│  │   │   .build()                                                 │    │   │
│  │   └────────────────────────────────────────────────────────────┘    │   │
│  │                          │                                          │   │
│  │                          ▼                                          │   │
│  │   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐       │   │
│  │   │ Thought 1 │  │ Thought 2 │  │ Thought 3 │  │ Meta-     │       │   │
│  │   │ (research)│  │ (analyze) │  │(synthesize│  │ Thought   │       │   │
│  │   │           │  │           │  │           │  │ (observe) │       │   │
│  │   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘       │   │
│  │         │              │              │              │              │   │
│  └─────────┼──────────────┼──────────────┼──────────────┼──────────────┘   │
│            │              │              │              │                   │
│            └──────────────┼──────────────┼──────────────┘                   │
│                           ▼              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      STRATEGY LAYER                                  │   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │   │ DEPTH-FIRST │  │BREADTH-FIRST│  │  DESIRE-    │                 │   │
│  │   │   VISITOR   │  │   VISITOR   │  │  GRADIENT   │                 │   │
│  │   │             │  │             │  │   VISITOR   │                 │   │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │   │
│  │          │                │                │                         │   │
│  │          └────────────────┼────────────────┘                         │   │
│  │                           ▼                                          │   │
│  │               ┌───────────────────────┐                              │   │
│  │               │   ADAPTIVE VISITOR    │  ← Switches strategy         │   │
│  │               │   (meta-strategy)     │    when stalling             │   │
│  │               └───────────────────────┘                              │   │
│  │                           │                                          │   │
│  └───────────────────────────┼──────────────────────────────────────────┘   │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    KNOWLEDGE LAYER                                   │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                  COGNITIVE GRAPH                             │   │   │
│  │   │                                                              │   │   │
│  │   │   [cat] ──IS-A──▶ [mammal] ──IS-A──▶ [animal]               │   │   │
│  │   │                       │                                      │   │   │
│  │   │                       │ (link)                               │   │   │
│  │   │                       ▼                                      │   │   │
│  │   │   [john] ──BELIEVES──▶ [cat-mammal-link]  ← Link TO link    │   │   │
│  │   │                                                              │   │   │
│  │   │   [inheritance] ──HAS-PROPERTY──▶ [transitivity]  ← Meta    │   │   │
│  │   │                                                              │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    WORKER LAYER                                      │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                    ORCHESTRATOR                              │   │   │
│  │   │   - Decomposes complex work                                  │   │   │
│  │   │   - Assigns to specialized workers                          │   │   │
│  │   │   - Gathers and synthesizes results                         │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                           │                                          │   │
│  │           ┌───────────────┼───────────────┐                          │   │
│  │           ▼               ▼               ▼                          │   │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                   │   │
│  │   │  Worker A   │ │  Worker B   │ │  Worker C   │                   │   │
│  │   │ (explorer)  │ │ (analyzer)  │ │(synthesizer)│                   │   │
│  │   │             │ │             │ │             │                   │   │
│  │   │ specialty:  │ │ specialty:  │ │ specialty:  │                   │   │
│  │   │ depth-first │ │ patterns    │ │ synthesis   │                   │   │
│  │   └─────────────┘ └─────────────┘ └─────────────┘                   │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  RESOURCE BOUNDARIES:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐         │   │
│  │   │   DEPTH     │      │  RESOURCE   │      │  GRADIENT   │         │   │
│  │   │   BUDGET    │ ───▶ │   BUDGET    │ ───▶ │   DESCENT   │         │   │
│  │   │ (grows on   │      │ (allocated  │      │ (stops when │         │   │
│  │   │  demand)    │      │  by value)  │      │  flat)      │         │   │
│  │   └─────────────┘      └─────────────┘      └─────────────┘         │   │
│  │                                                                      │   │
│  │   INFINITE KNOWLEDGE + FINITE RESOURCES = GRADIENT-GUIDED DEPTH     │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  DEPENDENCY INJECTION:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │   Container                                                          │   │
│  │   ├── register(CognitiveGraph)                                      │   │
│  │   ├── register(DesireManager)                                       │   │
│  │   ├── register(Orchestrator)                                        │   │
│  │   ├── register(WorkerPool)                                          │   │
│  │   └── register(AttentionAllocator)                                  │   │
│  │                                                                      │   │
│  │   ThoughtBuilder(container=container).auto_wire().build()           │   │
│  │   └── All dependencies resolved automatically                       │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  THE COGNITIVE CYCLE:                                                       │
│                                                                             │
│   DESIRE ──▶ ATTENTION ──▶ THOUGHT ──▶ STRATEGY ──▶ KNOWLEDGE ──┐          │
│      ▲                                                           │          │
│      └──────────────── SATISFACTION ◀────────────────────────────┘          │
│                                                                             │
│  "Understanding emerges from the coordinated dance of desire,               │
│   attention, strategy, and knowledge—each component essential,              │
│   none sufficient alone."                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
