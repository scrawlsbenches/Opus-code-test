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
    """
    id: str = field(default_factory=lambda: f"D-{uuid.uuid4().hex[:6]}")
    description: str = ""
    intensity: float = 0.5      # How much we want this (0-1)
    satisfaction: float = 0.0   # How satisfied we are (0-1)
    state: DesireState = DesireState.LATENT
    spawned_thoughts: List[str] = field(default_factory=list)
    parent_desire: Optional[str] = None  # Desires can derive from desires

    @property
    def urgency(self) -> float:
        """Urgency = intensity * (1 - satisfaction)."""
        return self.intensity * (1.0 - self.satisfaction)

    def partially_satisfy(self, amount: float) -> None:
        """Increase satisfaction (with diminishing returns)."""
        remaining = 1.0 - self.satisfaction
        self.satisfaction += remaining * amount
        if self.satisfaction > 0.95:
            self.state = DesireState.SATISFIED


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
# SUMMARY: The Architecture of Mind
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
