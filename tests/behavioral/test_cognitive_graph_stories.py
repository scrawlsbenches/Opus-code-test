"""
Behavioral tests for Cognitive Graph (Bio-Inspired Hypergraph).

As an AI system learning through interaction,
I want a hypergraph where links can point to links,
So that I can reason about relationships, not just entities.

As a developer building cognitive systems,
I want dependency injection and in-memory testing,
So that I can verify behavior without side effects.

Core Principles:
1. Links are first-class atoms (can be linked to)
2. Truth is probabilistic (strength, confidence)
3. Attention is finite (STI decays, spreads)
4. Identity emerges from consistent patterns

This test suite verifies:
- Atom creation (nodes and links)
- Hypergraph property (links to links)
- Truth value propagation
- Attention dynamics
- Container integration for testing
"""

import pytest
from typing import Protocol, List, Optional
from dataclasses import dataclass


# =============================================================================
# STORY: Basic Atom Creation
# =============================================================================


class TestAtomCreation:
    """
    Epic: Fundamental Building Blocks

    As a cognitive system,
    I want to create atoms (nodes and links) with truth values,
    So that I can represent knowledge with uncertainty.
    """

    def test_scenario_create_concept_node(self):
        """
        Scenario: Create a concept node

        Given a cognitive graph
        When I create a concept node "cat"
        Then I get an atom with name "cat"
        And it has default truth value (1.0, 0.0)
        Because new concepts start with full strength, no confidence
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType

        graph = CognitiveGraph()
        cat = graph.node("cat")

        assert cat is not None
        assert cat.name == "cat"
        assert cat.atom_type == AtomType.CONCEPT
        assert cat.tv.strength == 1.0
        assert cat.tv.confidence == 0.0

    def test_scenario_create_node_with_truth_value(self):
        """
        Scenario: Create a node with explicit truth value

        Given a cognitive graph
        When I create a concept with truth value (0.9, 0.8)
        Then the atom has that truth value
        Because we can specify our confidence in concepts
        """
        from cortical.cognitive.graph import CognitiveGraph, TruthValue

        graph = CognitiveGraph()
        cat = graph.node("cat", tv=TruthValue(0.9, 0.8))

        assert cat.tv.strength == 0.9
        assert cat.tv.confidence == 0.8

    def test_scenario_nodes_are_content_addressed(self):
        """
        Scenario: Same name returns same node

        Given a cognitive graph with node "cat"
        When I request node "cat" again
        Then I get the same atom instance
        Because atoms are content-addressed (same content = same atom)
        """
        from cortical.cognitive.graph import CognitiveGraph

        graph = CognitiveGraph()
        cat1 = graph.node("cat")
        cat2 = graph.node("cat")

        assert cat1 is cat2
        assert cat1.id == cat2.id


# =============================================================================
# STORY: Links Between Atoms
# =============================================================================


class TestLinkCreation:
    """
    Epic: Relationships as First-Class Objects

    As a cognitive system,
    I want to create links between atoms,
    So that I can represent relationships.
    """

    def test_scenario_create_inheritance_link(self):
        """
        Scenario: Create an inheritance link (IS-A)

        Given nodes "cat" and "animal"
        When I create an inheritance link from cat to animal
        Then I get a link atom connecting them
        And the link has its own truth value
        Because relationships are uncertain too
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType, TruthValue

        graph = CognitiveGraph()
        cat = graph.node("cat")
        animal = graph.node("animal")

        link = graph.link(
            AtomType.INHERITANCE,
            [cat, animal],
            tv=TruthValue(0.99, 0.95)
        )

        assert link is not None
        assert link.atom_type == AtomType.INHERITANCE
        assert link.is_link()
        assert len(link.outgoing) == 2
        assert link.tv.strength == 0.99
        assert link.tv.confidence == 0.95

    def test_scenario_links_are_content_addressed(self):
        """
        Scenario: Same link content returns same link

        Given an inheritance link cat->animal
        When I create the same link again
        Then I get the same atom
        Because links are content-addressed by type + outgoing
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType, TruthValue

        graph = CognitiveGraph()
        cat = graph.node("cat")
        animal = graph.node("animal")

        link1 = graph.link(AtomType.INHERITANCE, [cat, animal], TruthValue(0.9, 0.8))
        link2 = graph.link(AtomType.INHERITANCE, [cat, animal], TruthValue(0.8, 0.7))

        # Same link (content-addressed)
        assert link1 is link2

        # Truth values should be merged (more evidence)
        assert link1.tv.confidence > 0.8  # Confidence increased


# =============================================================================
# STORY: Links About Links (The Key Hypergraph Property)
# =============================================================================


class TestLinksAboutLinks:
    """
    Epic: Meta-Reasoning

    As a cognitive system,
    I want to create links that point to other links,
    So that I can reason about relationships themselves.

    THIS IS THE KEY INSIGHT:
    In a regular graph, you can only link nodes.
    In a hypergraph, links are atoms, so you can link TO links.
    """

    def test_scenario_belief_about_link(self):
        """
        Scenario: Create a belief about a relationship

        Given a link "cat IS-A animal"
        When I create a BELIEVES link from "john" to that link
        Then john's belief points directly to the relationship
        Because links are first-class atoms that can be referenced
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType, TruthValue

        graph = CognitiveGraph()

        # Create the base relationship
        cat = graph.node("cat")
        animal = graph.node("animal")
        cat_is_animal = graph.link(
            AtomType.INHERITANCE,
            [cat, animal],
            TruthValue(0.99, 0.9)
        )

        # Create a belief ABOUT that relationship
        john = graph.node("john", atom_type=AtomType.PERSON)
        john_believes = graph.link(
            AtomType.BELIEVES,
            [john, cat_is_animal],  # Second argument is a LINK, not a node
            TruthValue(1.0, 0.95)
        )

        assert john_believes is not None
        assert john_believes.is_link()
        # The second outgoing atom IS the original link
        assert john_believes.outgoing[1] == cat_is_animal.id

    def test_scenario_link_between_two_links(self):
        """
        Scenario: Create implication between two links

        Given link1 "bird IS-A animal"
        And link2 "tweety IS-A animal"
        When I create an IMPLIES link from link1 to link2
        Then I can express "if birds are animals, tweety is an animal"
        Because meta-reasoning requires linking links to links
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType, TruthValue

        graph = CognitiveGraph()

        # Base nodes
        bird = graph.node("bird")
        animal = graph.node("animal")
        tweety = graph.node("tweety")

        # Two inheritance links
        bird_is_animal = graph.link(AtomType.INHERITANCE, [bird, animal], TruthValue(1.0, 0.99))
        tweety_is_animal = graph.link(AtomType.INHERITANCE, [tweety, animal], TruthValue(0.95, 0.8))

        # Implication BETWEEN the links
        implication = graph.link(
            AtomType.IMPLIES,
            [bird_is_animal, tweety_is_animal],  # Both are LINKS
            TruthValue(0.9, 0.85)
        )

        assert implication is not None
        # Both outgoing atoms are links
        out1 = graph.get_atom(implication.outgoing[0])
        out2 = graph.get_atom(implication.outgoing[1])
        assert out1.is_link()
        assert out2.is_link()

    def test_scenario_evidence_chain(self):
        """
        Scenario: Build an evidence chain

        Given observation "tweety has feathers"
        And conclusion "tweety is a bird"
        When I link them with EVIDENCE_FOR
        Then I can track WHY I believe things
        Because justification requires meta-links
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType, TruthValue

        graph = CognitiveGraph()

        # Observation
        tweety = graph.node("tweety")
        has_feathers = graph.node("has_feathers")
        observation = graph.link(
            AtomType.INHERITANCE,
            [tweety, has_feathers],
            TruthValue(0.95, 0.99)  # High confidence observation
        )

        # Conclusion
        bird = graph.node("bird")
        conclusion = graph.link(
            AtomType.INHERITANCE,
            [tweety, bird],
            TruthValue(0.9, 0.8)
        )

        # Evidence relationship
        evidence = graph.link(
            AtomType.EVIDENCE_FOR,
            [observation, conclusion],  # Link pointing to two links
            TruthValue(0.85, 0.9)
        )

        assert evidence is not None
        # Can traverse the evidence chain
        obs_atom = graph.get_atom(evidence.outgoing[0])
        conc_atom = graph.get_atom(evidence.outgoing[1])
        assert obs_atom is observation
        assert conc_atom is conclusion


# =============================================================================
# STORY: Truth Value Operations
# =============================================================================


class TestTruthValueOperations:
    """
    Epic: Probabilistic Truth

    As a cognitive system reasoning under uncertainty,
    I want truth values that combine and propagate properly,
    So that my confidence reflects my evidence.
    """

    def test_scenario_truth_value_merge(self):
        """
        Scenario: Merging evidence increases confidence

        Given two observations with separate confidence
        When they are merged
        Then the combined confidence is higher than either alone
        Because more evidence = more confidence
        """
        from cortical.cognitive.graph import TruthValue

        tv1 = TruthValue(0.9, 0.5)  # Moderate confidence
        tv2 = TruthValue(0.85, 0.6)  # Moderate confidence

        merged = tv1.merge(tv2)

        # Confidence should increase
        assert merged.confidence > tv1.confidence
        assert merged.confidence > tv2.confidence

        # Strength is weighted average
        assert 0.85 <= merged.strength <= 0.9

    def test_scenario_truth_value_bounds(self):
        """
        Scenario: Truth values stay in valid range

        Given any operations on truth values
        When we compute results
        Then strength and confidence stay in [0, 1]
        Because probabilities must be valid
        """
        from cortical.cognitive.graph import TruthValue

        # Edge cases
        tv_high = TruthValue(1.5, 1.5)  # Above bounds
        assert tv_high.strength == 1.0
        assert tv_high.confidence == 1.0

        tv_low = TruthValue(-0.5, -0.5)  # Below bounds
        assert tv_low.strength == 0.0
        assert tv_low.confidence == 0.0


# =============================================================================
# STORY: Attention Dynamics
# =============================================================================


class TestAttentionDynamics:
    """
    Epic: Finite Attention

    As a cognitive system with limited resources,
    I want attention that decays and spreads,
    So that I focus on what matters.
    """

    def test_scenario_stimulate_atom(self):
        """
        Scenario: Stimulating an atom increases its attention

        Given a node with zero attention
        When I stimulate it
        Then its STI (short-term importance) increases
        Because stimulation directs attention
        """
        from cortical.cognitive.graph import CognitiveGraph

        graph = CognitiveGraph()
        cat = graph.node("cat")

        initial_sti = cat.sti
        graph.stimulate("cat", 10.0)

        assert cat.sti > initial_sti
        assert cat.sti == initial_sti + 10.0

    def test_scenario_attention_decays(self):
        """
        Scenario: Attention decays over time

        Given atoms with positive STI
        When a cognitive step occurs
        Then all STI values decay
        Because attention is transient
        """
        from cortical.cognitive.graph import CognitiveGraph

        graph = CognitiveGraph()
        cat = graph.node("cat")
        graph.stimulate("cat", 10.0)

        initial_sti = cat.sti
        graph.step()

        assert cat.sti < initial_sti
        assert cat.sti > 0  # Not instant zero

    def test_scenario_attention_spreads(self):
        """
        Scenario: Attention spreads through links

        Given node A with high STI linked to node B
        When a cognitive step occurs
        Then some attention spreads to B
        Because connected concepts prime each other
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType

        graph = CognitiveGraph()
        cat = graph.node("cat")
        animal = graph.node("animal")
        graph.link(AtomType.INHERITANCE, [cat, animal])

        # Stimulate cat
        graph.stimulate("cat", 20.0)
        initial_animal_sti = animal.sti

        # Step spreads attention
        graph.step()

        # Animal received some attention from cat
        assert animal.sti > initial_animal_sti


# =============================================================================
# STORY: Dependency Injection Integration
# =============================================================================


class TestContainerIntegration:
    """
    Epic: Testable Architecture

    As a developer testing cognitive systems,
    I want to inject dependencies and test in memory,
    So that tests are fast and isolated.
    """

    def test_scenario_resolve_graph_from_container(self):
        """
        Scenario: Resolve CognitiveGraph from container

        Given a container with cognitive module registered
        When I resolve CognitiveGraph
        Then I get a functional instance
        Because the container manages creation
        """
        from cortical.common import Container
        from cortical.cognitive.graph import CognitiveGraph, CognitiveGraphModule

        container = Container()
        container.apply_module(CognitiveGraphModule())

        graph = container.resolve(CognitiveGraph)

        assert graph is not None
        node = graph.node("test")
        assert node.name == "test"

    def test_scenario_child_container_for_test_isolation(self):
        """
        Scenario: Use child container for test isolation

        Given a parent container with shared configuration
        When I create a child container for testing
        Then each test gets isolated state
        Because child containers don't share instances
        """
        from cortical.common import Container, Lifecycle
        from cortical.cognitive.graph import CognitiveGraph, CognitiveGraphModule

        parent = Container()
        parent.apply_module(CognitiveGraphModule(lifecycle=Lifecycle.TRANSIENT))

        # Each child gets its own graph
        child1 = parent.create_child()
        child2 = parent.create_child()

        graph1 = child1.resolve(CognitiveGraph)
        graph2 = child2.resolve(CognitiveGraph)

        # Different instances
        assert graph1 is not graph2

        # Changes in one don't affect the other
        graph1.node("only_in_graph1")
        assert graph2.get_node("only_in_graph1") is None

    def test_scenario_inject_custom_storage_backend(self):
        """
        Scenario: Inject custom storage for testing

        Given a cognitive graph that uses a storage backend
        When I inject a mock storage
        Then the graph uses the mock
        Because dependencies are injected, not hardcoded
        """
        from cortical.common import Container
        from cortical.cognitive.graph import (
            CognitiveGraph,
            StorageBackend,
            InMemoryStorage,
        )

        # Custom mock storage
        class MockStorage(InMemoryStorage):
            def __init__(self):
                super().__init__()
                self.save_count = 0

            def save(self, atom):
                self.save_count += 1
                return super().save(atom)

        container = Container()
        mock_storage = MockStorage()
        container.register_instance(StorageBackend, mock_storage)
        # Use explicit dependency binding instead of auto-wire
        # because storage parameter has a default value
        container.register(
            CognitiveGraph,
            CognitiveGraph,
            storage=StorageBackend,
        )

        graph = container.resolve(CognitiveGraph)
        graph.node("test")

        # Our mock was used
        assert mock_storage.save_count > 0


# =============================================================================
# STORY: Query and Traversal
# =============================================================================


class TestGraphQueries:
    """
    Epic: Knowledge Retrieval

    As a cognitive system,
    I want to query and traverse the graph,
    So that I can find relevant knowledge.
    """

    def test_scenario_find_incoming_links(self):
        """
        Scenario: Find all links pointing to an atom

        Given multiple links to "animal"
        When I query incoming links
        Then I get all links where animal is a target
        Because I need to know what inherits from animal
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType

        graph = CognitiveGraph()
        animal = graph.node("animal")
        cat = graph.node("cat")
        dog = graph.node("dog")
        bird = graph.node("bird")

        graph.link(AtomType.INHERITANCE, [cat, animal])
        graph.link(AtomType.INHERITANCE, [dog, animal])
        graph.link(AtomType.INHERITANCE, [bird, animal])

        incoming = graph.get_incoming(animal.id)

        assert len(incoming) == 3
        sources = {graph.get_atom(link.outgoing[0]).name for link in incoming}
        assert sources == {"cat", "dog", "bird"}

    def test_scenario_find_by_type(self):
        """
        Scenario: Find all atoms of a type

        Given a graph with mixed atom types
        When I query by type INHERITANCE
        Then I get only inheritance links
        Because type-based queries enable pattern matching
        """
        from cortical.cognitive.graph import CognitiveGraph, AtomType

        graph = CognitiveGraph()
        cat = graph.node("cat")
        animal = graph.node("animal")
        john = graph.node("john", atom_type=AtomType.PERSON)

        cat_animal = graph.link(AtomType.INHERITANCE, [cat, animal])
        graph.link(AtomType.BELIEVES, [john, cat_animal])

        inheritance_links = graph.find_by_type(AtomType.INHERITANCE)

        assert len(inheritance_links) == 1
        assert inheritance_links[0] is cat_animal

    def test_scenario_get_attention_focus(self):
        """
        Scenario: Get atoms currently in attention

        Given atoms with varying STI values
        When I query the attention focus
        Then I get the top-K highest STI atoms
        Because I need to know what I'm focusing on
        """
        from cortical.cognitive.graph import CognitiveGraph

        graph = CognitiveGraph()
        cat = graph.node("cat")
        dog = graph.node("dog")
        bird = graph.node("bird")

        graph.stimulate("cat", 10.0)
        graph.stimulate("dog", 5.0)
        graph.stimulate("bird", 15.0)

        focus = graph.get_attention_focus(top_k=2)

        assert len(focus) == 2
        # Highest first
        assert focus[0].name == "bird"
        assert focus[1].name == "cat"


# =============================================================================
# STORY: Cognitive Processing
# =============================================================================


class TestCognitiveProcessing:
    """
    Epic: Thinking

    As a cognitive system,
    I want to process experiences and update my graph,
    So that I learn from interaction.
    """

    def test_scenario_process_experience(self):
        """
        Scenario: Process an experience and learn

        Given a cognitive graph
        When I process an experience with concepts
        Then new nodes and links are created
        And they are connected to existing knowledge
        Because processing builds the knowledge graph
        """
        from cortical.cognitive.graph import CognitiveGraph

        graph = CognitiveGraph()

        # Pre-existing knowledge
        graph.node("animal")

        # Process new experience
        result = graph.process_experience(
            concepts=["cat", "furry", "animal"],
            relations=[("cat", "is_a", "animal"), ("cat", "has", "furry")]
        )

        assert result["nodes_created"] >= 2  # cat, furry (animal existed)
        assert result["links_created"] >= 2  # two relations

        # Verify structure
        cat = graph.get_node("cat")
        assert cat is not None

    def test_scenario_observe_patterns(self):
        """
        Scenario: Observe patterns in experience

        Given multiple experiences over time
        When I observe my patterns
        Then I see consistent beliefs and behaviors
        Because self-observation enables meta-cognition
        """
        from cortical.cognitive.graph import CognitiveGraph

        graph = CognitiveGraph()

        # Multiple experiences
        for i in range(5):
            graph.process_experience(
                concepts=["cat", "animal"],
                relations=[("cat", "is_a", "animal")]
            )

        patterns = graph.observe_patterns()

        # Should notice the repeated pattern
        assert "cat" in patterns["frequent_concepts"]
        assert patterns["consistent_relations"]["cat_is_a_animal"] >= 5
