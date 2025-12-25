#!/usr/bin/env python3
"""
PRISM-GoT Demo: Predictive Reasoning through Incremental Synaptic Memory Graph of Thought

This demo showcases the biologically-inspired reasoning framework that combines:
1. Synaptic plasticity - connections strengthen/weaken based on usage
2. Incremental learning - graph structure built through experience
3. Predictive reasoning - anticipating likely next thoughts
4. Graph of Thought - network-based representation of reasoning

Run with: python examples/prism_got_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.reasoning import (
    NodeType,
    EdgeType,
    SynapticMemoryGraph,
    IncrementalReasoner,
    PlasticityRules,
)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_basic_synaptic_graph():
    """Demonstrate basic SynapticMemoryGraph operations."""
    print_section("1. BASIC SYNAPTIC MEMORY GRAPH")

    # Create a graph with custom plasticity rules
    rules = PlasticityRules(
        hebbian_rate=0.15,      # How much co-activation strengthens edges
        anti_hebbian_rate=0.05, # How much unused edges weaken
        reward_rate=0.25,       # How much reward affects edges
    )
    graph = SynapticMemoryGraph(plasticity_rules=rules)

    print("\nCreating a reasoning graph about authentication...")

    # Add nodes representing thoughts
    graph.add_node("Q1", NodeType.QUESTION, "How should we implement authentication?")
    graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens for stateless auth")
    graph.add_node("H2", NodeType.HYPOTHESIS, "Use session cookies for traditional auth")
    graph.add_node("E1", NodeType.EVIDENCE, "JWT allows horizontal scaling")
    graph.add_node("E2", NodeType.EVIDENCE, "Sessions require sticky sessions")
    graph.add_node("D1", NodeType.DECISION, "Implement JWT with refresh tokens")

    # Add synaptic edges with initial weights
    graph.add_synaptic_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.7)
    graph.add_synaptic_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.5)
    graph.add_synaptic_edge("H1", "E1", EdgeType.SUPPORTS, weight=0.8)
    graph.add_synaptic_edge("H2", "E2", EdgeType.SUPPORTS, weight=0.6)
    graph.add_synaptic_edge("E1", "D1", EdgeType.JUSTIFIES, weight=0.9)

    print(f"\nGraph created with {graph.node_count()} nodes and {len(graph.synaptic_edges)} synaptic edges")

    # Show initial edge weights
    print("\nInitial edge weights:")
    for (src, tgt, _), edge in graph.synaptic_edges.items():
        print(f"  {src} -> {tgt}: weight={edge.weight:.2f}")

    return graph


def demo_activation_and_learning(graph: SynapticMemoryGraph):
    """Demonstrate node activation and Hebbian learning."""
    print_section("2. ACTIVATION AND HEBBIAN LEARNING")

    print("\nSimulating a reasoning session...")
    print("(Activating nodes in sequence: Q1 -> H1 -> E1 -> D1)")

    # Activate nodes in a reasoning chain
    for node_id in ["Q1", "H1", "E1", "D1"]:
        graph.activate_node(node_id, context={"phase": "exploration"})
        trace = graph.activation_traces[node_id]
        print(f"  Activated {node_id}: total_activations={trace.total_activations}")

    print("\nApplying Hebbian learning (co-activation within 60s window)...")
    strengthened = graph.apply_hebbian_learning(time_window_seconds=60)
    print(f"  {strengthened} edges strengthened")

    print("\nEdge weights after Hebbian learning:")
    for (src, tgt, _), edge in graph.synaptic_edges.items():
        print(f"  {src} -> {tgt}: weight={edge.weight:.2f}, activations={edge.activation_count}")


def demo_prediction():
    """Demonstrate predictive reasoning."""
    print_section("3. PREDICTIVE REASONING")

    graph = SynapticMemoryGraph()

    # Build a graph with learned patterns
    graph.add_node("Q-auth", NodeType.QUESTION, "What auth method?")
    graph.add_node("H-jwt", NodeType.HYPOTHESIS, "Use JWT")
    graph.add_node("H-oauth", NodeType.HYPOTHESIS, "Use OAuth")
    graph.add_node("H-apikey", NodeType.HYPOTHESIS, "Use API Keys")

    # Create edges with different weights (simulating learned preferences)
    graph.add_synaptic_edge("Q-auth", "H-jwt", EdgeType.EXPLORES, weight=0.9)
    graph.add_synaptic_edge("Q-auth", "H-oauth", EdgeType.EXPLORES, weight=0.6)
    graph.add_synaptic_edge("Q-auth", "H-apikey", EdgeType.EXPLORES, weight=0.3)

    # Simulate some prediction history
    jwt_edge = graph.synaptic_edges[("Q-auth", "H-jwt", EdgeType.EXPLORES)]
    jwt_edge.record_prediction_outcome(correct=True)
    jwt_edge.record_prediction_outcome(correct=True)
    jwt_edge.record_prediction_outcome(correct=True)

    oauth_edge = graph.synaptic_edges[("Q-auth", "H-oauth", EdgeType.EXPLORES)]
    oauth_edge.record_prediction_outcome(correct=True)
    oauth_edge.record_prediction_outcome(correct=False)

    print("\nPredicting next thoughts from 'Q-auth'...")
    predictions = graph.predict_next_thoughts("Q-auth", top_n=3)

    print("\nTop predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred.node_id} ({pred.node.content})")
        print(f"     probability={pred.probability:.3f}, {pred.reasoning}")


def demo_incremental_reasoner():
    """Demonstrate the IncrementalReasoner for building graphs through experience."""
    print_section("4. INCREMENTAL REASONER")

    graph = SynapticMemoryGraph()
    reasoner = IncrementalReasoner(
        graph,
        auto_link_similar=True,
        similarity_threshold=0.5,
    )

    print("\nBuilding a reasoning graph incrementally...")

    # Process thoughts one at a time
    q1 = reasoner.process_thought(
        "How do we handle database migrations?",
        NodeType.QUESTION,
    )
    print(f"  1. Processed question: {q1.id}")

    h1 = reasoner.process_thought(
        "Use Alembic for SQLAlchemy migrations",
        NodeType.HYPOTHESIS,
        relation_to_focus=EdgeType.EXPLORES,
    )
    print(f"  2. Processed hypothesis: {h1.id}")

    e1 = reasoner.process_thought(
        "Team already uses SQLAlchemy ORM",
        NodeType.EVIDENCE,
        relation_to_focus=EdgeType.SUPPORTS,
    )
    print(f"  3. Processed evidence: {e1.id}")

    d1 = reasoner.process_thought(
        "Implement Alembic with auto-generation",
        NodeType.DECISION,
        relation_to_focus=EdgeType.JUSTIFIES,
    )
    print(f"  4. Processed decision: {d1.id}")

    # Mark the reasoning path as successful
    reasoner.mark_outcome_success(
        path=[q1.id, h1.id, e1.id, d1.id],
        reward=0.8,
    )
    print("\n  Marked path as successful (reward=0.8)")

    # Get summary
    summary = reasoner.get_summary()
    print(f"\nGraph summary:")
    print(f"  Total nodes: {summary['total_nodes']}")
    print(f"  Total edges: {summary['total_edges']}")
    print(f"  Nodes by type: {summary['nodes_by_type']}")
    print(f"  Average edge weight: {summary['average_edge_weight']:.2f}")

    return reasoner


def demo_decay_and_cleanup():
    """Demonstrate temporal decay of unused connections."""
    print_section("5. TEMPORAL DECAY")

    graph = SynapticMemoryGraph()

    # Create some edges with different decay rates
    graph.add_node("A", NodeType.CONCEPT, "Concept A")
    graph.add_node("B", NodeType.CONCEPT, "Concept B")
    graph.add_node("C", NodeType.CONCEPT, "Concept C")

    edge1 = graph.add_synaptic_edge("A", "B", EdgeType.REQUIRES, weight=1.0, decay_factor=0.9)
    edge2 = graph.add_synaptic_edge("A", "C", EdgeType.REQUIRES, weight=1.0, decay_factor=0.99)

    print("\nInitial weights:")
    print(f"  A->B: weight={edge1.weight:.3f} (fast decay: 0.9)")
    print(f"  A->C: weight={edge2.weight:.3f} (slow decay: 0.99)")

    print("\nApplying 10 decay cycles (simulating time passing without usage)...")
    for _ in range(10):
        graph.apply_global_decay()

    print("\nWeights after decay:")
    print(f"  A->B: weight={edge1.weight:.3f} (expected ~0.349)")
    print(f"  A->C: weight={edge2.weight:.3f} (expected ~0.904)")

    print("\n  Fast-decaying unused connections fade away,")
    print("  while frequently used connections resist decay through activation.")


def demo_persistence():
    """Demonstrate saving and loading synaptic state."""
    print_section("6. PERSISTENCE")

    # Create and populate a graph
    graph = SynapticMemoryGraph()
    graph.add_node("Q1", NodeType.QUESTION, "What framework?")
    graph.add_node("H1", NodeType.HYPOTHESIS, "Use FastAPI")
    edge = graph.add_synaptic_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.85)

    # Add some history
    graph.activate_node("Q1")
    graph.activate_node("H1")
    edge.record_prediction_outcome(correct=True)
    edge.record_prediction_outcome(correct=True)

    print("\nOriginal graph state:")
    print(f"  Nodes: {graph.node_count()}")
    print(f"  Edge Q1->H1: weight={edge.weight:.2f}, activations={edge.activation_count}")
    print(f"  Prediction accuracy: {edge.prediction_accuracy:.2f}")

    # Serialize to dict
    state = graph.to_dict()
    print(f"\nSerialized to dict with {len(state)} top-level keys")

    # Restore from dict
    restored = SynapticMemoryGraph.from_dict(state)
    restored_edge = restored.synaptic_edges[("Q1", "H1", EdgeType.EXPLORES)]

    print("\nRestored graph state:")
    print(f"  Nodes: {restored.node_count()}")
    print(f"  Edge Q1->H1: weight={restored_edge.weight:.2f}, activations={restored_edge.activation_count}")
    print(f"  Prediction accuracy: {restored_edge.prediction_accuracy:.2f}")

    print("\n  State fully preserved through serialization!")


def demo_visualization():
    """Demonstrate graph visualization."""
    print_section("7. VISUALIZATION")

    graph = SynapticMemoryGraph()

    # Build a small reasoning graph
    graph.add_node("Q1", NodeType.QUESTION, "API design?")
    graph.add_node("H1", NodeType.HYPOTHESIS, "REST")
    graph.add_node("H2", NodeType.HYPOTHESIS, "GraphQL")
    graph.add_node("D1", NodeType.DECISION, "Use REST")

    graph.add_synaptic_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.8)
    graph.add_synaptic_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.6)
    graph.add_synaptic_edge("H1", "D1", EdgeType.JUSTIFIES, weight=0.9)

    print("\nMermaid diagram (for docs/GitHub):")
    print("-" * 40)
    print(graph.to_mermaid())

    print("\n\nASCII tree view:")
    print("-" * 40)
    print(graph.to_ascii("Q1"))


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("  PRISM-GoT: Predictive Reasoning through Incremental")
    print("  Synaptic Memory Graph of Thought")
    print("="*60)
    print("\nThis demo showcases biologically-inspired reasoning with:")
    print("  - Synaptic plasticity (Hebbian learning)")
    print("  - Temporal decay (use it or lose it)")
    print("  - Predictive reasoning (anticipate next thoughts)")
    print("  - Reward-based learning (reinforce successful paths)")

    # Run demos
    graph = demo_basic_synaptic_graph()
    demo_activation_and_learning(graph)
    demo_prediction()
    demo_incremental_reasoner()
    demo_decay_and_cleanup()
    demo_persistence()
    demo_visualization()

    print_section("SUMMARY")
    print("""
PRISM-GoT provides a biologically-inspired reasoning framework:

1. SynapticEdge: Edges with activation tracking, decay, and prediction accuracy
2. SynapticMemoryGraph: ThoughtGraph with plasticity and learning
3. PlasticityRules: Hebbian, Anti-Hebbian, and Reward-based learning
4. IncrementalReasoner: Build graphs through experience

Key principles:
  - "Neurons that fire together wire together" (Hebbian learning)
  - "Use it or lose it" (temporal decay)
  - "Reinforce what works" (reward-based learning)
  - "Learn from experience" (incremental graph building)

Use cases:
  - Track reasoning patterns across sessions
  - Learn which approaches tend to succeed
  - Predict likely next steps in familiar domains
  - Adapt to user preferences over time
    """)


if __name__ == "__main__":
    main()
