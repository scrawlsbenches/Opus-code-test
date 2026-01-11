"""
Cognitive Graph Module.

Bio-inspired hypergraph for knowledge representation and reasoning.

Key Features:
- Links are first-class atoms (can be linked to)
- Truth values are probabilistic (strength, confidence)
- Attention is finite and dynamic (STI/LTI)
- Working memory with LRU eviction
- Prediction via co-occurrence learning
- Goal tracking with urgency prioritization
- Exploration/exploitation balance
- DI/IoC integration for testability

Usage:
    from cortical.cognitive.graph import CognitiveGraph, TruthValue, AtomType

    graph = CognitiveGraph()
    cat = graph.node("cat")
    animal = graph.node("animal")
    link = graph.link(AtomType.INHERITANCE, [cat, animal], TruthValue(0.99, 0.9))

    # Full cognitive agent:
    from cortical.cognitive.graph import CognitiveAgent

    agent = CognitiveAgent()
    agent.graph.node("concept")
    agent.attend("concept")
    metrics = agent.step()
"""

from cortical.cognitive.graph import (
    # Core types
    Atom,
    AtomType,
    TruthValue,
    # Graph
    CognitiveGraph,
    CognitiveGraphModule,
    StorageBackend,
    InMemoryStorage,
    # Cognitive layers
    Goal,
    WorkingMemory,
    AssociativePredictor,
    SurpriseTracker,
    GoalTracker,
    ExplorationController,
    Episode,
    EpisodicMemory,
    # Integrated agent
    CognitiveAgent,
    CognitiveAgentModule,
    # GoT integration
    GoTBridge,
    # Event system
    EventType,
    CognitiveEvent,
    EventBus,
)

__all__ = [
    # Core types
    "Atom",
    "AtomType",
    "TruthValue",
    # Graph
    "CognitiveGraph",
    "CognitiveGraphModule",
    "StorageBackend",
    "InMemoryStorage",
    # Cognitive layers
    "Goal",
    "WorkingMemory",
    "AssociativePredictor",
    "SurpriseTracker",
    "GoalTracker",
    "ExplorationController",
    "Episode",
    "EpisodicMemory",
    # Integrated agent
    "CognitiveAgent",
    "CognitiveAgentModule",
    # GoT integration
    "GoTBridge",
    # Event system
    "EventType",
    "CognitiveEvent",
    "EventBus",
]
