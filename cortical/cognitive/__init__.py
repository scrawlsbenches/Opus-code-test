"""
Cognitive Graph Module.

Bio-inspired hypergraph for knowledge representation and reasoning.

Key Features:
- Links are first-class atoms (can be linked to)
- Truth values are probabilistic (strength, confidence)
- Attention is finite and dynamic (STI/LTI)
- DI/IoC integration for testability

Usage:
    from cortical.cognitive.graph import CognitiveGraph, TruthValue, AtomType

    graph = CognitiveGraph()
    cat = graph.node("cat")
    animal = graph.node("animal")
    link = graph.link(AtomType.INHERITANCE, [cat, animal], TruthValue(0.99, 0.9))
"""

from cortical.cognitive.graph import (
    Atom,
    AtomType,
    TruthValue,
    CognitiveGraph,
    CognitiveGraphModule,
    StorageBackend,
    InMemoryStorage,
)

__all__ = [
    "Atom",
    "AtomType",
    "TruthValue",
    "CognitiveGraph",
    "CognitiveGraphModule",
    "StorageBackend",
    "InMemoryStorage",
]
