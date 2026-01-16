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

CLI Usage:
    python -m cortical.cognitive status
    python -m cortical.cognitive train samples/
    python -m cortical.cognitive reindex

Note on CLI and DI:
    The CLI uses __main__.py as entry point to avoid class identity issues
    with Python's -m flag. See training.py for detailed explanation.
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

# Lazy imports for training module to avoid class identity issues with -m flag.
# When running `python -m cortical.cognitive.training`, if we import these
# classes here, they get created before training.py runs as __main__, causing
# DI container resolution to fail (different class objects with same name).
#
# These are still available via:
#   from cortical.cognitive.training import IncrementalTrainer
#
# Or via lazy accessor:
#   from cortical.cognitive import get_trainer_class
#   TrainerClass = get_trainer_class()


def get_trainer_class():
    """Lazy import of IncrementalTrainer to avoid class identity issues."""
    from cortical.cognitive.training import IncrementalTrainer
    return IncrementalTrainer


def get_manifest_class():
    """Lazy import of TrainingManifest to avoid class identity issues."""
    from cortical.cognitive.training import TrainingManifest
    return TrainingManifest


def get_stats_class():
    """Lazy import of TrainingStats to avoid class identity issues."""
    from cortical.cognitive.training import TrainingStats
    return TrainingStats


# For backward compatibility, provide direct imports but document the caveat
# These work fine for normal imports, just not with `python -m` execution
def __getattr__(name):
    """Lazy attribute access for training classes."""
    if name == "IncrementalTrainer":
        from cortical.cognitive.training import IncrementalTrainer
        return IncrementalTrainer
    elif name == "TrainingManifest":
        from cortical.cognitive.training import TrainingManifest
        return TrainingManifest
    elif name == "TrainingStats":
        from cortical.cognitive.training import TrainingStats
        return TrainingStats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Memory system (CEL-based)
from cortical.cognitive.memory import CognitiveMemory

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
    # Training
    "IncrementalTrainer",
    "TrainingManifest",
    "TrainingStats",
    # Memory (CEL-based)
    "CognitiveMemory",
]
