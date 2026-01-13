"""
Cognitive Module - Cognitive Agent and Training Services.

Registers cognitive graph, agent, and training services.

Services Provided:
    - CognitiveGraph: The hypergraph with truth values
    - CognitiveAgent: Full cognitive agent stack
    - IncrementalTrainer: Incremental training orchestrator
    - TextToAtomsBridge: Text-to-graph processing

Usage:
    from cortical.core.modules import CognitiveModule

    container = Container()
    container.apply_module(CognitiveModule(model_dir=Path("models/agent")))

    agent = container.resolve(CognitiveAgent)
    trainer = container.resolve(IncrementalTrainer)
"""

from pathlib import Path
from typing import Optional

from cortical.common import Container, ContainerModule, Lifecycle, FileSystem


class CognitiveModule(ContainerModule):
    """
    Container module for Cognitive Agent services.

    The Cognitive layer provides the agent stack including graph,
    attention, working memory, prediction, and training.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        use_memory: bool = False,
        working_memory_size: int = 4,
        attention_focus_size: int = 7,
    ):
        """
        Initialize Cognitive module.

        Args:
            model_dir: Directory for model storage (default: models/cognitive_agent)
            use_memory: Use in-memory filesystem (for testing)
            working_memory_size: Capacity of working memory (default: 4)
            attention_focus_size: Size of attention focus (default: 7)
        """
        self.model_dir = model_dir or Path("models/cognitive_agent")
        self.use_memory = use_memory
        self.working_memory_size = working_memory_size
        self.attention_focus_size = attention_focus_size

    def register(self, container: Container) -> None:
        """Register Cognitive services with the container."""
        from cortical.cognitive.graph import (
            CognitiveGraph,
            CognitiveAgent,
            InMemoryStorage,
            StorageBackend,
        )
        from cortical.cognitive.text_bridge import TextToAtomsBridge
        from cortical.cognitive.training import IncrementalTrainer, TrainingConfig
        from cortical.common.filesystem import InMemoryFileSystem, RealFileSystem

        # Create filesystem based on use_memory flag
        if self.use_memory:
            filesystem: FileSystem = InMemoryFileSystem(self.model_dir)
            # Pre-create the model directory in memory
            filesystem.mkdir(self.model_dir, parents=True, exist_ok=True)
        else:
            filesystem = RealFileSystem(self.model_dir)
            self.model_dir.mkdir(parents=True, exist_ok=True)

        # Register filesystem for cognitive services
        # Use a named registration to avoid conflicts with CDG's filesystem
        container.register_instance("cognitive_filesystem", filesystem)

        # Register storage backend (always in-memory for graph)
        container.register(
            StorageBackend,
            InMemoryStorage,
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register CognitiveGraph
        def create_graph() -> CognitiveGraph:
            storage = container.resolve(StorageBackend)
            return CognitiveGraph(storage=storage)

        container.register(
            CognitiveGraph,
            create_graph,
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register CognitiveAgent
        def create_agent() -> CognitiveAgent:
            return CognitiveAgent(
                graph=container.resolve(CognitiveGraph),
                filesystem=filesystem,
                working_memory_size=self.working_memory_size,
                attention_focus_size=self.attention_focus_size,
            )

        container.register(
            CognitiveAgent,
            create_agent,
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register TextToAtomsBridge
        def create_bridge() -> TextToAtomsBridge:
            return TextToAtomsBridge(graph=container.resolve(CognitiveGraph))

        container.register(
            TextToAtomsBridge,
            create_bridge,
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register TrainingConfig
        container.register_instance(TrainingConfig, TrainingConfig())

        # Register IncrementalTrainer
        def create_trainer() -> IncrementalTrainer:
            return IncrementalTrainer(
                agent=container.resolve(CognitiveAgent),
                model_dir=self.model_dir,
                filesystem=filesystem,
                config=container.resolve(TrainingConfig),
            )

        container.register(
            IncrementalTrainer,
            create_trainer,
            lifecycle=Lifecycle.SINGLETON,
        )
