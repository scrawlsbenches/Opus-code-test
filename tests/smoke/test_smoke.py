"""
Smoke Tests - Quick Sanity Checks
=================================

These tests verify that the system fundamentally works.
They should complete in < 10 seconds total and catch critical breakage early.

If smoke tests fail, there's likely a critical issue that will affect everything.
Fix smoke test failures before investigating other test failures.

Run with: pytest tests/smoke/ -v

SMOKE TEST MANIFEST
-------------------
This manifest tracks which systems should have smoke test coverage.
When adding a major new system, add it here and create corresponding tests.

Systems with smoke tests:
    ✅ Cortical Core (CorticalTextProcessor, layers, search)
    ✅ GoT (Graph of Thought - task management, transactions)
    ✅ CDG (Core Data Graph - entity storage)
    ✅ BaseGraph (Unified graph architecture - SimpleGraph, DAGGraph, WeightedGraph)
    ✅ CognitiveGraph (Hypergraph with Atoms - nodes, links, meta-reasoning)
    ⚡ Hubris (MoE system - import check only, still evolving)
    ⚡ CEL (Event sourcing - import check only, still developing)

Systems intentionally without smoke tests:
    ⏸️ WovenMind (cognitive_bootstrap) - orchestration layer, let it stabilize
    ⏸️ Spark (language models) - experimental, tested via behavioral

Last manifest review: 2026-01-13
"""

import pytest


class TestCoreImports:
    """Verify core modules can be imported."""

    def test_import_cortical_package(self):
        """Main package imports successfully."""
        import cortical
        assert hasattr(cortical, 'CorticalTextProcessor')
        assert hasattr(cortical, 'CorticalLayer')

    def test_import_processor(self):
        """Processor module imports."""
        from cortical import CorticalTextProcessor
        assert CorticalTextProcessor is not None

    def test_import_analysis(self):
        """Analysis module imports."""
        from cortical import analysis
        assert hasattr(analysis, 'compute_pagerank')
        assert hasattr(analysis, 'compute_tfidf')

    def test_import_query(self):
        """Query module imports."""
        from cortical import query
        assert hasattr(query, 'find_documents_for_query')

    def test_import_tokenizer(self):
        """Tokenizer module imports."""
        from cortical.tokenizer import Tokenizer
        assert Tokenizer is not None


class TestProcessorCreation:
    """Verify processor can be created and used."""

    def test_create_empty_processor(self):
        """Empty processor can be instantiated."""
        from cortical import CorticalTextProcessor
        processor = CorticalTextProcessor()
        assert processor is not None
        assert len(processor.documents) == 0

    def test_create_with_config(self):
        """Processor accepts configuration."""
        from cortical import CorticalTextProcessor
        from cortical.config import CorticalConfig

        config = CorticalConfig(pagerank_damping=0.9)
        processor = CorticalTextProcessor(config=config)
        assert processor.config.pagerank_damping == 0.9

    def test_create_with_tokenizer(self):
        """Processor accepts custom tokenizer."""
        from cortical import CorticalTextProcessor
        from cortical.tokenizer import Tokenizer

        tokenizer = Tokenizer(filter_code_noise=True)
        processor = CorticalTextProcessor(tokenizer=tokenizer)
        assert processor is not None


class TestBasicWorkflow:
    """Verify the basic processing workflow works."""

    def test_process_single_document(self):
        """Single document can be processed."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        stats = processor.process_document("test", "Hello world test document.")

        assert stats['tokens'] > 0
        assert "test" in processor.documents

    def test_process_multiple_documents(self):
        """Multiple documents can be processed."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "First document content.")
        processor.process_document("doc2", "Second document content.")

        assert len(processor.documents) == 2

    def test_compute_all_completes(self):
        """compute_all() completes without error."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("test", "Test document for computation.")
        processor.compute_all(verbose=False)

        # Verify some computation happened
        from cortical import CorticalLayer
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        assert layer0.column_count() > 0


class TestBasicSearch:
    """Verify search functionality works."""

    def test_search_returns_results(self, small_processor):
        """Search returns results from corpus."""
        results = small_processor.find_documents_for_query("machine learning", top_n=5)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_search_empty_query_raises(self, small_processor):
        """Empty query raises ValueError."""
        with pytest.raises(ValueError):
            small_processor.find_documents_for_query("", top_n=5)

    def test_query_expansion_works(self, small_processor):
        """Query expansion returns related terms."""
        expanded = small_processor.expand_query("database", max_expansions=10)

        assert isinstance(expanded, dict)
        assert "database" in expanded or len(expanded) > 0


class TestBasicPersistence:
    """Verify save/load functionality works."""

    def test_save_and_load(self, tmp_path, small_processor):
        """Processor can be saved and loaded."""
        from cortical import CorticalTextProcessor

        save_path = tmp_path / "test_corpus.pkl"

        # Save
        small_processor.save(str(save_path))
        assert save_path.exists()

        # Load
        loaded = CorticalTextProcessor.load(str(save_path))
        assert len(loaded.documents) == len(small_processor.documents)


class TestLayerAccess:
    """Verify layer access works correctly."""

    def test_get_all_layers(self, small_processor):
        """All four layers are accessible."""
        from cortical import CorticalLayer

        for layer_type in CorticalLayer:
            layer = small_processor.get_layer(layer_type)
            assert layer is not None

    def test_token_layer_has_content(self, small_processor):
        """Token layer contains minicolumns."""
        from cortical import CorticalLayer

        layer0 = small_processor.get_layer(CorticalLayer.TOKENS)
        assert layer0.column_count() > 0

    def test_document_layer_has_content(self, small_processor):
        """Document layer contains all documents."""
        from cortical import CorticalLayer

        layer3 = small_processor.get_layer(CorticalLayer.DOCUMENTS)
        assert layer3.column_count() == len(small_processor.documents)


# =============================================================================
# GoT (Graph of Thought) Smoke Tests
# =============================================================================

class TestGoTImports:
    """Verify GoT modules can be imported."""

    def test_import_got_package(self):
        """GoT package imports successfully."""
        from cortical import got
        assert hasattr(got, 'GoTManager')
        assert hasattr(got, 'Task')
        assert hasattr(got, 'Decision')
        assert hasattr(got, 'Edge')

    def test_import_got_manager(self):
        """GoTManager imports directly."""
        from cortical.got import GoTManager
        assert GoTManager is not None

    def test_import_transaction_types(self):
        """Transaction types import."""
        from cortical.got import Transaction, TransactionManager
        assert Transaction is not None
        assert TransactionManager is not None


class TestGoTBasicOperations:
    """Verify GoT can create and query tasks (via container with in-memory storage)."""

    def test_create_manager(self):
        """GoTManager can be resolved from container."""
        from cortical.core.bootstrap import create_container
        from cortical.got import GoTManager
        # Use in-memory storage for fast smoke tests
        container = create_container(use_memory=True)
        manager = container.resolve(GoTManager)
        assert manager is not None

    def test_create_task(self):
        """Task can be created."""
        from cortical.core.bootstrap import create_container
        from cortical.got import GoTManager
        container = create_container(use_memory=True)
        manager = container.resolve(GoTManager)

        task = manager.create_task("Smoke test task", priority="medium")
        assert task is not None
        assert task.id.startswith("T-")
        assert task.title == "Smoke test task"

    def test_query_tasks(self):
        """Tasks can be queried."""
        from cortical.core.bootstrap import create_container
        from cortical.got import GoTManager
        container = create_container(use_memory=True)
        manager = container.resolve(GoTManager)

        manager.create_task("Task 1", priority="high")
        manager.create_task("Task 2", priority="low")

        tasks = manager.find_tasks()
        assert len(tasks) == 2

    def test_create_edge(self):
        """Edges can be created between tasks."""
        from cortical.core.bootstrap import create_container
        from cortical.got import GoTManager
        container = create_container(use_memory=True)
        manager = container.resolve(GoTManager)

        t1 = manager.create_task("Task A")
        t2 = manager.create_task("Task B")

        edge = manager.add_edge(t1.id, t2.id, "DEPENDS_ON")
        assert edge is not None


# =============================================================================
# CDG (Core Data Graph) Smoke Tests
# =============================================================================

class TestCDGImports:
    """Verify CDG modules can be imported."""

    def test_import_cdg_package(self):
        """CDG package imports successfully."""
        from cortical import cdg
        assert hasattr(cdg, 'CDGStore')
        assert hasattr(cdg, 'Entity')
        assert hasattr(cdg, 'Edge')

    def test_import_cdg_store(self):
        """CDGStore imports directly."""
        from cortical.cdg import CDGStore
        assert CDGStore is not None

    def test_import_cdg_types(self):
        """CDG types import."""
        from cortical.cdg import Entity, Edge, Transaction
        assert Entity is not None
        assert Edge is not None
        assert Transaction is not None


class TestCDGBasicOperations:
    """Verify CDG can store and retrieve entities (using in-memory storage)."""

    def test_create_store(self):
        """CDGStore with InMemoryFileSystem can be instantiated."""
        from cortical.cdg import CDGStore
        from cortical.common import InMemoryFileSystem
        from pathlib import Path
        fs = InMemoryFileSystem(Path("/test"))
        store = CDGStore(filesystem=fs)
        assert store is not None

    def test_write_and_read_entity(self):
        """Entity can be written and read back."""
        from cortical.cdg import CDGStore, Entity
        from cortical.common import InMemoryFileSystem
        from pathlib import Path

        fs = InMemoryFileSystem(Path("/test"))
        store = CDGStore(filesystem=fs)

        entity = Entity(id="smoke-test-001", entity_type="test")
        store.write(entity)

        loaded = store.read("smoke-test-001")
        assert loaded is not None
        assert loaded.id == "smoke-test-001"


# =============================================================================
# Hubris (MoE System) Smoke Tests - Import checks only
# =============================================================================

class TestHubrisImports:
    """Verify Hubris modules can be imported (light check - system evolving)."""

    def test_import_micro_expert_base(self):
        """MicroExpert base class imports."""
        import sys
        import os
        # Add scripts to path for hubris imports
        scripts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        if scripts_path not in sys.path:
            sys.path.insert(0, os.path.abspath(scripts_path))

        from hubris.micro_expert import MicroExpert
        assert MicroExpert is not None

    def test_import_expert_consolidator(self):
        """ExpertConsolidator imports."""
        import sys
        import os
        scripts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        if scripts_path not in sys.path:
            sys.path.insert(0, os.path.abspath(scripts_path))

        from hubris.expert_consolidator import ExpertConsolidator
        assert ExpertConsolidator is not None


# =============================================================================
# CEL (Event Sourcing) Smoke Tests - Import checks only
# =============================================================================

class TestCELImports:
    """Verify CEL modules can be imported (light check - system developing)."""

    def test_import_cel_package(self):
        """CEL package imports successfully."""
        from cortical import cel
        assert cel is not None

    def test_import_cel_core_types(self):
        """CEL core types import."""
        from cortical.cel import CognitiveEvent, EventStore
        assert CognitiveEvent is not None
        assert EventStore is not None


# =============================================================================
# BaseGraph (Unified Graph Architecture) Smoke Tests
# =============================================================================

class TestBaseGraphImports:
    """Verify BaseGraph modules can be imported."""

    def test_import_graph_package(self):
        """Graph package imports successfully."""
        from cortical import graph
        assert hasattr(graph, 'BaseGraph')
        assert hasattr(graph, 'SimpleGraph')
        assert hasattr(graph, 'DAGGraph')
        assert hasattr(graph, 'WeightedGraph')

    def test_import_protocols(self):
        """Graph protocols import."""
        from cortical.graph import NodeBase, EdgeBase, NodeProtocol, EdgeProtocol
        assert NodeBase is not None
        assert EdgeBase is not None
        assert NodeProtocol is not None
        assert EdgeProtocol is not None

    def test_import_storage(self):
        """Graph storage imports."""
        from cortical.graph import InMemoryGraphStorage
        assert InMemoryGraphStorage is not None

    def test_import_algorithms(self):
        """Algorithm mixins import."""
        from cortical.graph import (
            PageRankMixin,
            ClusteringMixin,
            SpreadingActivationMixin,
            CentralityMixin,
        )
        assert PageRankMixin is not None
        assert ClusteringMixin is not None
        assert SpreadingActivationMixin is not None
        assert CentralityMixin is not None


class TestBaseGraphBasicOperations:
    """Verify BaseGraph can create nodes and edges."""

    def test_create_simple_graph(self):
        """SimpleGraph can be instantiated."""
        from cortical.graph import SimpleGraph
        graph = SimpleGraph()
        assert graph is not None
        assert graph.node_count == 0

    def test_add_nodes_and_edges(self):
        """Nodes and edges can be added."""
        from cortical.graph import SimpleGraph
        graph = SimpleGraph()

        graph.add_node("A", content="Node A")
        graph.add_node("B", content="Node B")
        graph.add_edge("A", "B", edge_type="CONNECTS")

        assert graph.node_count == 2
        assert graph.edge_count == 1

    def test_basic_traversal(self):
        """BFS traversal works."""
        from cortical.graph import SimpleGraph
        graph = SimpleGraph()

        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        order = graph.bfs("A")
        assert order == ["A", "B", "C"]

    def test_dag_cycle_prevention(self):
        """DAGGraph prevents cycles."""
        from cortical.graph import DAGGraph
        import pytest

        dag = DAGGraph()
        dag.add_node("A")
        dag.add_node("B")
        dag.add_edge("A", "B")

        with pytest.raises(ValueError, match="cycle"):
            dag.add_edge("B", "A")

    def test_weighted_dijkstra(self):
        """WeightedGraph Dijkstra works."""
        from cortical.graph import WeightedGraph
        graph = WeightedGraph()

        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B", cost=1)
        graph.add_edge("B", "C", cost=2)

        path, cost = graph.dijkstra("A", "C")
        assert path == ["A", "B", "C"]
        assert cost == 3


# =============================================================================
# CognitiveGraph (Hypergraph with Atoms) Smoke Tests
# =============================================================================

class TestCognitiveGraphImports:
    """Verify CognitiveGraph modules can be imported."""

    def test_import_cognitive_graph(self):
        """CognitiveGraph imports successfully."""
        from cortical.cognitive.graph import CognitiveGraph
        assert CognitiveGraph is not None

    def test_import_atom_types(self):
        """Atom and AtomType import."""
        from cortical.cognitive.graph import Atom, AtomType, TruthValue
        assert Atom is not None
        assert AtomType is not None
        assert TruthValue is not None

    def test_import_storage_backend(self):
        """Storage backend imports."""
        from cortical.cognitive.graph import InMemoryStorage, StorageBackend
        assert InMemoryStorage is not None
        assert StorageBackend is not None


class TestCognitiveGraphBasicOperations:
    """Verify CognitiveGraph can create atoms and links."""

    def test_create_cognitive_graph(self):
        """CognitiveGraph can be instantiated."""
        from cortical.cognitive.graph import CognitiveGraph
        graph = CognitiveGraph()
        assert graph is not None

    def test_create_nodes(self):
        """Node atoms can be created."""
        from cortical.cognitive.graph import CognitiveGraph, AtomType
        graph = CognitiveGraph()

        cat = graph.node("cat", AtomType.CONCEPT)
        assert cat is not None
        assert cat.name == "cat"
        assert cat.is_node()

    def test_create_links(self):
        """Link atoms can be created (hypergraph capability)."""
        from cortical.cognitive.graph import CognitiveGraph, AtomType
        graph = CognitiveGraph()

        cat = graph.node("cat")
        animal = graph.node("animal")

        # Links are atoms too - hypergraph semantics
        link = graph.link(AtomType.INHERITANCE, [cat, animal])
        assert link is not None
        assert link.is_link()
        assert len(link.outgoing) == 2

    def test_truth_values(self):
        """TruthValue probabilistic logic works."""
        from cortical.cognitive.graph import TruthValue

        tv1 = TruthValue(strength=0.9, confidence=0.8)
        tv2 = TruthValue(strength=0.7, confidence=0.6)

        merged = tv1.merge(tv2)
        assert 0 <= merged.strength <= 1
        assert 0 <= merged.confidence <= 1

    def test_meta_links(self):
        """Links can point to other links (meta-reasoning)."""
        from cortical.cognitive.graph import CognitiveGraph, AtomType
        graph = CognitiveGraph()

        # Create a statement: "cats are animals"
        cat = graph.node("cat")
        animal = graph.node("animal")
        statement = graph.link(AtomType.INHERITANCE, [cat, animal])

        # Create a meta-statement: "John believes cats are animals"
        john = graph.node("john", AtomType.PERSON)
        belief = graph.link(AtomType.BELIEVES, [john, statement])

        assert belief is not None
        assert belief.is_link()
        # The belief links to the statement (which is itself a link)
        assert statement.id in belief.outgoing
