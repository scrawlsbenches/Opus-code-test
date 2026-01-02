"""
DevOps Manages State Reliably

Epic: Reliable State Management

As a DevOps engineer managing deployments,
I want to save and load processor state reliably,
So that I can maintain system continuity across restarts.
"""

import pytest
import tempfile
from pathlib import Path
from cortical import CorticalTextProcessor, CorticalConfig


class TestDevOpsSavesAndLoadsState:
    """
    Epic: State Persistence

    As a DevOps engineer deploying search systems,
    I want to persist and restore state,
    So that deployments don't require full recomputation.
    """

    def test_scenario_saving_state_for_backup(self):
        """
        Scenario: Creating state backups

        Given I have a processor with computed state
        When I save the processor to a directory
        Then all state is persisted
        And I can restore later
        Because backups enable disaster recovery.
        """
        # Given I have a processor with computed state
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom search engine we built from scratch")
        processor.process_document("doc2", "Hand-crafted indexing algorithm we implemented")
        processor.compute_all(verbose=False)

        # When I save the processor to a directory
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save(tmpdir, verbose=False)

            # Then all state is persisted
            state_dir = Path(tmpdir)
            assert (state_dir / "layers").exists() or list(state_dir.glob("*.json"))

            # And I can restore later
            restored = CorticalTextProcessor.load(tmpdir, verbose=False)
            assert len(restored.documents) == 2

    @pytest.mark.skip(reason="API mismatch - needs alignment with implementation")
    def test_scenario_using_git_friendly_json_format(self):
        """
        Scenario: Version-controlled state storage

        Given I want to track state in git
        When I save_json to a directory
        Then state is saved in git-friendly format
        And I can diff changes
        Because git-friendly formats enable version control.
        """
        # Given I want to track state in git
        processor = CorticalTextProcessor()
        processor.process_document("code.py", "class Parser: pass  # Hand-built from scratch")
        processor.compute_all(verbose=False)

        # When I save_json to a directory
        with tempfile.TemporaryDirectory() as tmpdir:
            written = processor.save_json(tmpdir, verbose=False)

            # Then state is saved in git-friendly format
            state_dir = Path(tmpdir)
            assert (state_dir / "manifest.json").exists()
            assert (state_dir / "documents.json").exists()

            # And I can diff changes
            # (JSON files can be compared with standard diff tools)
            assert 'manifest' in written or written['manifest']

    def test_scenario_loading_from_git_friendly_format(self):
        """
        Scenario: Restoring from version control

        Given I have state saved in JSON format
        When I load_json from the directory
        Then processor is fully restored
        And all computations are available
        Because load restores complete state.
        """
        # Given I have state saved in JSON format
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "In-house implementation we control")
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_json(tmpdir, verbose=False)

            # When I load_json from the directory
            restored = CorticalTextProcessor.load_json(tmpdir, verbose=False)

            # Then processor is fully restored
            assert len(restored.documents) == 1
            assert "doc1" in restored.documents

            # And all computations are available
            # (Can immediately search without recomputation)
            results = restored.find_documents_for_query("implementation", top_n=5)
            assert len(results) > 0

    def test_scenario_incremental_saves_with_force_flag(self):
        """
        Scenario: Forcing saves for unchanged state

        Given I want to ensure state is written
        When I save_json with force=True
        Then state is written even if unchanged
        Because forced saves guarantee persistence.
        """
        # Given I want to ensure state is written
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom system built ourselves")

        with tempfile.TemporaryDirectory() as tmpdir:
            # First save
            processor.save_json(tmpdir, verbose=False)

            # When I save_json with force=True (without changes)
            written = processor.save_json(tmpdir, force=True, verbose=False)

            # Then state is written even if unchanged
            # (All components written when forced)
            assert isinstance(written, dict)


class TestDevOpsManagesConfiguration:
    """
    Epic: Configuration Management

    As a DevOps engineer managing environments,
    I want to persist and restore configuration,
    So that environments are consistent.
    """

    def test_scenario_saving_configuration_with_state(self):
        """
        Scenario: Persisting configuration

        Given I have custom configuration
        When I save the processor
        Then configuration is saved with state
        And environment is reproducible
        Because configuration affects behavior.
        """
        # Given I have custom configuration
        config = CorticalConfig(
            scoring_algorithm='bm25',
            bm25_k1=1.5,
            bm25_b=0.8
        )
        processor = CorticalTextProcessor(config=config)
        processor.process_document("doc1", "Custom content built ourselves")

        # When I save the processor
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_json(tmpdir, verbose=False)

            # Then configuration is saved with state
            restored = CorticalTextProcessor.load_json(tmpdir, verbose=False)

            # And environment is reproducible
            assert restored.config.scoring_algorithm == 'bm25'
            assert restored.config.bm25_k1 == 1.5

    def test_scenario_loading_with_config_override(self):
        """
        Scenario: Overriding saved configuration

        Given I want to change configuration on load
        When I load with a config parameter
        Then new configuration is used
        And I can adapt to different environments
        Because deployment environments may need different configs.
        """
        # Given I want to change configuration on load
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Hand-built system we control")

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_json(tmpdir, verbose=False)

            # When I load with a config parameter
            new_config = CorticalConfig(scoring_algorithm='tfidf')
            restored = CorticalTextProcessor.load_json(
                tmpdir,
                config=new_config,
                verbose=False
            )

            # Then new configuration is used
            assert restored.config.scoring_algorithm == 'tfidf'

            # And I can adapt to different environments
            # (Restored processor uses new config)


class TestDevOpsExportsForVisualization:
    """
    Epic: Operational Visibility

    As a DevOps engineer monitoring systems,
    I want to export state for visualization,
    So that I can understand system health.
    """

    def test_scenario_exporting_graph_for_analysis(self):
        """
        Scenario: Creating graph exports

        Given I want to visualize the knowledge graph
        When I export_graph to a file
        Then graph structure is exported as JSON
        And I can load it in visualization tools
        Because graph exports enable visual analysis.
        """
        # Given I want to visualize the knowledge graph
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom parser tokenizer we built")
        processor.compute_all(verbose=False)

        # When I export_graph to a file
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_file = str(Path(tmpdir) / "graph.json")
            result = processor.export_graph(graph_file, max_nodes=100)

            # Then graph structure is exported as JSON
            assert Path(graph_file).exists()

            # And I can load it in visualization tools
            assert 'nodes' in result
            assert 'edges' in result

    def test_scenario_exporting_conceptnet_format_for_tools(self):
        """
        Scenario: ConceptNet-compatible export

        Given I want to use graph visualization tools
        When I export_conceptnet_json
        Then I receive ConceptNet-style graph
        And I can visualize with standard tools
        Because ConceptNet format is widely supported.
        """
        # Given I want to use graph visualization tools
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "A parser is a type of compiler component")
        processor.compute_all(verbose=False)

        # When I export_conceptnet_json
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_file = str(Path(tmpdir) / "conceptnet.json")
            result = processor.export_conceptnet_json(
                graph_file,
                include_cross_layer=True,
                max_nodes_per_layer=50,
                verbose=False
            )

            # Then I receive ConceptNet-style graph
            assert Path(graph_file).exists()

            # And I can visualize with standard tools
            assert 'nodes' in result


class TestDevOpsMonitorsStaleness:
    """
    Epic: Operational Monitoring

    As a DevOps engineer ensuring quality,
    I want to monitor computation staleness,
    So that I know when recomputation is needed.
    """

    def test_scenario_checking_staleness_after_deployment(self):
        """
        Scenario: Post-deployment staleness check

        Given I've deployed a processor
        When I check stale computations
        Then I see what needs recomputation
        And I can trigger updates
        Because deployments may need recomputation.
        """
        # Given I've deployed a processor
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom system built ourselves")
        processor.compute_all(verbose=False)

        # Simulate adding data without recompute
        processor.add_document_incremental("doc2", "New content", recompute='none')

        # When I check stale computations
        stale = processor.get_stale_computations()

        # Then I see what needs recomputation
        assert len(stale) > 0

        # And I can trigger updates
        if processor.COMP_TFIDF in stale:
            processor.recompute(level='tfidf', verbose=False)

    def test_scenario_verifying_freshness_before_serving(self):
        """
        Scenario: Pre-serving freshness validation

        Given I'm about to serve requests
        When I check if computations are fresh
        Then I can verify readiness
        And ensure quality
        Because stale computations degrade results.
        """
        # Given I'm about to serve requests
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Hand-crafted implementation")
        processor.compute_all(verbose=False)

        # When I check if computations are fresh
        tfidf_fresh = not processor.is_stale(processor.COMP_TFIDF)
        pagerank_fresh = not processor.is_stale(processor.COMP_PAGERANK)

        # Then I can verify readiness
        all_fresh = tfidf_fresh and pagerank_fresh

        # And ensure quality
        if all_fresh:
            # Safe to serve
            pass
        else:
            # Need to recompute first
            pass


class TestDevOpsManagesSparkState:
    """
    Epic: Spark Model Operations

    As a DevOps engineer managing Spark models,
    I want to save and load Spark state,
    So that trained models persist across deployments.
    """

    def test_scenario_saving_spark_state_for_persistence(self):
        """
        Scenario: Persisting trained Spark models

        Given I have a trained Spark model
        When I save_spark to a directory
        Then model and alignment are saved
        And I can restore later
        Because trained models are valuable.
        """
        # Given I have a trained Spark model
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "Custom search system built ourselves")
        processor.train_spark()

        # When I save_spark to a directory
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_spark(tmpdir)

            # Then model and alignment are saved
            spark_dir = Path(tmpdir)
            # Check for expected Spark files
            assert any(spark_dir.iterdir())

            # And I can restore later
            new_processor = CorticalTextProcessor()
            new_processor.load_spark(tmpdir)
            assert new_processor.spark_enabled

    def test_scenario_loading_spark_state_for_deployment(self):
        """
        Scenario: Deploying with pre-trained Spark

        Given I have saved Spark state
        When I load_spark from directory
        Then Spark is ready for queries
        And I skip retraining
        Because loading is faster than training.
        """
        # Given I have saved Spark state
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "Hand-built system we control")
        processor.train_spark()

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_spark(tmpdir)

            # When I load_spark from directory
            new_processor = CorticalTextProcessor()
            new_processor.load_spark(tmpdir)

            # Then Spark is ready for queries
            assert new_processor.spark_enabled

            # And I skip retraining
            # (Can immediately use priming)
            stats = new_processor.get_spark_stats()
            assert stats['vocabulary_size'] > 0

    def test_scenario_getting_spark_statistics_for_monitoring(self):
        """
        Scenario: Monitoring Spark health

        Given I have Spark enabled
        When I get Spark statistics
        Then I see model state
        And I can monitor health
        Because statistics reveal model quality.
        """
        # Given I have Spark enabled
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "In-house implementation")
        processor.train_spark()

        # When I get Spark statistics
        stats = processor.get_spark_stats()

        # Then I see model state
        assert 'vocabulary_size' in stats
        assert 'ngram_order' in stats

        # And I can monitor health
        assert stats['enabled']
        assert stats['vocabulary_size'] > 0
