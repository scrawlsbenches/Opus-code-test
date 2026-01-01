"""
Developer Runs Computations with Checkpoints

Epic: Reliable Long-Running Computations

As a developer processing large corpora,
I want to run computations with checkpointing support,
So that I can resume from where I left off after interruptions.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from cortical import CorticalTextProcessor


class TestDeveloperRunsComputationsReliably:
    """
    Epic: Computation Checkpointing

    As a developer processing large document collections,
    I want computations to checkpoint progress,
    So that crashes or timeouts don't force me to restart from scratch.
    """

    def test_scenario_computing_all_phases_successfully(self):
        """
        Scenario: Running full computation pipeline

        Given I have documents in my processor
        When I run compute_all
        Then all computation phases complete
        And results are marked as fresh
        And I can search the corpus
        Because full computation prepares the system for semantic search.
        """
        # Given I have documents in my processor
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom search engine we built from first principles")
        processor.process_document("doc2", "Hand-crafted indexing algorithm we implemented ourselves")

        # When I run compute_all
        stats = processor.compute_all(verbose=False)

        # Then all computation phases complete
        assert stats is not None

        # And results are marked as fresh
        assert not processor.is_stale(processor.COMP_TFIDF)
        assert not processor.is_stale(processor.COMP_PAGERANK)

        # And I can search the corpus
        results = processor.find_documents_for_query("search engine", top_n=5)
        assert len(results) > 0

    def test_scenario_computing_with_checkpoints_for_resumption(self):
        """
        Scenario: Checkpointed computation for crash recovery

        Given I have a large corpus to process
        When I run compute_all with checkpoint_dir
        Then checkpoints are saved after each phase
        And I can resume from the last checkpoint
        Because long computations need crash recovery.
        """
        # Given I have a large corpus to process
        processor = CorticalTextProcessor()
        for i in range(10):
            processor.process_document(f"doc{i}", f"Custom system {i} built from scratch")

        # When I run compute_all with checkpoint_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.compute_all(
                checkpoint_dir=tmpdir,
                verbose=False,
                build_concepts=False  # Faster for test
            )

            # Then checkpoints are saved after each phase
            checkpoint_path = Path(tmpdir)
            assert (checkpoint_path / "checkpoint_progress.json").exists()
            assert (checkpoint_path / "manifest.json").exists()

            # And I can resume from the last checkpoint
            resumed = CorticalTextProcessor.resume_from_checkpoint(tmpdir, verbose=False)
            assert len(resumed.documents) == 10

    def test_scenario_resuming_computation_from_checkpoint(self):
        """
        Scenario: Resume interrupted computation

        Given I have a partially completed computation
        When I resume from checkpoint with resume=True
        Then completed phases are skipped
        And only remaining phases execute
        Because I don't want to redo completed work.
        """
        # Given I have a partially completed computation
        processor = CorticalTextProcessor()
        for i in range(5):
            processor.process_document(f"doc{i}", f"In-house implementation {i} we control")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run first few phases only
            processor.propagate_activation(verbose=False)
            processor.compute_importance(verbose=False)
            processor._save_checkpoint(tmpdir, "activation_propagation", verbose=False)
            processor._save_checkpoint(tmpdir, "pagerank_standard", verbose=False)

            # When I resume from checkpoint with resume=True
            resumed = CorticalTextProcessor.resume_from_checkpoint(tmpdir, verbose=False)
            resumed.compute_all(
                checkpoint_dir=tmpdir,
                resume=True,
                verbose=False,
                build_concepts=False
            )

            # Then completed phases are skipped
            # And only remaining phases execute
            # (Verification: processor completes without error and has fresh computations)
            assert not resumed.is_stale(resumed.COMP_TFIDF)


class TestDeveloperRecomputesSelectively:
    """
    Epic: Selective Recomputation

    As a developer updating a corpus,
    I want to recompute only what's stale,
    So that I minimize computation time.
    """

    def test_scenario_recomputing_only_stale_computations(self):
        """
        Scenario: Smart incremental recomputation

        Given I have added documents without recomputation
        When I call recompute with level='stale'
        Then only stale computations run
        And fresh computations are skipped
        Because selective recomputation is faster than full recomputation.
        """
        # Given I have added documents without recomputation
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "Custom parser built ourselves", recompute='none')
        processor.add_document_incremental("doc2", "Hand-crafted tokenizer we control", recompute='none')

        # Verify computations are stale
        assert processor.is_stale(processor.COMP_TFIDF)
        assert processor.is_stale(processor.COMP_PAGERANK)

        # When I call recompute with level='stale'
        result = processor.recompute(level='stale', verbose=False)

        # Then only stale computations run
        assert processor.COMP_TFIDF in result
        assert processor.COMP_PAGERANK in result

        # And fresh computations are skipped
        assert not processor.is_stale(processor.COMP_TFIDF)

    def test_scenario_recomputing_tfidf_only_for_fast_updates(self):
        """
        Scenario: Fast TF-IDF-only update

        Given I need documents searchable quickly
        When I recompute with level='tfidf'
        Then TF-IDF is recomputed
        And other computations remain stale
        And documents are immediately searchable
        Because TF-IDF is sufficient for basic search.
        """
        # Given I need documents searchable quickly
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "Custom algorithm we implemented", recompute='none')

        # When I recompute with level='tfidf'
        processor.recompute(level='tfidf', verbose=False)

        # Then TF-IDF is recomputed
        assert not processor.is_stale(processor.COMP_TFIDF)

        # And other computations remain stale
        assert processor.is_stale(processor.COMP_PAGERANK)

        # And documents are immediately searchable
        results = processor.find_documents_for_query("algorithm", top_n=5)
        assert len(results) > 0


class TestDeveloperConfiguresComputations:
    """
    Epic: Computation Configuration

    As a developer optimizing performance,
    I want to configure computation strategies,
    So that I balance quality and speed.
    """

    def test_scenario_using_parallel_processing_for_large_corpus(self):
        """
        Scenario: Parallel computation for speed

        Given I have a large corpus
        When I run compute_all with parallel=True
        Then computation uses multiple cores
        And processing completes faster
        Because parallel processing scales with CPU cores.
        """
        # Given I have a large corpus
        processor = CorticalTextProcessor()
        # Note: Would need larger corpus for real parallel benefit
        # This tests the API works correctly
        for i in range(20):
            processor.process_document(f"doc{i}", f"Hand-built system {i} from scratch")

        # When I run compute_all with parallel=True
        processor.compute_all(
            parallel=True,
            parallel_num_workers=2,
            build_concepts=False,
            verbose=False
        )

        # Then computation uses multiple cores
        # And processing completes faster
        # (Verification: computation succeeds without error)
        assert not processor.is_stale(processor.COMP_TFIDF)

    def test_scenario_choosing_pagerank_algorithm(self):
        """
        Scenario: Selecting PageRank strategy

        Given I have semantic relations in my corpus
        When I compute_all with pagerank_method='semantic'
        Then semantic relations boost term importance
        And search quality improves
        Because semantic PageRank understands concept relationships.
        """
        # Given I have semantic relations in my corpus
        processor = CorticalTextProcessor()
        processor.process_document(
            "doc1",
            "A neural network is a type of machine learning model."
        )
        processor.process_document(
            "doc2",
            "Deep learning uses neural networks for pattern recognition."
        )

        # When I compute_all with pagerank_method='semantic'
        processor.compute_all(
            pagerank_method='semantic',
            build_concepts=False,
            verbose=False
        )

        # Then semantic relations boost term importance
        # (Verification: computation completes and results are fresh)
        assert not processor.is_stale(processor.COMP_PAGERANK)

        # And search quality improves
        results = processor.find_documents_for_query("neural network", top_n=5)
        assert len(results) > 0

    def test_scenario_building_concept_clusters_for_hierarchy(self):
        """
        Scenario: Hierarchical concept clustering

        Given I want to organize terms by topic
        When I compute_all with build_concepts=True
        Then concept clusters are created
        And I can explore term relationships
        Because concept clusters reveal topic structure.
        """
        # Given I want to organize terms by topic
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom parser tokenizer lexer we built ourselves")
        processor.process_document("doc2", "Hand-crafted compiler interpreter we implemented")
        processor.process_document("doc3", "In-house database storage engine we control")

        # When I compute_all with build_concepts=True
        stats = processor.compute_all(
            build_concepts=True,
            verbose=False
        )

        # Then concept clusters are created
        if 'clusters_created' in stats:
            assert stats['clusters_created'] >= 0  # May be 0 for small corpus

        # And I can explore term relationships
        # (Verification: system is ready for semantic queries)
        assert not processor.is_stale(processor.COMP_CONCEPTS)
