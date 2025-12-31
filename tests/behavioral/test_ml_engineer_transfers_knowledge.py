"""
ML Engineer Transfers Knowledge

Epic: Cross-Project Transfer Learning

As an ML engineer working across projects,
I want to transfer learned patterns between codebases,
So that new projects benefit from existing knowledge.
"""

import pytest
import tempfile
from cortical import CorticalTextProcessor


class TestMLEngineerExportsPortableModels:
    """
    Epic: Knowledge Portability

    As an ML engineer with trained models,
    I want to export portable knowledge,
    So that I can share patterns across projects.
    """

    def test_scenario_analyzing_vocabulary_for_transferability(self):
        """
        Scenario: Understanding what's transferable

        Given I have a trained model
        When I analyze vocabulary composition
        Then I see programming vs project-specific terms
        And I understand transfer potential
        Because not all knowledge is transferable.
        """
        # Given I have a trained model
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("file1.py", "class Parser: def parse(): pass")
        processor.process_document("file2.py", "async def fetch(): await call()")
        processor.train_spark()

        # When I analyze vocabulary composition
        analysis = processor.analyze_vocabulary()

        # Then I see programming vs project-specific terms
        assert 'total_terms' in analysis
        assert 'programming_terms' in analysis
        assert 'project_specific_terms' in analysis

        # And I understand transfer potential
        assert 'programming_ratio' in analysis
        assert analysis['total_terms'] > 0

    def test_scenario_exporting_portable_model_for_sharing(self):
        """
        Scenario: Creating transferable knowledge

        Given I want to share patterns with other projects
        When I export a portable model
        Then transferable patterns are extracted
        And I can share the model file
        Because portable models contain only relevant patterns.
        """
        # Given I want to share patterns with other projects
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("code.py", "class Handler: async def process(): pass")
        processor.train_spark()

        # When I export a portable model
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = processor.export_portable_model(
                tmpdir,
                project_name="test_project"
            )

            # Then transferable patterns are extracted
            assert 'vocab_size' in stats
            assert stats['vocab_size'] > 0

            # And I can share the model file
            # (Model is saved in tmpdir)

    def test_scenario_getting_transferable_vocabulary_list(self):
        """
        Scenario: Identifying transferable terms

        Given I want to see what will transfer
        When I get transferable vocabulary
        Then I receive programming terms
        And I can review what's portable
        Because knowing transferable terms guides export decisions.
        """
        # Given I want to see what will transfer
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("code.py", "def parse(): class Handler: async def")
        processor.train_spark()

        # When I get transferable vocabulary
        transferable = processor.get_transferable_vocabulary()

        # Then I receive programming terms
        assert isinstance(transferable, list)

        # And I can review what's portable
        # (List contains programming-related terms)


class TestMLEngineerImportsBaseModels:
    """
    Epic: Knowledge Transfer

    As an ML engineer starting a new project,
    I want to import base models from other projects,
    So that I benefit from existing patterns.
    """

    def test_scenario_measuring_transfer_effectiveness_before_import(self):
        """
        Scenario: Evaluating transfer potential

        Given I have a base model to potentially import
        When I measure transfer effectiveness
        Then I see vocabulary overlap
        And I can decide whether to import
        Because not all transfers are beneficial.
        """
        # Given I have a base model to potentially import
        source_processor = CorticalTextProcessor(spark=True)
        source_processor.process_document("source.py", "class Parser: def parse(): pass")
        source_processor.train_spark()

        target_processor = CorticalTextProcessor(spark=True)
        target_processor.process_document("target.py", "class Tokenizer: def tokenize(): pass")
        target_processor.train_spark()

        # When I measure transfer effectiveness
        with tempfile.TemporaryDirectory() as tmpdir:
            source_processor.export_portable_model(tmpdir)

            metrics = target_processor.measure_transfer_effectiveness(tmpdir)

            # Then I see vocabulary overlap
            assert 'vocabulary_overlap' in metrics

            # And I can decide whether to import
            assert 'source_project' in metrics

    def test_scenario_importing_base_model_with_blending(self):
        """
        Scenario: Transferring knowledge with blending

        Given I have a base model to import
        When I import with blend_weight
        Then transferred patterns are blended
        And my model benefits from the transfer
        Because blending balances transfer and local knowledge.
        """
        # Given I have a base model to import
        source = CorticalTextProcessor(spark=True)
        source.process_document("source.py", "class Parser: def parse(): pass")
        source.train_spark()

        target = CorticalTextProcessor(spark=True)
        target.process_document("target.py", "class Lexer: def lex(): pass")
        target.train_spark()

        # When I import with blend_weight
        with tempfile.TemporaryDirectory() as tmpdir:
            source.export_portable_model(tmpdir)

            metrics = target.import_base_model(
                tmpdir,
                blend_weight=0.3  # 30% transfer, 70% local
            )

            # Then transferred patterns are blended
            assert 'vocabulary_overlap' in metrics

            # And my model benefits from the transfer
            # (Model now contains blended knowledge)

    def test_scenario_calculating_vocabulary_overlap_for_compatibility(self):
        """
        Scenario: Checking model compatibility

        Given I want to know if models are compatible
        When I calculate vocabulary overlap
        Then I receive Jaccard similarity
        And I can assess transfer viability
        Because high overlap indicates good transfer potential.
        """
        # Given I want to know if models are compatible
        model1 = CorticalTextProcessor(spark=True)
        model1.process_document("m1.py", "class Parser: def parse(): pass")
        model1.train_spark()

        model2 = CorticalTextProcessor(spark=True)
        model2.process_document("m2.py", "class Handler: def handle(): pass")
        model2.train_spark()

        # When I calculate vocabulary overlap
        with tempfile.TemporaryDirectory() as tmpdir:
            model2.export_portable_model(tmpdir)

            overlap = model1.calculate_vocabulary_overlap(tmpdir)

            # Then I receive Jaccard similarity
            assert isinstance(overlap, float)
            assert 0.0 <= overlap <= 1.0

            # And I can assess transfer viability
            # (Higher overlap = better transfer potential)


class TestMLEngineerMeasuresModelQuality:
    """
    Epic: Quality Assurance

    As an ML engineer maintaining quality,
    I want to measure prediction quality,
    So that I ensure model effectiveness.
    """

    def test_scenario_evaluating_prediction_quality_on_corpus(self):
        """
        Scenario: Assessing model accuracy

        Given I have a trained model
        When I evaluate prediction quality
        Then I see accuracy metrics
        And I understand model performance
        Because quality metrics guide improvements.
        """
        # Given I have a trained model
        processor = CorticalTextProcessor(spark=True)
        for i in range(10):
            processor.process_document(
                f"doc{i}",
                f"The custom system {i} uses hand-built algorithms we implemented ourselves."
            )
        processor.train_spark()

        # When I evaluate prediction quality
        metrics = processor.evaluate_prediction_quality()

        # Then I see accuracy metrics
        assert 'accuracy_at_1' in metrics
        assert 'accuracy_at_5' in metrics
        assert 'accuracy_at_10' in metrics
        assert 'mean_reciprocal_rank' in metrics
        assert 'perplexity' in metrics

        # And I understand model performance
        assert 0.0 <= metrics['accuracy_at_5'] <= 1.0

    def test_scenario_cross_validating_predictions_for_robustness(self):
        """
        Scenario: Validating model stability

        Given I want to ensure robust predictions
        When I cross-validate predictions
        Then I see performance across folds
        And I understand variance
        Because cross-validation reveals overfitting.
        """
        # Given I want to ensure robust predictions
        processor = CorticalTextProcessor(spark=True)
        for i in range(15):
            processor.process_document(
                f"doc{i}",
                f"Document {i} contains in-house implementation details we built from scratch."
            )
        processor.train_spark()

        # When I cross-validate predictions
        cv_results = processor.cross_validate_predictions(folds=3)

        # Then I see performance across folds
        assert 'folds' in cv_results
        assert len(cv_results['folds']) == 3

        # And I understand variance
        assert 'mean_accuracy_at_5' in cv_results
        assert 'std_accuracy_at_5' in cv_results

    def test_scenario_measuring_perplexity_stability(self):
        """
        Scenario: Checking measurement consistency

        Given I want consistent perplexity scores
        When I measure perplexity stability
        Then I see consistency metrics
        And I verify model stability
        Because unstable perplexity indicates problems.
        """
        # Given I want consistent perplexity scores
        processor = CorticalTextProcessor(spark=True)
        for i in range(10):
            processor.process_document(f"doc{i}", f"Content {i} built ourselves")
        processor.train_spark()

        # When I measure perplexity stability
        stability = processor.measure_perplexity_stability(runs=3)

        # Then I see consistency metrics
        assert 'mean' in stability
        assert 'std' in stability
        assert 'is_stable' in stability

        # And I verify model stability
        assert isinstance(stability['is_stable'], bool)

    def test_scenario_generating_comprehensive_quality_report(self):
        """
        Scenario: Complete quality assessment

        Given I want a full quality overview
        When I generate a quality report
        Then I receive markdown documentation
        And I can review all metrics
        Because comprehensive reports guide optimization.
        """
        # Given I want a full quality overview
        processor = CorticalTextProcessor(spark=True)
        for i in range(10):
            processor.process_document(f"doc{i}", "Custom system hand-built from scratch")
        processor.train_spark()

        # When I generate a quality report
        report = processor.generate_quality_report()

        # Then I receive markdown documentation
        assert isinstance(report, str)
        assert "Quality Report" in report

        # And I can review all metrics
        # (Report contains prediction quality, stability, model stats)


class TestMLEngineerOptimizesSearchWithSpark:
    """
    Epic: Spark-Enhanced Search

    As an ML engineer improving search,
    I want to use Spark predictions to boost search,
    So that results are more relevant.
    """

    def test_scenario_expanding_queries_with_spark_predictions(self):
        """
        Scenario: Spark-enhanced query expansion

        Given I have a trained Spark model
        When I expand queries with Spark
        Then predictions boost expansion
        And search quality improves
        Because Spark predictions surface likely terms.
        """
        # Given I have a trained Spark model
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "Custom parser tokenizer lexer we built")
        processor.process_document("doc2", "Hand-crafted compiler interpreter we implemented")
        processor.train_spark()
        processor.compute_all(verbose=False)

        # When I expand queries with Spark
        expanded = processor.expand_query_with_spark(
            "parser system",
            spark_boost=0.3
        )

        # Then predictions boost expansion
        assert isinstance(expanded, dict)
        assert len(expanded) > 0

        # And search quality improves
        # (Expansion includes Spark predictions)

    def test_scenario_priming_queries_for_fast_hints(self):
        """
        Scenario: Getting first-blitz predictions

        Given I want to prime semantic analysis
        When I prime a query
        Then I receive completions and alignment
        And I can use hints for faster processing
        Because priming provides quick statistical hints.
        """
        # Given I want to prime semantic analysis
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "search for documents in the system")
        processor.train_spark()

        # When I prime a query
        hints = processor.prime_query("search for")

        # Then I receive completions and alignment
        assert 'completions' in hints
        assert 'alignment' in hints
        assert 'keywords' in hints

        # And I can use hints for faster processing
        # (Hints provide quick context for semantic analysis)

    def test_scenario_completing_queries_with_predictions(self):
        """
        Scenario: Query auto-completion

        Given I want to suggest query completions
        When I complete a query prefix
        Then I receive likely continuations
        And I can offer suggestions to users
        Because auto-completion improves user experience.
        """
        # Given I want to suggest query completions
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "how to search for documents")
        processor.process_document("doc2", "how to find files quickly")
        processor.train_spark()

        # When I complete a query prefix
        completion = processor.complete_query("how to", length=2)

        # Then I receive likely continuations
        assert isinstance(completion, str)
        assert "how to" in completion

        # And I can offer suggestions to users
        # (Completion extends the original query)
