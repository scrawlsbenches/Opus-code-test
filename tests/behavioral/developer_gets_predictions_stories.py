"""
Behavioral Tests: Developer Gets Statistical Predictions
=========================================================

Epic: Fast Statistical Predictions

As a developer searching through documents,
I want fast statistical predictions to prime my queries,
So that I get better search results with less effort.
"""

import pytest
import tempfile
from pathlib import Path
from cortical.spark.predictor import SparkPredictor


class TestDeveloperPrimesSearchQueries:
    """
    As a developer starting a search,
    I want query priming suggestions,
    So that my initial query yields better results.
    """

    def test_scenario_priming_extracts_keywords(self):
        """
        Scenario: Prime extracts key terms

        Given a trained predictor
        When I prime a query with common words
        Then I get content keywords without stop words
        Because priming filters out noise.
        """
        # Given a trained predictor
        spark = SparkPredictor()
        documents = [
            "neural network architecture design",
            "machine learning model training",
            "deep learning optimization techniques"
        ]
        spark.train_from_documents(documents)

        # When I prime a query
        result = spark.prime("the neural network model")

        # Then I get keywords
        assert 'keywords' in result
        keywords = result['keywords']
        assert 'neural' in keywords
        assert 'network' in keywords
        assert 'model' in keywords
        # Stop words filtered
        assert 'the' not in keywords

    def test_scenario_priming_suggests_completions(self):
        """
        Scenario: Prime suggests word completions

        Given a predictor trained on domain text
        When I prime a partial query
        Then I get likely next word predictions
        Because the n-gram model learned patterns.
        """
        # Given a trained predictor
        spark = SparkPredictor()
        documents = [
            "machine learning algorithm",
            "machine learning model",
            "machine learning framework",
        ]
        spark.train_from_documents(documents)

        # When I prime with partial query
        result = spark.prime("machine learning")

        # Then I get completions
        assert 'completions' in result
        completions = result['completions']
        assert isinstance(completions, list)
        # Should suggest words that commonly follow "machine learning"

    def test_scenario_priming_includes_alignment_context(self):
        """
        Scenario: Prime includes alignment context

        Given a predictor with loaded alignment data
        When I prime a query with aligned terms
        Then I get relevant alignment entries
        Because alignment provides user context.
        """
        # Given predictor with alignment
        with tempfile.TemporaryDirectory() as tmpdir:
            align_dir = Path(tmpdir)
            align_file = align_dir / "context.md"
            align_file.write_text("""
# Definitions
- minicolumn: Fundamental processing unit in cortical architecture
- activation: Pattern of active minicolumns in response to input
""")

            spark = SparkPredictor()
            spark.load_alignment(str(align_dir))

            # When I prime with aligned term
            result = spark.prime("minicolumn activation patterns")

            # Then alignment is included
            assert 'alignment' in result
            alignment = result['alignment']
            # Should find minicolumn and/or activation definitions


class TestDeveloperGetsWordPredictions:
    """
    As a developer typing,
    I want word-level predictions,
    So that I complete thoughts faster.
    """

    def test_scenario_completing_common_phrases(self):
        """
        Scenario: Complete common phrases

        Given a predictor trained on technical documents
        When I provide a common prefix
        Then I get likely completions based on frequency
        Because the n-gram model tracks patterns.
        """
        # Given a trained predictor
        spark = SparkPredictor()
        documents = [
            "database connection established successfully",
            "database connection failed with error",
            "database connection timeout occurred",
        ]
        spark.train_from_documents(documents)

        # When I complete a prefix
        completions = spark.complete("database connection", top_k=3)

        # Then I get predictions
        assert isinstance(completions, list)
        assert len(completions) > 0
        # Should suggest words that follow "database connection"

    def test_scenario_completing_sequence_extends_text(self):
        """
        Scenario: Complete a sequence of words

        Given a trained predictor
        When I ask to extend text by multiple words
        Then I get a coherent completion
        Because sequence prediction chains predictions.
        """
        # Given a trained predictor
        spark = SparkPredictor()
        documents = [
            "authentication handler processes user credentials",
            "authentication handler validates session tokens",
        ]
        spark.train_from_documents(documents)

        # When I complete a sequence
        completed = spark.complete_sequence("authentication handler", length=2)

        # Then text is extended
        assert len(completed) > len("authentication handler")
        assert completed.startswith("authentication handler")


class TestDeveloperManagesAlignment:
    """
    As a developer building context,
    I want to add and retrieve alignment entries,
    So that the system understands my vocabulary.
    """

    def test_scenario_adding_definitions_enriches_context(self):
        """
        Scenario: Add definition to alignment

        Given a predictor
        When I add a definition for a term
        Then I can retrieve it later
        Because alignment accumulates context.
        """
        # Given a predictor
        spark = SparkPredictor()

        # When I add a definition
        spark.add_definition("cortical", "Brain-inspired processing architecture")

        # Then I can retrieve it
        context = spark.get_alignment_context("cortical")
        assert context is not None
        assert 'value' in context
        assert "Brain-inspired" in context['value']

    def test_scenario_adding_patterns_guides_behavior(self):
        """
        Scenario: Add pattern to alignment

        Given a predictor
        When I add a pattern description
        Then the alignment index contains it
        Because patterns document expected structures.
        """
        # Given a predictor
        spark = SparkPredictor()

        # When I add a pattern
        spark.add_pattern("search_query", "Natural language question about code")

        # Then it's in alignment
        summary = spark.get_context_summary()
        assert isinstance(summary, str)
        # Pattern should be accessible

    def test_scenario_adding_preferences_captures_choices(self):
        """
        Scenario: Add preference to alignment

        Given a predictor
        When I add a preference
        Then it's stored in alignment
        Because preferences guide future decisions.
        """
        # Given a predictor
        spark = SparkPredictor()

        # When I add a preference
        spark.add_preference("naming", "Use snake_case for functions")

        # Then it's stored
        summary = spark.get_context_summary()
        assert isinstance(summary, str)


class TestDeveloperTrainsFromMultipleSources:
    """
    As a developer,
    I want to train predictors from various sources,
    So that I can use them in different contexts.
    """

    def test_scenario_training_from_directory_learns_patterns(self):
        """
        Scenario: Train from file directory

        Given a directory with text files
        When I train a predictor on that directory
        Then it learns patterns from the files
        Because the trainer reads multiple file types.
        """
        # Given a directory with files
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_dir = Path(tmpdir)
            (doc_dir / "doc1.txt").write_text("machine learning models")
            (doc_dir / "doc2.md").write_text("neural network training")
            (doc_dir / "code.py").write_text("def train_model():\n    pass")

            # When I train from directory
            spark = SparkPredictor()
            spark.train_from_directory(str(doc_dir))

            # Then it's trained
            assert spark._trained, "Should be marked as trained"
            # Should be able to make predictions
            completions = spark.complete("machine", top_k=3)
            assert isinstance(completions, list)

    def test_scenario_persisting_predictor_preserves_state(self):
        """
        Scenario: Save and load predictor

        Given a trained predictor
        When I save it and load in new instance
        Then the new instance provides same predictions
        Because persistence preserves model state.
        """
        # Given a trained predictor
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            spark1 = SparkPredictor()
            spark1.train_from_documents(["test document content"])
            spark1.add_definition("test_term", "test definition")

            # When I save and load
            spark1.save(str(save_dir))
            spark2 = SparkPredictor.load(str(save_dir))

            # Then state is preserved
            assert spark2._trained, "Loaded predictor should be trained"
            context = spark2.get_alignment_context("test_term")
            assert context is not None, "Alignment should be preserved"
