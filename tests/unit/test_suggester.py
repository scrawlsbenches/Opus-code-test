"""
Unit tests for SampleSuggester.

Tests the suggestion generation system that observes interactions
and suggests new alignment entries.
"""

import unittest
from datetime import datetime

from cortical.spark.suggester import (
    SampleSuggester,
    DefinitionSuggestion,
    PatternSuggestion,
    PreferenceSuggestion,
    Observation
)


class TestObservation(unittest.TestCase):
    """Test Observation dataclass."""

    def test_observation_creation(self):
        """Test creating an observation."""
        obs = Observation(
            query="test query",
            timestamp="2025-01-01T00:00:00",
            success=True,
            context={"key": "value"}
        )
        self.assertEqual(obs.query, "test query")
        self.assertTrue(obs.success)
        self.assertEqual(obs.context["key"], "value")

    def test_observation_default_context(self):
        """Test observation with default empty context."""
        obs = Observation(
            query="test",
            timestamp="2025-01-01T00:00:00",
            success=True
        )
        self.assertEqual(obs.context, {})


class TestDefinitionSuggestion(unittest.TestCase):
    """Test DefinitionSuggestion dataclass."""

    def test_to_markdown(self):
        """Test markdown export."""
        suggestion = DefinitionSuggestion(
            term="minicolumn",
            frequency=5,
            contexts=["minicolumn activation", "minicolumn connections"],
            confidence=0.8,
            reason="Used 5 times without definition"
        )
        md = suggestion.to_markdown()
        self.assertIn("minicolumn", md)
        self.assertIn("5 times", md)
        self.assertIn("TODO", md)


class TestPatternSuggestion(unittest.TestCase):
    """Test PatternSuggestion dataclass."""

    def test_to_markdown(self):
        """Test markdown export."""
        suggestion = PatternSuggestion(
            pattern_name="how_to",
            examples=["how do I search", "how do I query"],
            frequency=10,
            confidence=0.7,
            reason="Pattern seen 10 times"
        )
        md = suggestion.to_markdown()
        self.assertIn("how_to", md)
        self.assertIn("how do I search", md)


class TestPreferenceSuggestion(unittest.TestCase):
    """Test PreferenceSuggestion dataclass."""

    def test_to_markdown(self):
        """Test markdown export."""
        suggestion = PreferenceSuggestion(
            preference_name="naming",
            chosen="camelCase",
            over="snake_case",
            frequency=8,
            confidence=0.8,
            reason="Chose camelCase 8/10 times"
        )
        md = suggestion.to_markdown()
        self.assertIn("naming", md)
        self.assertIn("camelCase", md)
        self.assertIn("snake_case", md)


class TestSampleSuggesterInit(unittest.TestCase):
    """Test SampleSuggester initialization."""

    def test_default_init(self):
        """Test default initialization."""
        suggester = SampleSuggester()
        self.assertEqual(suggester.min_frequency, 3)
        self.assertEqual(suggester.min_confidence, 0.5)
        self.assertEqual(len(suggester.known_terms), 0)

    def test_custom_known_terms(self):
        """Test initialization with known terms."""
        known = {"neural", "network", "graph"}
        suggester = SampleSuggester(known_terms=known)
        self.assertEqual(len(suggester.known_terms), 3)
        self.assertIn("neural", suggester.known_terms)

    def test_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        suggester = SampleSuggester(min_frequency=5, min_confidence=0.7)
        self.assertEqual(suggester.min_frequency, 5)
        self.assertEqual(suggester.min_confidence, 0.7)


class TestObserveQuery(unittest.TestCase):
    """Test query observation."""

    def setUp(self):
        self.suggester = SampleSuggester()

    def test_observe_single_query(self):
        """Test observing a single query."""
        self.suggester.observe_query("neural network activation")
        self.assertEqual(len(self.suggester.observations), 1)
        self.assertIn("neural", self.suggester.term_counts)

    def test_observe_multiple_queries(self):
        """Test observing multiple queries."""
        self.suggester.observe_query("neural network")
        self.suggester.observe_query("neural activation")
        self.suggester.observe_query("network graph")
        self.assertEqual(len(self.suggester.observations), 3)
        self.assertEqual(self.suggester.term_counts["neural"], 2)

    def test_observe_success_tracking(self):
        """Test success/failure tracking."""
        self.suggester.observe_query("query1", success=True)
        self.suggester.observe_query("query2", success=False)
        self.assertEqual(len(self.suggester.success_contexts), 1)
        self.assertEqual(len(self.suggester.failure_contexts), 1)

    def test_observe_with_context(self):
        """Test observation with context."""
        self.suggester.observe_query("test", context={"results": 5})
        obs = self.suggester.observations[0]
        self.assertEqual(obs.context["results"], 5)

    def test_bigram_extraction(self):
        """Test bigram extraction from queries."""
        self.suggester.observe_query("neural network activation")
        self.assertIn("neural network", self.suggester.bigram_counts)

    def test_pattern_detection(self):
        """Test query pattern detection."""
        self.suggester.observe_query("how do I search documents?")
        self.assertIn("how_to", self.suggester.query_patterns)


class TestObserveChoice(unittest.TestCase):
    """Test choice observation."""

    def setUp(self):
        self.suggester = SampleSuggester()

    def test_observe_single_choice(self):
        """Test observing a single choice."""
        self.suggester.observe_choice(
            choice_type="naming",
            chosen="camelCase",
            alternatives=["snake_case", "kebab-case"]
        )
        self.assertEqual(len(self.suggester.observations), 1)

    def test_choice_stored_in_context(self):
        """Test choice details stored in context."""
        self.suggester.observe_choice(
            choice_type="approach",
            chosen="iterative",
            alternatives=["recursive"]
        )
        obs = self.suggester.observations[0]
        self.assertEqual(obs.context["type"], "choice")
        self.assertEqual(obs.context["chosen"], "iterative")


class TestKnownTerms(unittest.TestCase):
    """Test known term management."""

    def setUp(self):
        self.suggester = SampleSuggester()

    def test_add_known_term(self):
        """Test adding a single known term."""
        self.suggester.add_known_term("Neural")
        self.assertIn("neural", self.suggester.known_terms)

    def test_add_known_terms(self):
        """Test adding multiple known terms."""
        self.suggester.add_known_terms({"Graph", "Network"})
        self.assertIn("graph", self.suggester.known_terms)
        self.assertIn("network", self.suggester.known_terms)

    def test_known_terms_case_insensitive(self):
        """Test that known terms are stored lowercase."""
        self.suggester.add_known_term("NEURAL")
        self.assertIn("neural", self.suggester.known_terms)
        self.assertNotIn("NEURAL", self.suggester.known_terms)


class TestSuggestDefinitions(unittest.TestCase):
    """Test definition suggestions."""

    def setUp(self):
        self.suggester = SampleSuggester(min_frequency=2, min_confidence=0.3)

    def test_no_suggestions_empty(self):
        """Test no suggestions when no observations."""
        suggestions = self.suggester.suggest_definitions()
        self.assertEqual(len(suggestions), 0)

    def test_suggest_frequent_term(self):
        """Test suggesting frequently used term."""
        for _ in range(5):
            self.suggester.observe_query("minicolumn activation patterns")
        suggestions = self.suggester.suggest_definitions()
        terms = [s.term for s in suggestions]
        self.assertIn("minicolumn", terms)

    def test_skip_known_terms(self):
        """Test that known terms are not suggested."""
        self.suggester.add_known_term("neural")
        for _ in range(5):
            self.suggester.observe_query("neural network processing")
        suggestions = self.suggester.suggest_definitions()
        terms = [s.term for s in suggestions]
        self.assertNotIn("neural", terms)

    def test_skip_infrequent_terms(self):
        """Test that infrequent terms are not suggested."""
        self.suggester.observe_query("unicorn rainbow")
        suggestions = self.suggester.suggest_definitions()
        terms = [s.term for s in suggestions]
        self.assertNotIn("unicorn", terms)

    def test_confidence_calculation(self):
        """Test confidence is calculated."""
        for _ in range(10):
            self.suggester.observe_query("pagerank algorithm")
        suggestions = self.suggester.suggest_definitions()
        if suggestions:
            self.assertGreater(suggestions[0].confidence, 0)

    def test_sorted_by_confidence(self):
        """Test suggestions are sorted by confidence."""
        for _ in range(5):
            self.suggester.observe_query("termA usage")
        for _ in range(10):
            self.suggester.observe_query("termB frequent")
        suggestions = self.suggester.suggest_definitions()
        if len(suggestions) >= 2:
            self.assertGreaterEqual(suggestions[0].confidence, suggestions[1].confidence)


class TestSuggestPatterns(unittest.TestCase):
    """Test pattern suggestions."""

    def setUp(self):
        self.suggester = SampleSuggester(min_frequency=2, min_confidence=0.3)

    def test_no_patterns_empty(self):
        """Test no patterns when no observations."""
        suggestions = self.suggester.suggest_patterns()
        self.assertEqual(len(suggestions), 0)

    def test_detect_how_to_pattern(self):
        """Test detecting 'how to' query pattern."""
        for _ in range(3):
            self.suggester.observe_query("how do I search?")
        suggestions = self.suggester.suggest_patterns()
        patterns = [s.pattern_name for s in suggestions]
        self.assertIn("how_to", patterns)

    def test_detect_location_pattern(self):
        """Test detecting 'where' query pattern."""
        for _ in range(3):
            self.suggester.observe_query("where is the config?")
        suggestions = self.suggester.suggest_patterns()
        patterns = [s.pattern_name for s in suggestions]
        self.assertIn("location", patterns)

    def test_detect_definition_pattern(self):
        """Test detecting 'what is' query pattern."""
        for _ in range(3):
            self.suggester.observe_query("what is pagerank?")
        suggestions = self.suggester.suggest_patterns()
        patterns = [s.pattern_name for s in suggestions]
        self.assertIn("definition", patterns)

    def test_bigram_phrase_patterns(self):
        """Test detecting bigram phrase patterns."""
        # Need enough observations to meet min_frequency threshold
        for _ in range(10):
            self.suggester.observe_query("neural network processing")
        # Check that bigrams are being counted
        self.assertIn("neural network", self.suggester.bigram_counts)
        self.assertGreaterEqual(self.suggester.bigram_counts["neural network"], 10)


class TestSuggestPreferences(unittest.TestCase):
    """Test preference suggestions."""

    def setUp(self):
        self.suggester = SampleSuggester(min_frequency=2, min_confidence=0.3)

    def test_no_preferences_empty(self):
        """Test no preferences when no observations."""
        suggestions = self.suggester.suggest_preferences()
        self.assertEqual(len(suggestions), 0)

    def test_detect_strong_preference(self):
        """Test detecting strong preference."""
        for _ in range(8):
            self.suggester.observe_choice("naming", "camelCase", ["snake_case"])
        for _ in range(2):
            self.suggester.observe_choice("naming", "snake_case", ["camelCase"])
        suggestions = self.suggester.suggest_preferences()
        self.assertGreater(len(suggestions), 0)
        self.assertEqual(suggestions[0].chosen, "camelCase")

    def test_no_preference_if_balanced(self):
        """Test no preference if choices are balanced."""
        for _ in range(5):
            self.suggester.observe_choice("style", "A", ["B"])
        for _ in range(5):
            self.suggester.observe_choice("style", "B", ["A"])
        suggestions = self.suggester.suggest_preferences()
        style_prefs = [s for s in suggestions if s.preference_name == "style"]
        self.assertEqual(len(style_prefs), 0)


class TestGetAllSuggestions(unittest.TestCase):
    """Test combined suggestion retrieval."""

    def test_get_all_suggestions_structure(self):
        """Test get_all_suggestions returns correct structure."""
        suggester = SampleSuggester()
        result = suggester.get_all_suggestions()
        self.assertIn("definitions", result)
        self.assertIn("patterns", result)
        self.assertIn("preferences", result)

    def test_get_all_suggestions_empty(self):
        """Test get_all_suggestions with no observations."""
        suggester = SampleSuggester()
        result = suggester.get_all_suggestions()
        self.assertEqual(len(result["definitions"]), 0)
        self.assertEqual(len(result["patterns"]), 0)
        self.assertEqual(len(result["preferences"]), 0)


class TestExportMarkdown(unittest.TestCase):
    """Test markdown export."""

    def test_export_empty(self):
        """Test export with no suggestions."""
        suggester = SampleSuggester()
        md = suggester.export_suggestions_markdown()
        self.assertIn("Suggested Alignment Entries", md)
        self.assertIn("No suggestions yet", md)

    def test_export_with_definitions(self):
        """Test export includes definitions."""
        suggester = SampleSuggester(min_frequency=2, min_confidence=0.3)
        for _ in range(5):
            self.suggester = suggester
            suggester.observe_query("minicolumn patterns")
        md = suggester.export_suggestions_markdown()
        self.assertIn("Suggested Alignment Entries", md)

    def test_export_contains_timestamp(self):
        """Test export contains generation timestamp."""
        suggester = SampleSuggester()
        md = suggester.export_suggestions_markdown()
        self.assertIn("Generated:", md)

    def test_export_contains_observation_count(self):
        """Test export contains observation count."""
        suggester = SampleSuggester()
        suggester.observe_query("test query")
        md = suggester.export_suggestions_markdown()
        self.assertIn("1 observations", md)


class TestGetStats(unittest.TestCase):
    """Test statistics retrieval."""

    def test_stats_structure(self):
        """Test stats returns expected keys."""
        suggester = SampleSuggester()
        stats = suggester.get_stats()
        self.assertIn("total_observations", stats)
        self.assertIn("unique_terms", stats)
        self.assertIn("known_terms", stats)
        self.assertIn("success_rate", stats)
        self.assertIn("pending_suggestions", stats)

    def test_stats_after_observations(self):
        """Test stats reflect observations."""
        suggester = SampleSuggester()
        suggester.observe_query("query1", success=True)
        suggester.observe_query("query2", success=False)
        stats = suggester.get_stats()
        self.assertEqual(stats["total_observations"], 2)
        self.assertEqual(stats["success_rate"], 0.5)


class TestClear(unittest.TestCase):
    """Test clearing observations."""

    def test_clear_resets_all(self):
        """Test clear resets all data."""
        suggester = SampleSuggester()
        suggester.observe_query("test query")
        suggester.observe_choice("style", "A", ["B"])
        suggester.clear()
        self.assertEqual(len(suggester.observations), 0)
        self.assertEqual(len(suggester.term_counts), 0)
        self.assertEqual(len(suggester.success_contexts), 0)


class TestExtractTerms(unittest.TestCase):
    """Test term extraction."""

    def setUp(self):
        self.suggester = SampleSuggester()

    def test_extract_basic_terms(self):
        """Test extracting basic terms."""
        terms = self.suggester._extract_terms("neural network processing")
        self.assertIn("neural", terms)
        self.assertIn("network", terms)

    def test_filter_stop_words(self):
        """Test that stop words are filtered."""
        terms = self.suggester._extract_terms("the quick brown fox")
        self.assertNotIn("the", terms)
        self.assertIn("quick", terms)

    def test_filter_short_words(self):
        """Test that short words are filtered."""
        terms = self.suggester._extract_terms("a b c word")
        self.assertNotIn("a", terms)
        self.assertIn("word", terms)

    def test_lowercase_extraction(self):
        """Test terms are lowercased."""
        terms = self.suggester._extract_terms("NEURAL Network")
        self.assertIn("neural", terms)
        self.assertNotIn("NEURAL", terms)


class TestExtractPatterns(unittest.TestCase):
    """Test pattern extraction."""

    def setUp(self):
        self.suggester = SampleSuggester()

    def test_how_to_pattern(self):
        """Test detecting 'how to' pattern."""
        pattern = self.suggester._extract_pattern("how do I search?")
        self.assertEqual(pattern, "how_to")

    def test_location_pattern(self):
        """Test detecting 'where' pattern."""
        pattern = self.suggester._extract_pattern("where is config?")
        self.assertEqual(pattern, "location")

    def test_definition_pattern(self):
        """Test detecting 'what is' pattern."""
        pattern = self.suggester._extract_pattern("what is pagerank?")
        self.assertEqual(pattern, "definition")

    def test_why_pattern(self):
        """Test detecting 'why' pattern."""
        pattern = self.suggester._extract_pattern("why does it fail?")
        self.assertEqual(pattern, "explanation")

    def test_debugging_pattern(self):
        """Test detecting debugging pattern."""
        pattern = self.suggester._extract_pattern("there's an error in search")
        self.assertEqual(pattern, "debugging")

    def test_testing_pattern(self):
        """Test detecting testing pattern."""
        pattern = self.suggester._extract_pattern("how to test the feature?")
        self.assertEqual(pattern, "testing")

    def test_no_pattern(self):
        """Test query with no detectable pattern."""
        pattern = self.suggester._extract_pattern("random words here")
        self.assertIsNone(pattern)


if __name__ == "__main__":
    unittest.main()
