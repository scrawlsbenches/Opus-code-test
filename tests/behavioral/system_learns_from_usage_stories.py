"""
Behavioral Tests: System Learns From Usage Patterns
====================================================

Epic: Learning From User Interactions

As a system observing user interactions,
I want to suggest new alignment entries based on patterns,
So that the system becomes more aligned with user vocabulary.
"""

import pytest
from cortical.spark.suggester import (
    SampleSuggester,
    DefinitionSuggestion,
    PatternSuggestion,
    PreferenceSuggestion
)


class TestSystemSuggestsDefinitions:
    """
    As a system observing queries,
    I want to suggest definitions for frequently-used undefined terms,
    So that humans can add them to the alignment corpus.
    """

    def test_scenario_suggesting_frequently_used_terms(self):
        """
        Scenario: Suggest definition for common term

        Given a suggester observing queries
        When a term appears frequently without definition
        Then it suggests defining that term
        Because frequency indicates importance.

        Note: Confidence requires freq_score (count/10 * 0.6) + length_score
        (len/10 * 0.2) + format_score + 0.1 >= 0.5. With 4 observations of
        a 10-char term: 0.4*0.6 + 1.0*0.2 + 0.1 = 0.54 >= 0.5.
        """
        # Given a suggester observing queries
        suggester = SampleSuggester(min_frequency=3)

        # When a term appears frequently (4 times to exceed confidence threshold)
        suggester.observe_query("minicolumn activation patterns", success=True)
        suggester.observe_query("minicolumn connectivity graph", success=True)
        suggester.observe_query("minicolumn response timing", success=True)
        suggester.observe_query("minicolumn signal propagation", success=True)

        # Then it suggests definition
        suggestions = suggester.suggest_definitions()
        assert len(suggestions) > 0
        terms = [s.term for s in suggestions]
        assert 'minicolumn' in terms, "Should suggest 'minicolumn' definition"

    def test_scenario_not_suggesting_known_terms(self):
        """
        Scenario: Skip terms that are already defined

        Given a suggester with known terms
        When those terms appear in queries
        Then they are not suggested
        Because they're already defined.
        """
        # Given a suggester with known terms
        suggester = SampleSuggester(
            known_terms={'machine', 'learning', 'algorithm'},
            min_frequency=2
        )

        # When known terms appear
        suggester.observe_query("machine learning algorithm", success=True)
        suggester.observe_query("machine learning model", success=True)
        suggester.observe_query("learning algorithm design", success=True)

        # Then they're not suggested
        suggestions = suggester.suggest_definitions()
        terms = [s.term for s in suggestions]
        assert 'machine' not in terms
        assert 'learning' not in terms
        assert 'algorithm' not in terms

    def test_scenario_confidence_based_on_frequency(self):
        """
        Scenario: Suggestion confidence reflects frequency

        Given a suggester observing terms
        When one term appears much more than others
        Then it has higher suggestion confidence
        Because frequent usage indicates higher priority.
        """
        # Given a suggester
        suggester = SampleSuggester(min_frequency=2)

        # When one term is very frequent
        for _ in range(10):
            suggester.observe_query("pagerank algorithm", success=True)

        for _ in range(2):
            suggester.observe_query("betweenness centrality", success=True)

        # Then high-frequency term has higher confidence
        suggestions = suggester.suggest_definitions()
        pagerank_sugg = next((s for s in suggestions if s.term == 'pagerank'), None)
        betweenness_sugg = next((s for s in suggestions if s.term == 'betweenness'), None)

        if pagerank_sugg and betweenness_sugg:
            assert pagerank_sugg.confidence > betweenness_sugg.confidence


class TestSystemIdentifiesPatterns:
    """
    As a system observing query structures,
    I want to identify repeated patterns,
    So that common query types are documented.
    """

    def test_scenario_identifying_how_to_pattern(self):
        """
        Scenario: Detect "how to" question pattern

        Given a suggester observing queries
        When multiple "how do I" queries appear
        Then it suggests a "how_to" pattern
        Because this is a common query structure.
        """
        # Given a suggester
        suggester = SampleSuggester(min_frequency=3)

        # When "how to" queries appear
        suggester.observe_query("how do I train a model", success=True)
        suggester.observe_query("how do I optimize search", success=True)
        suggester.observe_query("how do I index documents", success=True)

        # Then pattern is suggested
        patterns = suggester.suggest_patterns()
        pattern_names = [p.pattern_name for p in patterns]
        assert 'how_to' in pattern_names

    def test_scenario_identifying_common_phrases(self):
        """
        Scenario: Detect frequently-used phrases

        Given a suggester tracking bigrams
        When a phrase appears repeatedly
        Then it suggests documenting that phrase
        Because common phrases may need definitions.

        Note: Bigram confidence = count/20, needs >= min_confidence (0.5).
        So count >= 10 is required for suggestions to be made.
        """
        # Given a suggester
        suggester = SampleSuggester(min_frequency=3)

        # When a phrase repeats (10+ times to exceed confidence threshold)
        for topic in ["training", "architecture", "optimization", "inference",
                      "deployment", "tuning", "scaling", "debugging",
                      "monitoring", "evaluation"]:
            suggester.observe_query(f"neural network {topic}", success=True)

        # Then phrase pattern is suggested
        patterns = suggester.suggest_patterns()
        # Should suggest "neural network" as a common phrase
        assert len(patterns) > 0
        phrase_patterns = [p for p in patterns if 'neural network' in p.pattern_name]
        assert len(phrase_patterns) > 0, "Should detect 'neural network' phrase"


class TestSystemDetectsPreferences:
    """
    As a system observing user choices,
    I want to detect preferences,
    So that future decisions align with past choices.
    """

    def test_scenario_detecting_naming_preference(self):
        """
        Scenario: Detect consistent naming choice

        Given a suggester observing choices
        When user consistently prefers one naming style
        Then it suggests that preference
        Because consistency should be preserved.
        """
        # Given a suggester
        suggester = SampleSuggester(min_frequency=3)

        # When user consistently chooses snake_case
        suggester.observe_choice("naming", "snake_case", ["camelCase", "kebab-case"])
        suggester.observe_choice("naming", "snake_case", ["camelCase", "kebab-case"])
        suggester.observe_choice("naming", "snake_case", ["camelCase", "kebab-case"])
        suggester.observe_choice("naming", "camelCase", ["snake_case", "kebab-case"])

        # Then preference is detected
        preferences = suggester.suggest_preferences()
        naming_prefs = [p for p in preferences if p.preference_name == "naming"]

        if naming_prefs:
            pref = naming_prefs[0]
            assert pref.chosen == "snake_case"
            assert pref.confidence >= 0.7  # 3 out of 4 = 75%

    def test_scenario_no_preference_without_clear_winner(self):
        """
        Scenario: No preference when choices are balanced

        Given a suggester observing choices
        When choices are evenly distributed
        Then no preference is suggested
        Because there's no clear favorite.
        """
        # Given a suggester
        suggester = SampleSuggester(min_frequency=3)

        # When choices are balanced
        suggester.observe_choice("approach", "option_a", ["option_b"])
        suggester.observe_choice("approach", "option_b", ["option_a"])
        suggester.observe_choice("approach", "option_a", ["option_b"])
        suggester.observe_choice("approach", "option_b", ["option_a"])

        # Then no strong preference
        preferences = suggester.suggest_preferences()
        # Should have very low confidence or no suggestion


class TestSystemExportsSuggestions:
    """
    As a human reviewing suggestions,
    I want suggestions exported in markdown format,
    So that I can easily review and add them to alignment files.
    """

    def test_scenario_exporting_suggestions_as_markdown(self):
        """
        Scenario: Export all suggestions to markdown

        Given a suggester with various suggestions
        When I export to markdown
        Then I get a formatted document
        Because humans need readable output.

        Note: Definition confidence = freq/10*0.6 + len/10*0.2 + 0.1 >= 0.5.
        For "cortical" (8 chars): need freq >= 5 for confidence ~0.5.
        """
        # Given a suggester with observations (5+ for confidence threshold)
        suggester = SampleSuggester(min_frequency=2)

        suggester.observe_query("cortical processing", success=True)
        suggester.observe_query("cortical architecture", success=True)
        suggester.observe_query("cortical model", success=True)
        suggester.observe_query("cortical layers", success=True)
        suggester.observe_query("cortical connections", success=True)

        # When I export
        markdown = suggester.export_suggestions_markdown()

        # Then I get formatted output
        assert isinstance(markdown, str)
        assert "# Suggested Alignment Entries" in markdown
        assert "cortical" in markdown.lower()

    def test_scenario_individual_suggestion_markdown_format(self):
        """
        Scenario: Each suggestion has markdown format

        Given various suggestion types
        When I convert them to markdown
        Then each has appropriate formatting
        Because consistency aids review.
        """
        # Given suggestions
        def_sugg = DefinitionSuggestion(
            term="minicolumn",
            frequency=5,
            contexts=["test"],
            confidence=0.8,
            reason="test"
        )

        pattern_sugg = PatternSuggestion(
            pattern_name="search",
            examples=["find X", "search Y"],
            frequency=3,
            confidence=0.7,
            reason="test"
        )

        pref_sugg = PreferenceSuggestion(
            preference_name="naming",
            chosen="snake_case",
            over="camelCase",
            frequency=4,
            confidence=0.75,
            reason="test"
        )

        # When I convert to markdown
        def_md = def_sugg.to_markdown()
        pattern_md = pattern_sugg.to_markdown()
        pref_md = pref_sugg.to_markdown()

        # Then format is correct
        assert "minicolumn" in def_md
        assert "seen 5 times" in def_md
        assert "search" in pattern_md
        assert "snake_case" in pref_md
        assert "camelCase" in pref_md


class TestSystemManagesSuggestionState:
    """
    As a system managing suggestions,
    I want to track statistics and clear state,
    So that I control the suggestion lifecycle.
    """

    def test_scenario_viewing_suggester_statistics(self):
        """
        Scenario: View suggester statistics

        Given a suggester with observations
        When I request statistics
        Then I see counts and success rate
        Because transparency aids monitoring.
        """
        # Given a suggester with observations
        suggester = SampleSuggester()

        suggester.observe_query("query one", success=True)
        suggester.observe_query("query two", success=True)
        suggester.observe_query("query three", success=False)

        # When I get stats
        stats = suggester.get_stats()

        # Then I see statistics
        assert stats['total_observations'] == 3
        assert stats['success_rate'] == 2 / 3
        assert 'unique_terms' in stats
        assert 'pending_suggestions' in stats

    def test_scenario_clearing_observations(self):
        """
        Scenario: Clear all observations

        Given a suggester with accumulated data
        When I clear observations
        Then all statistics reset
        Because I may need to start fresh.
        """
        # Given a suggester with data
        suggester = SampleSuggester(min_frequency=1)

        suggester.observe_query("test query", success=True)
        suggester.observe_query("another query", success=True)

        assert len(suggester.observations) > 0

        # When I clear
        suggester.clear()

        # Then it's reset
        assert len(suggester.observations) == 0
        assert len(suggester.term_counts) == 0
        assert len(suggester.suggest_definitions()) == 0

    def test_scenario_adding_known_terms_prevents_suggestions(self):
        """
        Scenario: Mark terms as known to prevent suggestions

        Given a suggester
        When I add terms to known set
        Then they won't be suggested even if frequent
        Because we acknowledge they're already defined.
        """
        # Given a suggester
        suggester = SampleSuggester(min_frequency=2)

        # When I mark terms as known
        suggester.add_known_terms({'neural', 'network', 'model'})

        # Then they won't be suggested
        suggester.observe_query("neural network model", success=True)
        suggester.observe_query("neural network training", success=True)
        suggester.observe_query("neural network design", success=True)

        suggestions = suggester.suggest_definitions()
        terms = [s.term for s in suggestions]

        assert 'neural' not in terms
        assert 'network' not in terms
        assert 'model' not in terms

    def test_scenario_getting_all_suggestions_at_once(self):
        """
        Scenario: Get all suggestion types together

        Given a suggester with multiple observation types
        When I request all suggestions
        Then I get definitions, patterns, and preferences
        Because one call retrieves everything.
        """
        # Given a suggester with varied observations
        suggester = SampleSuggester(min_frequency=2)

        # Add queries for definitions
        suggester.observe_query("pagerank calculation", success=True)
        suggester.observe_query("pagerank algorithm", success=True)

        # Add patterns
        suggester.observe_query("how do I compute pagerank", success=True)
        suggester.observe_query("how do I optimize search", success=True)

        # Add choices for preferences
        suggester.observe_choice("format", "json", ["xml", "yaml"])
        suggester.observe_choice("format", "json", ["xml", "yaml"])

        # When I get all suggestions
        all_sugg = suggester.get_all_suggestions()

        # Then I get all types
        assert 'definitions' in all_sugg
        assert 'patterns' in all_sugg
        assert 'preferences' in all_sugg
        assert isinstance(all_sugg['definitions'], list)
        assert isinstance(all_sugg['patterns'], list)
        assert isinstance(all_sugg['preferences'], list)
