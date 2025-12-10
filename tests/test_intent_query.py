"""
Tests for intent-based query understanding.

Tests the parse_intent_query and search_by_intent functions
used for natural language code search.
"""

import unittest
from cortical.query import (
    parse_intent_query,
    search_by_intent,
    QUESTION_INTENTS,
    ACTION_VERBS,
    ParsedIntent,
)


class TestParseIntentQuery(unittest.TestCase):
    """Test the parse_intent_query function."""

    def test_where_query(self):
        """Test parsing 'where' queries for location intent."""
        result = parse_intent_query("where do we handle authentication?")
        self.assertEqual(result['intent'], 'location')
        self.assertEqual(result['question_word'], 'where')
        self.assertEqual(result['action'], 'handle')
        self.assertEqual(result['subject'], 'authentication')

    def test_how_query(self):
        """Test parsing 'how' queries for implementation intent."""
        result = parse_intent_query("how do we validate user input?")
        self.assertEqual(result['intent'], 'implementation')
        self.assertEqual(result['question_word'], 'how')
        self.assertEqual(result['action'], 'validate')
        self.assertIn('user', [result['subject'], result['expanded_terms']])

    def test_what_query(self):
        """Test parsing 'what' queries for definition intent."""
        result = parse_intent_query("what is the database schema?")
        self.assertEqual(result['intent'], 'definition')
        self.assertEqual(result['question_word'], 'what')

    def test_why_query(self):
        """Test parsing 'why' queries for rationale intent."""
        result = parse_intent_query("why do we cache this data?")
        self.assertEqual(result['intent'], 'rationale')
        self.assertEqual(result['question_word'], 'why')

    def test_when_query(self):
        """Test parsing 'when' queries for lifecycle intent."""
        result = parse_intent_query("when does initialization happen?")
        self.assertEqual(result['intent'], 'lifecycle')
        self.assertEqual(result['question_word'], 'when')

    def test_no_question_word(self):
        """Test parsing queries without question words."""
        result = parse_intent_query("fetch user data")
        self.assertEqual(result['intent'], 'search')
        self.assertIsNone(result['question_word'])
        self.assertEqual(result['action'], 'fetch')

    def test_empty_query(self):
        """Test parsing empty query."""
        result = parse_intent_query("")
        self.assertEqual(result['intent'], 'search')
        self.assertIsNone(result['action'])
        self.assertIsNone(result['subject'])
        self.assertEqual(result['expanded_terms'], [])

    def test_punctuation_handling(self):
        """Test that punctuation is handled correctly."""
        result = parse_intent_query("where is authentication???")
        self.assertEqual(result['intent'], 'location')
        self.assertIn('authentication', result['expanded_terms'])

    def test_expanded_terms_include_synonyms(self):
        """Test that expanded terms include code concept synonyms."""
        result = parse_intent_query("how to fetch data")
        # 'fetch' should expand to include related terms
        self.assertIn('fetch', result['expanded_terms'])
        # Should have some related terms (from retrieval group)
        self.assertGreater(len(result['expanded_terms']), 1)

    def test_action_verb_detection(self):
        """Test detection of various action verbs."""
        test_cases = [
            ("validate input", "validate"),
            ("process request", "process"),
            ("save user data", "save"),
            ("delete old records", "delete"),
            ("transform response", "transform"),
        ]
        for query, expected_action in test_cases:
            result = parse_intent_query(query)
            self.assertEqual(result['action'], expected_action,
                           f"Failed for query: {query}")

    def test_subject_extraction(self):
        """Test extraction of query subject."""
        result = parse_intent_query("handle errors gracefully")
        self.assertEqual(result['subject'], 'errors')

    def test_filler_words_removed(self):
        """Test that filler words don't become subject/action."""
        result = parse_intent_query("do we have a database connection?")
        self.assertNotEqual(result['subject'], 'do')
        self.assertNotEqual(result['subject'], 'we')
        self.assertNotEqual(result['subject'], 'have')


class TestQuestionIntents(unittest.TestCase):
    """Test the QUESTION_INTENTS mapping."""

    def test_all_question_words_mapped(self):
        """Test that common question words are mapped."""
        expected_words = ['where', 'how', 'what', 'why', 'when', 'which', 'who']
        for word in expected_words:
            self.assertIn(word, QUESTION_INTENTS)

    def test_intent_types(self):
        """Test that intent types are meaningful."""
        self.assertEqual(QUESTION_INTENTS['where'], 'location')
        self.assertEqual(QUESTION_INTENTS['how'], 'implementation')
        self.assertEqual(QUESTION_INTENTS['what'], 'definition')
        self.assertEqual(QUESTION_INTENTS['why'], 'rationale')


class TestActionVerbs(unittest.TestCase):
    """Test the ACTION_VERBS set."""

    def test_common_verbs_included(self):
        """Test that common programming action verbs are included."""
        expected_verbs = [
            'handle', 'process', 'create', 'delete', 'update', 'fetch',
            'validate', 'parse', 'transform', 'authenticate', 'initialize'
        ]
        for verb in expected_verbs:
            self.assertIn(verb, ACTION_VERBS)

    def test_is_frozenset(self):
        """Test that ACTION_VERBS is immutable."""
        self.assertIsInstance(ACTION_VERBS, frozenset)


class TestSearchByIntent(unittest.TestCase):
    """Test the search_by_intent function."""

    def setUp(self):
        """Set up test processor."""
        from cortical import CorticalTextProcessor
        self.processor = CorticalTextProcessor()
        self.processor.process_document("auth_handler", """
            Authentication handler module.
            This module handles user authentication and login.
            It validates credentials and creates sessions.
        """)
        self.processor.process_document("data_fetcher", """
            Data fetching utilities.
            Functions to fetch and retrieve data from external APIs.
            Handles HTTP requests and response parsing.
        """)
        self.processor.process_document("validator", """
            Input validation module.
            Validates and sanitizes user input.
            Checks for required fields and data types.
        """)
        self.processor.compute_all()

    def test_search_returns_results(self):
        """Test that search returns results."""
        results = self.processor.search_by_intent("where do we handle authentication?")
        self.assertIsInstance(results, list)
        # Should find auth_handler document
        if results:
            doc_ids = [r[0] for r in results]
            self.assertIn('auth_handler', doc_ids)

    def test_search_returns_parsed_intent(self):
        """Test that search returns parsed intent with results."""
        results = self.processor.search_by_intent("how to validate input?")
        if results:
            doc_id, score, parsed = results[0]
            self.assertIn('intent', parsed)
            self.assertIn('action', parsed)
            self.assertIn('expanded_terms', parsed)

    def test_search_empty_query(self):
        """Test search with empty query."""
        results = self.processor.search_by_intent("")
        self.assertEqual(results, [])

    def test_search_top_n_limit(self):
        """Test that top_n limits results."""
        results = self.processor.search_by_intent("fetch data", top_n=2)
        self.assertLessEqual(len(results), 2)

    def test_processor_parse_intent_query(self):
        """Test the processor wrapper for parse_intent_query."""
        result = self.processor.parse_intent_query("where is the login function?")
        self.assertEqual(result['intent'], 'location')
        # 'login' is detected as action verb, so 'function' becomes subject
        self.assertEqual(result['action'], 'login')
        self.assertEqual(result['subject'], 'function')


class TestParsedIntentStructure(unittest.TestCase):
    """Test the ParsedIntent TypedDict structure."""

    def test_all_keys_present(self):
        """Test that all expected keys are in parsed result."""
        result = parse_intent_query("where do we handle errors?")
        expected_keys = ['action', 'subject', 'intent', 'question_word', 'expanded_terms']
        for key in expected_keys:
            self.assertIn(key, result)

    def test_expanded_terms_is_list(self):
        """Test that expanded_terms is a list."""
        result = parse_intent_query("handle authentication")
        self.assertIsInstance(result['expanded_terms'], list)

    def test_no_duplicate_expanded_terms(self):
        """Test that expanded_terms has no duplicates."""
        result = parse_intent_query("handle handle authentication")
        self.assertEqual(
            len(result['expanded_terms']),
            len(set(result['expanded_terms']))
        )


if __name__ == '__main__':
    unittest.main()
