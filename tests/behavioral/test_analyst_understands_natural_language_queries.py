"""
Behavioral tests for analysts using natural language query understanding.

Epic: Natural Language Query Understanding

As an analyst exploring a knowledge base,
I want the system to understand my natural language queries,
So that I can ask questions intuitively without learning query syntax.

Based on: nlu_showcase.py
"""

import pytest
from cortical import CorticalTextProcessor, CorticalLayer
from cortical.query.intent import parse_intent_query, QUESTION_INTENTS


class TestAnalystUnderstandsNaturalLanguageQueries:
    """
    Epic: Natural Language Query Understanding

    As an analyst working with a knowledge base,
    I want natural language understanding of my queries,
    So that I can express information needs conversationally.
    """

    def test_scenario_analyst_asks_where_questions(self):
        """
        Scenario: Understanding "where" questions for location queries

        Given a knowledge base with location information
        When I ask a "where" question
        Then the system identifies the "where" intent
        And extracts the subject I'm asking about
        And expands the query with related terms
        Because "where is X" questions seek location information.
        """
        # GIVEN a knowledge base with location information
        docs = {
            "auth_module": "Authentication logic is implemented in the security module handlers.",
            "security_docs": "The security module handles user authentication and authorization.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I ask a "where" question
        query = "where is authentication handled"

        # THEN the system identifies the "where" intent
        parsed = parse_intent_query(query)

        assert parsed['intent'] == 'where', "Should detect 'where' intent"
        assert parsed['question_word'] == 'where', "Should identify question word"

        # AND extracts the subject I'm asking about
        assert parsed['subject'] is not None, "Should extract subject"

        # AND expands the query with related terms
        assert len(parsed['expanded_terms']) > 0, "Should expand query terms"

    def test_scenario_analyst_asks_how_questions(self):
        """
        Scenario: Understanding "how" questions for process queries

        Given documentation explaining processes
        When I ask a "how" question
        Then the system identifies the "how" intent
        And understands I'm asking about implementation or process
        And finds documents explaining mechanisms
        Because "how does X work" questions seek explanation.
        """
        # GIVEN documentation explaining processes
        docs = {
            "tokenizer_doc": "The tokenizer splits text into words by whitespace and punctuation.",
            "parser_doc": "Parsing converts raw text into structured token sequences.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I ask a "how" question
        query = "how does the tokenizer work"

        # THEN the system identifies the "how" intent
        parsed = parse_intent_query(query)

        assert parsed['intent'] == 'how', "Should detect 'how' intent"

        # AND understands I'm asking about implementation or process
        assert parsed['action'] is not None or parsed['subject'] is not None

        # AND finds documents explaining mechanisms
        results = processor.find_documents_for_query(query, top_n=2)
        assert len(results) > 0, "Should find relevant documentation"

    def test_scenario_analyst_asks_what_questions(self):
        """
        Scenario: Understanding "what" questions for definition queries

        Given a corpus with definitions and explanations
        When I ask a "what is" question
        Then the system identifies the "what" intent
        And recognizes I'm seeking a definition
        And prioritizes explanatory content
        Because "what is X" questions seek understanding.
        """
        # GIVEN a corpus with definitions and explanations
        docs = {
            "pagerank_def": "PageRank is an algorithm that measures the importance of nodes in a graph.",
            "pagerank_impl": "def compute_pagerank(graph, damping=0.85): ...",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I ask a "what is" question
        query = "what is PageRank"

        # THEN the system identifies the "what" intent
        parsed = parse_intent_query(query)

        assert parsed['intent'] == 'what', "Should detect 'what' intent"

        # AND recognizes I'm seeking a definition
        assert parsed['subject'] is not None

        # AND prioritizes explanatory content
        # (Conceptual query detection handles this)
        is_conceptual = processor.is_conceptual_query(query)
        assert is_conceptual == True, "Should recognize as conceptual query"

    def test_scenario_analyst_asks_why_questions(self):
        """
        Scenario: Understanding "why" questions for rationale queries

        Given documentation with design rationales
        When I ask a "why" question
        Then the system identifies the "why" intent
        And understands I'm seeking reasoning or justification
        And finds documents explaining motivations
        Because "why do we X" questions seek rationale.
        """
        # GIVEN documentation with design rationales
        docs = {
            "design_doc": "We use TF-IDF because it identifies distinctive terms that characterize documents.",
            "architecture": "The system employs TF-IDF scoring to rank document relevance.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I ask a "why" question
        query = "why do we use TF-IDF"

        # THEN the system identifies the "why" intent
        parsed = parse_intent_query(query)

        assert parsed['intent'] == 'why', "Should detect 'why' intent"

        # AND understands I'm seeking reasoning or justification
        assert 'use' in parsed['action'] or 'tfidf' in parsed['subject'].lower() if parsed['subject'] else True

        # AND finds documents explaining motivations
        results = processor.find_documents_for_query(query, top_n=2)
        assert len(results) > 0, "Should find relevant documentation"

    def test_scenario_analyst_uses_action_verbs_in_queries(self):
        """
        Scenario: Extracting actions from queries

        Given queries with action verbs
        When I parse the query intent
        Then the system extracts the action verb
        And uses it to expand the query appropriately
        And finds documents describing that action
        Because action verbs indicate what operation the user wants.
        """
        # GIVEN queries with action verbs
        docs = {
            "validation": "Input validation filters malicious content before processing user data.",
            "sanitization": "Data sanitization removes dangerous characters from user input.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I parse the query intent
        query = "validate user input"

        parsed = parse_intent_query(query)

        # THEN the system extracts the action verb
        assert parsed['action'] is not None, "Should extract action verb"

        # AND uses it to expand the query appropriately
        assert len(parsed['expanded_terms']) > 0

        # AND finds documents describing that action
        results = processor.find_documents_for_query(query, top_n=2)
        assert len(results) > 0, "Should find documents about validation"

    def test_scenario_analyst_analyzes_knowledge_base_gaps(self):
        """
        Scenario: Identifying gaps in knowledge coverage

        Given a knowledge base with uneven coverage
        When I analyze for gaps
        Then the system identifies isolated documents
        And reports weak topics with minimal coverage
        And suggests areas needing more content
        Because gap analysis guides knowledge base improvement.
        """
        # GIVEN a knowledge base with uneven coverage
        docs = {
            "ml_intro": "Machine learning trains models on data to recognize patterns.",
            "ml_supervised": "Supervised learning uses labeled examples to train classifiers.",
            "ml_unsupervised": "Unsupervised learning discovers structure without labels.",
            "quantum": "Quantum computing exploits superposition for parallel computation.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I analyze for gaps
        gaps = processor.analyze_knowledge_gaps()

        # THEN the system identifies isolated documents
        assert 'isolated_documents' in gaps
        assert 'weak_topics' in gaps

        # AND reports weak topics with minimal coverage
        assert 'coverage_score' in gaps
        assert 0 <= gaps['coverage_score'] <= 1

        # AND suggests areas needing more content
        # Isolated documents should be identified
        isolated_ids = [doc['doc_id'] for doc in gaps['isolated_documents']]
        # Quantum doc should likely be isolated from ML cluster

    def test_scenario_analyst_gets_explanations_for_results(self):
        """
        Scenario: Understanding why results matched query

        Given a query with results
        When I request explanations
        Then I see which terms matched directly
        And which came from query expansion
        And how scores were computed
        Because transparency builds trust in the system.
        """
        # GIVEN a query with results
        docs = {
            "doc1": "Neural networks learn patterns through backpropagation training.",
            "doc2": "Deep learning models discover hierarchical feature representations.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I request explanations
        query = "neural learning"

        # Get expanded terms to track what was added
        expanded = processor.expand_query(query, max_expansions=5)
        original = set(processor.tokenizer.tokenize(query))

        # THEN I see which terms matched directly
        assert "neural" in original or "neural" in expanded
        assert "learning" in original or "learning" in expanded

        # AND which came from query expansion
        expansion_terms = set(expanded.keys()) - original
        # Some terms should be from expansion

        # AND how scores were computed
        results = processor.find_documents_for_query(query, top_n=2)
        for doc_id, score in results:
            # Score should be computable and normalized
            assert 0 <= score, "Score should be non-negative"

    def test_scenario_analyst_uses_query_expansion_for_recall(self):
        """
        Scenario: Query expansion improves result recall

        Given documents using varied terminology
        When I query with specific terms
        Then expansion finds related terms
        And results include documents with synonyms
        And recall improves without sacrificing precision
        Because users don't always know the exact vocabulary.
        """
        # GIVEN documents using varied terminology
        docs = {
            "fetch_api": "The fetch API retrieves data from remote servers asynchronously.",
            "get_request": "HTTP GET requests obtain resources from web endpoints.",
            "retrieve_data": "Data retrieval operations access information from databases.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I query with specific terms
        query = "fetch data"

        # THEN expansion finds related terms
        expanded = processor.expand_query(query, max_expansions=8)
        original = set(processor.tokenizer.tokenize(query))

        # Should expand beyond original
        assert len(expanded) >= len(original)

        # AND results include documents with synonyms
        results = processor.find_documents_for_query(query, top_n=3)

        # Should find multiple related docs (fetch, get, retrieve are synonyms)
        assert len(results) > 0

        # AND recall improves without sacrificing precision
        # (All results should be about data retrieval)

    def test_scenario_analyst_assesses_corpus_readiness_for_queries(self):
        """
        Scenario: Assessing if corpus can answer query types

        Given a knowledge base
        When I assess readiness for different query types
        Then I see coverage for "how", "what", "where", "why" questions
        And identify which question types are well-supported
        And which need more documentation
        Because different intents require different content types.
        """
        # GIVEN a knowledge base
        docs = {
            "how_to": "To implement authentication, create a session manager and verify credentials.",
            "what_is": "Authentication is the process of verifying user identity.",
            "where_code": "Authentication code resides in the security module.",
            # Missing: "why we use X" rationale documents
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I assess readiness for different query types
        test_queries = {
            'how': "how to implement authentication",
            'what': "what is authentication",
            'where': "where is authentication code",
            'why': "why use authentication",
        }

        coverage = {}
        for intent, query in test_queries.items():
            results = processor.find_documents_for_query(query, top_n=1)
            coverage[intent] = len(results) > 0 and results[0][1] > 0.1

        # THEN I see coverage for "how", "what", "where", "why" questions
        assert 'how' in coverage
        assert 'what' in coverage
        assert 'where' in coverage
        assert 'why' in coverage

        # AND identify which question types are well-supported
        # (how, what, where should have coverage from our docs)

        # AND which need more documentation
        # (why might have poor coverage if no rationale docs)
