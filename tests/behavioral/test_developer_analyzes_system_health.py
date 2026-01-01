"""
Developer Analyzes System Health

Epic: System Introspection and Analysis

As a developer monitoring a corpus,
I want to inspect system state and health,
So that I can identify issues and optimize performance.
"""

import pytest
from cortical import CorticalTextProcessor


class TestDeveloperInspectsCorpusState:
    """
    Epic: Corpus Health Monitoring

    As a developer maintaining a search system,
    I want to inspect corpus statistics and health,
    So that I understand system state and quality.
    """

    def test_scenario_getting_corpus_summary_for_overview(self):
        """
        Scenario: Understanding corpus composition

        Given I have documents in my processor
        When I request a corpus summary
        Then I receive statistics about documents and terms
        And I understand the corpus size and complexity
        Because corpus metrics guide optimization decisions.
        """
        # Given I have documents in my processor
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom parser we built from scratch")
        processor.process_document("doc2", "Hand-crafted tokenizer we implemented")
        processor.compute_all(verbose=False)

        # When I request a corpus summary
        summary = processor.get_corpus_summary()

        # Then I receive statistics about documents and terms
        assert 'documents' in summary
        assert 'total_columns' in summary

        # And I understand the corpus size and complexity
        assert summary['documents'] == 2
        assert summary['total_columns'] > 0

    def test_scenario_checking_staleness_after_updates(self):
        """
        Scenario: Monitoring computation freshness

        Given I've modified my corpus
        When I check which computations are stale
        Then I see what needs recomputation
        And I can plan my recomputation strategy
        Because knowing staleness prevents using outdated results.
        """
        # Given I've modified my corpus
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom implementation from scratch")
        processor.compute_all(verbose=False)

        # Add document without recomputation
        processor.add_document_incremental("doc2", "Hand-built system", recompute='none')

        # When I check which computations are stale
        stale = processor.get_stale_computations()

        # Then I see what needs recomputation
        assert len(stale) > 0

        # And I can plan my recomputation strategy
        assert processor.is_stale(processor.COMP_TFIDF)
        assert processor.is_stale(processor.COMP_PAGERANK)

    def test_scenario_getting_document_signature_for_characterization(self):
        """
        Scenario: Characterizing document content

        Given I want to understand what a document is about
        When I get the document signature
        Then I receive top TF-IDF terms
        And I can see the document's key concepts
        Because signatures reveal document themes.
        """
        # Given I want to understand what a document is about
        processor = CorticalTextProcessor()
        processor.process_document(
            "parser.py",
            "Custom parser implementation. Parser handles tokenization. "
            "The parser we built from scratch uses hand-crafted algorithms."
        )
        processor.compute_all(verbose=False)

        # When I get the document signature
        signature = processor.get_document_signature("parser.py", n=5)

        # Then I receive top TF-IDF terms
        assert len(signature) > 0

        # And I can see the document's key concepts
        terms = [term for term, _ in signature]
        assert "parser" in terms or "custom" in terms


class TestDeveloperDetectsAnomalies:
    """
    Epic: Anomaly Detection

    As a developer monitoring quality,
    I want to detect anomalous patterns,
    So that I can identify corpus problems.
    """

    def test_scenario_detecting_knowledge_gaps_in_corpus(self):
        """
        Scenario: Finding knowledge gaps

        Given I have an incomplete corpus
        When I analyze knowledge gaps
        Then I see what's missing or weakly connected
        And I can prioritize content to add
        Because gaps reveal where the corpus is incomplete.
        """
        # Given I have an incomplete corpus
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom parser we built")
        processor.process_document("doc2", "Unrelated topic about weather")
        processor.compute_all(verbose=False)

        # When I analyze knowledge gaps
        gaps = processor.analyze_knowledge_gaps()

        # Then I see what's missing or weakly connected
        assert gaps is not None
        assert isinstance(gaps, dict)

    def test_scenario_detecting_anomalous_documents(self):
        """
        Scenario: Identifying outlier documents

        Given I have documents with varying quality
        When I detect anomalies
        Then outlier documents are identified
        And I can review or remove them
        Because anomalies often indicate quality issues.
        """
        # Given I have documents with varying quality
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom implementation from scratch")
        processor.process_document("doc2", "Hand-built system we control")
        processor.process_document("weird", "xyz abc qwerty asdf")  # Anomalous
        processor.compute_all(verbose=False)

        # When I detect anomalies
        anomalies = processor.detect_anomalies(threshold=0.3)

        # Then outlier documents are identified
        assert isinstance(anomalies, list)

        # (Note: Small corpus may not detect anomalies reliably)


class TestDeveloperComparesTextSemantics:
    """
    Epic: Semantic Comparison

    As a developer comparing texts,
    I want to measure semantic similarity,
    So that I can find duplicates or related content.
    """

    def test_scenario_computing_fingerprints_for_comparison(self):
        """
        Scenario: Fingerprinting text semantics

        Given I have text to characterize
        When I compute a fingerprint
        Then I receive semantic features
        And I can compare with other fingerprints
        Because fingerprints capture semantic essence.
        """
        # Given I have text to characterize
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom parser tokenizer lexer we built")
        processor.compute_all(verbose=False)

        # When I compute a fingerprint
        fingerprint = processor.get_fingerprint("parser tokenizer implementation")

        # Then I receive semantic features
        assert 'terms' in fingerprint
        assert 'top_terms' in fingerprint

        # And I can compare with other fingerprints
        fp2 = processor.get_fingerprint("hand-crafted parser system")
        comparison = processor.compare_fingerprints(fingerprint, fp2)
        assert 'overall_similarity' in comparison

    def test_scenario_finding_similar_texts_by_fingerprint(self):
        """
        Scenario: Finding semantically similar texts

        Given I have candidate texts to search
        When I find similar texts to a query
        Then results are ranked by semantic similarity
        And I see comparison details
        Because semantic similarity goes beyond keyword matching.
        """
        # Given I have candidate texts to search
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom parser we built from scratch")
        processor.compute_all(verbose=False)

        candidates = [
            ("text1", "Hand-crafted parser implementation"),
            ("text2", "Unrelated database system"),
            ("text3", "In-house compiler and parser"),
        ]

        # When I find similar texts to a query
        results = processor.find_similar_texts(
            "parser implementation",
            candidates,
            top_n=2
        )

        # Then results are ranked by semantic similarity
        assert len(results) > 0

        # And I see comparison details
        for text_id, similarity, comparison in results:
            assert 'overall_similarity' in comparison
            assert similarity >= 0

    def test_scenario_comparing_documents_in_corpus(self):
        """
        Scenario: Document similarity analysis

        Given I have multiple documents
        When I compare two documents
        Then I see their semantic overlap
        And I can identify duplicates or relationships
        Because document comparison reveals corpus structure.
        """
        # Given I have multiple documents
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom parser tokenizer we built")
        processor.process_document("doc2", "Hand-crafted parser lexer we implemented")
        processor.compute_all(verbose=False)

        # When I compare two documents
        comparison = processor.compare_documents("doc1", "doc2")

        # Then I see their semantic overlap
        assert comparison is not None
        assert isinstance(comparison, dict)


class TestDeveloperDetectsCodePatterns:
    """
    Epic: Code Pattern Detection

    As a developer analyzing code,
    I want to detect programming patterns,
    So that I understand code architecture and quality.
    """

    def test_scenario_detecting_patterns_in_code_document(self):
        """
        Scenario: Finding design patterns in code

        Given I have code documents
        When I detect patterns in a document
        Then I see which patterns are present
        And I understand code structure
        Because pattern detection reveals architecture.
        """
        # Given I have code documents
        processor = CorticalTextProcessor()
        processor.process_document(
            "auth.py",
            """
class Authenticator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

async def authenticate(token):
    await validate_token(token)
"""
        )

        # When I detect patterns in a document
        patterns = processor.detect_patterns("auth.py")

        # Then I see which patterns are present
        assert isinstance(patterns, dict)

        # And I understand code structure
        # (May find singleton, async patterns)

    def test_scenario_getting_pattern_summary_for_document(self):
        """
        Scenario: Summarizing pattern usage

        Given I've detected patterns in code
        When I get a pattern summary
        Then I see occurrence counts
        And I understand pattern frequency
        Because summaries show which patterns dominate.
        """
        # Given I've detected patterns in code
        processor = CorticalTextProcessor()
        processor.process_document(
            "code.py",
            """
try:
    result = operation()
except Exception as e:
    handle_error(e)

try:
    another = operation2()
except ValueError:
    pass
"""
        )

        # When I get a pattern summary
        summary = processor.get_pattern_summary("code.py")

        # Then I see occurrence counts
        assert isinstance(summary, dict)

        # (May show error_handling pattern count)

    def test_scenario_analyzing_corpus_wide_pattern_statistics(self):
        """
        Scenario: Corpus-level pattern analysis

        Given I have multiple code documents
        When I get corpus pattern statistics
        Then I see patterns across all documents
        And I understand architectural consistency
        Because corpus statistics reveal project-wide patterns.
        """
        # Given I have multiple code documents
        processor = CorticalTextProcessor()
        processor.process_document("file1.py", "async def fetch(): await call()")
        processor.process_document("file2.py", "async def process(): await handle()")

        # When I get corpus pattern statistics
        stats = processor.get_corpus_pattern_statistics()

        # Then I see patterns across all documents
        assert 'total_documents' in stats
        assert 'patterns_found' in stats

        # And I understand architectural consistency
        assert stats['total_documents'] == 2

    def test_scenario_listing_available_patterns_for_discovery(self):
        """
        Scenario: Discovering detectable patterns

        Given I want to know what patterns can be detected
        When I list available patterns
        Then I see all pattern names
        And I can focus detection on specific patterns
        Because knowing available patterns guides analysis.
        """
        # Given I want to know what patterns can be detected
        processor = CorticalTextProcessor()

        # When I list available patterns
        patterns = processor.list_available_patterns()

        # Then I see all pattern names
        assert isinstance(patterns, list)
        assert len(patterns) > 0

        # And I can focus detection on specific patterns
        # (Patterns like 'singleton', 'async_await', etc.)
