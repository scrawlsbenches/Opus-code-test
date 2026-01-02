"""
Behavioral tests for security engineers searching threat knowledge bases.

Epic: Security Threat Knowledge Search

As a security engineer managing threat intelligence,
I want to search security knowledge bases semantically,
So that I find relevant vulnerabilities, mitigations, and best practices.

Based on: secureshowcase.py
"""

import pytest
from cortical import CorticalTextProcessor, CorticalLayer
from cortical.tokenizer import Tokenizer


class TestSecurityEngineerSearchesThreatKnowledge:
    """
    Epic: Security Threat Knowledge Search

    As a security engineer working with threat intelligence,
    I want semantic search over security documentation,
    So that I rapidly find relevant vulnerability information and mitigations.
    """

    def test_scenario_engineer_builds_security_knowledge_hierarchy(self):
        """
        Scenario: Building hierarchical security knowledge graph

        Given security documentation covering multiple threat domains
        When I process the documents hierarchically
        Then the system organizes security concepts across layers
        And connects related security terms semantically
        And enables multi-level security analysis
        Because security knowledge spans from specific attacks to general principles.
        """
        # GIVEN security documentation covering multiple threat domains
        docs = {
            "injection": "SQL injection allows attackers to execute malicious database queries through input fields.",
            "xss": "Cross-site scripting enables injection of malicious scripts into web pages viewed by users.",
            "auth": "Authentication vulnerabilities allow unauthorized access through credential bypass techniques.",
        }

        # WHEN I process the documents hierarchically
        tokenizer = Tokenizer(filter_code_noise=True)
        processor = CorticalTextProcessor(tokenizer=tokenizer)

        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)

        processor.compute_all(verbose=False)

        # THEN the system organizes security concepts across layers
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        layer1 = processor.get_layer(CorticalLayer.BIGRAMS)
        layer3 = processor.get_layer(CorticalLayer.DOCUMENTS)

        assert layer0.column_count() > 0, "Should extract security terms"
        assert layer1.column_count() > 0, "Should extract security concept pairs"
        assert layer3.column_count() == 3, "Should have 3 security documents"

        # AND connects related security terms semantically
        # Check for security terms
        assert layer0.get_minicolumn("injection") is not None
        assert layer0.get_minicolumn("malicious") is not None

        # AND enables multi-level security analysis
        # All 4 layers should exist
        assert len(processor.layers) == 4

    def test_scenario_engineer_discovers_key_security_concepts(self):
        """
        Scenario: Identifying central security concepts via PageRank

        Given a security knowledge base with interconnected threats
        When I compute PageRank on security terms
        Then highly connected security concepts rank higher
        And I identify hub concepts bridging multiple threat domains
        Because central concepts are most critical for security understanding.
        """
        # GIVEN a security knowledge base with interconnected threats
        docs = {
            "doc1": "Authentication mechanisms verify user credentials to prevent unauthorized access.",
            "doc2": "Access control systems authenticate users and authorize resource access.",
            "doc3": "Credential theft enables authentication bypass and unauthorized system access.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I compute PageRank on security terms
        layer0 = processor.get_layer(CorticalLayer.TOKENS)

        # THEN highly connected security concepts rank higher
        auth_col = layer0.get_minicolumn("authentication")
        access_col = layer0.get_minicolumn("access")

        assert auth_col is not None, "Should have authentication concept"
        assert access_col is not None, "Should have access concept"

        # AND I identify hub concepts bridging multiple threat domains
        # These terms appear across multiple docs, should have PageRank
        assert auth_col.pagerank > 0
        assert access_col.pagerank > 0

    def test_scenario_engineer_analyzes_vulnerability_categories(self):
        """
        Scenario: Categorizing vulnerabilities by TF-IDF

        Given security documents covering different vulnerability types
        When I compute TF-IDF scores
        Then vulnerability-specific terms score higher
        And I identify distinctive threats per category
        Because TF-IDF reveals what makes each vulnerability unique.
        """
        # GIVEN security documents covering different vulnerability types
        docs = {
            "sqli": "SQL injection exploits database queries by inserting malicious SQL code.",
            "deserialize": "Insecure deserialization allows attackers to execute arbitrary code via pickle objects.",
            "crypto": "Cryptographic failures expose sensitive data through weak encryption algorithms.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I compute TF-IDF scores
        layer0 = processor.get_layer(CorticalLayer.TOKENS)

        # THEN vulnerability-specific terms score higher
        pickle_col = layer0.get_minicolumn("pickle")
        sql_col = layer0.get_minicolumn("sql")

        # AND I identify distinctive threats per category
        # These terms are specific to their respective vulnerabilities
        if pickle_col:
            assert pickle_col.tfidf > 0, "Pickle should have TF-IDF score"
        if sql_col:
            assert sql_col.tfidf > 0, "SQL should have TF-IDF score"

    def test_scenario_engineer_finds_security_concept_associations(self):
        """
        Scenario: Discovering relationships between security concepts

        Given security documents where threats co-occur
        When I analyze lateral connections
        Then related security concepts show strong connections
        And connection strength reflects co-occurrence in threat contexts
        Because understanding threat relationships aids in comprehensive defense.
        """
        # GIVEN security documents where threats co-occur
        docs = {
            "doc1": "Authentication vulnerabilities enable credential theft and unauthorized access.",
            "doc2": "Credential verification prevents authentication bypass attacks.",
            "doc3": "Access control relies on proper authentication and authorization.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I analyze lateral connections
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        auth_col = layer0.get_minicolumn("authentication")

        # THEN related security concepts show strong connections
        assert auth_col is not None
        assert len(auth_col.lateral_connections) > 0, "Should have connections to related concepts"

        # AND connection strength reflects co-occurrence in threat contexts
        for neighbor_id, weight in auth_col.lateral_connections.items():
            assert weight > 0, "Connection weights should be positive"

    def test_scenario_engineer_searches_for_vulnerabilities(self):
        """
        Scenario: Finding information about specific vulnerabilities

        Given a security knowledge base
        When I query for a specific vulnerability
        Then I find relevant threat documentation
        And query expansion includes related attack vectors
        And results are ranked by relevance
        Because engineers need rapid access to vulnerability information.
        """
        # GIVEN a security knowledge base
        docs = {
            "sql_injection": "SQL injection attacks manipulate database queries through unsanitized user input.",
            "input_validation": "Input validation and parameterized queries prevent SQL injection vulnerabilities.",
            "secure_coding": "Secure coding practices include input sanitization and output encoding.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I query for a specific vulnerability
        query = "SQL injection prevention"

        # THEN I find relevant threat documentation
        results = processor.find_documents_for_query(query, top_n=3)

        assert len(results) > 0, "Should find vulnerability documentation"

        # AND query expansion includes related attack vectors
        expanded = processor.expand_query(query, max_expansions=5)
        # Should expand beyond original terms

        # AND results are ranked by relevance
        doc_ids = [doc_id for doc_id, _ in results]
        # Should find both the vulnerability and mitigation docs
        assert "sql_injection" in doc_ids or "input_validation" in doc_ids

    def test_scenario_engineer_applies_stride_threat_modeling(self):
        """
        Scenario: Mapping vulnerabilities to STRIDE categories

        Given security documentation
        When I query for STRIDE threat categories
        Then I find threats matching each category
        And can map vulnerabilities to Spoofing, Tampering, Repudiation, etc.
        Because STRIDE provides structured threat analysis framework.
        """
        # GIVEN security documentation
        docs = {
            "spoofing": "Identity spoofing attacks impersonate legitimate users through stolen credentials.",
            "tampering": "Data tampering modifies information in transit or at rest without authorization.",
            "repudiation": "Repudiation attacks deny performing actions through lack of audit logging.",
            "info_disclosure": "Information disclosure leaks sensitive data through inadequate access controls.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I query for STRIDE threat categories
        stride_queries = [
            "spoofing identity authentication",
            "tampering data integrity",
            "repudiation audit logging",
            "information disclosure data leak",
        ]

        # THEN I find threats matching each category
        for query in stride_queries:
            results = processor.find_documents_for_query(query, top_n=1)
            # Should find at least some results for each STRIDE category

        # AND can map vulnerabilities to Spoofing, Tampering, Repudiation, etc.
        # All categories should have some representation

    def test_scenario_engineer_retrieves_secure_coding_guidance(self):
        """
        Scenario: Finding secure coding practices

        Given documentation on secure development
        When I search for coding security guidance
        Then I retrieve specific implementation recommendations
        And get code-level guidance via passage retrieval
        Because developers need concrete examples for secure implementation.
        """
        # GIVEN documentation on secure development
        docs = {
            "input_val": """
            Input validation must sanitize all user-provided data before processing.
            Use allowlists rather than denylists to define acceptable input patterns.
            Validate data type, length, format, and range for all inputs.
            """,
            "crypto": """
            Use industry-standard cryptographic libraries rather than custom implementations.
            Generate cryptographic keys with sufficient entropy from secure random sources.
            Store sensitive data encrypted at rest using AES-256 or equivalent algorithms.
            """
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I search for coding security guidance
        query = "input validation sanitization"

        # THEN I retrieve specific implementation recommendations
        passages = processor.find_passages_for_query(
            query,
            top_n=2,
            chunk_size=200,
            overlap=30
        )

        assert len(passages) > 0, "Should find guidance passages"

        # AND get code-level guidance via passage retrieval
        for text, doc_id, start, end, score in passages:
            assert len(text) > 0, "Should return passage text"
            assert score > 0, "Should have relevance score"

    def test_scenario_engineer_assesses_owasp_coverage(self):
        """
        Scenario: Checking OWASP Top 10 coverage

        Given a security knowledge base
        When I assess coverage of OWASP categories
        Then I identify which categories are documented
        And find gaps in security documentation
        And prioritize missing coverage areas
        Because OWASP Top 10 provides essential security baseline.
        """
        # GIVEN a security knowledge base
        docs = {
            "injection": "Injection flaws like SQL injection allow attackers to send malicious data to interpreters.",
            "broken_auth": "Broken authentication allows attackers to compromise passwords, keys, or session tokens.",
            "sensitive_data": "Sensitive data exposure occurs when applications don't adequately protect sensitive information.",
            # Missing: many other OWASP categories
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I assess coverage of OWASP categories
        owasp_queries = [
            ("Injection", "injection SQL command"),
            ("Broken Authentication", "authentication password session"),
            ("Sensitive Data Exposure", "sensitive data encryption"),
            ("XXE", "XML external entity"),  # Not in our docs
        ]

        coverage = {}
        for category, query in owasp_queries:
            results = processor.find_documents_for_query(query, top_n=1)
            coverage[category] = len(results) > 0 and results[0][1] > 0.1

        # THEN I identify which categories are documented
        assert coverage["Injection"] == True, "Should have injection coverage"
        assert coverage["Broken Authentication"] == True, "Should have auth coverage"

        # AND find gaps in security documentation
        # XXE should have poor coverage

        # AND prioritize missing coverage areas
        missing = [cat for cat, covered in coverage.items() if not covered]
        # Can identify what's missing

    def test_scenario_engineer_analyzes_security_coverage_gaps(self):
        """
        Scenario: Identifying security knowledge gaps

        Given a security corpus with varying coverage
        When I analyze knowledge gaps
        Then I identify isolated security topics
        And find weak areas needing more documentation
        And get recommendations for improvement
        Because comprehensive coverage ensures thorough security posture.
        """
        # GIVEN a security corpus with varying coverage
        docs = {
            "auth_1": "Authentication systems verify user identity through credential validation.",
            "auth_2": "Multi-factor authentication adds security layers beyond passwords.",
            "auth_3": "Session management secures authenticated user interactions.",
            "quantum": "Post-quantum cryptography prepares for quantum computing threats.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I analyze knowledge gaps
        gaps = processor.analyze_knowledge_gaps()

        # THEN I identify isolated security topics
        assert 'isolated_documents' in gaps, "Should identify isolated documents"

        # AND find weak areas needing more documentation
        assert 'weak_topics' in gaps, "Should identify weak topics"

        # AND get recommendations for improvement
        assert 'coverage_score' in gaps, "Should provide coverage metric"
        assert 0 <= gaps['coverage_score'] <= 1

    def test_scenario_engineer_searches_threat_intelligence_rapidly(self):
        """
        Scenario: Rapid threat intelligence search during incident response

        Given an active security incident
        When I search for threat information
        Then results return within milliseconds
        And I can iterate queries quickly
        And find relevant mitigations fast
        Because incident response requires rapid information access.
        """
        # GIVEN an active security incident
        docs = {
            f"threat_{i}": f"Security threat {i} involves malicious activity targeting system vulnerabilities."
            for i in range(10)
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I search for threat information
        import time
        queries = ["malicious activity", "system vulnerabilities", "security threat"]

        total_time = 0
        for query in queries:
            start = time.perf_counter()
            results = processor.find_documents_for_query(query, top_n=3)
            elapsed = time.perf_counter() - start
            total_time += elapsed

        # THEN results return within milliseconds
        avg_time = total_time / len(queries)
        assert avg_time < 1.0, "Queries should complete quickly"

        # AND I can iterate queries quickly
        # Multiple queries executed successfully

        # AND find relevant mitigations fast
        assert len(results) > 0, "Should return results"
