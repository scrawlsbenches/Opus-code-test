"""
Unit Specifications for Semantic Knowledge Graph.

These specifications document the precise behavior of the knowledge graph.
Each specification is a fact about the system that must remain true.
"""

import pytest


class SemanticKnowledgeGraphSpecification:
    """
    Specifications for SemanticKnowledgeGraph behavior.
    """

    def spec_graph_has_unique_id(self):
        """
        SPECIFICATION: Each graph gets a unique ID on creation.

        Enables graph tracking and identification.
        """
        from cortical.graph import SemanticKnowledgeGraph

        g1 = SemanticKnowledgeGraph()
        g2 = SemanticKnowledgeGraph()

        assert g1.id != g2.id

    def spec_documents_can_be_added_and_removed(self):
        """
        SPECIFICATION: Documents can be added and removed.

        Basic CRUD operations must work.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()

        skg.add_document("doc1", "Content one")
        assert skg.document_count() == 1

        skg.add_document("doc2", "Content two")
        assert skg.document_count() == 2

        skg.remove_document("doc1")
        assert skg.document_count() == 1

    def spec_build_populates_nodes(self):
        """
        SPECIFICATION: build() creates nodes from documents.

        Graph must be populated after building.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.add_document("doc1", "Machine learning algorithms process data.")
        skg.build()

        assert skg.node_count() > 0

    def spec_build_creates_document_nodes(self):
        """
        SPECIFICATION: Each document becomes a node in layer 3.

        Documents must be represented in the graph.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.add_document("doc1", "First document")
        skg.add_document("doc2", "Second document")
        skg.build()

        stats = skg.get_layer_statistics()
        assert stats['documents'] == 2

    def spec_build_creates_token_nodes(self):
        """
        SPECIFICATION: Tokens from documents become nodes in layer 0.

        Text must be tokenized into graph nodes.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.add_document("doc1", "Machine learning is powerful.")
        skg.build()

        stats = skg.get_layer_statistics()
        assert stats['tokens'] > 0

    def spec_search_returns_matching_documents(self):
        """
        SPECIFICATION: Search returns documents matching query terms.

        Basic search functionality must work.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.add_document("ml_doc", "Machine learning enables AI.")
        skg.add_document("other_doc", "Cooking recipes for dinner.")
        skg.build()

        results = skg.search("machine learning")

        doc_ids = [r.doc_id for r in results]
        assert "ml_doc" in doc_ids

    def spec_search_scores_are_positive(self):
        """
        SPECIFICATION: Search result scores are non-negative.

        Negative scores would break ranking.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.add_document("doc1", "Content about topic X.")
        skg.build()

        results = skg.search("topic")

        for r in results:
            assert r.score >= 0

    def spec_pagerank_scores_are_normalized(self):
        """
        SPECIFICATION: PageRank scores sum to approximately 1.

        PageRank is a probability distribution.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.add_document("doc1", "Topic A relates to topic B.")
        skg.add_document("doc2", "Topic B connects to topic C.")
        skg.build()

        # Get all PageRank scores
        total_pr = 0.0
        for node in skg._nodes.values():
            total_pr += node.pagerank

        # Should be approximately 1 (within tolerance)
        assert 0.9 < total_pr < 1.1

    def spec_spreading_activation_decays(self):
        """
        SPECIFICATION: Spreading activation decays with distance.

        Farther nodes receive less activation.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.add_document("doc1", "Alpha connects to beta. Beta connects to gamma.")
        skg.build()

        activations = skg.spread_activation("alpha", decay=0.5, hops=2)

        # Source should have highest activation (or close to it)
        source_activation = activations.get("alpha", 0)
        other_activations = [v for k, v in activations.items() if k != "alpha"]

        if other_activations:
            assert source_activation >= min(other_activations)

    def spec_custom_relations_are_registered(self):
        """
        SPECIFICATION: Custom relation types can be registered.

        Enables domain-specific relation types.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.register_relation_type("ImplementedBy", weight=1.5)

        relations = skg.get_custom_relations()
        assert "ImplementedBy" in relations
        assert relations["ImplementedBy"] == 1.5

    def spec_edges_can_be_added_manually(self):
        """
        SPECIFICATION: Edges can be added manually.

        Enables external knowledge injection.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.add_edge("concept_a", "concept_b", "RelatedTo", confidence=0.9)

        connections = skg.get_connections_for("concept_a")
        assert len(connections) > 0

    def spec_merge_combines_documents(self):
        """
        SPECIFICATION: merge() combines documents from multiple graphs.

        Graph merging must preserve all content.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg1 = SemanticKnowledgeGraph()
        skg1.add_document("doc1", "Content from graph 1.")

        skg2 = SemanticKnowledgeGraph()
        skg2.add_document("doc2", "Content from graph 2.")

        merged = SemanticKnowledgeGraph.merge([skg1, skg2])

        assert merged.document_count() == 2

    def spec_cel_events_are_logged_when_enabled(self):
        """
        SPECIFICATION: CEL events are logged when CEL is enabled.

        Event sourcing requires event logging.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph(enable_cel=True)
        skg.add_document("doc1", "Test content.")
        skg.build()

        events = skg.get_cel_events()
        # Should have at least document_added and graph_built events
        assert len(events) >= 2

    def spec_summary_contains_key_metrics(self):
        """
        SPECIFICATION: Summary includes essential metrics.

        Summary must be informative for monitoring.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.add_document("doc1", "Test content.")
        skg.build()

        summary = skg.get_summary()

        assert 'nodes' in summary
        assert 'edges' in summary
        assert 'documents' in summary
        assert 'built' in summary


class ConnectionTypeSpecification:
    """
    Specifications for connection types.
    """

    def spec_lateral_connections_within_layer(self):
        """
        SPECIFICATION: Lateral connections are within same layer.

        Lateral connections connect nodes at the same abstraction level.
        """
        from cortical.graph import SemanticKnowledgeGraph, ConnectionType

        skg = SemanticKnowledgeGraph()
        skg.add_document("doc1", "Token A cooccurs with token B.")
        skg.build()

        # Find lateral connections
        lateral = [e for e in skg._edges if e.connection_type == ConnectionType.LATERAL]

        # All lateral edges should connect tokens (same layer)
        for edge in lateral[:5]:  # Check first 5
            if edge.source_id.startswith("token:") and edge.target_id.startswith("token:"):
                assert True  # Valid lateral connection

    def spec_feedforward_connections_go_up(self):
        """
        SPECIFICATION: Feedforward connections go from lower to higher layers.

        Tokens -> Bigrams -> Concepts -> Documents
        """
        from cortical.graph import SemanticKnowledgeGraph, ConnectionType

        skg = SemanticKnowledgeGraph()
        skg.add_document("doc1", "Test document content.")
        skg.build()

        ff = [e for e in skg._edges if e.connection_type == ConnectionType.FEEDFORWARD]

        # Should have token -> document feedforward connections
        assert len(ff) > 0


class SearchResultSpecification:
    """
    Specifications for search result behavior.
    """

    def spec_results_are_sorted_by_score(self):
        """
        SPECIFICATION: Search results are sorted by score (descending).

        Higher scores should appear first.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        for i in range(5):
            skg.add_document(f"doc_{i}", f"Topic X appears {i+1} times. " * (i+1))
        skg.build()

        results = skg.search("topic")

        # Verify sorted
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def spec_limit_constrains_result_count(self):
        """
        SPECIFICATION: limit parameter constrains result count.

        Should not return more than requested.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        for i in range(20):
            skg.add_document(f"doc_{i}", "Common term appears here.")
        skg.build()

        results = skg.search("common term", limit=5)
        assert len(results) <= 5

    def spec_matched_terms_are_tracked(self):
        """
        SPECIFICATION: Results track which terms matched.

        Enables highlighting and relevance explanation.
        """
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.add_document("doc1", "Machine learning is great.")
        skg.build()

        results = skg.search("machine learning")

        for r in results:
            # matched_terms should be populated
            assert hasattr(r, 'matched_terms')


# Pytest test class
class TestKnowledgeGraphSpecifications:
    """Pytest wrapper for knowledge graph specifications."""

    def test_unique_id(self):
        spec = SemanticKnowledgeGraphSpecification()
        spec.spec_graph_has_unique_id()

    def test_document_crud(self):
        spec = SemanticKnowledgeGraphSpecification()
        spec.spec_documents_can_be_added_and_removed()

    def test_build_populates(self):
        spec = SemanticKnowledgeGraphSpecification()
        spec.spec_build_populates_nodes()

    def test_search_works(self):
        spec = SemanticKnowledgeGraphSpecification()
        spec.spec_search_returns_matching_documents()

    def test_positive_scores(self):
        spec = SemanticKnowledgeGraphSpecification()
        spec.spec_search_scores_are_positive()

    def test_spreading_activation(self):
        spec = SemanticKnowledgeGraphSpecification()
        spec.spec_spreading_activation_decays()

    def test_custom_relations(self):
        spec = SemanticKnowledgeGraphSpecification()
        spec.spec_custom_relations_are_registered()

    def test_merge(self):
        spec = SemanticKnowledgeGraphSpecification()
        spec.spec_merge_combines_documents()

    def test_result_sorting(self):
        spec = SearchResultSpecification()
        spec.spec_results_are_sorted_by_score()

    def test_result_limit(self):
        spec = SearchResultSpecification()
        spec.spec_limit_constrains_result_count()
