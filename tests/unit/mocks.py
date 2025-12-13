"""
Unit Test Mocks and Test Doubles
================================

Task #150: Create unit test fixtures and mocks for core data structures.

This module provides test doubles that allow testing algorithm logic in isolation,
without requiring a full CorticalTextProcessor with populated layers.

Classes:
    MockMinicolumn: Test double with controllable attributes
    MockHierarchicalLayer: Supports get_minicolumn(), get_by_id(), column_count()
    MockLayers: Factory with common test scenarios
    LayerBuilder: Fluent API for custom test data construction

Usage:
    from tests.unit.mocks import MockMinicolumn, MockHierarchicalLayer, MockLayers

    # Simple mock minicolumn
    col = MockMinicolumn(id="L0_test", content="test", pagerank=0.5)

    # Factory for common scenarios
    layers = MockLayers.two_connected_terms("neural", "networks", weight=0.9)

    # Builder for custom scenarios
    layer = LayerBuilder().with_term("a", pagerank=0.8).with_connection("a", "b", 0.5).build()
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class MockEdge:
    """
    Mock edge for testing typed connections.

    Mirrors cortical.minicolumn.Edge for test purposes.
    """
    target_id: str
    weight: float = 1.0
    relation_type: str = 'co_occurrence'
    confidence: float = 1.0
    source: str = 'corpus'


@dataclass
class MockMinicolumn:
    """
    Test double for Minicolumn with controllable attributes.

    Unlike the real Minicolumn which uses __slots__, this dataclass
    allows easy construction with only the attributes needed for a test.

    All attributes have sensible defaults, so tests only specify what matters.

    Attributes:
        id: Unique identifier (default: auto-generated from content)
        content: The actual content
        layer: Layer number (0-3)
        activation: Current activation level
        occurrence_count: How many times observed
        document_ids: Which documents contain this
        lateral_connections: Simple weight dict
        typed_connections: Dict of MockEdge objects
        feedforward_connections: Links to lower layer
        feedback_connections: Links to higher layer
        tfidf: Global TF-IDF score
        tfidf_per_doc: Per-document TF-IDF scores
        pagerank: Importance score
        cluster_id: Cluster assignment
        doc_occurrence_counts: Per-document counts

    Example:
        # Minimal mock - just what the test needs
        col = MockMinicolumn(content="test", pagerank=0.5)

        # Mock with specific attributes
        col = MockMinicolumn(
            id="L0_neural",
            content="neural",
            pagerank=0.8,
            lateral_connections={"L0_networks": 0.9}
        )
    """
    content: str = "mock"
    id: Optional[str] = None
    layer: int = 0
    activation: float = 0.0
    occurrence_count: int = 1
    document_ids: Set[str] = field(default_factory=set)
    lateral_connections: Dict[str, float] = field(default_factory=dict)
    typed_connections: Dict[str, MockEdge] = field(default_factory=dict)
    feedforward_sources: Set[str] = field(default_factory=set)
    feedforward_connections: Dict[str, float] = field(default_factory=dict)
    feedback_connections: Dict[str, float] = field(default_factory=dict)
    tfidf: float = 0.0
    tfidf_per_doc: Dict[str, float] = field(default_factory=dict)
    pagerank: float = 1.0
    cluster_id: Optional[int] = None
    doc_occurrence_counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-generate ID from layer and content if not provided."""
        if self.id is None:
            self.id = f"L{self.layer}_{self.content}"

    def add_lateral_connection(self, target_id: str, weight: float = 1.0) -> None:
        """Add or strengthen lateral connection (mimics real Minicolumn)."""
        self.lateral_connections[target_id] = (
            self.lateral_connections.get(target_id, 0) + weight
        )

    def add_typed_connection(
        self,
        target_id: str,
        weight: float = 1.0,
        relation_type: str = 'co_occurrence',
        confidence: float = 1.0,
        source: str = 'corpus'
    ) -> None:
        """Add or update typed connection (mimics real Minicolumn)."""
        if target_id in self.typed_connections:
            existing = self.typed_connections[target_id]
            self.typed_connections[target_id] = MockEdge(
                target_id=target_id,
                weight=existing.weight + weight,
                relation_type=relation_type if relation_type != 'co_occurrence' else existing.relation_type,
                confidence=confidence,
                source=source
            )
        else:
            self.typed_connections[target_id] = MockEdge(
                target_id=target_id,
                weight=weight,
                relation_type=relation_type,
                confidence=confidence,
                source=source
            )
        # Also update lateral for backward compat
        self.lateral_connections[target_id] = (
            self.lateral_connections.get(target_id, 0) + weight
        )

    def connection_count(self) -> int:
        """Return number of lateral connections."""
        return len(self.lateral_connections)

    def top_connections(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get strongest lateral connections."""
        sorted_conns = sorted(
            self.lateral_connections.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_conns[:n]

    def get_typed_connection(self, target_id: str) -> Optional[MockEdge]:
        """Get typed connection by target ID."""
        return self.typed_connections.get(target_id)


class MockHierarchicalLayer:
    """
    Test double for HierarchicalLayer with preset minicolumns.

    Supports the essential methods: get_minicolumn(), get_by_id(), column_count()

    Example:
        mock_cols = [MockMinicolumn(content="a"), MockMinicolumn(content="b")]
        layer = MockHierarchicalLayer(mock_cols)

        assert layer.get_minicolumn("a") is not None
        assert layer.get_by_id("L0_a") is not None
        assert layer.column_count() == 2
    """

    def __init__(
        self,
        minicolumns: Optional[List[MockMinicolumn]] = None,
        level: int = 0
    ):
        """
        Initialize with a list of MockMinicolumn objects.

        Args:
            minicolumns: List of MockMinicolumn objects
            level: Layer level (0-3)
        """
        self.level = level
        self.minicolumns: Dict[str, MockMinicolumn] = {}
        self._id_index: Dict[str, str] = {}

        if minicolumns:
            for col in minicolumns:
                self.minicolumns[col.content] = col
                self._id_index[col.id] = col.content

    def get_minicolumn(self, content: str) -> Optional[MockMinicolumn]:
        """Get minicolumn by content, or None if not found."""
        return self.minicolumns.get(content)

    def get_by_id(self, col_id: str) -> Optional[MockMinicolumn]:
        """Get minicolumn by ID in O(1) time."""
        content = self._id_index.get(col_id)
        return self.minicolumns.get(content) if content else None

    def get_or_create_minicolumn(self, content: str) -> MockMinicolumn:
        """Get existing or create new minicolumn."""
        if content not in self.minicolumns:
            col = MockMinicolumn(content=content, layer=self.level)
            self.minicolumns[content] = col
            self._id_index[col.id] = content
        return self.minicolumns[content]

    def remove_minicolumn(self, content: str) -> bool:
        """Remove minicolumn from layer."""
        if content not in self.minicolumns:
            return False
        col = self.minicolumns[content]
        if col.id in self._id_index:
            del self._id_index[col.id]
        del self.minicolumns[content]
        return True

    def column_count(self) -> int:
        """Return number of minicolumns."""
        return len(self.minicolumns)

    def total_connections(self) -> int:
        """Return total lateral connections."""
        return sum(col.connection_count() for col in self.minicolumns.values())

    def average_activation(self) -> float:
        """Calculate average activation."""
        if not self.minicolumns:
            return 0.0
        return sum(col.activation for col in self.minicolumns.values()) / len(self.minicolumns)

    def top_by_pagerank(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top minicolumns by PageRank."""
        sorted_cols = sorted(
            self.minicolumns.values(),
            key=lambda c: c.pagerank,
            reverse=True
        )
        return [(col.content, col.pagerank) for col in sorted_cols[:n]]

    def top_by_tfidf(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top minicolumns by TF-IDF."""
        sorted_cols = sorted(
            self.minicolumns.values(),
            key=lambda c: c.tfidf,
            reverse=True
        )
        return [(col.content, col.tfidf) for col in sorted_cols[:n]]

    def __iter__(self):
        """Iterate over minicolumns."""
        return iter(self.minicolumns.values())

    def __len__(self):
        """Return number of minicolumns."""
        return len(self.minicolumns)

    def __contains__(self, content: str) -> bool:
        """Check if content exists."""
        return content in self.minicolumns


class MockLayers:
    """
    Factory class providing common test scenarios.

    These factory methods create pre-configured layer dictionaries that
    can be passed to analysis functions for unit testing.

    Example:
        # Two terms connected with a specific weight
        layers = MockLayers.two_connected_terms("neural", "networks", weight=0.9)

        # Document with specific terms
        layers = MockLayers.document_with_terms("doc1", ["term1", "term2"])

        # Clustered terms
        layers = MockLayers.clustered_terms({"cluster1": ["a", "b"], "cluster2": ["c", "d"]})
    """

    # Layer enum values for convenience
    TOKENS = 0
    BIGRAMS = 1
    CONCEPTS = 2
    DOCUMENTS = 3

    @classmethod
    def empty(cls) -> Dict[int, MockHierarchicalLayer]:
        """
        Create empty layers for all 4 levels.

        Returns:
            Dict mapping layer number to empty MockHierarchicalLayer
        """
        return {
            cls.TOKENS: MockHierarchicalLayer(level=cls.TOKENS),
            cls.BIGRAMS: MockHierarchicalLayer(level=cls.BIGRAMS),
            cls.CONCEPTS: MockHierarchicalLayer(level=cls.CONCEPTS),
            cls.DOCUMENTS: MockHierarchicalLayer(level=cls.DOCUMENTS),
        }

    @classmethod
    def single_term(
        cls,
        term: str,
        pagerank: float = 1.0,
        tfidf: float = 1.0,
        doc_ids: Optional[List[str]] = None
    ) -> Dict[int, MockHierarchicalLayer]:
        """
        Create layers with a single term.

        Args:
            term: The term content
            pagerank: PageRank score
            tfidf: TF-IDF score
            doc_ids: List of document IDs containing this term

        Returns:
            Dict mapping layer number to MockHierarchicalLayer
        """
        col = MockMinicolumn(
            content=term,
            layer=cls.TOKENS,
            pagerank=pagerank,
            tfidf=tfidf,
            document_ids=set(doc_ids) if doc_ids else {"doc1"}
        )

        layers = cls.empty()
        layers[cls.TOKENS] = MockHierarchicalLayer([col], level=cls.TOKENS)
        return layers

    @classmethod
    def two_connected_terms(
        cls,
        term1: str,
        term2: str,
        weight: float = 1.0,
        pagerank1: float = 0.5,
        pagerank2: float = 0.5
    ) -> Dict[int, MockHierarchicalLayer]:
        """
        Create layers with two terms connected bidirectionally.

        Args:
            term1: First term
            term2: Second term
            weight: Connection weight in both directions
            pagerank1: PageRank for term1
            pagerank2: PageRank for term2

        Returns:
            Dict mapping layer number to MockHierarchicalLayer
        """
        col1 = MockMinicolumn(
            content=term1,
            layer=cls.TOKENS,
            pagerank=pagerank1,
            lateral_connections={f"L0_{term2}": weight}
        )
        col2 = MockMinicolumn(
            content=term2,
            layer=cls.TOKENS,
            pagerank=pagerank2,
            lateral_connections={f"L0_{term1}": weight}
        )

        layers = cls.empty()
        layers[cls.TOKENS] = MockHierarchicalLayer([col1, col2], level=cls.TOKENS)
        return layers

    @classmethod
    def connected_chain(
        cls,
        terms: List[str],
        weights: Optional[List[float]] = None
    ) -> Dict[int, MockHierarchicalLayer]:
        """
        Create layers with terms connected in a chain: a -> b -> c -> ...

        Args:
            terms: List of term strings
            weights: Connection weights (defaults to 1.0 for all)

        Returns:
            Dict mapping layer number to MockHierarchicalLayer
        """
        if weights is None:
            weights = [1.0] * (len(terms) - 1)

        cols = []
        for i, term in enumerate(terms):
            connections = {}
            if i > 0:
                connections[f"L0_{terms[i-1]}"] = weights[i-1]
            if i < len(terms) - 1:
                connections[f"L0_{terms[i+1]}"] = weights[i]

            cols.append(MockMinicolumn(
                content=term,
                layer=cls.TOKENS,
                lateral_connections=connections
            ))

        layers = cls.empty()
        layers[cls.TOKENS] = MockHierarchicalLayer(cols, level=cls.TOKENS)
        return layers

    @classmethod
    def complete_graph(
        cls,
        terms: List[str],
        weight: float = 1.0
    ) -> Dict[int, MockHierarchicalLayer]:
        """
        Create layers where all terms are connected to all other terms.

        Args:
            terms: List of term strings
            weight: Connection weight for all edges

        Returns:
            Dict mapping layer number to MockHierarchicalLayer
        """
        cols = []
        for term in terms:
            connections = {
                f"L0_{other}": weight
                for other in terms if other != term
            }
            cols.append(MockMinicolumn(
                content=term,
                layer=cls.TOKENS,
                lateral_connections=connections
            ))

        layers = cls.empty()
        layers[cls.TOKENS] = MockHierarchicalLayer(cols, level=cls.TOKENS)
        return layers

    @classmethod
    def disconnected_terms(
        cls,
        terms: List[str],
        pageranks: Optional[List[float]] = None
    ) -> Dict[int, MockHierarchicalLayer]:
        """
        Create layers with multiple disconnected terms (no connections).

        Args:
            terms: List of term strings
            pageranks: PageRank scores (defaults to 1.0 for all)

        Returns:
            Dict mapping layer number to MockHierarchicalLayer
        """
        if pageranks is None:
            pageranks = [1.0] * len(terms)

        cols = [
            MockMinicolumn(content=term, layer=cls.TOKENS, pagerank=pr)
            for term, pr in zip(terms, pageranks)
        ]

        layers = cls.empty()
        layers[cls.TOKENS] = MockHierarchicalLayer(cols, level=cls.TOKENS)
        return layers

    @classmethod
    def document_with_terms(
        cls,
        doc_id: str,
        terms: List[str],
        term_counts: Optional[Dict[str, int]] = None
    ) -> Dict[int, MockHierarchicalLayer]:
        """
        Create layers simulating a document with specific terms.

        Args:
            doc_id: Document identifier
            terms: List of terms in the document
            term_counts: Optional counts per term (defaults to 1)

        Returns:
            Dict mapping layer number to MockHierarchicalLayer
        """
        if term_counts is None:
            term_counts = {t: 1 for t in terms}

        # Create token layer
        term_cols = [
            MockMinicolumn(
                content=term,
                layer=cls.TOKENS,
                document_ids={doc_id},
                occurrence_count=term_counts.get(term, 1),
                doc_occurrence_counts={doc_id: term_counts.get(term, 1)}
            )
            for term in terms
        ]

        # Create document layer
        doc_col = MockMinicolumn(
            content=doc_id,
            id=f"L3_{doc_id}",
            layer=cls.DOCUMENTS,
            feedforward_connections={f"L0_{t}": 1.0 for t in terms}
        )

        layers = cls.empty()
        layers[cls.TOKENS] = MockHierarchicalLayer(term_cols, level=cls.TOKENS)
        layers[cls.DOCUMENTS] = MockHierarchicalLayer([doc_col], level=cls.DOCUMENTS)
        return layers

    @classmethod
    def multi_document_corpus(
        cls,
        documents: Dict[str, List[str]]
    ) -> Dict[int, MockHierarchicalLayer]:
        """
        Create layers simulating multiple documents.

        Args:
            documents: Dict mapping doc_id to list of terms

        Returns:
            Dict mapping layer number to MockHierarchicalLayer
        """
        # Aggregate term occurrences
        term_docs: Dict[str, Set[str]] = {}
        term_counts: Dict[str, Dict[str, int]] = {}

        for doc_id, terms in documents.items():
            for term in terms:
                if term not in term_docs:
                    term_docs[term] = set()
                    term_counts[term] = {}
                term_docs[term].add(doc_id)
                term_counts[term][doc_id] = term_counts[term].get(doc_id, 0) + 1

        # Create term columns
        term_cols = [
            MockMinicolumn(
                content=term,
                layer=cls.TOKENS,
                document_ids=doc_ids,
                occurrence_count=sum(term_counts[term].values()),
                doc_occurrence_counts=term_counts[term]
            )
            for term, doc_ids in term_docs.items()
        ]

        # Create document columns
        doc_cols = [
            MockMinicolumn(
                content=doc_id,
                id=f"L3_{doc_id}",
                layer=cls.DOCUMENTS,
                feedforward_connections={f"L0_{t}": 1.0 for t in terms}
            )
            for doc_id, terms in documents.items()
        ]

        layers = cls.empty()
        layers[cls.TOKENS] = MockHierarchicalLayer(term_cols, level=cls.TOKENS)
        layers[cls.DOCUMENTS] = MockHierarchicalLayer(doc_cols, level=cls.DOCUMENTS)
        return layers

    @classmethod
    def clustered_terms(
        cls,
        clusters: Dict[str, List[str]],
        intra_weight: float = 2.0,
        inter_weight: float = 0.1
    ) -> Dict[int, MockHierarchicalLayer]:
        """
        Create layers with terms pre-assigned to clusters.

        Terms within a cluster are strongly connected (intra_weight).
        Terms between clusters are weakly connected (inter_weight).

        Args:
            clusters: Dict mapping cluster_id to list of terms
            intra_weight: Connection weight within clusters
            inter_weight: Connection weight between clusters

        Returns:
            Dict mapping layer number to MockHierarchicalLayer
        """
        # Build all terms and their cluster assignments
        all_terms = []
        term_to_cluster = {}
        for cluster_id, terms in clusters.items():
            for term in terms:
                all_terms.append(term)
                term_to_cluster[term] = cluster_id

        # Create columns with appropriate connections
        cols = []
        for term in all_terms:
            connections = {}
            my_cluster = term_to_cluster[term]

            for other_term in all_terms:
                if other_term == term:
                    continue
                other_cluster = term_to_cluster[other_term]
                weight = intra_weight if my_cluster == other_cluster else inter_weight
                connections[f"L0_{other_term}"] = weight

            # Map cluster_id string to integer
            cluster_ids = list(clusters.keys())
            cluster_int = cluster_ids.index(my_cluster)

            cols.append(MockMinicolumn(
                content=term,
                layer=cls.TOKENS,
                lateral_connections=connections,
                cluster_id=cluster_int
            ))

        layers = cls.empty()
        layers[cls.TOKENS] = MockHierarchicalLayer(cols, level=cls.TOKENS)
        return layers

    @classmethod
    def with_bigrams(
        cls,
        terms: List[str],
        bigrams: List[Tuple[str, str]]
    ) -> Dict[int, MockHierarchicalLayer]:
        """
        Create layers with both token and bigram layers populated.

        Args:
            terms: List of individual terms
            bigrams: List of (term1, term2) tuples for bigrams

        Returns:
            Dict mapping layer number to MockHierarchicalLayer
        """
        term_cols = [MockMinicolumn(content=t, layer=cls.TOKENS) for t in terms]

        bigram_cols = []
        for t1, t2 in bigrams:
            bigram_content = f"{t1} {t2}"  # Space separator per codebase convention
            bigram_id = f"L1_{bigram_content}"
            col = MockMinicolumn(
                content=bigram_content,
                id=bigram_id,
                layer=cls.BIGRAMS,
                feedforward_connections={f"L0_{t1}": 1.0, f"L0_{t2}": 1.0}
            )
            bigram_cols.append(col)

        layers = cls.empty()
        layers[cls.TOKENS] = MockHierarchicalLayer(term_cols, level=cls.TOKENS)
        layers[cls.BIGRAMS] = MockHierarchicalLayer(bigram_cols, level=cls.BIGRAMS)
        return layers


class LayerBuilder:
    """
    Fluent builder for constructing custom test layer configurations.

    Provides a chainable API for building complex test scenarios step by step.

    Example:
        layers = LayerBuilder() \\
            .with_term("neural", pagerank=0.8, tfidf=2.5) \\
            .with_term("networks", pagerank=0.6, tfidf=1.8) \\
            .with_connection("neural", "networks", 0.9) \\
            .with_document("doc1", ["neural", "networks"]) \\
            .build()
    """

    def __init__(self):
        """Initialize empty builder state."""
        self._terms: Dict[str, Dict[str, Any]] = {}
        self._connections: List[Tuple[str, str, float]] = []
        self._documents: Dict[str, List[str]] = {}
        self._bigrams: List[Tuple[str, str]] = []
        self._clusters: Dict[str, int] = {}

    def with_term(
        self,
        term: str,
        pagerank: float = 1.0,
        tfidf: float = 0.0,
        activation: float = 0.0,
        occurrence_count: int = 1
    ) -> 'LayerBuilder':
        """
        Add a term with specified attributes.

        Args:
            term: Term content
            pagerank: PageRank score
            tfidf: TF-IDF score
            activation: Activation level
            occurrence_count: Occurrence count

        Returns:
            self for chaining
        """
        self._terms[term] = {
            'pagerank': pagerank,
            'tfidf': tfidf,
            'activation': activation,
            'occurrence_count': occurrence_count
        }
        return self

    def with_terms(self, terms: List[str], **kwargs) -> 'LayerBuilder':
        """
        Add multiple terms with the same attributes.

        Args:
            terms: List of term strings
            **kwargs: Attributes to apply to all terms

        Returns:
            self for chaining
        """
        for term in terms:
            self.with_term(term, **kwargs)
        return self

    def with_connection(
        self,
        term1: str,
        term2: str,
        weight: float = 1.0,
        bidirectional: bool = True
    ) -> 'LayerBuilder':
        """
        Add a connection between two terms.

        Args:
            term1: Source term
            term2: Target term
            weight: Connection weight
            bidirectional: If True, add connection in both directions

        Returns:
            self for chaining
        """
        self._connections.append((term1, term2, weight))
        if bidirectional:
            self._connections.append((term2, term1, weight))

        # Ensure terms exist
        if term1 not in self._terms:
            self._terms[term1] = {}
        if term2 not in self._terms:
            self._terms[term2] = {}

        return self

    def with_document(self, doc_id: str, terms: List[str]) -> 'LayerBuilder':
        """
        Add a document with its terms.

        Args:
            doc_id: Document identifier
            terms: List of terms in the document

        Returns:
            self for chaining
        """
        self._documents[doc_id] = terms
        # Ensure terms exist
        for term in terms:
            if term not in self._terms:
                self._terms[term] = {}
        return self

    def with_bigram(self, term1: str, term2: str) -> 'LayerBuilder':
        """
        Add a bigram.

        Args:
            term1: First term
            term2: Second term

        Returns:
            self for chaining
        """
        self._bigrams.append((term1, term2))
        # Ensure terms exist
        if term1 not in self._terms:
            self._terms[term1] = {}
        if term2 not in self._terms:
            self._terms[term2] = {}
        return self

    def with_cluster(self, term: str, cluster_id: int) -> 'LayerBuilder':
        """
        Assign a term to a cluster.

        Args:
            term: Term to assign
            cluster_id: Cluster ID

        Returns:
            self for chaining
        """
        self._clusters[term] = cluster_id
        if term not in self._terms:
            self._terms[term] = {}
        return self

    def build(self) -> Dict[int, MockHierarchicalLayer]:
        """
        Build the final layer configuration.

        Returns:
            Dict mapping layer number to MockHierarchicalLayer
        """
        # Build connection map
        conn_map: Dict[str, Dict[str, float]] = {}
        for term1, term2, weight in self._connections:
            if term1 not in conn_map:
                conn_map[term1] = {}
            conn_map[term1][f"L0_{term2}"] = weight

        # Build document membership
        term_docs: Dict[str, Set[str]] = {}
        for doc_id, terms in self._documents.items():
            for term in terms:
                if term not in term_docs:
                    term_docs[term] = set()
                term_docs[term].add(doc_id)

        # Create token columns
        term_cols = []
        for term, attrs in self._terms.items():
            col = MockMinicolumn(
                content=term,
                layer=MockLayers.TOKENS,
                pagerank=attrs.get('pagerank', 1.0),
                tfidf=attrs.get('tfidf', 0.0),
                activation=attrs.get('activation', 0.0),
                occurrence_count=attrs.get('occurrence_count', 1),
                lateral_connections=conn_map.get(term, {}),
                document_ids=term_docs.get(term, set()),
                cluster_id=self._clusters.get(term)
            )
            term_cols.append(col)

        # Create bigram columns
        bigram_cols = []
        for t1, t2 in self._bigrams:
            bigram_content = f"{t1} {t2}"
            col = MockMinicolumn(
                content=bigram_content,
                id=f"L1_{bigram_content}",
                layer=MockLayers.BIGRAMS,
                feedforward_connections={f"L0_{t1}": 1.0, f"L0_{t2}": 1.0}
            )
            bigram_cols.append(col)

        # Create document columns
        doc_cols = []
        for doc_id, terms in self._documents.items():
            col = MockMinicolumn(
                content=doc_id,
                id=f"L3_{doc_id}",
                layer=MockLayers.DOCUMENTS,
                feedforward_connections={f"L0_{t}": 1.0 for t in terms}
            )
            doc_cols.append(col)

        # Assemble layers
        layers = MockLayers.empty()
        if term_cols:
            layers[MockLayers.TOKENS] = MockHierarchicalLayer(term_cols, level=MockLayers.TOKENS)
        if bigram_cols:
            layers[MockLayers.BIGRAMS] = MockHierarchicalLayer(bigram_cols, level=MockLayers.BIGRAMS)
        if doc_cols:
            layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer(doc_cols, level=MockLayers.DOCUMENTS)

        return layers

    def build_token_layer(self) -> MockHierarchicalLayer:
        """
        Build only the token layer (convenience method).

        Returns:
            MockHierarchicalLayer for tokens only
        """
        return self.build()[MockLayers.TOKENS]


# =============================================================================
# GRAPH REPRESENTATION HELPERS
# =============================================================================

def layers_to_graph(layers: Dict[int, MockHierarchicalLayer]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extract a simple graph representation from mock layers.

    Useful for testing core algorithms that take graph input.

    Args:
        layers: Mock layers dict

    Returns:
        Dict mapping node content to list of (target_content, weight) tuples
    """
    graph = {}
    token_layer = layers.get(MockLayers.TOKENS)

    if token_layer:
        for col in token_layer:
            # Extract target content from L0_xxx format
            edges = []
            for target_id, weight in col.lateral_connections.items():
                # target_id is like "L0_networks"
                if target_id.startswith("L0_"):
                    target_content = target_id[3:]  # Remove "L0_" prefix
                    edges.append((target_content, weight))
            graph[col.content] = edges

    return graph


def layers_to_adjacency(layers: Dict[int, MockHierarchicalLayer]) -> Dict[str, Dict[str, float]]:
    """
    Extract adjacency dict representation from mock layers.

    Args:
        layers: Mock layers dict

    Returns:
        Dict mapping node content to dict of {target: weight}
    """
    adj = {}
    token_layer = layers.get(MockLayers.TOKENS)

    if token_layer:
        for col in token_layer:
            neighbors = {}
            for target_id, weight in col.lateral_connections.items():
                if target_id.startswith("L0_"):
                    target_content = target_id[3:]
                    neighbors[target_content] = weight
            adj[col.content] = neighbors

    return adj
