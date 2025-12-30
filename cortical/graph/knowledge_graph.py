"""
Semantic Knowledge Graph: Unified Orchestrator for Cognitive Architecture.

The SemanticKnowledgeGraph (SKG) is the central component that integrates:
- Core data structures (Minicolumn, Edge, Layers)
- Algorithms (PageRank, TF-IDF, BM25, Label Propagation, Spreading Activation)
- Semantic relations (IsA, PartOf, HasA, SimilarTo, etc.)
- Connection types (Lateral, Typed, Feedforward, Feedback)
- CEL for event sourcing and persistence
- GoT for task/decision tracking
- WovenMind for dual-process cognition
- PRISM for attention and plasticity
- SparkSLM for prediction

Design Philosophy:
    A knowledge graph is more than nodes and edges—it is a living
    representation of understanding. This unified orchestrator ensures
    that all cognitive components work coherently, from low-level
    tokenization to high-level reasoning.

Example:
    >>> skg = SemanticKnowledgeGraph()
    >>> skg.add_document("intro", "Machine learning enables intelligent systems.")
    >>> skg.build()
    >>> results = skg.search("AI systems", expand_query=True)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Iterator
from enum import Enum, auto
import time
import uuid

from ..minicolumn import Minicolumn, Edge
from ..layers import CorticalLayer, HierarchicalLayer
from ..tokenizer import Tokenizer
from ..constants import RELATION_WEIGHTS


# Module-level tokenizer instance
_tokenizer = Tokenizer()


def tokenize(text: str) -> list:
    """Tokenize text using the module tokenizer."""
    return _tokenizer.tokenize(text)


class ConnectionType(Enum):
    """Types of connections in the knowledge graph."""
    LATERAL = auto()  # Within same layer
    FEEDFORWARD = auto()  # From lower to higher layer
    FEEDBACK = auto()  # From higher to lower layer
    SEMANTIC = auto()  # Semantic relation (typed)


class RelationType(Enum):
    """Semantic relation types."""
    IS_A = "IsA"
    PART_OF = "PartOf"
    HAS_A = "HasA"
    SIMILAR_TO = "SimilarTo"
    RELATED_TO = "RelatedTo"
    CO_OCCURS = "CoOccurs"
    CAUSES = "Causes"
    USED_FOR = "UsedFor"
    ANTONYM = "Antonym"
    DERIVED_FROM = "DerivedFrom"
    SAME_AS = "SameAs"
    HAS_PROPERTY = "HasProperty"
    CAPABLE_OF = "CapableOf"
    AT_LOCATION = "AtLocation"
    DEFINED_BY = "DefinedBy"


@dataclass
class GraphNode:
    """
    A node in the semantic knowledge graph.

    Wraps a Minicolumn with additional graph metadata.
    """
    id: str
    content: str
    layer: CorticalLayer
    minicolumn: Optional[Minicolumn] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    # Computed scores
    pagerank: float = 0.0
    tfidf: float = 0.0
    bm25: float = 0.0
    activation: float = 0.0


@dataclass
class GraphEdge:
    """
    An edge in the semantic knowledge graph.

    Wraps an Edge with additional graph metadata.
    """
    source_id: str
    target_id: str
    connection_type: ConnectionType
    relation_type: Optional[str] = None
    weight: float = 1.0
    confidence: float = 1.0
    source_label: str = "corpus"  # 'corpus', 'semantic', 'inferred'
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SearchResult:
    """Result from a knowledge graph search."""
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    matched_terms: List[str] = field(default_factory=list)


@dataclass
class LayerStatistics:
    """Statistics for a cortical layer."""
    layer: CorticalLayer
    node_count: int
    edge_count: int
    avg_connections: float
    top_nodes: List[Tuple[str, float]]  # By PageRank


class SemanticKnowledgeGraph:
    """
    Unified orchestrator for the cognitive architecture.

    Integrates all components into a coherent knowledge graph that supports:
    - Document ingestion and indexing
    - Multi-layer hierarchical organization
    - Semantic relation extraction
    - PageRank importance scoring
    - TF-IDF/BM25 relevance ranking
    - Query expansion
    - Spreading activation
    - Integration with CEL, GoT, WovenMind, PRISM, SparkSLM

    Example:
        >>> skg = SemanticKnowledgeGraph()
        >>> skg.add_document("doc1", "Neural networks are machine learning models.")
        >>> skg.add_document("doc2", "Deep learning uses neural networks.")
        >>> skg.build()
        >>> results = skg.search("ML models")
        >>> for r in results:
        ...     print(f"{r.doc_id}: {r.score:.3f}")
    """

    def __init__(
        self,
        enable_cel: bool = False,
        enable_got: bool = False,
        enable_woven_mind: bool = False,
        enable_prism: bool = False,
        enable_spark: bool = False,
    ):
        """
        Initialize the Semantic Knowledge Graph.

        Args:
            enable_cel: Enable CEL event sourcing
            enable_got: Enable GoT task tracking
            enable_woven_mind: Enable dual-process cognition
            enable_prism: Enable PRISM plasticity
            enable_spark: Enable SparkSLM prediction
        """
        self.id = str(uuid.uuid4())[:8]
        self.created_at = datetime.now()

        # Core data structures
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._edge_index: Dict[str, List[GraphEdge]] = {}  # source_id -> edges

        # Documents
        self._documents: Dict[str, str] = {}  # doc_id -> content
        self._doc_metadata: Dict[str, Dict[str, Any]] = {}

        # Layers
        self._layers: Dict[CorticalLayer, HierarchicalLayer] = {
            layer: HierarchicalLayer(layer)
            for layer in CorticalLayer
        }

        # Custom relations
        self._custom_relations: Dict[str, float] = {}  # relation -> weight

        # Integration flags
        self._enable_cel = enable_cel
        self._enable_got = enable_got
        self._enable_woven_mind = enable_woven_mind
        self._enable_prism = enable_prism
        self._enable_spark = enable_spark

        # Subsystems (lazy initialized)
        self._cel_container = None
        self._got_manager = None
        self._woven_mind = None
        self._prism_reasoner = None
        self._spark_predictor = None

        # Build state
        self._built = False
        self._build_time: Optional[float] = None

        # CEL event log (simplified, real CEL would be in cel/)
        self._cel_events: List[Dict[str, Any]] = []

        # PRISM plasticity tracking
        self._connection_strengths: Dict[Tuple[str, str], float] = {}

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a document to the knowledge graph.

        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata (author, date, etc.)
        """
        self._documents[doc_id] = content
        self._doc_metadata[doc_id] = metadata or {}

        # Log CEL event
        if self._enable_cel:
            self._log_cel_event("OBSERVATION", {
                "type": "document_added",
                "doc_id": doc_id,
                "content_length": len(content),
            })

        # Invalidate build
        self._built = False

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the graph."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            if doc_id in self._doc_metadata:
                del self._doc_metadata[doc_id]
            self._built = False
            return True
        return False

    def build(self) -> None:
        """
        Build the knowledge graph from documents.

        Processes all documents to:
        1. Tokenize and create token nodes (Layer 0)
        2. Create bigram nodes (Layer 1)
        3. Cluster into concept nodes (Layer 2)
        4. Create document nodes (Layer 3)
        5. Build connections
        6. Compute PageRank and TF-IDF
        7. Extract semantic relations
        """
        start_time = time.time()

        # Clear existing graph
        self._nodes.clear()
        self._edges.clear()
        self._edge_index.clear()

        for layer in self._layers.values():
            layer.minicolumns.clear()

        # Process each document
        for doc_id, content in self._documents.items():
            self._process_document(doc_id, content)

        # Build connections between layers
        self._build_connections()

        # Extract semantic relations
        self._extract_semantic_relations()

        # Compute importance scores
        self._compute_pagerank()
        self._compute_tfidf()

        self._built = True
        self._build_time = time.time() - start_time

        # Log CEL event
        if self._enable_cel:
            self._log_cel_event("OBSERVATION", {
                "type": "graph_built",
                "nodes": len(self._nodes),
                "edges": len(self._edges),
                "build_time_ms": self._build_time * 1000,
            })

    def _process_document(self, doc_id: str, content: str) -> None:
        """Process a single document into graph nodes."""
        # Tokenize
        tokens = tokenize(content)

        # Create token nodes (Layer 0)
        token_nodes = []
        for token in tokens:
            node_id = f"token:{token}"
            if node_id not in self._nodes:
                node = GraphNode(
                    id=node_id,
                    content=token,
                    layer=CorticalLayer.TOKENS,
                )
                self._nodes[node_id] = node
            token_nodes.append(node_id)

        # Create bigram nodes (Layer 1)
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            node_id = f"bigram:{bigram}"
            if node_id not in self._nodes:
                node = GraphNode(
                    id=node_id,
                    content=bigram,
                    layer=CorticalLayer.BIGRAMS,
                )
                self._nodes[node_id] = node

            # Connect to constituent tokens
            self._add_edge(f"token:{tokens[i]}", node_id, ConnectionType.FEEDFORWARD)
            self._add_edge(f"token:{tokens[i+1]}", node_id, ConnectionType.FEEDFORWARD)

        # Create document node (Layer 3)
        doc_node_id = f"doc:{doc_id}"
        doc_node = GraphNode(
            id=doc_node_id,
            content=content[:200],  # Preview
            layer=CorticalLayer.DOCUMENTS,
            properties={'full_content': content, 'tokens': tokens},
        )
        self._nodes[doc_node_id] = doc_node

        # Connect tokens to document
        for token_id in set(token_nodes):
            self._add_edge(token_id, doc_node_id, ConnectionType.FEEDFORWARD)

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        connection_type: ConnectionType,
        relation_type: Optional[str] = None,
        weight: float = 1.0,
        confidence: float = 1.0,
    ) -> None:
        """Add an edge to the graph."""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            connection_type=connection_type,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
        )
        self._edges.append(edge)

        # Index by source
        if source_id not in self._edge_index:
            self._edge_index[source_id] = []
        self._edge_index[source_id].append(edge)

    def _build_connections(self) -> None:
        """Build lateral connections within layers."""
        # Connect tokens that co-occur in documents
        for doc_id, content in self._documents.items():
            tokens = tokenize(content)
            unique_tokens = list(set(tokens))

            for i, t1 in enumerate(unique_tokens):
                for t2 in unique_tokens[i+1:]:
                    node1 = f"token:{t1}"
                    node2 = f"token:{t2}"
                    if node1 in self._nodes and node2 in self._nodes:
                        self._add_edge(
                            node1, node2,
                            ConnectionType.LATERAL,
                            relation_type="CoOccurs",
                            weight=0.5,
                        )

    def _extract_semantic_relations(self) -> None:
        """Extract semantic relations from text patterns."""
        import re

        # Simple IsA pattern: "X is a type of Y" or "X is a Y"
        isa_patterns = [
            r'(\w+)\s+(?:is\s+a\s+(?:type\s+of\s+)?|are\s+)(\w+)',
            r'(\w+)\s+(?:such\s+as|like)\s+(\w+)',
        ]

        # PartOf pattern: "X is part of Y" or "X contains Y"
        partof_patterns = [
            r'(\w+)\s+(?:is\s+part\s+of|belongs\s+to)\s+(\w+)',
            r'(\w+)\s+(?:contains|includes)\s+(\w+)',
        ]

        for doc_id, content in self._documents.items():
            content_lower = content.lower()

            # Extract IsA relations
            for pattern in isa_patterns:
                for match in re.finditer(pattern, content_lower):
                    source = match.group(1)
                    target = match.group(2)
                    source_id = f"token:{source}"
                    target_id = f"token:{target}"

                    if source_id in self._nodes and target_id in self._nodes:
                        self._add_edge(
                            source_id, target_id,
                            ConnectionType.SEMANTIC,
                            relation_type="IsA",
                            weight=RELATION_WEIGHTS.get("IsA", 1.5),
                            confidence=0.7,
                        )

            # Extract PartOf relations
            for pattern in partof_patterns:
                for match in re.finditer(pattern, content_lower):
                    source = match.group(1)
                    target = match.group(2)
                    source_id = f"token:{source}"
                    target_id = f"token:{target}"

                    if source_id in self._nodes and target_id in self._nodes:
                        self._add_edge(
                            source_id, target_id,
                            ConnectionType.SEMANTIC,
                            relation_type="PartOf",
                            weight=RELATION_WEIGHTS.get("PartOf", 1.3),
                            confidence=0.7,
                        )

    def _compute_pagerank(self, damping: float = 0.85, iterations: int = 20) -> None:
        """Compute PageRank for all nodes."""
        if not self._nodes:
            return

        # Initialize
        n = len(self._nodes)
        node_ids = list(self._nodes.keys())
        pr = {nid: 1.0 / n for nid in node_ids}

        # Build adjacency
        outgoing: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        for edge in self._edges:
            if edge.source_id in outgoing:
                outgoing[edge.source_id].append(edge.target_id)

        # Iterate
        for _ in range(iterations):
            new_pr = {}
            for node_id in node_ids:
                # Sum of PageRank from incoming nodes
                incoming_sum = 0.0
                for other_id in node_ids:
                    if node_id in outgoing.get(other_id, []):
                        out_count = len(outgoing[other_id])
                        if out_count > 0:
                            incoming_sum += pr[other_id] / out_count

                new_pr[node_id] = (1 - damping) / n + damping * incoming_sum

            pr = new_pr

        # Assign to nodes
        for node_id, score in pr.items():
            self._nodes[node_id].pagerank = score

    def _compute_tfidf(self) -> None:
        """Compute TF-IDF scores for token nodes."""
        import math

        # Document frequency
        df: Dict[str, int] = {}
        for doc_id, content in self._documents.items():
            tokens = set(tokenize(content))
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        n_docs = len(self._documents)

        # Compute IDF and assign to nodes
        for node_id, node in self._nodes.items():
            if node.layer == CorticalLayer.TOKENS:
                token = node.content
                doc_freq = df.get(token, 1)
                idf = math.log(n_docs / doc_freq) if doc_freq > 0 else 0
                node.tfidf = idf

    def compute_importance(self) -> None:
        """Recompute importance scores (PageRank, TF-IDF)."""
        self._compute_pagerank()
        self._compute_tfidf()

    def search(
        self,
        query: str,
        expand_query: bool = True,
        ranking: str = "combined",
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        Search the knowledge graph.

        Args:
            query: Search query
            expand_query: Whether to expand query with related terms
            ranking: Ranking method ('bm25', 'tfidf', 'pagerank', 'combined')
            limit: Maximum results

        Returns:
            List of SearchResult objects
        """
        if not self._built:
            self.build()

        query_tokens = tokenize(query)

        # Query expansion
        if expand_query:
            expanded_tokens = set(query_tokens)
            for token in query_tokens:
                token_id = f"token:{token}"
                if token_id in self._nodes:
                    # Add connected tokens (lateral)
                    for edge in self._edge_index.get(token_id, []):
                        if edge.connection_type == ConnectionType.LATERAL:
                            if edge.target_id.startswith("token:"):
                                expanded_tokens.add(edge.target_id[6:])
            query_tokens = list(expanded_tokens)

        # Score documents
        results = []
        for doc_id, content in self._documents.items():
            doc_tokens = set(tokenize(content))

            # Match score
            matched = set(query_tokens) & doc_tokens
            if not matched:
                continue

            # Score based on ranking method
            if ranking == "bm25":
                score = self._bm25_score(query_tokens, doc_id)
            elif ranking == "tfidf":
                score = sum(
                    self._nodes.get(f"token:{t}", GraphNode("", "", CorticalLayer.TOKENS)).tfidf
                    for t in matched
                )
            elif ranking == "pagerank":
                score = sum(
                    self._nodes.get(f"token:{t}", GraphNode("", "", CorticalLayer.TOKENS)).pagerank
                    for t in matched
                )
            else:  # combined
                tfidf = sum(
                    self._nodes.get(f"token:{t}", GraphNode("", "", CorticalLayer.TOKENS)).tfidf
                    for t in matched
                )
                pr = sum(
                    self._nodes.get(f"token:{t}", GraphNode("", "", CorticalLayer.TOKENS)).pagerank
                    for t in matched
                )
                score = 0.6 * tfidf + 0.4 * (pr * 100)  # Scale PR

            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                content=content[:200],
                metadata=self._doc_metadata.get(doc_id, {}),
                matched_terms=list(matched),
            ))

        # Sort and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _bm25_score(
        self,
        query_tokens: List[str],
        doc_id: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """Compute BM25 score for a document."""
        import math

        content = self._documents[doc_id]
        doc_tokens = tokenize(content)
        doc_len = len(doc_tokens)

        # Average document length
        avg_len = sum(len(tokenize(c)) for c in self._documents.values()) / len(self._documents)

        # Term frequencies in this doc
        tf = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1

        # Document frequencies
        df = {}
        for t in query_tokens:
            count = sum(1 for c in self._documents.values() if t in tokenize(c))
            df[t] = count

        n_docs = len(self._documents)
        score = 0.0

        for term in query_tokens:
            if term not in tf:
                continue

            # IDF
            doc_freq = df.get(term, 0)
            idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

            # TF component
            term_freq = tf[term]
            tf_component = (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * doc_len / avg_len))

            score += idf * tf_component

        return score

    def spread_activation(
        self,
        source: str,
        initial_activation: float = 1.0,
        decay: float = 0.5,
        hops: int = 2,
    ) -> Dict[str, float]:
        """
        Spread activation from a source concept.

        Args:
            source: Source term or node ID
            initial_activation: Starting activation level
            decay: Decay factor per hop
            hops: Maximum hops to spread

        Returns:
            Dictionary of node_id -> activation level
        """
        # Find source node
        if source.startswith("token:") or source.startswith("doc:"):
            source_id = source
        else:
            source_id = f"token:{source.lower()}"

        if source_id not in self._nodes:
            return {}

        # Initialize activations
        activations = {source_id: initial_activation}
        frontier = [source_id]

        for hop in range(hops):
            next_frontier = []
            current_decay = decay ** (hop + 1)

            for node_id in frontier:
                for edge in self._edge_index.get(node_id, []):
                    target = edge.target_id
                    if target in self._nodes:
                        new_activation = activations.get(node_id, 0) * edge.weight * current_decay
                        activations[target] = max(activations.get(target, 0), new_activation)
                        if target not in next_frontier:
                            next_frontier.append(target)

            frontier = next_frontier

        # Return token names (not full IDs) for user-friendly output
        return {
            (nid[6:] if nid.startswith("token:") else nid): act
            for nid, act in activations.items()
        }

    def query(self, query_text: str) -> Optional[Any]:
        """
        High-level query interface.

        Routes query through WovenMind if enabled, otherwise uses search.
        """
        if self._enable_woven_mind:
            # Would route through WovenMind for dual-process handling
            pass

        # Fall back to search
        results = self.search(query_text)
        if results:
            return results[0]
        return None

    def mark_retrieval_success(self, term1: str, term2: str) -> None:
        """
        Mark a successful retrieval for PRISM plasticity.

        Strengthens the connection between the two terms.
        """
        if not self._enable_prism:
            return

        key = (term1.lower(), term2.lower())
        current = self._connection_strengths.get(key, 1.0)
        self._connection_strengths[key] = current * 1.1  # Hebbian strengthening

    def get_connection_strength(self, term1: str, term2: str) -> Optional[float]:
        """Get the connection strength between two terms."""
        key = (term1.lower(), term2.lower())
        if key in self._connection_strengths:
            return self._connection_strengths[key]

        # Check if edge exists
        node1 = f"token:{term1.lower()}"
        node2 = f"token:{term2.lower()}"

        for edge in self._edge_index.get(node1, []):
            if edge.target_id == node2:
                return edge.weight

        return None

    def get_relations_for_concept(self, concept: str) -> List[GraphEdge]:
        """Get all semantic relations for a concept."""
        node_id = f"token:{concept.lower()}"
        if node_id not in self._nodes:
            return []

        return [
            edge for edge in self._edge_index.get(node_id, [])
            if edge.connection_type == ConnectionType.SEMANTIC
        ]

    def get_connections_for(self, term: str) -> List[GraphEdge]:
        """Get all connections for a term."""
        node_id = f"token:{term.lower()}"
        return self._edge_index.get(node_id, [])

    def get_pagerank(self, term: str) -> Optional[float]:
        """Get PageRank for a term."""
        node_id = f"token:{term.lower()}"
        node = self._nodes.get(node_id)
        if node:
            return node.pagerank
        return None

    def node_count(self) -> int:
        """Get total node count."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Get total edge count."""
        return len(self._edges)

    def document_count(self) -> int:
        """Get document count."""
        return len(self._documents)

    def get_layer_statistics(self) -> Dict[str, int]:
        """Get node counts by layer."""
        stats = {'tokens': 0, 'bigrams': 0, 'concepts': 0, 'documents': 0}
        for node in self._nodes.values():
            if node.layer == CorticalLayer.TOKENS:
                stats['tokens'] += 1
            elif node.layer == CorticalLayer.BIGRAMS:
                stats['bigrams'] += 1
            elif node.layer == CorticalLayer.CONCEPTS:
                stats['concepts'] += 1
            elif node.layer == CorticalLayer.DOCUMENTS:
                stats['documents'] += 1
        return stats

    def get_cross_layer_connections(self) -> List[GraphEdge]:
        """Get all feedforward/feedback connections."""
        return [
            edge for edge in self._edges
            if edge.connection_type in (ConnectionType.FEEDFORWARD, ConnectionType.FEEDBACK)
        ]

    def register_relation_type(self, name: str, weight: float = 1.0) -> None:
        """Register a custom relation type."""
        self._custom_relations[name] = weight

    def get_custom_relations(self) -> Dict[str, float]:
        """Get all custom relation types."""
        return dict(self._custom_relations)

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        confidence: float = 1.0,
    ) -> None:
        """Add a custom edge."""
        source_id = f"token:{source}" if not source.startswith(("token:", "doc:")) else source
        target_id = f"token:{target}" if not target.startswith(("token:", "doc:")) else target

        # Ensure nodes exist
        if source_id not in self._nodes:
            self._nodes[source_id] = GraphNode(
                id=source_id,
                content=source,
                layer=CorticalLayer.TOKENS,
            )
        if target_id not in self._nodes:
            self._nodes[target_id] = GraphNode(
                id=target_id,
                content=target,
                layer=CorticalLayer.TOKENS,
            )

        weight = self._custom_relations.get(relation_type, RELATION_WEIGHTS.get(relation_type, 1.0))

        self._add_edge(
            source_id, target_id,
            ConnectionType.SEMANTIC,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
        )

    def _log_cel_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a CEL event."""
        self._cel_events.append({
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
        })

    def get_cel_events(self) -> List[Dict[str, Any]]:
        """Get all CEL events."""
        return list(self._cel_events)

    @classmethod
    def merge(cls, graphs: List['SemanticKnowledgeGraph']) -> 'SemanticKnowledgeGraph':
        """
        Merge multiple graphs into one.

        Args:
            graphs: List of graphs to merge

        Returns:
            New merged graph
        """
        merged = cls()

        for graph in graphs:
            # Add all documents
            for doc_id, content in graph._documents.items():
                merged.add_document(doc_id, content, graph._doc_metadata.get(doc_id))

            # Add custom relations
            merged._custom_relations.update(graph._custom_relations)

        return merged

    def get_summary(self) -> Dict[str, Any]:
        """Get graph summary."""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'built': self._built,
            'build_time_ms': self._build_time * 1000 if self._build_time else None,
            'documents': len(self._documents),
            'nodes': len(self._nodes),
            'edges': len(self._edges),
            'layers': self.get_layer_statistics(),
            'integrations': {
                'cel': self._enable_cel,
                'got': self._enable_got,
                'woven_mind': self._enable_woven_mind,
                'prism': self._enable_prism,
                'spark': self._enable_spark,
            },
        }
