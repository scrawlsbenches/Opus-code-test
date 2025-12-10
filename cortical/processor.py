"""
Cortical Text Processor - Main processor class that orchestrates all components.
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from .tokenizer import Tokenizer
from .minicolumn import Minicolumn
from .layers import CorticalLayer, HierarchicalLayer
from . import analysis
from . import semantics
from . import embeddings as emb_module
from . import query as query_module
from . import gaps as gaps_module
from . import persistence


class CorticalTextProcessor:
    """Neocortex-inspired text processing system."""

    # Computation types for tracking staleness
    COMP_TFIDF = 'tfidf'
    COMP_PAGERANK = 'pagerank'
    COMP_ACTIVATION = 'activation'
    COMP_DOC_CONNECTIONS = 'doc_connections'
    COMP_BIGRAM_CONNECTIONS = 'bigram_connections'
    COMP_CONCEPTS = 'concepts'
    COMP_EMBEDDINGS = 'embeddings'
    COMP_SEMANTICS = 'semantics'

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.layers: Dict[CorticalLayer, HierarchicalLayer] = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS),
            CorticalLayer.DOCUMENTS: HierarchicalLayer(CorticalLayer.DOCUMENTS),
        }
        self.documents: Dict[str, str] = {}
        self.document_metadata: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.semantic_relations: List[Tuple[str, str, str, float]] = []
        # Track which computations are stale and need recomputation
        self._stale_computations: set = set()

    def process_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Process a document and add it to the corpus.

        Args:
            doc_id: Unique identifier for the document
            content: Document text content
            metadata: Optional metadata dict (source, timestamp, author, etc.)

        Returns:
            Dict with processing statistics (tokens, bigrams, unique_tokens)

        Raises:
            ValueError: If doc_id or content is empty or not a string
        """
        # Input validation
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if not content.strip():
            raise ValueError("content must not be empty or whitespace-only")

        self.documents[doc_id] = content

        # Store metadata if provided
        if metadata:
            self.document_metadata[doc_id] = metadata.copy()
        elif doc_id not in self.document_metadata:
            self.document_metadata[doc_id] = {}

        tokens = self.tokenizer.tokenize(content)
        bigrams = self.tokenizer.extract_ngrams(tokens, n=2)
        
        layer0 = self.layers[CorticalLayer.TOKENS]
        layer1 = self.layers[CorticalLayer.BIGRAMS]
        layer3 = self.layers[CorticalLayer.DOCUMENTS]
        
        doc_col = layer3.get_or_create_minicolumn(doc_id)
        doc_col.occurrence_count += 1
        
        for token in tokens:
            col = layer0.get_or_create_minicolumn(token)
            col.occurrence_count += 1
            col.document_ids.add(doc_id)
            col.activation += 1.0
            # Weighted feedforward: document → token (weight by occurrence count)
            doc_col.add_feedforward_connection(col.id, 1.0)
            # Weighted feedback: token → document (weight by occurrence count)
            col.add_feedback_connection(doc_col.id, 1.0)
            # Track per-document occurrence count for accurate TF-IDF
            col.doc_occurrence_counts[doc_id] = col.doc_occurrence_counts.get(doc_id, 0) + 1
        
        for i, token in enumerate(tokens):
            col = layer0.get_minicolumn(token)
            if col:
                for j in range(max(0, i-3), min(len(tokens), i+4)):
                    if i != j:
                        other = layer0.get_minicolumn(tokens[j])
                        if other:
                            col.add_lateral_connection(other.id, 1.0)
        
        for bigram in bigrams:
            col = layer1.get_or_create_minicolumn(bigram)
            col.occurrence_count += 1
            col.document_ids.add(doc_id)
            col.activation += 1.0
            for part in bigram.split():
                token_col = layer0.get_minicolumn(part)
                if token_col:
                    # Weighted feedforward: bigram → tokens (weight 1.0 per occurrence)
                    col.add_feedforward_connection(token_col.id, 1.0)
                    # Weighted feedback: token → bigram (weight 1.0 per occurrence)
                    token_col.add_feedback_connection(col.id, 1.0)

        # Mark all computations as stale since document corpus changed
        self._mark_all_stale()

        return {'tokens': len(tokens), 'bigrams': len(bigrams), 'unique_tokens': len(set(tokens))}

    def set_document_metadata(self, doc_id: str, **kwargs) -> None:
        """
        Set or update metadata for a document.

        Args:
            doc_id: Document identifier
            **kwargs: Metadata key-value pairs to set

        Example:
            >>> processor.set_document_metadata("doc1",
            ...     source="https://example.com",
            ...     author="John Doe",
            ...     timestamp="2025-12-09"
            ... )
        """
        if doc_id not in self.document_metadata:
            self.document_metadata[doc_id] = {}
        self.document_metadata[doc_id].update(kwargs)

    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """
        Get metadata for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Metadata dict (empty dict if no metadata set)
        """
        return self.document_metadata.get(doc_id, {})

    def get_all_document_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all documents.

        Returns:
            Dict mapping doc_id to metadata dict (deep copy)
        """
        import copy
        return copy.deepcopy(self.document_metadata)

    def _mark_all_stale(self) -> None:
        """Mark all computations as stale (needing recomputation)."""
        self._stale_computations = {
            self.COMP_TFIDF,
            self.COMP_PAGERANK,
            self.COMP_ACTIVATION,
            self.COMP_DOC_CONNECTIONS,
            self.COMP_BIGRAM_CONNECTIONS,
            self.COMP_CONCEPTS,
            self.COMP_EMBEDDINGS,
            self.COMP_SEMANTICS,
        }

    def _mark_fresh(self, *computation_types: str) -> None:
        """Mark specified computations as fresh (up-to-date)."""
        for comp in computation_types:
            self._stale_computations.discard(comp)

    def is_stale(self, computation_type: str) -> bool:
        """
        Check if a specific computation is stale.

        Args:
            computation_type: One of COMP_TFIDF, COMP_PAGERANK, etc.

        Returns:
            True if the computation needs to be run again
        """
        return computation_type in self._stale_computations

    def get_stale_computations(self) -> set:
        """
        Get the set of computations that are currently stale.

        Returns:
            Set of computation type strings that need recomputation
        """
        return self._stale_computations.copy()

    def add_document_incremental(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        recompute: str = 'tfidf'
    ) -> Dict[str, int]:
        """
        Add a document with selective recomputation for efficiency.

        Unlike process_document() + compute_all(), this method only recomputes
        what's necessary based on the recompute parameter. This is more efficient
        for RAG systems with frequent document updates.

        Args:
            doc_id: Unique identifier for the document
            content: Document text content
            metadata: Optional metadata dict (source, timestamp, author, etc.)
            recompute: Level of recomputation to perform:
                - 'none': Just add document, mark all computations stale
                - 'tfidf': Recompute TF-IDF only (fast, updates term weights)
                - 'full': Run compute_all() (slowest, most accurate)

        Returns:
            Dict with processing statistics (tokens, bigrams, unique_tokens)

        Example:
            >>> # Quick update for search without full recomputation
            >>> processor.add_document_incremental("new_doc", "content", recompute='tfidf')
            >>>
            >>> # Just queue document, recompute later in batch
            >>> processor.add_document_incremental("doc1", "content1", recompute='none')
            >>> processor.add_document_incremental("doc2", "content2", recompute='none')
            >>> processor.recompute(level='full')  # Batch recomputation
        """
        stats = self.process_document(doc_id, content, metadata)

        if recompute == 'tfidf':
            self.compute_tfidf(verbose=False)
            self._mark_fresh(self.COMP_TFIDF)
        elif recompute == 'full':
            self.compute_all(verbose=False)
            self._stale_computations.clear()
        # 'none' leaves all computations marked as stale

        return stats

    def add_documents_batch(
        self,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]],
        recompute: str = 'full',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Add multiple documents with a single recomputation.

        More efficient than calling add_document_incremental() multiple times
        when adding many documents at once.

        Args:
            documents: List of (doc_id, content, metadata) tuples.
                       metadata can be None for documents without metadata.
            recompute: Level of recomputation after all documents are added:
                - 'none': Just add documents, mark all computations stale
                - 'tfidf': Recompute TF-IDF only
                - 'full': Run compute_all()
            verbose: Print progress messages

        Returns:
            Dict with batch statistics:
                - documents_added: Number of documents added
                - total_tokens: Total tokens across all documents
                - recomputation: Type of recomputation performed

        Example:
            >>> docs = [
            ...     ("doc1", "First document content", {"source": "web"}),
            ...     ("doc2", "Second document content", None),
            ...     ("doc3", "Third document content", {"author": "AI"}),
            ... ]
            >>> processor.add_documents_batch(docs, recompute='full')

        Raises:
            ValueError: If documents list is invalid or recompute level is unknown
        """
        # Input validation
        if not isinstance(documents, list):
            raise ValueError("documents must be a list")
        if not documents:
            raise ValueError("documents list must not be empty")

        valid_recompute = {'none', 'tfidf', 'full'}
        if recompute not in valid_recompute:
            raise ValueError(f"recompute must be one of {valid_recompute}")

        for i, doc in enumerate(documents):
            if not isinstance(doc, (tuple, list)) or len(doc) < 2:
                raise ValueError(
                    f"documents[{i}] must be a tuple of (doc_id, content) or "
                    f"(doc_id, content, metadata)"
                )
            doc_id, content = doc[0], doc[1]
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"documents[{i}][0] (doc_id) must be a non-empty string")
            if not isinstance(content, str):
                raise ValueError(f"documents[{i}][1] (content) must be a string")

        total_tokens = 0
        total_bigrams = 0

        if verbose:
            print(f"Adding {len(documents)} documents...")

        for doc_id, content, metadata in documents:
            # Use process_document directly (not add_document_incremental)
            # to avoid per-document recomputation
            stats = self.process_document(doc_id, content, metadata)
            total_tokens += stats['tokens']
            total_bigrams += stats['bigrams']

        if verbose:
            print(f"Processed {total_tokens} tokens, {total_bigrams} bigrams")

        # Perform single recomputation for entire batch
        if recompute == 'tfidf':
            if verbose:
                print("Recomputing TF-IDF...")
            self.compute_tfidf(verbose=False)
            self._mark_fresh(self.COMP_TFIDF)
        elif recompute == 'full':
            if verbose:
                print("Running full recomputation...")
            self.compute_all(verbose=False)
            self._stale_computations.clear()

        if verbose:
            print("Done.")

        return {
            'documents_added': len(documents),
            'total_tokens': total_tokens,
            'total_bigrams': total_bigrams,
            'recomputation': recompute
        }

    def recompute(
        self,
        level: str = 'stale',
        verbose: bool = True
    ) -> Dict[str, bool]:
        """
        Recompute specified analysis levels.

        Use this after adding documents with recompute='none' to batch
        the recomputation step.

        Args:
            level: What to recompute:
                - 'stale': Only recompute what's marked as stale
                - 'tfidf': Only TF-IDF (marks others stale)
                - 'full': Run complete compute_all()
            verbose: Print progress messages

        Returns:
            Dict indicating what was recomputed

        Example:
            >>> # Add documents without recomputation
            >>> processor.add_document_incremental("doc1", "content", recompute='none')
            >>> processor.add_document_incremental("doc2", "content", recompute='none')
            >>> # Batch recompute
            >>> processor.recompute(level='full')
        """
        recomputed = {}

        if level == 'full':
            self.compute_all(verbose=verbose)
            self._stale_computations.clear()
            recomputed = {
                self.COMP_ACTIVATION: True,
                self.COMP_PAGERANK: True,
                self.COMP_TFIDF: True,
                self.COMP_DOC_CONNECTIONS: True,
                self.COMP_BIGRAM_CONNECTIONS: True,
                self.COMP_CONCEPTS: True,
            }
        elif level == 'tfidf':
            self.compute_tfidf(verbose=verbose)
            self._mark_fresh(self.COMP_TFIDF)
            recomputed[self.COMP_TFIDF] = True
        elif level == 'stale':
            # Recompute only what's stale, in dependency order
            if self.COMP_ACTIVATION in self._stale_computations:
                self.propagate_activation(verbose=verbose)
                self._mark_fresh(self.COMP_ACTIVATION)
                recomputed[self.COMP_ACTIVATION] = True

            if self.COMP_PAGERANK in self._stale_computations:
                self.compute_importance(verbose=verbose)
                self._mark_fresh(self.COMP_PAGERANK)
                recomputed[self.COMP_PAGERANK] = True

            if self.COMP_TFIDF in self._stale_computations:
                self.compute_tfidf(verbose=verbose)
                self._mark_fresh(self.COMP_TFIDF)
                recomputed[self.COMP_TFIDF] = True

            if self.COMP_DOC_CONNECTIONS in self._stale_computations:
                self.compute_document_connections(verbose=verbose)
                self._mark_fresh(self.COMP_DOC_CONNECTIONS)
                recomputed[self.COMP_DOC_CONNECTIONS] = True

            if self.COMP_BIGRAM_CONNECTIONS in self._stale_computations:
                self.compute_bigram_connections(verbose=verbose)
                self._mark_fresh(self.COMP_BIGRAM_CONNECTIONS)
                recomputed[self.COMP_BIGRAM_CONNECTIONS] = True

            if self.COMP_CONCEPTS in self._stale_computations:
                self.build_concept_clusters(verbose=verbose)
                self._mark_fresh(self.COMP_CONCEPTS)
                recomputed[self.COMP_CONCEPTS] = True

            if self.COMP_EMBEDDINGS in self._stale_computations:
                self.compute_graph_embeddings(verbose=verbose)
                self._mark_fresh(self.COMP_EMBEDDINGS)
                recomputed[self.COMP_EMBEDDINGS] = True

            if self.COMP_SEMANTICS in self._stale_computations:
                self.extract_corpus_semantics(verbose=verbose)
                self._mark_fresh(self.COMP_SEMANTICS)
                recomputed[self.COMP_SEMANTICS] = True

        return recomputed

    def compute_all(
        self,
        verbose: bool = True,
        build_concepts: bool = True,
        pagerank_method: str = 'standard',
        connection_strategy: str = 'document_overlap',
        cluster_strictness: float = 1.0,
        bridge_weight: float = 0.0
    ) -> Dict[str, Any]:
        """
        Run all computation steps.

        Args:
            verbose: Print progress messages
            build_concepts: Build concept clusters in Layer 2 (default True)
                           This enables topic-based filtering and hierarchical search.
            pagerank_method: PageRank algorithm to use:
                - 'standard': Traditional PageRank using connection weights
                - 'semantic': ConceptNet-style PageRank with relation type weighting.
                              Requires semantic relations (extracts automatically if needed).
                - 'hierarchical': Cross-layer PageRank with importance propagation
                                  between layers (tokens ↔ bigrams ↔ concepts ↔ documents).
            connection_strategy: Strategy for connecting Layer 2 concepts:
                - 'document_overlap': Traditional Jaccard similarity (default)
                - 'semantic': Connect via semantic relations between members
                - 'embedding': Connect via embedding centroid similarity
                - 'hybrid': Combine all three strategies for maximum connectivity
            cluster_strictness: Controls clustering aggressiveness (0.0-1.0).
                Lower values create fewer, larger clusters with more connections.
            bridge_weight: Weight for inter-document token bridging (0.0-1.0).
                Higher values help bridge topic-isolated clusters.

        Returns:
            Dict with computation statistics (concept_stats, etc.)

        Example:
            >>> # Default behavior
            >>> processor.compute_all()
            >>>
            >>> # Maximum connectivity for diverse documents
            >>> processor.compute_all(
            ...     connection_strategy='hybrid',
            ...     cluster_strictness=0.5,
            ...     bridge_weight=0.3
            ... )
        """
        stats: Dict[str, Any] = {}

        if verbose:
            print("Computing activation propagation...")
        self.propagate_activation(verbose=False)

        if pagerank_method == 'semantic':
            # Extract semantic relations if not already done
            if not self.semantic_relations:
                if verbose:
                    print("Extracting semantic relations...")
                self.extract_corpus_semantics(verbose=False)
            if verbose:
                print("Computing importance (Semantic PageRank)...")
            self.compute_semantic_importance(verbose=False)
        elif pagerank_method == 'hierarchical':
            if verbose:
                print("Computing importance (Hierarchical PageRank)...")
            self.compute_hierarchical_importance(verbose=False)
        else:
            if verbose:
                print("Computing importance (PageRank)...")
            self.compute_importance(verbose=False)
        if verbose:
            print("Computing TF-IDF...")
        self.compute_tfidf(verbose=False)
        if verbose:
            print("Computing document connections...")
        self.compute_document_connections(verbose=False)
        if verbose:
            print("Computing bigram connections...")
        self.compute_bigram_connections(verbose=False)

        if build_concepts:
            if verbose:
                print("Building concept clusters...")
            clusters = self.build_concept_clusters(
                cluster_strictness=cluster_strictness,
                bridge_weight=bridge_weight,
                verbose=False
            )
            stats['clusters_created'] = len(clusters)

            # Determine connection parameters based on strategy
            use_member_semantics = connection_strategy in ('semantic', 'hybrid')
            use_embedding_similarity = connection_strategy in ('embedding', 'hybrid')

            # For semantic/embedding strategies, extract/compute prerequisites
            if use_member_semantics and not self.semantic_relations:
                if verbose:
                    print("Extracting semantic relations...")
                self.extract_corpus_semantics(verbose=False)

            if use_embedding_similarity and not self.embeddings:
                if verbose:
                    print("Computing graph embeddings...")
                self.compute_graph_embeddings(verbose=False)

            # Set thresholds based on strategy
            if connection_strategy == 'hybrid':
                min_shared_docs = 0
                min_jaccard = 0.0
            elif connection_strategy in ('semantic', 'embedding'):
                min_shared_docs = 0
                min_jaccard = 0.0
            else:  # document_overlap
                min_shared_docs = 1
                min_jaccard = 0.1

            if verbose:
                print(f"Computing concept connections ({connection_strategy})...")
            concept_stats = self.compute_concept_connections(
                use_member_semantics=use_member_semantics,
                use_embedding_similarity=use_embedding_similarity,
                min_shared_docs=min_shared_docs,
                min_jaccard=min_jaccard,
                verbose=False
            )
            stats['concept_connections'] = concept_stats

        # Mark core computations as fresh
        fresh_comps = [
            self.COMP_ACTIVATION,
            self.COMP_PAGERANK,
            self.COMP_TFIDF,
            self.COMP_DOC_CONNECTIONS,
            self.COMP_BIGRAM_CONNECTIONS,
        ]
        if build_concepts:
            fresh_comps.append(self.COMP_CONCEPTS)
        self._mark_fresh(*fresh_comps)
        if verbose:
            print("Done.")

        return stats
    
    def propagate_activation(self, iterations: int = 3, decay: float = 0.8, verbose: bool = True) -> None:
        analysis.propagate_activation(self.layers, iterations, decay)
        if verbose: print(f"Propagated activation ({iterations} iterations)")
    
    def compute_importance(self, verbose: bool = True) -> None:
        for layer_enum in [CorticalLayer.TOKENS, CorticalLayer.BIGRAMS]:
            analysis.compute_pagerank(self.layers[layer_enum])
        if verbose: print("Computed PageRank importance")

    def compute_semantic_importance(
        self,
        relation_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute PageRank with semantic relation weighting.

        Uses semantic relations to weight edges in the PageRank graph.
        Edges with stronger semantic relationships (e.g., IsA, PartOf) receive
        higher weights, affecting importance propagation.

        Args:
            relation_weights: Optional custom relation type weights dict.
                Defaults to built-in weights (IsA: 1.5, PartOf: 1.3, etc.)
            verbose: Print progress messages

        Returns:
            Dict with statistics:
            - total_edges_with_relations: Sum across layers
            - token_layer: Stats for token layer
            - bigram_layer: Stats for bigram layer

        Example:
            >>> # Use default relation weights
            >>> stats = processor.compute_semantic_importance()
            >>> print(f"Found {stats['total_edges_with_relations']} semantic edges")
            >>>
            >>> # Custom weights
            >>> weights = {'IsA': 2.0, 'RelatedTo': 0.5}
            >>> processor.compute_semantic_importance(relation_weights=weights)
        """
        if not self.semantic_relations:
            # Fall back to standard PageRank if no semantic relations
            self.compute_importance(verbose=verbose)
            return {
                'total_edges_with_relations': 0,
                'token_layer': {'edges_with_relations': 0},
                'bigram_layer': {'edges_with_relations': 0}
            }

        total_edges = 0
        layer_stats = {}

        for layer_enum in [CorticalLayer.TOKENS, CorticalLayer.BIGRAMS]:
            result = analysis.compute_semantic_pagerank(
                self.layers[layer_enum],
                self.semantic_relations,
                relation_weights=relation_weights
            )
            layer_name = 'token_layer' if layer_enum == CorticalLayer.TOKENS else 'bigram_layer'
            layer_stats[layer_name] = {
                'iterations_run': result['iterations_run'],
                'edges_with_relations': result['edges_with_relations']
            }
            total_edges += result['edges_with_relations']

        if verbose:
            print(f"Computed semantic PageRank ({total_edges} relation-weighted edges)")

        return {
            'total_edges_with_relations': total_edges,
            **layer_stats
        }

    def compute_hierarchical_importance(
        self,
        layer_iterations: int = 10,
        global_iterations: int = 5,
        cross_layer_damping: float = 0.7,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute PageRank with cross-layer propagation.

        This hierarchical PageRank allows importance to flow between layers:
        - Upward: tokens → bigrams → concepts → documents
        - Downward: documents → concepts → bigrams → tokens

        Important tokens boost their containing bigrams and concepts.
        Important documents boost their contained terms. This creates
        a more holistic importance score that considers the full hierarchy.

        Args:
            layer_iterations: Max iterations for intra-layer PageRank (default 10)
            global_iterations: Max iterations for cross-layer propagation (default 5)
            cross_layer_damping: Damping factor at layer boundaries (default 0.7)
            verbose: Print progress messages

        Returns:
            Dict with statistics:
            - iterations_run: Number of global iterations
            - converged: Whether the algorithm converged
            - layer_stats: Per-layer statistics (nodes, max/min/avg PageRank)

        Example:
            >>> stats = processor.compute_hierarchical_importance()
            >>> print(f"Converged: {stats['converged']}")
            >>> for layer, info in stats['layer_stats'].items():
            ...     print(f"{layer}: {info['nodes']} nodes, max PR={info['max_pagerank']:.4f}")
        """
        result = analysis.compute_hierarchical_pagerank(
            self.layers,
            layer_iterations=layer_iterations,
            global_iterations=global_iterations,
            cross_layer_damping=cross_layer_damping
        )

        if verbose:
            status = "converged" if result['converged'] else "did not converge"
            print(f"Computed hierarchical PageRank ({result['iterations_run']} iterations, {status})")

        return result

    def compute_tfidf(self, verbose: bool = True) -> None:
        analysis.compute_tfidf(self.layers, self.documents)
        if verbose: print("Computed TF-IDF scores")
    
    def compute_document_connections(self, min_shared_terms: int = 3, verbose: bool = True) -> None:
        analysis.compute_document_connections(self.layers, self.documents, min_shared_terms)
        if verbose: print("Computed document connections")

    def compute_bigram_connections(
        self,
        min_shared_docs: int = 1,
        component_weight: float = 0.5,
        chain_weight: float = 0.7,
        cooccurrence_weight: float = 0.3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Build lateral connections between bigrams based on shared components and co-occurrence.

        Bigrams are connected when they:
        - Share a component term ("neural_networks" ↔ "neural_processing")
        - Form chains ("machine_learning" ↔ "learning_algorithms")
        - Co-occur in the same documents

        Args:
            min_shared_docs: Minimum shared documents for co-occurrence connection
            component_weight: Weight for shared component connections (default 0.5)
            chain_weight: Weight for chain connections (default 0.7)
            cooccurrence_weight: Weight for document co-occurrence (default 0.3)
            verbose: Print progress messages

        Returns:
            Statistics about connections created:
            - connections_created: Total bidirectional connections
            - component_connections: Connections from shared components
            - chain_connections: Connections from chains
            - cooccurrence_connections: Connections from document co-occurrence

        Example:
            >>> stats = processor.compute_bigram_connections()
            >>> print(f"Created {stats['connections_created']} bigram connections")
            >>> print(f"  Component: {stats['component_connections']}")
            >>> print(f"  Chain: {stats['chain_connections']}")
            >>> print(f"  Co-occurrence: {stats['cooccurrence_connections']}")
        """
        stats = analysis.compute_bigram_connections(
            self.layers,
            min_shared_docs=min_shared_docs,
            component_weight=component_weight,
            chain_weight=chain_weight,
            cooccurrence_weight=cooccurrence_weight
        )
        if verbose:
            print(f"Created {stats['connections_created']} bigram connections "
                  f"(component: {stats['component_connections']}, "
                  f"chain: {stats['chain_connections']}, "
                  f"cooccur: {stats['cooccurrence_connections']})")
        return stats

    def build_concept_clusters(
        self,
        min_cluster_size: int = 3,
        cluster_strictness: float = 1.0,
        bridge_weight: float = 0.0,
        verbose: bool = True
    ) -> Dict[int, List[str]]:
        """
        Build concept clusters from token layer using label propagation.

        Args:
            min_cluster_size: Minimum tokens per cluster (default 3)
            cluster_strictness: Controls clustering aggressiveness (0.0-1.0).
                - 1.0 (default): Strict clustering, topics stay separate
                - 0.5: Moderate mixing, allows some cross-topic clustering
                - 0.0: Minimal clustering, most tokens group together
                Lower values create fewer, larger clusters with more connections.
            bridge_weight: Weight for synthetic inter-document connections (0.0-1.0).
                When > 0, adds weak connections between tokens from different
                documents, helping bridge topic-isolated clusters.
                - 0.0 (default): No bridging
                - 0.3: Light bridging
                - 0.7: Strong bridging
            verbose: Print progress messages

        Returns:
            Dictionary mapping cluster_id to list of token contents

        Example:
            >>> # Default strict clustering
            >>> clusters = processor.build_concept_clusters()
            >>>
            >>> # Looser clustering for more cross-topic connections
            >>> clusters = processor.build_concept_clusters(
            ...     cluster_strictness=0.5,
            ...     bridge_weight=0.3
            ... )
        """
        clusters = analysis.cluster_by_label_propagation(
            self.layers[CorticalLayer.TOKENS],
            min_cluster_size=min_cluster_size,
            cluster_strictness=cluster_strictness,
            bridge_weight=bridge_weight
        )
        analysis.build_concept_clusters(self.layers, clusters)
        if verbose:
            print(f"Built {len(clusters)} concept clusters")
        return clusters

    def compute_concept_connections(
        self,
        use_semantics: bool = True,
        min_shared_docs: int = 1,
        min_jaccard: float = 0.1,
        use_member_semantics: bool = False,
        use_embedding_similarity: bool = False,
        embedding_threshold: float = 0.3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Build lateral connections between concepts based on document overlap and semantics.

        Multiple connection strategies can be combined:
        1. Document overlap (default): Jaccard similarity of document sets
        2. Semantic boost: Boost overlapping connections with semantic relations
        3. Member semantics: Connect via semantic relations even without doc overlap
        4. Embedding similarity: Connect via concept centroid similarity

        Args:
            use_semantics: Use semantic relations to boost connection weights
            min_shared_docs: Minimum shared documents for connection (0 to disable)
            min_jaccard: Minimum Jaccard similarity threshold (0.0 to disable)
            use_member_semantics: Connect concepts via member token semantic relations,
                                  independent of document overlap
            use_embedding_similarity: Connect concepts via embedding centroid similarity
            embedding_threshold: Minimum cosine similarity for embedding connections
            verbose: Print progress messages

        Returns:
            Statistics about connections created:
            - connections_created: Total connections
            - concepts: Number of concepts
            - doc_overlap_connections: Connections from document overlap
            - semantic_connections: Connections from member semantics
            - embedding_connections: Connections from embedding similarity

        Example:
            >>> # Traditional document overlap only
            >>> stats = processor.compute_concept_connections()
            >>>
            >>> # Enable all connection strategies
            >>> processor.compute_graph_embeddings()
            >>> processor.extract_corpus_semantics()
            >>> stats = processor.compute_concept_connections(
            ...     use_member_semantics=True,
            ...     use_embedding_similarity=True,
            ...     min_shared_docs=0,
            ...     min_jaccard=0.0
            ... )
        """
        semantic_rels = self.semantic_relations if use_semantics else None
        emb = self.embeddings if use_embedding_similarity else None
        stats = analysis.compute_concept_connections(
            self.layers,
            semantic_relations=semantic_rels,
            min_shared_docs=min_shared_docs,
            min_jaccard=min_jaccard,
            use_member_semantics=use_member_semantics,
            use_embedding_similarity=use_embedding_similarity,
            embedding_threshold=embedding_threshold,
            embeddings=emb
        )
        if verbose:
            parts = [f"Created {stats['connections_created']} concept connections"]
            if stats.get('doc_overlap_connections', 0) > 0:
                parts.append(f"doc_overlap: {stats['doc_overlap_connections']}")
            if stats.get('semantic_connections', 0) > 0:
                parts.append(f"semantic: {stats['semantic_connections']}")
            if stats.get('embedding_connections', 0) > 0:
                parts.append(f"embedding: {stats['embedding_connections']}")
            print(", ".join(parts) if len(parts) > 1 else parts[0])
        return stats

    def extract_corpus_semantics(
        self,
        use_pattern_extraction: bool = True,
        min_pattern_confidence: float = 0.6,
        verbose: bool = True
    ) -> int:
        """
        Extract semantic relations from the corpus.

        Combines co-occurrence analysis with pattern-based extraction to discover
        semantic relationships like IsA, HasA, UsedFor, Causes, etc.

        Args:
            use_pattern_extraction: Extract relations from text patterns (e.g., "X is a Y")
            min_pattern_confidence: Minimum confidence for pattern-based relations
            verbose: Print progress messages

        Returns:
            Number of relations extracted

        Example:
            >>> count = processor.extract_corpus_semantics(verbose=False)
            >>> print(f"Found {count} semantic relations")
        """
        self.semantic_relations = semantics.extract_corpus_semantics(
            self.layers,
            self.documents,
            self.tokenizer,
            use_pattern_extraction=use_pattern_extraction,
            min_pattern_confidence=min_pattern_confidence
        )
        if verbose:
            print(f"Extracted {len(self.semantic_relations)} semantic relations")
        return len(self.semantic_relations)

    def extract_pattern_relations(
        self,
        min_confidence: float = 0.6,
        verbose: bool = True
    ) -> List[Tuple[str, str, str, float]]:
        """
        Extract semantic relations using pattern matching only.

        Uses regex patterns to identify commonsense relations from text patterns
        like "X is a type of Y" → IsA, "X is used for Y" → UsedFor, etc.

        Args:
            min_confidence: Minimum confidence for extracted relations
            verbose: Print progress messages

        Returns:
            List of (term1, relation_type, term2, confidence) tuples

        Example:
            >>> relations = processor.extract_pattern_relations(verbose=False)
            >>> for t1, rel, t2, conf in relations[:5]:
            ...     print(f"{t1} --{rel}--> {t2} ({conf:.2f})")
        """
        layer0 = self.get_layer(CorticalLayer.TOKENS)
        valid_terms = set(layer0.minicolumns.keys())

        relations = semantics.extract_pattern_relations(
            self.documents,
            valid_terms,
            min_confidence=min_confidence
        )

        if verbose:
            stats = semantics.get_pattern_statistics(relations)
            print(f"Extracted {stats['total_relations']} pattern-based relations")
            print(f"  Types: {stats['relation_type_counts']}")

        return relations
    
    def retrofit_connections(self, iterations: int = 10, alpha: float = 0.3, verbose: bool = True) -> Dict:
        if not self.semantic_relations: self.extract_corpus_semantics(verbose=False)
        stats = semantics.retrofit_connections(self.layers, self.semantic_relations, iterations, alpha)
        if verbose: print(f"Retrofitted {stats['tokens_affected']} tokens")
        return stats

    def compute_property_inheritance(
        self,
        decay_factor: float = 0.7,
        max_depth: int = 5,
        apply_to_connections: bool = True,
        boost_factor: float = 0.3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute property inheritance based on IsA hierarchy.

        If "dog IsA animal" and "animal HasProperty living", then "dog" inherits
        "living" with a decayed weight. This enables similarity computation between
        terms that share inherited properties.

        Args:
            decay_factor: Weight multiplier per inheritance level (default 0.7)
            max_depth: Maximum inheritance depth (default 5)
            apply_to_connections: Boost lateral connections for shared properties
            boost_factor: Weight boost for shared inherited properties
            verbose: Print progress messages

        Returns:
            Dict with statistics:
            - terms_with_inheritance: Number of terms that inherited properties
            - total_properties_inherited: Total property inheritance relationships
            - connections_boosted: Connections boosted (if apply_to_connections=True)
            - inherited: The full inheritance mapping (for advanced use)

        Example:
            >>> processor.extract_corpus_semantics()
            >>> stats = processor.compute_property_inheritance()
            >>> print(f"{stats['terms_with_inheritance']} terms inherited properties")
            >>>
            >>> # Check inherited properties for a term
            >>> inherited = stats['inherited']
            >>> if 'dog' in inherited:
            ...     for prop, (weight, source, depth) in inherited['dog'].items():
            ...         print(f"  {prop}: {weight:.2f} (from {source}, depth {depth})")
        """
        if not self.semantic_relations:
            self.extract_corpus_semantics(verbose=False)

        inherited = semantics.inherit_properties(
            self.semantic_relations,
            decay_factor=decay_factor,
            max_depth=max_depth
        )

        total_props = sum(len(props) for props in inherited.values())

        result = {
            'terms_with_inheritance': len(inherited),
            'total_properties_inherited': total_props,
            'inherited': inherited
        }

        if apply_to_connections and inherited:
            conn_stats = semantics.apply_inheritance_to_connections(
                self.layers,
                inherited,
                boost_factor=boost_factor
            )
            result['connections_boosted'] = conn_stats['connections_boosted']
            result['total_boost'] = conn_stats['total_boost']
        else:
            result['connections_boosted'] = 0
            result['total_boost'] = 0.0

        if verbose:
            print(f"Computed property inheritance: {result['terms_with_inheritance']} terms, "
                  f"{total_props} properties, {result['connections_boosted']} connections boosted")

        return result

    def compute_property_similarity(self, term1: str, term2: str) -> float:
        """
        Compute similarity between terms based on shared properties (direct + inherited).

        Requires that compute_property_inheritance() or extract_corpus_semantics()
        has been called first.

        Args:
            term1: First term
            term2: Second term

        Returns:
            Similarity score (0.0-1.0) based on Jaccard-like overlap of properties

        Example:
            >>> processor.extract_corpus_semantics()
            >>> stats = processor.compute_property_inheritance()
            >>> sim = processor.compute_property_similarity("dog", "cat")
            >>> # Both inherit "living" from "animal", so similarity > 0
        """
        if not self.semantic_relations:
            return 0.0

        # Compute inherited properties on the fly if needed
        inherited = semantics.inherit_properties(self.semantic_relations)

        return semantics.compute_property_similarity(term1, term2, inherited)
    
    def compute_graph_embeddings(self, dimensions: int = 64, method: str = 'adjacency', verbose: bool = True) -> Dict:
        self.embeddings, stats = emb_module.compute_graph_embeddings(self.layers, dimensions, method)
        if verbose: print(f"Computed {stats['terms_embedded']} embeddings ({method})")
        return stats
    
    def retrofit_embeddings(self, iterations: int = 10, alpha: float = 0.4, verbose: bool = True) -> Dict:
        if not self.embeddings: self.compute_graph_embeddings(verbose=False)
        if not self.semantic_relations: self.extract_corpus_semantics(verbose=False)
        stats = semantics.retrofit_embeddings(self.embeddings, self.semantic_relations, iterations, alpha)
        if verbose: print(f"Retrofitted embeddings (moved {stats['total_movement']:.2f} total)")
        return stats
    
    def embedding_similarity(self, term1: str, term2: str) -> float:
        return emb_module.embedding_similarity(self.embeddings, term1, term2)
    
    def find_similar_by_embedding(self, term: str, top_n: int = 10) -> List[Tuple[str, float]]:
        return emb_module.find_similar_by_embedding(self.embeddings, term, top_n)
    
    def expand_query(
        self,
        query_text: str,
        max_expansions: int = 10,
        use_variants: bool = True,
        use_code_concepts: bool = False,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Expand a query using lateral connections and concept clusters.

        Args:
            query_text: Original query string
            max_expansions: Maximum expansion terms to add
            use_variants: Try word variants when direct match fails
            use_code_concepts: Include programming synonym expansions

        Returns:
            Dict mapping terms to weights
        """
        return query_module.expand_query(
            query_text,
            self.layers,
            self.tokenizer,
            max_expansions=max_expansions,
            use_variants=use_variants,
            use_code_concepts=use_code_concepts
        )

    def expand_query_for_code(self, query_text: str, max_expansions: int = 15) -> Dict[str, float]:
        """
        Expand a query optimized for code search.

        Enables code concept expansion to find programming synonyms
        (e.g., "fetch" also matches "get", "load", "retrieve").

        Args:
            query_text: Original query string
            max_expansions: Maximum expansion terms to add

        Returns:
            Dict mapping terms to weights
        """
        return query_module.expand_query(
            query_text,
            self.layers,
            self.tokenizer,
            max_expansions=max_expansions,
            use_variants=True,
            use_code_concepts=True
        )

    def expand_query_semantic(self, query_text: str, max_expansions: int = 10) -> Dict[str, float]:
        return query_module.expand_query_semantic(query_text, self.layers, self.tokenizer, self.semantic_relations, max_expansions)

    def complete_analogy(
        self,
        term_a: str,
        term_b: str,
        term_c: str,
        top_n: int = 5,
        use_embeddings: bool = True,
        use_relations: bool = True
    ) -> List[Tuple[str, float, str]]:
        """
        Complete an analogy: "a is to b as c is to ?"

        Uses multiple strategies to find the best completion:
        1. Relation matching: Find what relation connects a→b, then find terms
           with the same relation from c
        2. Vector arithmetic: Use embeddings to compute d = c + (b - a)
        3. Pattern matching: Find terms that co-occur with c similarly to how
           b co-occurs with a

        Args:
            term_a: First term of the known pair (e.g., "king")
            term_b: Second term of the known pair (e.g., "queen")
            term_c: First term of the query pair (e.g., "man")
            top_n: Number of candidates to return
            use_embeddings: Whether to use embedding-based completion
            use_relations: Whether to use relation-based completion

        Returns:
            List of (candidate_term, confidence, method) tuples, where method
            describes which approach found this candidate ('relation:IsA',
            'embedding', 'pattern')

        Example:
            >>> processor.extract_corpus_semantics()
            >>> processor.compute_graph_embeddings()
            >>> results = processor.complete_analogy("neural", "networks", "knowledge")
            >>> for term, score, method in results:
            ...     print(f"{term}: {score:.3f} ({method})")

        Raises:
            ValueError: If any term is empty or top_n is not positive
        """
        # Input validation
        for name, term in [('term_a', term_a), ('term_b', term_b), ('term_c', term_c)]:
            if not isinstance(term, str) or not term.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer")

        if not self.semantic_relations:
            self.extract_corpus_semantics(verbose=False)

        return query_module.complete_analogy(
            term_a, term_b, term_c,
            self.layers,
            self.semantic_relations,
            embeddings=self.embeddings,
            top_n=top_n,
            use_embeddings=use_embeddings,
            use_relations=use_relations
        )

    def complete_analogy_simple(
        self,
        term_a: str,
        term_b: str,
        term_c: str,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Simplified analogy completion using only term relationships.

        A lighter version that doesn't require embeddings. Uses bigram patterns
        and co-occurrence to find analogies.

        Args:
            term_a: First term of the known pair
            term_b: Second term of the known pair
            term_c: First term of the query pair
            top_n: Number of candidates to return

        Returns:
            List of (candidate_term, confidence) tuples

        Example:
            >>> results = processor.complete_analogy_simple("neural", "networks", "knowledge")
            >>> for term, score in results:
            ...     print(f"{term}: {score:.3f}")
        """
        return query_module.complete_analogy_simple(
            term_a, term_b, term_c,
            self.layers,
            self.tokenizer,
            semantic_relations=self.semantic_relations,
            top_n=top_n
        )

    def expand_query_multihop(
        self,
        query_text: str,
        max_hops: int = 2,
        max_expansions: int = 15,
        decay_factor: float = 0.5,
        min_path_score: float = 0.2
    ) -> Dict[str, float]:
        """
        Expand query using multi-hop semantic inference.

        Unlike single-hop expansion that only follows direct connections,
        this follows relation chains to discover semantically related terms
        through transitive relationships.

        Example inference chains:
            "dog" → IsA → "animal" → HasProperty → "living"
            "neural" → RelatedTo → "network" → RelatedTo → "deep"

        Args:
            query_text: Original query string
            max_hops: Maximum number of relation hops (default: 2)
            max_expansions: Maximum expansion terms to return
            decay_factor: Weight decay per hop (default: 0.5, so hop2 = 0.25)
            min_path_score: Minimum path validity score to include (default: 0.2)

        Returns:
            Dict mapping terms to weights (original terms get weight 1.0,
            expansions get decayed weights based on hop distance and path validity)

        Example:
            >>> # Extract semantic relations first
            >>> processor.extract_corpus_semantics()
            >>>
            >>> # Multi-hop expansion
            >>> expanded = processor.expand_query_multihop("neural", max_hops=2)
            >>> for term, weight in sorted(expanded.items(), key=lambda x: -x[1]):
            ...     print(f"{term}: {weight:.3f}")
        """
        if not self.semantic_relations:
            # Fall back to regular expansion if no semantic relations
            return self.expand_query(query_text, max_expansions=max_expansions)

        return query_module.expand_query_multihop(
            query_text,
            self.layers,
            self.tokenizer,
            self.semantic_relations,
            max_hops=max_hops,
            max_expansions=max_expansions,
            decay_factor=decay_factor,
            min_path_score=min_path_score
        )
    
    def find_documents_for_query(
        self,
        query_text: str,
        top_n: int = 5,
        use_expansion: bool = True,
        use_semantic: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Find documents most relevant to a query.

        Args:
            query_text: Search query
            top_n: Number of documents to return
            use_expansion: Whether to expand query terms using lateral connections
            use_semantic: Whether to use semantic relations for expansion (if available)

        Returns:
            List of (doc_id, score) tuples ranked by relevance

        Raises:
            ValueError: If query_text is empty or top_n is not positive
        """
        # Input validation
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer")

        return query_module.find_documents_for_query(
            query_text,
            self.layers,
            self.tokenizer,
            top_n=top_n,
            use_expansion=use_expansion,
            semantic_relations=self.semantic_relations if use_semantic else None,
            use_semantic=use_semantic
        )

    def find_passages_for_query(
        self,
        query_text: str,
        top_n: int = 5,
        chunk_size: int = 512,
        overlap: int = 128,
        use_expansion: bool = True,
        doc_filter: Optional[List[str]] = None,
        use_semantic: bool = True
    ) -> List[Tuple[str, str, int, int, float]]:
        """
        Find text passages most relevant to a query (for RAG systems).

        Instead of returning just document IDs, this returns actual text passages
        with position information suitable for context windows and citations.

        Args:
            query_text: Search query
            top_n: Number of passages to return
            chunk_size: Size of each chunk in characters (default 512)
            overlap: Overlap between chunks in characters (default 128)
            use_expansion: Whether to expand query terms
            doc_filter: Optional list of doc_ids to restrict search to
            use_semantic: Whether to use semantic relations for expansion (if available)

        Returns:
            List of (passage_text, doc_id, start_char, end_char, score) tuples
            ranked by relevance

        Example:
            >>> results = processor.find_passages_for_query("neural networks")
            >>> for passage, doc_id, start, end, score in results:
            ...     print(f"[{doc_id}:{start}-{end}] {passage[:50]}... (score: {score:.3f})")
        """
        return query_module.find_passages_for_query(
            query_text,
            self.layers,
            self.tokenizer,
            self.documents,
            top_n=top_n,
            chunk_size=chunk_size,
            overlap=overlap,
            use_expansion=use_expansion,
            doc_filter=doc_filter,
            semantic_relations=self.semantic_relations if use_semantic else None,
            use_semantic=use_semantic
        )

    def find_documents_batch(
        self,
        queries: List[str],
        top_n: int = 5,
        use_expansion: bool = True,
        use_semantic: bool = True
    ) -> List[List[Tuple[str, float]]]:
        """
        Find documents for multiple queries efficiently.

        More efficient than calling find_documents_for_query() multiple times
        because it shares tokenization and expansion caching across queries.

        Args:
            queries: List of search query strings
            top_n: Number of documents to return per query
            use_expansion: Whether to expand query terms using lateral connections
            use_semantic: Whether to use semantic relations for expansion

        Returns:
            List of results, one per query. Each result is a list of (doc_id, score) tuples.

        Example:
            >>> queries = ["neural networks", "machine learning", "data processing"]
            >>> results = processor.find_documents_batch(queries, top_n=3)
            >>> for query, docs in zip(queries, results):
            ...     print(f"{query}: {[doc_id for doc_id, _ in docs]}")
        """
        return query_module.find_documents_batch(
            queries,
            self.layers,
            self.tokenizer,
            top_n=top_n,
            use_expansion=use_expansion,
            semantic_relations=self.semantic_relations if use_semantic else None,
            use_semantic=use_semantic
        )

    def find_passages_batch(
        self,
        queries: List[str],
        top_n: int = 5,
        chunk_size: int = 512,
        overlap: int = 128,
        use_expansion: bool = True,
        doc_filter: Optional[List[str]] = None,
        use_semantic: bool = True
    ) -> List[List[Tuple[str, str, int, int, float]]]:
        """
        Find passages for multiple queries efficiently.

        More efficient than calling find_passages_for_query() multiple times
        because it shares chunk computation and expansion caching across queries.

        Args:
            queries: List of search query strings
            top_n: Number of passages to return per query
            chunk_size: Size of each chunk in characters (default 512)
            overlap: Overlap between chunks in characters (default 128)
            use_expansion: Whether to expand query terms
            doc_filter: Optional list of doc_ids to restrict search to
            use_semantic: Whether to use semantic relations for expansion

        Returns:
            List of results, one per query. Each result is a list of
            (passage_text, doc_id, start_char, end_char, score) tuples.

        Example:
            >>> queries = ["neural networks", "deep learning"]
            >>> results = processor.find_passages_batch(queries)
            >>> for query, passages in zip(queries, results):
            ...     print(f"{query}: {len(passages)} passages found")
        """
        return query_module.find_passages_batch(
            queries,
            self.layers,
            self.tokenizer,
            self.documents,
            top_n=top_n,
            chunk_size=chunk_size,
            overlap=overlap,
            use_expansion=use_expansion,
            doc_filter=doc_filter,
            semantic_relations=self.semantic_relations if use_semantic else None,
            use_semantic=use_semantic
        )

    def multi_stage_rank(
        self,
        query_text: str,
        top_n: int = 5,
        chunk_size: int = 512,
        overlap: int = 128,
        concept_boost: float = 0.3,
        use_expansion: bool = True,
        use_semantic: bool = True
    ) -> List[Tuple[str, str, int, int, float, Dict[str, float]]]:
        """
        Multi-stage ranking pipeline for improved RAG performance.

        Uses a 4-stage pipeline combining concept, document, and chunk signals:
        1. Concepts: Filter by topic relevance using Layer 2 clusters
        2. Documents: Rank documents within relevant topics
        3. Chunks: Rank passages within top documents
        4. Rerank: Combine all signals for final scoring

        Args:
            query_text: Search query
            top_n: Number of passages to return
            chunk_size: Size of each chunk in characters (default 512)
            overlap: Overlap between chunks in characters (default 128)
            concept_boost: Weight for concept relevance (0.0-1.0, default 0.3)
            use_expansion: Whether to expand query terms
            use_semantic: Whether to use semantic relations for expansion

        Returns:
            List of (passage_text, doc_id, start_char, end_char, final_score, stage_scores)
            tuples. stage_scores contains: concept_score, doc_score, chunk_score, final_score

        Example:
            >>> results = processor.multi_stage_rank("neural networks", top_n=5)
            >>> for passage, doc_id, start, end, score, stages in results:
            ...     print(f"[{doc_id}] Final: {score:.3f}, Concept: {stages['concept_score']:.3f}")
        """
        return query_module.multi_stage_rank(
            query_text,
            self.layers,
            self.tokenizer,
            self.documents,
            top_n=top_n,
            chunk_size=chunk_size,
            overlap=overlap,
            concept_boost=concept_boost,
            use_expansion=use_expansion,
            semantic_relations=self.semantic_relations if use_semantic else None,
            use_semantic=use_semantic
        )

    def multi_stage_rank_documents(
        self,
        query_text: str,
        top_n: int = 5,
        concept_boost: float = 0.3,
        use_expansion: bool = True,
        use_semantic: bool = True
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Multi-stage ranking for documents (without chunk scoring).

        Uses stages 1-2 of the pipeline for document-level ranking:
        1. Concepts: Filter by topic relevance
        2. Documents: Rank by combined concept + TF-IDF scores

        Args:
            query_text: Search query
            top_n: Number of documents to return
            concept_boost: Weight for concept relevance (0.0-1.0, default 0.3)
            use_expansion: Whether to expand query terms
            use_semantic: Whether to use semantic relations

        Returns:
            List of (doc_id, final_score, stage_scores) tuples.
            stage_scores contains: concept_score, tfidf_score, combined_score

        Example:
            >>> results = processor.multi_stage_rank_documents("neural networks")
            >>> for doc_id, score, stages in results:
            ...     print(f"{doc_id}: {score:.3f} (concept: {stages['concept_score']:.3f})")
        """
        return query_module.multi_stage_rank_documents(
            query_text,
            self.layers,
            self.tokenizer,
            top_n=top_n,
            concept_boost=concept_boost,
            use_expansion=use_expansion,
            semantic_relations=self.semantic_relations if use_semantic else None,
            use_semantic=use_semantic
        )

    def query_expanded(self, query_text: str, top_n: int = 10, max_expansions: int = 8) -> List[Tuple[str, float]]:
        return query_module.query_with_spreading_activation(query_text, self.layers, self.tokenizer, top_n, max_expansions)
    
    def find_related_documents(self, doc_id: str) -> List[Tuple[str, float]]:
        return query_module.find_related_documents(doc_id, self.layers)
    
    def analyze_knowledge_gaps(self) -> Dict:
        return gaps_module.analyze_knowledge_gaps(self.layers, self.documents)
    
    def detect_anomalies(self, threshold: float = 0.3) -> List[Dict]:
        return gaps_module.detect_anomalies(self.layers, self.documents, threshold)
    
    def get_layer(self, layer: CorticalLayer) -> HierarchicalLayer:
        return self.layers[layer]
    
    def get_document_signature(self, doc_id: str, n: int = 10) -> List[Tuple[str, float]]:
        layer0 = self.layers[CorticalLayer.TOKENS]
        terms = [(col.content, col.tfidf_per_doc.get(doc_id, col.tfidf)) 
                 for col in layer0.minicolumns.values() if doc_id in col.document_ids]
        return sorted(terms, key=lambda x: x[1], reverse=True)[:n]
    
    def get_corpus_summary(self) -> Dict:
        return persistence.get_state_summary(self.layers, self.documents)
    
    def save(self, filepath: str, verbose: bool = True) -> None:
        """
        Save processor state to a file.

        Saves all computed state including embeddings and semantic relations,
        so they don't need to be recomputed when loading.
        """
        metadata = {
            'has_embeddings': bool(self.embeddings),
            'has_relations': bool(self.semantic_relations)
        }
        persistence.save_processor(
            filepath,
            self.layers,
            self.documents,
            self.document_metadata,
            self.embeddings,
            self.semantic_relations,
            metadata,
            verbose
        )

    @classmethod
    def load(cls, filepath: str, verbose: bool = True) -> 'CorticalTextProcessor':
        """
        Load processor state from a file.

        Restores all computed state including embeddings and semantic relations.
        """
        result = persistence.load_processor(filepath, verbose)
        layers, documents, document_metadata, embeddings, semantic_relations, metadata = result
        processor = cls()
        processor.layers = layers
        processor.documents = documents
        processor.document_metadata = document_metadata
        processor.embeddings = embeddings
        processor.semantic_relations = semantic_relations
        return processor
    
    def export_graph(self, filepath: str, layer: Optional[CorticalLayer] = None, max_nodes: int = 500) -> Dict:
        return persistence.export_graph_json(filepath, self.layers, layer, max_nodes=max_nodes)

    def export_conceptnet_json(
        self,
        filepath: str,
        include_cross_layer: bool = True,
        include_typed_edges: bool = True,
        min_weight: float = 0.0,
        min_confidence: float = 0.0,
        max_nodes_per_layer: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Export ConceptNet-style graph for visualization.

        Creates a rich graph format with:
        - Color-coded nodes by layer (tokens=blue, bigrams=green, concepts=orange, docs=red)
        - Typed edges with relation types and confidence scores
        - Cross-layer connections (feedforward/feedback)
        - D3.js/Cytoscape-compatible output

        Args:
            filepath: Output file path (JSON)
            include_cross_layer: Include feedforward/feedback edges
            include_typed_edges: Include typed_connections with relation types
            min_weight: Minimum edge weight to include
            min_confidence: Minimum confidence for typed edges
            max_nodes_per_layer: Maximum nodes per layer (by PageRank)
            verbose: Print progress messages

        Returns:
            The exported graph data

        Example:
            >>> processor.extract_corpus_semantics(verbose=False)
            >>> graph = processor.export_conceptnet_json("graph.json")
            >>> # Open graph.json in D3.js or Cytoscape for visualization
        """
        return persistence.export_conceptnet_json(
            filepath,
            self.layers,
            semantic_relations=self.semantic_relations,
            include_cross_layer=include_cross_layer,
            include_typed_edges=include_typed_edges,
            min_weight=min_weight,
            min_confidence=min_confidence,
            max_nodes_per_layer=max_nodes_per_layer,
            verbose=verbose
        )

    def summarize_document(self, doc_id: str, num_sentences: int = 3) -> str:
        if doc_id not in self.documents: return ""
        content = self.documents[doc_id]
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) <= num_sentences: return content
        
        layer0 = self.layers[CorticalLayer.TOKENS]
        scored = []
        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            score = sum(layer0.get_minicolumn(t).tfidf if layer0.get_minicolumn(t) else 0 for t in tokens)
            scored.append((sent, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [s for s, _ in scored[:num_sentences]]
        return ' '.join([s for s in sentences if s in top])
    
    def __repr__(self) -> str:
        stats = self.get_corpus_summary()
        return f"CorticalTextProcessor(documents={stats['documents']}, columns={stats['total_columns']})"
