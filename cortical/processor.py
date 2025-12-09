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
        """
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
            doc_col.feedforward_sources.add(col.id)
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
                    col.feedforward_sources.add(token_col.id)
        
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

    def compute_all(self, verbose: bool = True, build_concepts: bool = True) -> None:
        """
        Run all computation steps.

        Args:
            verbose: Print progress messages
            build_concepts: Build concept clusters in Layer 2 (default True)
                           This enables topic-based filtering and hierarchical search.
        """
        if verbose:
            print("Computing activation propagation...")
        self.propagate_activation(verbose=False)
        if verbose:
            print("Computing importance (PageRank)...")
        self.compute_importance(verbose=False)
        if verbose:
            print("Computing TF-IDF...")
        self.compute_tfidf(verbose=False)
        if verbose:
            print("Computing document connections...")
        self.compute_document_connections(verbose=False)
        if build_concepts:
            if verbose:
                print("Building concept clusters...")
            self.build_concept_clusters(verbose=False)
        if verbose:
            print("Done.")
    
    def propagate_activation(self, iterations: int = 3, decay: float = 0.8, verbose: bool = True) -> None:
        analysis.propagate_activation(self.layers, iterations, decay)
        if verbose: print(f"Propagated activation ({iterations} iterations)")
    
    def compute_importance(self, verbose: bool = True) -> None:
        for layer_enum in [CorticalLayer.TOKENS, CorticalLayer.BIGRAMS]:
            analysis.compute_pagerank(self.layers[layer_enum])
        if verbose: print("Computed PageRank importance")
    
    def compute_tfidf(self, verbose: bool = True) -> None:
        analysis.compute_tfidf(self.layers, self.documents)
        if verbose: print("Computed TF-IDF scores")
    
    def compute_document_connections(self, min_shared_terms: int = 3, verbose: bool = True) -> None:
        analysis.compute_document_connections(self.layers, self.documents, min_shared_terms)
        if verbose: print("Computed document connections")
    
    def build_concept_clusters(self, verbose: bool = True) -> Dict[int, List[str]]:
        clusters = analysis.cluster_by_label_propagation(self.layers[CorticalLayer.TOKENS])
        analysis.build_concept_clusters(self.layers, clusters)
        if verbose: print(f"Built {len(clusters)} concept clusters")
        return clusters
    
    def extract_corpus_semantics(self, verbose: bool = True) -> int:
        self.semantic_relations = semantics.extract_corpus_semantics(self.layers, self.documents, self.tokenizer)
        if verbose: print(f"Extracted {len(self.semantic_relations)} semantic relations")
        return len(self.semantic_relations)
    
    def retrofit_connections(self, iterations: int = 10, alpha: float = 0.3, verbose: bool = True) -> Dict:
        if not self.semantic_relations: self.extract_corpus_semantics(verbose=False)
        stats = semantics.retrofit_connections(self.layers, self.semantic_relations, iterations, alpha)
        if verbose: print(f"Retrofitted {stats['tokens_affected']} tokens")
        return stats
    
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
    
    def expand_query(self, query_text: str, max_expansions: int = 10, use_variants: bool = True, verbose: bool = False) -> Dict[str, float]:
        return query_module.expand_query(query_text, self.layers, self.tokenizer, max_expansions=max_expansions, use_variants=use_variants)
    
    def expand_query_semantic(self, query_text: str, max_expansions: int = 10) -> Dict[str, float]:
        return query_module.expand_query_semantic(query_text, self.layers, self.tokenizer, self.semantic_relations, max_expansions)
    
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
        """
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
