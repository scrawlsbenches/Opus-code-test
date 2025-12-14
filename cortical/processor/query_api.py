"""
Query API: search, expansion, and retrieval methods.

This module contains all query-related methods that delegate to the query module.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

from .. import query as query_module
from ..observability import timed

logger = logging.getLogger(__name__)


class QueryMixin:
    """
    Mixin providing query functionality.

    Requires CoreMixin to be present (provides layers, documents, tokenizer,
    config, semantic_relations, embeddings, _query_expansion_cache, _query_cache_max_size).
    """

    def expand_query(
        self,
        query_text: str,
        max_expansions: Optional[int] = None,
        use_variants: bool = True,
        use_code_concepts: bool = False,
        filter_code_stop_words: bool = False,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Expand a query using lateral connections and concept clusters.

        Args:
            query_text: Original query string
            max_expansions: Maximum expansion terms to add (default from config)
            use_variants: Try word variants when direct match fails
            use_code_concepts: Include programming synonym expansions
            filter_code_stop_words: Filter ubiquitous code tokens (self, cls, etc.)

        Returns:
            Dict mapping terms to weights

        Raises:
            ValueError: If query_text is empty or max_expansions is negative
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        if max_expansions is None:
            max_expansions = self.config.max_query_expansions
        if not isinstance(max_expansions, int) or max_expansions < 0:
            raise ValueError("max_expansions must be a non-negative integer")

        return query_module.expand_query(
            query_text,
            self.layers,
            self.tokenizer,
            max_expansions=max_expansions,
            use_variants=use_variants,
            use_code_concepts=use_code_concepts,
            filter_code_stop_words=filter_code_stop_words
        )

    def expand_query_for_code(self, query_text: str, max_expansions: Optional[int] = None) -> Dict[str, float]:
        """
        Expand a query optimized for code search.

        Args:
            query_text: Original query string
            max_expansions: Maximum expansion terms to add (default from config + 5)

        Returns:
            Dict mapping terms to weights
        """
        if max_expansions is None:
            max_expansions = self.config.max_query_expansions + 5

        return query_module.expand_query(
            query_text,
            self.layers,
            self.tokenizer,
            max_expansions=max_expansions,
            use_variants=True,
            use_code_concepts=True,
            filter_code_stop_words=True
        )

    def expand_query_cached(
        self,
        query_text: str,
        max_expansions: Optional[int] = None,
        use_variants: bool = True,
        use_code_concepts: bool = False
    ) -> Dict[str, float]:
        """
        Expand a query with caching for faster repeated lookups.

        Args:
            query_text: Original query string
            max_expansions: Maximum expansion terms to add (default from config)
            use_variants: Try word variants when direct match fails
            use_code_concepts: Include programming synonym expansions

        Returns:
            Dict mapping terms to weights
        """
        if max_expansions is None:
            max_expansions = self.config.max_query_expansions

        cache_key = f"{query_text}|{max_expansions}|{use_variants}|{use_code_concepts}"

        if cache_key in self._query_expansion_cache:
            self._metrics.record_count("query_cache_hits")
            return self._query_expansion_cache[cache_key].copy()

        self._metrics.record_count("query_cache_misses")
        result = query_module.expand_query(
            query_text,
            self.layers,
            self.tokenizer,
            max_expansions=max_expansions,
            use_variants=use_variants,
            use_code_concepts=use_code_concepts
        )

        if len(self._query_expansion_cache) >= self._query_cache_max_size:
            oldest_key = next(iter(self._query_expansion_cache))
            del self._query_expansion_cache[oldest_key]

        self._query_expansion_cache[cache_key] = result.copy()
        return result

    def clear_query_cache(self) -> int:
        """
        Clear the query expansion cache.

        Returns:
            Number of cache entries cleared
        """
        count = len(self._query_expansion_cache)
        self._query_expansion_cache.clear()
        return count

    def set_query_cache_size(self, max_size: int) -> None:
        """
        Set the maximum size of the query expansion cache.

        Args:
            max_size: Maximum number of queries to cache (must be > 0)

        Raises:
            ValueError: If max_size <= 0
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._query_cache_max_size = max_size

        while len(self._query_expansion_cache) > max_size:
            oldest_key = next(iter(self._query_expansion_cache))
            del self._query_expansion_cache[oldest_key]

    def parse_intent_query(self, query_text: str) -> Dict:
        """
        Parse a natural language query to extract intent and searchable terms.

        Args:
            query_text: Natural language query string

        Returns:
            Dict with 'action', 'subject', 'intent', 'question_word', 'expanded_terms'
        """
        return query_module.parse_intent_query(query_text)

    def search_by_intent(self, query_text: str, top_n: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search the corpus using intent-based query understanding.

        Args:
            query_text: Natural language query string
            top_n: Number of results to return

        Returns:
            List of (doc_id, score, parsed_intent) tuples
        """
        return query_module.search_by_intent(
            query_text,
            self.layers,
            self.tokenizer,
            top_n=top_n
        )

    def expand_query_semantic(self, query_text: str, max_expansions: int = 10) -> Dict[str, float]:
        return query_module.expand_query_semantic(
            query_text, self.layers, self.tokenizer, self.semantic_relations, max_expansions
        )

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

        Args:
            term_a: First term of the known pair
            term_b: Second term of the known pair
            term_c: First term of the query pair
            top_n: Number of candidates to return
            use_embeddings: Whether to use embedding-based completion
            use_relations: Whether to use relation-based completion

        Returns:
            List of (candidate_term, confidence, method) tuples

        Raises:
            ValueError: If any term is empty or top_n is not positive
        """
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

        Args:
            term_a: First term of the known pair
            term_b: Second term of the known pair
            term_c: First term of the query pair
            top_n: Number of candidates to return

        Returns:
            List of (candidate_term, confidence) tuples
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

        Args:
            query_text: Original query string
            max_hops: Maximum number of relation hops (default: 2)
            max_expansions: Maximum expansion terms to return
            decay_factor: Weight decay per hop (default: 0.5)
            min_path_score: Minimum path validity score (default: 0.2)

        Returns:
            Dict mapping terms to weights
        """
        if not self.semantic_relations:
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

    @timed("find_documents_for_query", include_args=True)
    def find_documents_for_query(
        self,
        query_text: str,
        top_n: int = 5,
        use_expansion: bool = True,
        use_semantic: bool = True,
        filter_code_stop_words: bool = True,
        test_file_penalty: float = 0.8
    ) -> List[Tuple[str, float]]:
        """
        Find documents most relevant to a query.

        Args:
            query_text: Search query
            top_n: Number of documents to return
            use_expansion: Whether to expand query terms
            use_semantic: Whether to use semantic relations for expansion
            filter_code_stop_words: Filter ubiquitous code tokens (self, def, return)
                                    from expansion. Reduces noise in code search. (default True)
            test_file_penalty: Multiplier for test files to rank them lower (default 0.8).
                               Set to 1.0 to disable penalty.

        Returns:
            List of (doc_id, score) tuples ranked by relevance

        Raises:
            ValueError: If query_text is empty or top_n is not positive
        """
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
            use_semantic=use_semantic,
            filter_code_stop_words=filter_code_stop_words,
            test_file_penalty=test_file_penalty
        )

    def fast_find_documents(
        self,
        query_text: str,
        top_n: int = 5,
        candidate_multiplier: int = 3,
        use_code_concepts: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Fast document search using candidate filtering.

        Args:
            query_text: Search query
            top_n: Number of results to return
            candidate_multiplier: Multiplier for candidate set size
            use_code_concepts: Whether to use code concept expansion

        Returns:
            List of (doc_id, score) tuples ranked by relevance

        Raises:
            ValueError: If query_text is empty or top_n is not positive
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer")
        if not isinstance(candidate_multiplier, int) or candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be a positive integer")

        return query_module.fast_find_documents(
            query_text,
            self.layers,
            self.tokenizer,
            top_n=top_n,
            candidate_multiplier=candidate_multiplier,
            use_code_concepts=use_code_concepts
        )

    def quick_search(self, query: str, top_n: int = 5) -> List[str]:
        """
        One-call document search with sensible defaults.

        Args:
            query: Search query string
            top_n: Number of results to return (default 5)

        Returns:
            List of document IDs ranked by relevance
        """
        results = self.find_documents_for_query(query, top_n=top_n)
        return [doc_id for doc_id, _score in results]

    def rag_retrieve(
        self,
        query: str,
        top_n: int = 3,
        max_chars_per_passage: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Retrieve passages optimized for RAG.

        Args:
            query: Search query string
            top_n: Number of passages to return (default 3)
            max_chars_per_passage: Maximum characters per passage (default 500)

        Returns:
            List of passage dictionaries with text, doc_id, start, end, score
        """
        results = self.find_passages_for_query(
            query,
            top_n=top_n,
            chunk_size=max_chars_per_passage
        )
        return [
            {
                'text': text,
                'doc_id': doc_id,
                'start': start,
                'end': end,
                'score': score
            }
            for text, doc_id, start, end, score in results
        ]

    def explore(self, query: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Search with query expansion visibility.

        Args:
            query: Search query string
            top_n: Number of results to return (default 5)

        Returns:
            Dictionary with results, expansion, original_terms
        """
        expansion = self.expand_query(query)
        results = self.find_documents_for_query(query, top_n=top_n)
        original_terms = list(self.tokenizer.tokenize(query))

        return {
            'results': results,
            'expansion': expansion,
            'original_terms': original_terms
        }

    def find_documents_with_boost(
        self,
        query_text: str,
        top_n: int = 5,
        auto_detect_intent: bool = True,
        prefer_docs: bool = False,
        custom_boosts: Optional[Dict[str, float]] = None,
        use_expansion: bool = True,
        use_semantic: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Find documents with optional document-type boosting.

        Args:
            query_text: Search query
            top_n: Number of results to return
            auto_detect_intent: Auto-boost docs for conceptual queries
            prefer_docs: Always boost documentation
            custom_boosts: Optional custom boost factors per doc_type
            use_expansion: Whether to expand query terms
            use_semantic: Whether to use semantic relations

        Returns:
            List of (doc_id, score) tuples ranked by relevance
        """
        return query_module.find_documents_with_boost(
            query_text,
            self.layers,
            self.tokenizer,
            top_n=top_n,
            doc_metadata=self.document_metadata,
            auto_detect_intent=auto_detect_intent,
            prefer_docs=prefer_docs,
            custom_boosts=custom_boosts,
            use_expansion=use_expansion,
            semantic_relations=self.semantic_relations if use_semantic else None,
            use_semantic=use_semantic
        )

    def is_conceptual_query(self, query_text: str) -> bool:
        """Check if a query appears to be conceptual."""
        return query_module.is_conceptual_query(query_text)

    def build_search_index(self) -> Dict[str, Dict[str, float]]:
        """Build an optimized inverted index for fast querying."""
        return query_module.build_document_index(self.layers)

    def search_with_index(
        self,
        query_text: str,
        index: Dict[str, Dict[str, float]],
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Search using a pre-built inverted index."""
        return query_module.search_with_index(
            query_text,
            index,
            self.tokenizer,
            top_n=top_n
        )

    def find_passages_for_query(
        self,
        query_text: str,
        top_n: int = 5,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        use_expansion: bool = True,
        doc_filter: Optional[List[str]] = None,
        use_semantic: bool = True,
        use_definition_search: bool = True,
        definition_boost: float = 5.0,
        apply_doc_boost: bool = True,
        auto_detect_intent: bool = True,
        prefer_docs: bool = False,
        custom_boosts: Optional[Dict[str, float]] = None,
        use_code_aware_chunks: bool = True,
        filter_code_stop_words: bool = True,
        test_file_penalty: float = 0.8
    ) -> List[Tuple[str, str, int, int, float]]:
        """
        Find text passages most relevant to a query (for RAG systems).

        Args:
            query_text: Search query
            top_n: Number of passages to return
            chunk_size: Size of each chunk in characters (default from config)
            overlap: Overlap between chunks in characters (default from config)
            use_expansion: Whether to expand query terms
            doc_filter: Optional list of doc_ids to restrict search to
            use_semantic: Whether to use semantic relations for expansion
            use_definition_search: Whether to search for definition patterns
            definition_boost: Score boost for definition matches
            apply_doc_boost: Whether to apply document-type boosting
            auto_detect_intent: Auto-detect conceptual queries and boost docs
            prefer_docs: Always boost documentation
            custom_boosts: Optional custom boost factors for doc types
            use_code_aware_chunks: Use semantic boundaries for code files
            filter_code_stop_words: Filter ubiquitous code tokens (self, def, return)
                                    from expansion. Reduces noise in code search. (default True)
            test_file_penalty: Multiplier for test files to rank them lower (default 0.8).
                               Set to 1.0 to disable penalty.

        Returns:
            List of (passage_text, doc_id, start_char, end_char, score) tuples

        Raises:
            ValueError: If query_text is empty or parameters are invalid
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer")

        if chunk_size is None:
            chunk_size = self.config.chunk_size
        else:
            if not isinstance(chunk_size, int) or chunk_size < 1:
                raise ValueError("chunk_size must be a positive integer")

        if overlap is None:
            overlap = self.config.chunk_overlap
        else:
            if not isinstance(overlap, int) or overlap < 0:
                raise ValueError("overlap must be a non-negative integer")
            if overlap >= chunk_size:
                raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

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
            use_semantic=use_semantic,
            use_definition_search=use_definition_search,
            definition_boost=definition_boost,
            apply_doc_boost=apply_doc_boost,
            doc_metadata=self.document_metadata,
            auto_detect_intent=auto_detect_intent,
            prefer_docs=prefer_docs,
            custom_boosts=custom_boosts,
            use_code_aware_chunks=use_code_aware_chunks,
            filter_code_stop_words=filter_code_stop_words,
            test_file_penalty=test_file_penalty
        )

    def is_definition_query(self, query_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Detect if a query is looking for a code definition."""
        return query_module.is_definition_query(query_text)

    def find_definition_passages(
        self,
        query_text: str,
        context_chars: int = 500,
        boost: float = 5.0
    ) -> List[Tuple[str, str, int, int, float]]:
        """Find definition passages for a definition query."""
        return query_module.find_definition_passages(
            query_text, self.documents, context_chars, boost
        )

    def find_documents_batch(
        self,
        queries: List[str],
        top_n: int = 5,
        use_expansion: bool = True,
        use_semantic: bool = True
    ) -> List[List[Tuple[str, float]]]:
        """Find documents for multiple queries efficiently."""
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
        """Find passages for multiple queries efficiently."""
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
        """Multi-stage ranking pipeline for improved RAG performance."""
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
        """Multi-stage ranking for documents (without chunk scoring)."""
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
        return query_module.query_with_spreading_activation(
            query_text, self.layers, self.tokenizer, top_n, max_expansions
        )

    def find_related_documents(self, doc_id: str) -> List[Tuple[str, float]]:
        return query_module.find_related_documents(doc_id, self.layers)
