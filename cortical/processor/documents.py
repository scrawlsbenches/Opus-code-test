"""
Document management: processing, adding, removing, and metadata handling.

This module contains all methods related to managing documents in the corpus.
"""

import copy
import logging
from typing import Dict, List, Tuple, Optional, Any

from ..layers import CorticalLayer
from ..observability import timed

logger = logging.getLogger(__name__)


class DocumentsMixin:
    """
    Mixin providing document management functionality.

    Requires CoreMixin to be present (provides tokenizer, layers, documents,
    document_metadata, _mark_all_stale, _query_expansion_cache).
    """

    @timed("process_document", include_args=True)
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
        # Cache tokenized document name for fast doc_name_boost in search
        # This avoids re-tokenizing the doc_id on every query
        doc_col.name_tokens = set(self.tokenizer.tokenize(doc_id.replace('_', ' ')))

        for token in tokens:
            col = layer0.get_or_create_minicolumn(token)
            col.occurrence_count += 1
            col.document_ids.add(doc_id)
            col.activation += 1.0
            # Weighted feedforward: document -> token (weight by occurrence count)
            doc_col.add_feedforward_connection(col.id, 1.0)
            # Weighted feedback: token -> document (weight by occurrence count)
            col.add_feedback_connection(doc_col.id, 1.0)
            # Track per-document occurrence count for accurate TF-IDF
            col.doc_occurrence_counts[doc_id] = col.doc_occurrence_counts.get(doc_id, 0) + 1

        for i, token in enumerate(tokens):
            col = layer0.get_minicolumn(token)
            if col:
                for j in range(max(0, i - 3), min(len(tokens), i + 4)):
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
                    # Weighted feedforward: bigram -> tokens (weight 1.0 per occurrence)
                    col.add_feedforward_connection(token_col.id, 1.0)
                    # Weighted feedback: token -> bigram (weight 1.0 per occurrence)
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
        return copy.deepcopy(self.document_metadata)

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
            logger.info(f"Adding {len(documents)} documents...")

        for doc_id, content, metadata in documents:
            # Use process_document directly (not add_document_incremental)
            # to avoid per-document recomputation
            stats = self.process_document(doc_id, content, metadata)
            total_tokens += stats['tokens']
            total_bigrams += stats['bigrams']

        if verbose:
            logger.info(f"Processed {total_tokens} tokens, {total_bigrams} bigrams")

        # Perform single recomputation for entire batch
        if recompute == 'tfidf':
            if verbose:
                logger.info("Recomputing TF-IDF...")
            self.compute_tfidf(verbose=False)
            self._mark_fresh(self.COMP_TFIDF)
        elif recompute == 'full':
            if verbose:
                logger.info("Running full recomputation...")
            self.compute_all(verbose=False)
            self._stale_computations.clear()

        if verbose:
            logger.info("Done.")

        return {
            'documents_added': len(documents),
            'total_tokens': total_tokens,
            'total_bigrams': total_bigrams,
            'recomputation': recompute
        }

    def remove_document(self, doc_id: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Remove a document from the corpus.

        Removes the document and cleans up all references to it in the layers:
        - Removes from documents dict and metadata
        - Removes document minicolumn from Layer 3
        - Removes doc_id from token and bigram document_ids sets
        - Decrements occurrence counts appropriately
        - Cleans up feedforward/feedback connections

        Args:
            doc_id: Document identifier to remove
            verbose: Print progress messages

        Returns:
            Dict with removal statistics:
                - found: Whether the document existed
                - tokens_affected: Number of tokens that referenced this document
                - bigrams_affected: Number of bigrams that referenced this document

        Example:
            >>> processor.remove_document("old_doc")
            {'found': True, 'tokens_affected': 42, 'bigrams_affected': 35}
        """
        if doc_id not in self.documents:
            return {'found': False, 'tokens_affected': 0, 'bigrams_affected': 0}

        if verbose:
            logger.info(f"Removing document: {doc_id}")

        # Remove from documents and metadata
        del self.documents[doc_id]
        if doc_id in self.document_metadata:
            del self.document_metadata[doc_id]

        # Remove document minicolumn from Layer 3
        layer3 = self.layers[CorticalLayer.DOCUMENTS]
        doc_col = layer3.get_minicolumn(doc_id)
        if doc_col:
            # Get tokens/bigrams that were connected to this document
            connected_ids = set(doc_col.feedforward_connections.keys())
            layer3.remove_minicolumn(doc_id)

        # Clean up token references in Layer 0
        layer0 = self.layers[CorticalLayer.TOKENS]
        tokens_affected = 0
        for content, col in list(layer0.minicolumns.items()):
            if doc_id in col.document_ids:
                col.document_ids.discard(doc_id)
                tokens_affected += 1

                # Decrement occurrence count by per-doc count
                if doc_id in col.doc_occurrence_counts:
                    col.occurrence_count -= col.doc_occurrence_counts[doc_id]
                    del col.doc_occurrence_counts[doc_id]

                # Clean up feedback connections to document
                doc_col_id = f"L3_{doc_id}"
                if doc_col_id in col.feedback_connections:
                    del col.feedback_connections[doc_col_id]

        # Clean up bigram references in Layer 1
        layer1 = self.layers[CorticalLayer.BIGRAMS]
        bigrams_affected = 0
        for content, col in list(layer1.minicolumns.items()):
            if doc_id in col.document_ids:
                col.document_ids.discard(doc_id)
                bigrams_affected += 1

                # Decrement occurrence count (approximate since we don't track per-doc for bigrams)
                if doc_id in col.doc_occurrence_counts:
                    col.occurrence_count -= col.doc_occurrence_counts[doc_id]
                    del col.doc_occurrence_counts[doc_id]

        # Mark all computations as stale
        self._mark_all_stale()

        # Invalidate query cache since corpus changed
        if hasattr(self, '_query_expansion_cache'):
            self._query_expansion_cache.clear()

        if verbose:
            logger.info(f"  Affected: {tokens_affected} tokens, {bigrams_affected} bigrams")

        return {
            'found': True,
            'tokens_affected': tokens_affected,
            'bigrams_affected': bigrams_affected
        }

    def remove_documents_batch(
        self,
        doc_ids: List[str],
        recompute: str = 'none',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Remove multiple documents efficiently with single recomputation.

        Args:
            doc_ids: List of document identifiers to remove
            recompute: Level of recomputation after removal:
                - 'none': Just remove documents, mark computations stale
                - 'tfidf': Recompute TF-IDF only
                - 'full': Run full compute_all()
            verbose: Print progress messages

        Returns:
            Dict with removal statistics:
                - documents_removed: Number of documents actually removed
                - documents_not_found: Number of doc_ids that didn't exist
                - total_tokens_affected: Total tokens affected
                - total_bigrams_affected: Total bigrams affected

        Example:
            >>> processor.remove_documents_batch(["old1", "old2", "old3"])
        """
        removed = 0
        not_found = 0
        total_tokens = 0
        total_bigrams = 0

        if verbose:
            logger.info(f"Removing {len(doc_ids)} documents...")

        for doc_id in doc_ids:
            result = self.remove_document(doc_id, verbose=False)
            if result['found']:
                removed += 1
                total_tokens += result['tokens_affected']
                total_bigrams += result['bigrams_affected']
            else:
                not_found += 1

        if verbose:
            logger.info(f"  Removed: {removed}, Not found: {not_found}")
            logger.info(f"  Affected: {total_tokens} tokens, {total_bigrams} bigrams")

        # Perform recomputation
        if recompute == 'tfidf':
            if verbose:
                logger.info("Recomputing TF-IDF...")
            self.compute_tfidf(verbose=False)
            self._mark_fresh(self.COMP_TFIDF)
        elif recompute == 'full':
            if verbose:
                logger.info("Running full recomputation...")
            self.compute_all(verbose=False)
            self._stale_computations.clear()

        return {
            'documents_removed': removed,
            'documents_not_found': not_found,
            'total_tokens_affected': total_tokens,
            'total_bigrams_affected': total_bigrams,
            'recomputation': recompute
        }
