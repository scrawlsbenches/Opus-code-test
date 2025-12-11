"""
Search Engine Module - Sample code for demonstrating code search features.

This module implements a simple search engine with indexing and ranking.
"""

from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import math


class SearchIndex:
    """Inverted index for text search.

    The SearchIndex maintains an inverted index mapping terms to documents,
    enabling fast full-text search with TF-IDF ranking.

    Example:
        index = SearchIndex()
        index.add_document("doc1", "hello world")
        results = index.search("hello")
    """

    def __init__(self):
        """Initialize empty search index."""
        self._inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self._documents: Dict[str, str] = {}
        self._term_frequencies: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._document_lengths: Dict[str, int] = {}

    def add_document(self, doc_id: str, content: str) -> None:
        """Add a document to the index.

        Args:
            doc_id: Unique document identifier
            content: Text content to index
        """
        self._documents[doc_id] = content
        tokens = self._tokenize(content)
        self._document_lengths[doc_id] = len(tokens)

        for token in tokens:
            self._inverted_index[token].add(doc_id)
            self._term_frequencies[doc_id][token] += 1

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index.

        Args:
            doc_id: The document ID to remove

        Returns:
            True if document was removed, False if not found
        """
        if doc_id not in self._documents:
            return False

        # Remove from inverted index
        for term, doc_ids in self._inverted_index.items():
            doc_ids.discard(doc_id)

        # Clean up empty term entries
        empty_terms = [t for t, ids in self._inverted_index.items() if not ids]
        for term in empty_terms:
            del self._inverted_index[term]

        del self._documents[doc_id]
        del self._term_frequencies[doc_id]
        del self._document_lengths[doc_id]
        return True

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for documents matching the query.

        Args:
            query: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of (doc_id, score) tuples sorted by relevance
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = defaultdict(float)

        for token in query_tokens:
            if token not in self._inverted_index:
                continue

            idf = self._compute_idf(token)
            for doc_id in self._inverted_index[token]:
                tf = self._compute_tf(token, doc_id)
                scores[doc_id] += tf * idf

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_document(self, doc_id: str) -> Optional[str]:
        """Retrieve document content by ID.

        Args:
            doc_id: The document identifier

        Returns:
            Document content string if found, None otherwise
        """
        return self._documents.get(doc_id)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase terms.

        Args:
            text: Input text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Simple whitespace tokenization
        return [t.lower().strip('.,!?;:()[]{}') for t in text.split() if t.strip()]

    def _compute_tf(self, term: str, doc_id: str) -> float:
        """Compute term frequency for a term in a document.

        Args:
            term: The term to compute TF for
            doc_id: The document identifier

        Returns:
            Normalized term frequency
        """
        raw_tf = self._term_frequencies[doc_id][term]
        doc_length = self._document_lengths[doc_id]
        return raw_tf / doc_length if doc_length > 0 else 0

    def _compute_idf(self, term: str) -> float:
        """Compute inverse document frequency for a term.

        Args:
            term: The term to compute IDF for

        Returns:
            IDF value using log scaling
        """
        n_docs = len(self._documents)
        doc_freq = len(self._inverted_index.get(term, set()))
        if doc_freq == 0:
            return 0
        return math.log(n_docs / doc_freq)


class QueryParser:
    """Parser for search queries with advanced syntax.

    Supports:
        - Simple terms: hello world
        - Phrase queries: "hello world"
        - Required terms: +important
        - Excluded terms: -spam
    """

    def __init__(self):
        """Initialize the query parser."""
        self._operators = {'+', '-', '"'}

    def parse(self, query: str) -> Dict[str, List[str]]:
        """Parse a query string into components.

        Args:
            query: The query string to parse

        Returns:
            Dictionary with keys:
                - required: Terms that must appear
                - excluded: Terms that must not appear
                - optional: Regular search terms
                - phrases: Exact phrase matches
        """
        result = {
            'required': [],
            'excluded': [],
            'optional': [],
            'phrases': []
        }

        i = 0
        tokens = query.split()

        while i < len(tokens):
            token = tokens[i]

            if token.startswith('+'):
                result['required'].append(token[1:].lower())
            elif token.startswith('-'):
                result['excluded'].append(token[1:].lower())
            elif token.startswith('"'):
                # Handle phrase - collect until closing quote
                phrase_tokens = [token[1:]]
                i += 1
                while i < len(tokens) and not tokens[i].endswith('"'):
                    phrase_tokens.append(tokens[i])
                    i += 1
                if i < len(tokens):
                    phrase_tokens.append(tokens[i][:-1])
                result['phrases'].append(' '.join(phrase_tokens).lower())
            else:
                result['optional'].append(token.lower())

            i += 1

        return result


def compute_bm25_score(
    term: str,
    doc_id: str,
    index: SearchIndex,
    k1: float = 1.5,
    b: float = 0.75
) -> float:
    """Compute BM25 score for a term in a document.

    BM25 is a ranking function used by search engines to estimate
    the relevance of documents to a given search query.

    Args:
        term: The search term
        doc_id: The document identifier
        index: The SearchIndex to use
        k1: Term frequency saturation parameter
        b: Length normalization parameter

    Returns:
        BM25 score for the term-document pair
    """
    tf = index._term_frequencies[doc_id].get(term, 0)
    doc_length = index._document_lengths[doc_id]
    avg_doc_length = sum(index._document_lengths.values()) / len(index._document_lengths)

    idf = index._compute_idf(term)

    numerator = tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * doc_length / avg_doc_length)

    return idf * numerator / denominator


def highlight_matches(text: str, query_terms: List[str], marker: str = '**') -> str:
    """Highlight query term matches in text.

    Args:
        text: The text to highlight
        query_terms: List of terms to highlight
        marker: String to use for highlighting (wraps matches)

    Returns:
        Text with query terms wrapped in markers
    """
    result = text
    for term in query_terms:
        # Case-insensitive replacement
        import re
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(f'{marker}{term}{marker}', result)
    return result
