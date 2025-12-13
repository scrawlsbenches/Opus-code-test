"""
Result Dataclasses for Cortical Text Processor
===============================================

Strongly-typed result containers for query operations that provide
IDE autocomplete and type checking support.

Example:
    # Document search results
    matches = processor.find_documents_for_query("neural networks")
    document_matches = [DocumentMatch.from_tuple(doc_id, score)
                        for doc_id, score in matches]
    for match in document_matches:
        print(f"{match.doc_id}: {match.score:.3f}")

    # Passage retrieval results
    passages = processor.find_passages_for_query("PageRank algorithm")
    passage_matches = [PassageMatch.from_tuple(*p) for p in passages]
    for match in passage_matches:
        print(f"[{match.doc_id}:{match.start}-{match.end}] {match.text[:50]}...")
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union


@dataclass(frozen=True)
class DocumentMatch:
    """
    A document search result with relevance score.

    Attributes:
        doc_id: Document identifier
        score: Relevance score (higher is more relevant)
        metadata: Optional metadata dict for additional information

    Example:
        >>> match = DocumentMatch("doc1.txt", 0.95)
        >>> print(match.doc_id)
        'doc1.txt'
        >>> print(f"Score: {match.score:.2f}")
        'Score: 0.95'
        >>> match_dict = match.to_dict()
    """
    doc_id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        """Pretty string representation."""
        if self.metadata:
            return f"DocumentMatch(doc_id='{self.doc_id}', score={self.score:.4f}, metadata={self.metadata})"
        return f"DocumentMatch(doc_id='{self.doc_id}', score={self.score:.4f})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary with doc_id, score, and metadata fields

        Example:
            >>> match = DocumentMatch("doc1", 0.8)
            >>> match.to_dict()
            {'doc_id': 'doc1', 'score': 0.8, 'metadata': None}
        """
        return asdict(self)

    def to_tuple(self) -> tuple:
        """
        Convert to tuple format (doc_id, score).

        Returns:
            Tuple of (doc_id, score) for compatibility with legacy code

        Example:
            >>> match = DocumentMatch("doc1", 0.8)
            >>> match.to_tuple()
            ('doc1', 0.8)
        """
        return (self.doc_id, self.score)

    @classmethod
    def from_tuple(cls, doc_id: str, score: float, metadata: Optional[Dict[str, Any]] = None) -> 'DocumentMatch':
        """
        Create from tuple format (doc_id, score).

        Args:
            doc_id: Document identifier
            score: Relevance score
            metadata: Optional metadata dict

        Returns:
            DocumentMatch instance

        Example:
            >>> match = DocumentMatch.from_tuple("doc1", 0.8)
            >>> match.doc_id
            'doc1'
        """
        return cls(doc_id=doc_id, score=score, metadata=metadata)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMatch':
        """
        Create from dictionary.

        Args:
            data: Dictionary with doc_id, score, and optional metadata fields

        Returns:
            DocumentMatch instance

        Example:
            >>> data = {'doc_id': 'doc1', 'score': 0.8}
            >>> match = DocumentMatch.from_dict(data)
            >>> match.score
            0.8
        """
        return cls(
            doc_id=data['doc_id'],
            score=data['score'],
            metadata=data.get('metadata')
        )


@dataclass(frozen=True)
class PassageMatch:
    """
    A passage retrieval result with text, location, and relevance score.

    Suitable for RAG (Retrieval-Augmented Generation) systems where you need
    actual text passages with position information for citations.

    Attributes:
        doc_id: Document identifier
        text: Passage text content
        score: Relevance score (higher is more relevant)
        start: Start character position in document
        end: End character position in document
        metadata: Optional metadata dict for additional information

    Example:
        >>> match = PassageMatch(
        ...     doc_id="doc1.py",
        ...     text="def compute_pagerank():\\n    ...",
        ...     score=0.92,
        ...     start=100,
        ...     end=150
        ... )
        >>> print(f"[{match.doc_id}:{match.start}-{match.end}]")
        '[doc1.py:100-150]'
        >>> print(match.text[:30])
        'def compute_pagerank():\n    ...'
    """
    doc_id: str
    text: str
    score: float
    start: int
    end: int
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        """Pretty string representation with truncated text."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        text_preview = text_preview.replace('\n', '\\n')
        if self.metadata:
            return (f"PassageMatch(doc_id='{self.doc_id}', text='{text_preview}', "
                   f"score={self.score:.4f}, start={self.start}, end={self.end}, "
                   f"metadata={self.metadata})")
        return (f"PassageMatch(doc_id='{self.doc_id}', text='{text_preview}', "
               f"score={self.score:.4f}, start={self.start}, end={self.end})")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary with all fields

        Example:
            >>> match = PassageMatch("doc1", "text here", 0.9, 0, 9)
            >>> match.to_dict()
            {'doc_id': 'doc1', 'text': 'text here', 'score': 0.9, 'start': 0, 'end': 9, 'metadata': None}
        """
        return asdict(self)

    def to_tuple(self) -> tuple:
        """
        Convert to tuple format (doc_id, text, start, end, score).

        Returns:
            Tuple for compatibility with legacy code

        Example:
            >>> match = PassageMatch("doc1", "text", 0.8, 0, 4)
            >>> match.to_tuple()
            ('doc1', 'text', 0, 4, 0.8)
        """
        return (self.doc_id, self.text, self.start, self.end, self.score)

    @property
    def location(self) -> str:
        """
        Get citation-style location string.

        Returns:
            Location in format "doc_id:start-end"

        Example:
            >>> match = PassageMatch("doc1.py", "text", 0.8, 100, 150)
            >>> match.location
            'doc1.py:100-150'
        """
        return f"{self.doc_id}:{self.start}-{self.end}"

    @property
    def length(self) -> int:
        """
        Get passage length in characters.

        Returns:
            Number of characters in passage

        Example:
            >>> match = PassageMatch("doc1", "hello", 0.8, 0, 5)
            >>> match.length
            5
        """
        return self.end - self.start

    @classmethod
    def from_tuple(
        cls,
        doc_id: str,
        text: str,
        start: int,
        end: int,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'PassageMatch':
        """
        Create from tuple format (doc_id, text, start, end, score).

        Args:
            doc_id: Document identifier
            text: Passage text
            start: Start character position
            end: End character position
            score: Relevance score
            metadata: Optional metadata dict

        Returns:
            PassageMatch instance

        Example:
            >>> match = PassageMatch.from_tuple("doc1", "hello", 0, 5, 0.9)
            >>> match.text
            'hello'
        """
        return cls(
            doc_id=doc_id,
            text=text,
            score=score,
            start=start,
            end=end,
            metadata=metadata
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PassageMatch':
        """
        Create from dictionary.

        Args:
            data: Dictionary with required fields

        Returns:
            PassageMatch instance

        Example:
            >>> data = {'doc_id': 'doc1', 'text': 'hi', 'score': 0.8, 'start': 0, 'end': 2}
            >>> match = PassageMatch.from_dict(data)
            >>> match.length
            2
        """
        return cls(
            doc_id=data['doc_id'],
            text=data['text'],
            score=data['score'],
            start=data['start'],
            end=data['end'],
            metadata=data.get('metadata')
        )


@dataclass(frozen=True)
class QueryResult:
    """
    Complete query result with matches and metadata.

    Wraps search results with additional context like query expansion terms
    and timing information. Useful for analyzing search quality and debugging.

    Attributes:
        query: Original query text
        matches: List of DocumentMatch or PassageMatch results
        expansion_terms: Optional dict of expanded terms and their weights
        timing_ms: Optional query execution time in milliseconds
        metadata: Optional metadata dict for additional information

    Example:
        >>> doc_matches = [DocumentMatch("doc1", 0.9), DocumentMatch("doc2", 0.7)]
        >>> result = QueryResult(
        ...     query="neural networks",
        ...     matches=doc_matches,
        ...     expansion_terms={"neural": 1.0, "network": 0.8, "deep": 0.5},
        ...     timing_ms=15.3
        ... )
        >>> print(f"Found {len(result.matches)} matches in {result.timing_ms}ms")
        'Found 2 matches in 15.3ms'
        >>> result.top_match
        DocumentMatch(doc_id='doc1', score=0.9000)
    """
    query: str
    matches: Union[List[DocumentMatch], List[PassageMatch]]
    expansion_terms: Optional[Dict[str, float]] = None
    timing_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        """Pretty string representation."""
        match_type = "DocumentMatch" if self.matches and isinstance(self.matches[0], DocumentMatch) else "PassageMatch"
        return (f"QueryResult(query='{self.query}', "
               f"matches={len(self.matches)} x {match_type}, "
               f"expansion_terms={len(self.expansion_terms) if self.expansion_terms else 0}, "
               f"timing_ms={self.timing_ms})")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with nested match dicts.

        Returns:
            Dictionary representation

        Example:
            >>> result = QueryResult("test", [DocumentMatch("doc1", 0.9)])
            >>> d = result.to_dict()
            >>> d['query']
            'test'
        """
        return {
            'query': self.query,
            'matches': [m.to_dict() for m in self.matches],
            'expansion_terms': self.expansion_terms,
            'timing_ms': self.timing_ms,
            'metadata': self.metadata
        }

    @property
    def top_match(self) -> Union[DocumentMatch, PassageMatch, None]:
        """
        Get the highest-scoring match.

        Returns:
            Top match or None if no matches

        Example:
            >>> result = QueryResult("test", [DocumentMatch("doc1", 0.5), DocumentMatch("doc2", 0.9)])
            >>> result.top_match.doc_id
            'doc2'
        """
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.score)

    @property
    def match_count(self) -> int:
        """
        Get number of matches.

        Returns:
            Count of matches

        Example:
            >>> result = QueryResult("test", [DocumentMatch("doc1", 0.9)])
            >>> result.match_count
            1
        """
        return len(self.matches)

    @property
    def average_score(self) -> float:
        """
        Get average relevance score across all matches.

        Returns:
            Average score or 0.0 if no matches

        Example:
            >>> result = QueryResult("test", [DocumentMatch("doc1", 0.8), DocumentMatch("doc2", 0.6)])
            >>> result.average_score
            0.7
        """
        if not self.matches:
            return 0.0
        return sum(m.score for m in self.matches) / len(self.matches)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryResult':
        """
        Create from dictionary.

        Args:
            data: Dictionary with query, matches, and optional fields

        Returns:
            QueryResult instance

        Example:
            >>> data = {
            ...     'query': 'test',
            ...     'matches': [{'doc_id': 'doc1', 'score': 0.9, 'metadata': None}],
            ...     'expansion_terms': {'test': 1.0},
            ...     'timing_ms': 10.0
            ... }
            >>> result = QueryResult.from_dict(data)
            >>> result.query
            'test'
        """
        # Determine match type from first match
        matches = []
        if data['matches']:
            first_match = data['matches'][0]
            if 'text' in first_match:
                matches = [PassageMatch.from_dict(m) for m in data['matches']]
            else:
                matches = [DocumentMatch.from_dict(m) for m in data['matches']]

        return cls(
            query=data['query'],
            matches=matches,
            expansion_terms=data.get('expansion_terms'),
            timing_ms=data.get('timing_ms'),
            metadata=data.get('metadata')
        )


# Helper functions for batch conversions
def convert_document_matches(
    results: List[tuple],
    metadata: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[DocumentMatch]:
    """
    Convert list of (doc_id, score) tuples to DocumentMatch objects.

    Args:
        results: List of (doc_id, score) tuples
        metadata: Optional dict mapping doc_id to metadata dict

    Returns:
        List of DocumentMatch objects

    Example:
        >>> results = [("doc1", 0.9), ("doc2", 0.7)]
        >>> matches = convert_document_matches(results)
        >>> matches[0].doc_id
        'doc1'
    """
    if metadata:
        return [DocumentMatch(doc_id, score, metadata.get(doc_id))
                for doc_id, score in results]
    return [DocumentMatch(doc_id, score) for doc_id, score in results]


def convert_passage_matches(
    results: List[tuple],
    metadata: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[PassageMatch]:
    """
    Convert list of (doc_id, text, start, end, score) tuples to PassageMatch objects.

    Args:
        results: List of (doc_id, text, start, end, score) tuples
        metadata: Optional dict mapping doc_id to metadata dict

    Returns:
        List of PassageMatch objects

    Example:
        >>> results = [("doc1", "text here", 0, 9, 0.9)]
        >>> matches = convert_passage_matches(results)
        >>> matches[0].text
        'text here'
    """
    if metadata:
        return [PassageMatch(doc_id, text, score, start, end, metadata.get(doc_id))
                for doc_id, text, start, end, score in results]
    return [PassageMatch(doc_id, text, score, start, end)
            for doc_id, text, start, end, score in results]
