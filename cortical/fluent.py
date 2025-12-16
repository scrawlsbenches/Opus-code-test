"""
Fluent API for CorticalTextProcessor - chainable method interface.

Example:
    from cortical import FluentProcessor

    # Simple usage
    results = (FluentProcessor()
        .add_document("doc1", "Neural networks process information")
        .add_document("doc2", "Deep learning uses neural architectures")
        .build()
        .search("neural processing", top_n=5))

    # From files
    results = (FluentProcessor
        .from_files(["file1.txt", "file2.txt"])
        .build()
        .search("query"))

    # Advanced configuration
    processor = (FluentProcessor()
        .add_documents({
            "doc1": "content 1",
            "doc2": "content 2"
        })
        .build(verbose=True, build_concepts=True)
        .save("corpus.pkl"))
"""

import os
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

from .processor import CorticalTextProcessor
from .tokenizer import Tokenizer
from .config import CorticalConfig


class FluentProcessor:
    """
    Fluent/chainable API wrapper for CorticalTextProcessor.

    Provides a builder pattern interface for constructing and querying
    text processors with method chaining.

    Example:
        >>> processor = (FluentProcessor()
        ...     .add_document("doc1", "text")
        ...     .build()
        ...     .search("query"))
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        config: Optional[CorticalConfig] = None
    ):
        """
        Initialize a fluent processor.

        Args:
            tokenizer: Optional custom tokenizer
            config: Optional configuration object
        """
        self._processor = CorticalTextProcessor(tokenizer=tokenizer, config=config)
        self._is_built = False

    @classmethod
    def from_existing(cls, processor: CorticalTextProcessor) -> 'FluentProcessor':
        """
        Create a FluentProcessor from an existing CorticalTextProcessor.

        Args:
            processor: Existing CorticalTextProcessor instance

        Returns:
            FluentProcessor wrapping the existing processor

        Example:
            >>> proc = CorticalTextProcessor()
            >>> fluent = FluentProcessor.from_existing(proc)
        """
        instance = cls.__new__(cls)
        instance._processor = processor
        instance._is_built = False
        return instance

    @classmethod
    def from_files(
        cls,
        file_paths: List[Union[str, Path]],
        tokenizer: Optional[Tokenizer] = None,
        config: Optional[CorticalConfig] = None
    ) -> 'FluentProcessor':
        """
        Create a processor from a list of files.

        Args:
            file_paths: List of file paths to process
            tokenizer: Optional custom tokenizer
            config: Optional configuration object

        Returns:
            FluentProcessor with documents added from files

        Example:
            >>> processor = FluentProcessor.from_files(["doc1.txt", "doc2.txt"])
        """
        instance = cls(tokenizer=tokenizer, config=config)
        for path in file_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if not path_obj.is_file():
                raise ValueError(f"Not a file: {path}")

            doc_id = path_obj.stem  # Use filename without extension as doc_id
            with open(path_obj, 'r', encoding='utf-8') as f:
                content = f.read()

            instance._processor.process_document(doc_id, content, metadata={'source': str(path)})

        return instance

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        pattern: str = "*.txt",
        recursive: bool = False,
        tokenizer: Optional[Tokenizer] = None,
        config: Optional[CorticalConfig] = None
    ) -> 'FluentProcessor':
        """
        Create a processor from all files in a directory.

        Args:
            directory: Directory path to scan
            pattern: Glob pattern for file matching (default: "*.txt")
            recursive: Whether to search subdirectories
            tokenizer: Optional custom tokenizer
            config: Optional configuration object

        Returns:
            FluentProcessor with documents added from directory

        Example:
            >>> processor = FluentProcessor.from_directory("./docs", pattern="*.md")
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Find files matching pattern
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        if not files:
            raise ValueError(f"No files matching pattern '{pattern}' found in {directory}")

        return cls.from_files(files, tokenizer=tokenizer, config=config)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FluentProcessor':
        """
        Load a processor from a saved file.

        Args:
            path: Path to saved processor file

        Returns:
            FluentProcessor loaded from file

        Example:
            >>> processor = FluentProcessor.load("corpus.pkl")
        """
        loaded = CorticalTextProcessor.load(str(path))
        instance = cls.from_existing(loaded)
        instance._is_built = True  # Loaded processors are already built
        return instance

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'FluentProcessor':
        """
        Add a document to the processor (chainable).

        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata dictionary

        Returns:
            Self for method chaining

        Example:
            >>> processor = (FluentProcessor()
            ...     .add_document("doc1", "content")
            ...     .add_document("doc2", "more content"))
        """
        self._processor.process_document(doc_id, content, metadata)
        self._is_built = False  # Mark as needing rebuild
        return self

    def add_documents(
        self,
        documents: Union[Dict[str, str], List[Tuple[str, str]], List[Tuple[str, str, Dict]]]
    ) -> 'FluentProcessor':
        """
        Add multiple documents at once (chainable).

        Args:
            documents: Can be:
                - Dict mapping doc_id -> content
                - List of (doc_id, content) tuples
                - List of (doc_id, content, metadata) tuples

        Returns:
            Self for method chaining

        Example:
            >>> # From dict
            >>> processor = FluentProcessor().add_documents({
            ...     "doc1": "content 1",
            ...     "doc2": "content 2"
            ... })

            >>> # From list of tuples
            >>> processor = FluentProcessor().add_documents([
            ...     ("doc1", "content 1"),
            ...     ("doc2", "content 2", {"author": "Alice"})
            ... ])
        """
        if isinstance(documents, dict):
            for doc_id, content in documents.items():
                self._processor.process_document(doc_id, content)
        elif isinstance(documents, list):
            for item in documents:
                if len(item) == 2:
                    doc_id, content = item
                    self._processor.process_document(doc_id, content)
                elif len(item) == 3:
                    doc_id, content, metadata = item
                    self._processor.process_document(doc_id, content, metadata)
                else:
                    raise ValueError(f"Invalid document tuple: {item}. Expected (doc_id, content) or (doc_id, content, metadata)")
        else:
            raise TypeError("documents must be a dict or list of tuples")

        self._is_built = False
        return self

    def with_config(self, config: CorticalConfig) -> 'FluentProcessor':
        """
        Set configuration (chainable).

        Args:
            config: CorticalConfig object

        Returns:
            Self for method chaining

        Example:
            >>> from cortical import CorticalConfig
            >>> config = CorticalConfig(min_token_length=2)
            >>> processor = FluentProcessor().with_config(config)
        """
        self._processor.config = config
        return self

    def with_tokenizer(self, tokenizer: Tokenizer) -> 'FluentProcessor':
        """
        Set custom tokenizer (chainable).

        Args:
            tokenizer: Custom Tokenizer instance

        Returns:
            Self for method chaining

        Example:
            >>> from cortical import Tokenizer
            >>> tokenizer = Tokenizer(split_identifiers=True)
            >>> processor = FluentProcessor().with_tokenizer(tokenizer)
        """
        self._processor.tokenizer = tokenizer
        return self

    def build(
        self,
        verbose: bool = True,
        build_concepts: bool = True,
        pagerank_method: str = 'standard',
        connection_strategy: str = 'document_overlap',
        cluster_strictness: float = 1.0,
        bridge_weight: float = 0.0,
        show_progress: bool = False
    ) -> 'FluentProcessor':
        """
        Build the processor by computing all analysis phases (chainable).

        This calls compute_all() on the underlying processor to perform:
        - TF-IDF computation
        - PageRank importance
        - Bigram connections
        - Document connections
        - Concept clustering (optional)

        Args:
            verbose: Print debug messages via Python logging (complementary to show_progress)
            build_concepts: Build concept clusters (Layer 2)
            pagerank_method: 'standard', 'semantic', or 'hierarchical'
            connection_strategy: 'document_overlap', 'semantic', 'embedding', or 'hybrid'
            cluster_strictness: Controls clustering aggressiveness (0.0-1.0)
            bridge_weight: Weight for inter-document token bridging (0.0-1.0)
            show_progress: Show progress bar on console

        Returns:
            Self for method chaining

        Example:
            >>> processor = (FluentProcessor()
            ...     .add_document("doc1", "content")
            ...     .build(verbose=False))
        """
        self._processor.compute_all(
            verbose=verbose,
            build_concepts=build_concepts,
            pagerank_method=pagerank_method,
            connection_strategy=connection_strategy,
            cluster_strictness=cluster_strictness,
            bridge_weight=bridge_weight,
            show_progress=show_progress
        )
        self._is_built = True
        return self

    def save(self, path: Union[str, Path]) -> 'FluentProcessor':
        """
        Save the processor to disk (chainable).

        Args:
            path: File path to save to

        Returns:
            Self for method chaining

        Example:
            >>> processor = (FluentProcessor()
            ...     .add_document("doc1", "content")
            ...     .build()
            ...     .save("corpus.pkl"))
        """
        self._processor.save(str(path))
        return self

    # ========== Terminal operations (return results, not self) ==========

    def search(
        self,
        query: str,
        top_n: int = 5,
        use_expansion: bool = True,
        use_semantic: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query string
            top_n: Number of results to return
            use_expansion: Use query expansion
            use_semantic: Use semantic expansion

        Returns:
            List of (doc_id, score) tuples sorted by relevance

        Example:
            >>> results = (FluentProcessor()
            ...     .add_document("doc1", "neural networks")
            ...     .build()
            ...     .search("neural", top_n=10))
        """
        return self._processor.find_documents_for_query(
            query, top_n=top_n, use_expansion=use_expansion, use_semantic=use_semantic
        )

    def fast_search(
        self,
        query: str,
        top_n: int = 5,
        candidate_multiplier: int = 3,
        use_code_concepts: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Fast document search with pre-filtering.

        Args:
            query: Search query string
            top_n: Number of results to return
            candidate_multiplier: Candidate pool size multiplier
            use_code_concepts: Use code concept expansion

        Returns:
            List of (doc_id, score) tuples sorted by relevance

        Example:
            >>> results = processor.build().fast_search("authentication", top_n=5)
        """
        return self._processor.fast_find_documents(
            query, top_n=top_n, candidate_multiplier=candidate_multiplier,
            use_code_concepts=use_code_concepts
        )

    def search_passages(
        self,
        query: str,
        top_n: int = 5,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        use_expansion: bool = True
    ) -> List[Tuple[str, str, int, int, float]]:
        """
        Search for passage chunks matching the query.

        Args:
            query: Search query string
            top_n: Number of passage results
            chunk_size: Token count per chunk (default from config)
            overlap: Token overlap between chunks (default from config)
            use_expansion: Use query expansion

        Returns:
            List of (doc_id, passage_text, start_pos, end_pos, score) tuples

        Example:
            >>> passages = processor.build().search_passages("neural networks", top_n=3)
        """
        return self._processor.find_passages_for_query(
            query, top_n=top_n, chunk_size=chunk_size,
            overlap=overlap, use_expansion=use_expansion
        )

    def expand(
        self,
        query: str,
        max_expansions: Optional[int] = None,
        use_variants: bool = True,
        use_code_concepts: bool = False
    ) -> Dict[str, float]:
        """
        Expand a query with related terms.

        Args:
            query: Query string to expand
            max_expansions: Maximum number of expansion terms
            use_variants: Include term variants
            use_code_concepts: Use code concept synonyms

        Returns:
            Dict mapping terms to expansion weights

        Example:
            >>> expansions = processor.build().expand("neural networks")
            >>> # {'neural': 1.0, 'networks': 1.0, 'network': 0.8, ...}
        """
        return self._processor.expand_query(
            query, max_expansions=max_expansions,
            use_variants=use_variants, use_code_concepts=use_code_concepts
        )

    # ========== Property access to underlying processor ==========

    @property
    def processor(self) -> CorticalTextProcessor:
        """
        Access the underlying CorticalTextProcessor instance.

        Returns:
            The wrapped CorticalTextProcessor

        Example:
            >>> fluent = FluentProcessor().add_document("doc1", "text")
            >>> raw_processor = fluent.processor
            >>> raw_processor.compute_importance()
        """
        return self._processor

    @property
    def is_built(self) -> bool:
        """
        Check if the processor has been built.

        Returns:
            True if build() has been called
        """
        return self._is_built

    def __repr__(self) -> str:
        """String representation."""
        doc_count = len(self._processor.documents)
        status = "built" if self._is_built else "not built"
        return f"FluentProcessor(documents={doc_count}, status={status})"
