"""
Spark Predictor
===============

Unified facade for SparkSLM - the statistical first-blitz predictor.

Combines:
- N-gram model for word prediction
- Alignment index for user context
- Fast priming for search enhancement

Example:
    >>> from cortical.spark import SparkPredictor
    >>> spark = SparkPredictor()
    >>> spark.train_from_processor(processor)
    >>> spark.load_alignment("samples/alignment/")
    >>>
    >>> # Prime a query
    >>> primed = spark.prime("authentication handler")
    >>> print(primed['keywords'])
    >>> print(primed['completions'])
    >>>
    >>> # Get alignment context
    >>> context = spark.get_alignment_context("spark")
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os

from .ngram import NGramModel
from .alignment import AlignmentIndex


class SparkPredictor:
    """
    SparkSLM: Statistical First-Blitz Predictor.

    Provides fast, lightweight predictions to "prime" deeper search.
    Not a replacement for full analysis - a spark that guides it.
    """

    def __init__(self, ngram_order: int = 3):
        """
        Initialize SparkPredictor.

        Args:
            ngram_order: Order of n-gram model (default 3 = trigram)
        """
        self.ngram = NGramModel(n=ngram_order)
        self.alignment = AlignmentIndex()
        self._trained = False
        self._alignment_loaded = False

    def train_from_processor(self, processor: 'CorticalTextProcessor') -> 'SparkPredictor':
        """
        Train on documents from a CorticalTextProcessor.

        Args:
            processor: Processor with loaded documents

        Returns:
            self for method chaining
        """
        from ..layers import CorticalLayer

        # Get document layer
        doc_layer = processor.layers.get(CorticalLayer.DOCUMENTS)
        if not doc_layer:
            return self

        # Get token layer for vocabulary
        token_layer = processor.layers.get(CorticalLayer.TOKENS)

        # Collect document texts
        # Documents store their content in metadata or we reconstruct from tokens
        documents = []

        for col in doc_layer.minicolumns.values():
            doc_id = col.content
            # Try to get document metadata
            metadata = processor.get_document_metadata(doc_id) if hasattr(processor, 'get_document_metadata') else {}

            if 'text' in metadata:
                documents.append(metadata['text'])
            elif hasattr(processor, '_document_texts') and doc_id in processor._document_texts:
                documents.append(processor._document_texts[doc_id])

        # If we have documents, train on them
        if documents:
            self.ngram.train(documents)
            self._trained = True

        # Also train on token vocabulary (fallback)
        if token_layer and not documents:
            # Use token content as vocabulary hints
            tokens = [col.content for col in token_layer.minicolumns.values()]
            # Create synthetic documents from token pairs
            for i in range(0, len(tokens) - 2, 3):
                synthetic = ' '.join(tokens[i:i+10])
                documents.append(synthetic)

            if documents:
                self.ngram.train(documents)
                self._trained = True

        return self

    def train_from_documents(self, documents: List[str]) -> 'SparkPredictor':
        """
        Train directly on document texts.

        Args:
            documents: List of document texts

        Returns:
            self for method chaining
        """
        self.ngram.train(documents)
        self._trained = True
        return self

    def train_from_directory(self, directory: str, extensions: List[str] = None) -> 'SparkPredictor':
        """
        Train on files from a directory.

        Args:
            directory: Directory path
            extensions: File extensions to include (default: .txt, .md, .py)

        Returns:
            self for method chaining
        """
        if extensions is None:
            extensions = ['.txt', '.md', '.py', '.rst']

        documents = []
        dir_path = Path(directory)

        if not dir_path.exists():
            return self

        for ext in extensions:
            for file_path in dir_path.rglob(f'*{ext}'):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content.strip():
                            documents.append(content)
                except Exception:
                    continue

        if documents:
            self.ngram.train(documents)
            self._trained = True

        return self

    def load_alignment(self, path: str) -> 'SparkPredictor':
        """
        Load alignment context from file or directory.

        Args:
            path: Path to alignment file (.json) or directory with .md files

        Returns:
            self for method chaining
        """
        path_obj = Path(path)

        if path_obj.is_file():
            if path.endswith('.json'):
                self.alignment = AlignmentIndex.load(path)
            elif path.endswith('.md'):
                self.alignment.load_from_markdown(path)
        elif path_obj.is_dir():
            # Load all markdown files in directory
            for md_file in path_obj.glob('*.md'):
                self.alignment.load_from_markdown(str(md_file))

        self._alignment_loaded = len(self.alignment) > 0
        return self

    def prime(self, query: str) -> Dict[str, Any]:
        """
        Generate first-blitz suggestions for a query.

        This is the main entry point for priming search.

        Args:
            query: User query

        Returns:
            Dictionary with:
            - keywords: Key terms extracted from query
            - completions: Predicted next words
            - alignment: Relevant alignment context
            - topics: Suggested topic classifications
        """
        result = {
            'query': query,
            'keywords': [],
            'completions': [],
            'alignment': [],
            'topics': [],
            'is_trained': self._trained,
        }

        # Extract keywords (simple: content words)
        words = query.lower().split()
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'under', 'again', 'further',
                     'then', 'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                     'because', 'until', 'while', 'although', 'though', 'what',
                     'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                     'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                     'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                     'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                     'itself', 'they', 'them', 'their', 'theirs', 'themselves'}

        result['keywords'] = [w for w in words if w not in stop_words and len(w) > 1]

        # Get completions if trained
        if self._trained and words:
            completions = self.ngram.predict(words, top_k=5)
            result['completions'] = completions

        # Get alignment context
        if self._alignment_loaded:
            alignment_matches = self.alignment.search(query, top_k=3)
            result['alignment'] = [
                {'key': key, 'type': entry.entry_type, 'value': entry.value}
                for key, entry in alignment_matches
            ]

        return result

    def complete(self, prefix: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next words given a prefix.

        Args:
            prefix: Text prefix
            top_k: Number of predictions

        Returns:
            List of (word, probability) tuples
        """
        if not self._trained:
            return []

        words = prefix.lower().split()
        return self.ngram.predict(words, top_k=top_k)

    def complete_sequence(self, prefix: str, length: int = 3) -> str:
        """
        Complete a prefix with predicted words.

        Args:
            prefix: Text prefix
            length: Number of words to add

        Returns:
            Completed text
        """
        if not self._trained:
            return prefix

        words = prefix.lower().split()
        predictions = self.ngram.predict_sequence(words, length=length)
        return prefix + ' ' + ' '.join(predictions)

    def get_alignment_context(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Look up alignment context for a term.

        Args:
            term: Term to look up

        Returns:
            Alignment entry dict or None
        """
        entries = self.alignment.lookup(term)
        if entries:
            e = entries[0]
            return {
                'key': e.key,
                'value': e.value,
                'type': e.entry_type,
                'source': e.source,
            }
        return None

    def get_context_summary(self) -> str:
        """
        Get full alignment context summary.

        Returns:
            Markdown-formatted alignment summary
        """
        return self.alignment.get_context_summary()

    def add_definition(self, term: str, meaning: str) -> None:
        """Add a definition to alignment index."""
        self.alignment.add_definition(term, meaning, source='session')
        self._alignment_loaded = True

    def add_pattern(self, name: str, description: str) -> None:
        """Add a pattern to alignment index."""
        self.alignment.add_pattern(name, description, source='session')
        self._alignment_loaded = True

    def add_preference(self, topic: str, preference: str) -> None:
        """Add a preference to alignment index."""
        self.alignment.add_preference(topic, preference, source='session')
        self._alignment_loaded = True

    def save(self, directory: str) -> None:
        """
        Save SparkPredictor state.

        Args:
            directory: Directory to save to
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save n-gram model
        self.ngram.save(str(dir_path / 'ngram.json'))

        # Save alignment index
        self.alignment.save(str(dir_path / 'alignment.json'))

    @classmethod
    def load(cls, directory: str) -> 'SparkPredictor':
        """
        Load SparkPredictor from saved state.

        Args:
            directory: Directory to load from

        Returns:
            Loaded SparkPredictor
        """
        dir_path = Path(directory)
        spark = cls()

        ngram_path = dir_path / 'ngram.json'
        if ngram_path.exists():
            spark.ngram = NGramModel.load(str(ngram_path))
            spark._trained = True

        alignment_path = dir_path / 'alignment.json'
        if alignment_path.exists():
            spark.alignment = AlignmentIndex.load(str(alignment_path))
            spark._alignment_loaded = len(spark.alignment) > 0

        return spark

    def __repr__(self) -> str:
        return (
            f"SparkPredictor("
            f"trained={self._trained}, "
            f"ngram={self.ngram}, "
            f"alignment={self.alignment})"
        )
