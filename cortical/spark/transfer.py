"""
Cross-Project Transfer: Vocabulary analysis and model portability.

This module enables knowledge transfer between projects by:
1. Analyzing vocabulary composition (shared vs project-specific)
2. Creating portable models that transfer across projects
3. Adapting transferred knowledge to new domains

The key insight: programming language constructs are shared across projects,
while domain-specific terms are project-specific. By separating these,
we can transfer the shared knowledge.
"""

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from .ngram import NGramModel


# Common programming terms that appear across projects
# These are good candidates for transfer
PROGRAMMING_VOCABULARY = {
    # Control flow
    'if', 'else', 'elif', 'for', 'while', 'return', 'break', 'continue',
    'try', 'except', 'finally', 'raise', 'with', 'yield', 'await', 'async',

    # Declarations
    'def', 'class', 'import', 'from', 'as', 'global', 'nonlocal', 'lambda',

    # Operators and values
    'and', 'or', 'not', 'in', 'is', 'true', 'false', 'none', 'null',

    # Common types
    'int', 'str', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
    'string', 'integer', 'boolean', 'array', 'object', 'function',

    # Common methods
    'get', 'set', 'add', 'remove', 'update', 'delete', 'create', 'read',
    'write', 'open', 'close', 'init', 'start', 'stop', 'run', 'execute',
    'load', 'save', 'parse', 'format', 'convert', 'validate', 'check',
    'find', 'search', 'filter', 'sort', 'map', 'reduce', 'count', 'sum',

    # Common patterns
    'error', 'exception', 'warning', 'debug', 'info', 'log', 'print',
    'test', 'assert', 'mock', 'stub', 'fixture', 'setup', 'teardown',
    'config', 'setting', 'option', 'parameter', 'argument', 'value',
    'key', 'name', 'id', 'type', 'data', 'result', 'response', 'request',
    'input', 'output', 'source', 'target', 'path', 'file', 'directory',

    # Documentation
    'todo', 'fixme', 'note', 'see', 'param', 'returns', 'raises', 'example',
}


@dataclass
class VocabularyAnalysis:
    """Analysis of a vocabulary's composition."""

    total_terms: int
    programming_terms: int
    project_specific_terms: int
    programming_ratio: float
    top_programming_terms: List[Tuple[str, int]]
    top_project_terms: List[Tuple[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_terms': self.total_terms,
            'programming_terms': self.programming_terms,
            'project_specific_terms': self.project_specific_terms,
            'programming_ratio': self.programming_ratio,
            'top_programming_terms': self.top_programming_terms,
            'top_project_terms': self.top_project_terms,
        }


@dataclass
class TransferMetrics:
    """Metrics measuring transfer effectiveness."""

    vocabulary_overlap: float  # % of terms shared between projects
    ngram_coverage: float  # % of n-grams that apply to new project
    perplexity_improvement: float  # Reduction in perplexity from transfer
    adapted_terms: int  # Number of terms successfully adapted

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'vocabulary_overlap': self.vocabulary_overlap,
            'ngram_coverage': self.ngram_coverage,
            'perplexity_improvement': self.perplexity_improvement,
            'adapted_terms': self.adapted_terms,
        }


class VocabularyAnalyzer:
    """
    Analyze vocabulary composition for transfer learning.

    Separates programming language constructs (transferable) from
    project-specific terms (domain-specific).

    Example:
        >>> analyzer = VocabularyAnalyzer()
        >>> analysis = analyzer.analyze(ngram_model)
        >>> print(f"Programming terms: {analysis.programming_ratio:.1%}")
    """

    def __init__(
        self,
        programming_vocab: Optional[Set[str]] = None,
        min_frequency: int = 2
    ):
        """
        Initialize analyzer.

        Args:
            programming_vocab: Set of known programming terms.
                              Defaults to PROGRAMMING_VOCABULARY.
            min_frequency: Minimum term frequency to consider.
        """
        self.programming_vocab = programming_vocab or PROGRAMMING_VOCABULARY
        self.min_frequency = min_frequency

    def analyze(self, ngram_model: NGramModel) -> VocabularyAnalysis:
        """
        Analyze vocabulary composition.

        Args:
            ngram_model: Trained n-gram model to analyze.

        Returns:
            VocabularyAnalysis with breakdown of term types.
        """
        # Exclude special tokens
        special_tokens = {'<s>', '</s>', '<unk>'}
        vocab = {t for t in ngram_model.vocab if t not in special_tokens}

        # Count term frequencies from n-gram counts
        term_counts = Counter()
        for context, next_tokens in ngram_model.counts.items():
            for token, count in next_tokens.items():
                if token not in special_tokens:
                    term_counts[token] += count

        # If no counts, use vocabulary with frequency 1
        if not term_counts:
            for term in vocab:
                term_counts[term] = 1

        # Separate programming vs project-specific
        programming_counts = Counter()
        project_counts = Counter()

        for term, count in term_counts.items():
            if count < self.min_frequency:
                continue
            if term.lower() in self.programming_vocab:
                programming_counts[term] = count
            else:
                project_counts[term] = count

        total = len(programming_counts) + len(project_counts)
        programming_ratio = len(programming_counts) / max(total, 1)

        return VocabularyAnalysis(
            total_terms=total,
            programming_terms=len(programming_counts),
            project_specific_terms=len(project_counts),
            programming_ratio=programming_ratio,
            top_programming_terms=programming_counts.most_common(20),
            top_project_terms=project_counts.most_common(20),
        )

    def get_transferable_terms(
        self,
        ngram_model: NGramModel,
        min_frequency: int = 3
    ) -> Set[str]:
        """
        Get terms that are good candidates for transfer.

        Returns programming terms that appear frequently enough
        to have learned meaningful patterns.

        Args:
            ngram_model: Trained n-gram model.
            min_frequency: Minimum frequency for transfer.

        Returns:
            Set of transferable terms.
        """
        special_tokens = {'<s>', '</s>', '<unk>'}
        transferable = set()

        # Get term frequencies from n-gram counts
        term_counts = Counter()
        for context, next_tokens in ngram_model.counts.items():
            for token, count in next_tokens.items():
                if token not in special_tokens:
                    term_counts[token] += count

        for term, count in term_counts.items():
            if count >= min_frequency:
                if term.lower() in self.programming_vocab:
                    transferable.add(term)

        return transferable

    def calculate_overlap(
        self,
        model_a: NGramModel,
        model_b: NGramModel
    ) -> float:
        """
        Calculate vocabulary overlap between two models.

        Excludes special tokens like <s> and </s>.

        Args:
            model_a: First n-gram model.
            model_b: Second n-gram model.

        Returns:
            Jaccard similarity (0.0 to 1.0).
        """
        special_tokens = {'<s>', '</s>', '<unk>'}
        vocab_a = {t for t in model_a.vocab if t not in special_tokens}
        vocab_b = {t for t in model_b.vocab if t not in special_tokens}

        intersection = len(vocab_a & vocab_b)
        union = len(vocab_a | vocab_b)

        return intersection / max(union, 1)


@dataclass
class PortableModel:
    """
    A portable model that can be transferred between projects.

    Contains:
    - Shared n-gram patterns (programming constructs)
    - Vocabulary metadata
    - Transfer weights
    """

    ngram_order: int
    shared_counts: Dict[Tuple[str, ...], Dict[str, int]] = field(default_factory=dict)
    shared_vocab: Set[str] = field(default_factory=set)
    source_project: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ngram_model(
        cls,
        ngram_model: NGramModel,
        analyzer: Optional[VocabularyAnalyzer] = None,
        source_project: str = ""
    ) -> 'PortableModel':
        """
        Extract portable model from a trained n-gram model.

        Args:
            ngram_model: Trained model to extract from.
            analyzer: Vocabulary analyzer (uses default if None).
            source_project: Name of source project.

        Returns:
            PortableModel containing transferable patterns.
        """
        if analyzer is None:
            analyzer = VocabularyAnalyzer()

        # Get transferable terms
        transferable = analyzer.get_transferable_terms(ngram_model)

        # Extract only n-grams involving transferable terms
        shared_counts = {}
        for context, next_tokens in ngram_model.counts.items():
            # Check if context involves transferable terms
            context_transferable = all(
                t.lower() in analyzer.programming_vocab or t in transferable
                for t in context
            )

            if context_transferable:
                # Filter next tokens to transferable ones
                filtered_next = {
                    t: c for t, c in next_tokens.items()
                    if t.lower() in analyzer.programming_vocab or t in transferable
                }

                if filtered_next:
                    shared_counts[context] = filtered_next

        # Get analysis for metadata
        analysis = analyzer.analyze(ngram_model)

        return cls(
            ngram_order=ngram_model.n,
            shared_counts=shared_counts,
            shared_vocab=transferable,
            source_project=source_project,
            metadata={
                'total_contexts': len(shared_counts),
                'vocab_size': len(transferable),
                'analysis': analysis.to_dict(),
            }
        )

    def save(self, path: str) -> None:
        """
        Save portable model to disk.

        Args:
            path: Directory path to save to.
        """
        os.makedirs(path, exist_ok=True)

        # Convert tuple keys to strings for JSON
        counts_serializable = {
            '|'.join(k): v
            for k, v in self.shared_counts.items()
        }

        data = {
            'ngram_order': self.ngram_order,
            'shared_counts': counts_serializable,
            'shared_vocab': list(self.shared_vocab),
            'source_project': self.source_project,
            'metadata': self.metadata,
        }

        with open(os.path.join(path, 'portable_model.json'), 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PortableModel':
        """
        Load portable model from disk.

        Args:
            path: Directory path to load from.

        Returns:
            Loaded PortableModel.
        """
        with open(os.path.join(path, 'portable_model.json'), 'r') as f:
            data = json.load(f)

        # Convert string keys back to tuples
        shared_counts = {
            tuple(k.split('|')) if k else (): v
            for k, v in data['shared_counts'].items()
        }

        return cls(
            ngram_order=data['ngram_order'],
            shared_counts=shared_counts,
            shared_vocab=set(data['shared_vocab']),
            source_project=data['source_project'],
            metadata=data['metadata'],
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the portable model."""
        return {
            'ngram_order': self.ngram_order,
            'context_count': len(self.shared_counts),
            'vocab_size': len(self.shared_vocab),
            'source_project': self.source_project,
            **self.metadata,
        }


class TransferAdapter:
    """
    Adapt a portable model to a new project.

    Combines transferred knowledge with project-specific training
    to bootstrap learning on a new codebase.

    Example:
        >>> adapter = TransferAdapter(portable_model)
        >>> adapted = adapter.adapt(new_project_model)
        >>> metrics = adapter.measure_effectiveness(new_project_model)
    """

    def __init__(
        self,
        portable_model: PortableModel,
        blend_weight: float = 0.3
    ):
        """
        Initialize adapter.

        Args:
            portable_model: Pre-trained portable model to transfer.
            blend_weight: How much to weight transferred knowledge (0-1).
                         0.3 means 30% transfer, 70% local.
        """
        self.portable = portable_model
        self.blend_weight = blend_weight
        self._metrics: Optional[TransferMetrics] = None

    def adapt(
        self,
        target_model: NGramModel,
        in_place: bool = False
    ) -> NGramModel:
        """
        Adapt transferred knowledge to target model.

        Blends the portable model's n-gram counts with the target
        model's counts, weighted by blend_weight.

        Args:
            target_model: Model to adapt to.
            in_place: If True, modify target_model directly.

        Returns:
            Adapted model (either target_model or new copy).
        """
        if not in_place:
            # Create a copy by re-training on same data
            adapted = NGramModel(n=target_model.n)
            # Copy vocabulary and counts
            adapted.vocab = target_model.vocab.copy()
            adapted.counts = {
                k: dict(v) for k, v in target_model.counts.items()
            }
            adapted.total_tokens = target_model.total_tokens
            adapted.total_documents = target_model.total_documents
        else:
            adapted = target_model

        # Merge transferred counts
        adapted_count = 0
        for context, next_tokens in self.portable.shared_counts.items():
            # Adjust context length if needed
            if len(context) > adapted.n - 1:
                context = context[-(adapted.n - 1):]

            if context not in adapted.counts:
                adapted.counts[context] = {}

            for token, count in next_tokens.items():
                # Blend: existing + transfer * weight
                transfer_count = int(count * self.blend_weight)
                if transfer_count > 0:
                    existing = adapted.counts[context].get(token, 0)
                    adapted.counts[context][token] = existing + transfer_count
                    adapted.total_tokens += transfer_count
                    adapted_count += 1

                    # Add to vocabulary if new
                    adapted.vocab.add(token)

        # Store metrics
        self._adapted_count = adapted_count

        return adapted

    def measure_effectiveness(
        self,
        target_model: NGramModel,
        test_texts: Optional[List[str]] = None
    ) -> TransferMetrics:
        """
        Measure how effective the transfer was.

        Args:
            target_model: The target model (before adaptation).
            test_texts: Optional texts to measure perplexity on.

        Returns:
            TransferMetrics with effectiveness measures.
        """
        # Calculate vocabulary overlap
        target_vocab = set(target_model.vocab)
        portable_vocab = self.portable.shared_vocab

        intersection = len(target_vocab & portable_vocab)
        union = len(target_vocab | portable_vocab)
        vocab_overlap = intersection / max(union, 1)

        # Calculate n-gram coverage
        target_contexts = set(target_model.counts.keys())
        portable_contexts = set(self.portable.shared_counts.keys())

        # Adjust for context length differences
        adjusted_portable = set()
        for ctx in portable_contexts:
            if len(ctx) <= target_model.n - 1:
                adjusted_portable.add(ctx)
            else:
                adjusted_portable.add(ctx[-(target_model.n - 1):])

        matching_contexts = len(target_contexts & adjusted_portable)
        ngram_coverage = matching_contexts / max(len(portable_contexts), 1)

        # Measure perplexity improvement if test texts provided
        perplexity_improvement = 0.0
        if test_texts:
            # Before adaptation
            before_perplexity = sum(
                target_model.perplexity(t) for t in test_texts
            ) / len(test_texts)

            # After adaptation
            adapted = self.adapt(target_model, in_place=False)
            after_perplexity = sum(
                adapted.perplexity(t) for t in test_texts
            ) / len(test_texts)

            # Improvement as percentage reduction
            if before_perplexity > 0:
                perplexity_improvement = (
                    (before_perplexity - after_perplexity) / before_perplexity
                )

        metrics = TransferMetrics(
            vocabulary_overlap=vocab_overlap,
            ngram_coverage=ngram_coverage,
            perplexity_improvement=perplexity_improvement,
            adapted_terms=getattr(self, '_adapted_count', 0),
        )

        self._metrics = metrics
        return metrics

    def get_metrics(self) -> Optional[TransferMetrics]:
        """Get the most recent transfer metrics."""
        return self._metrics

    def get_transfer_summary(self) -> str:
        """Get human-readable transfer summary."""
        if not self._metrics:
            return "No transfer metrics available. Call measure_effectiveness() first."

        m = self._metrics
        lines = [
            "Transfer Summary",
            "=" * 40,
            f"Source Project: {self.portable.source_project or 'Unknown'}",
            f"Blend Weight: {self.blend_weight:.1%}",
            "",
            "Metrics:",
            f"  Vocabulary Overlap: {m.vocabulary_overlap:.1%}",
            f"  N-gram Coverage: {m.ngram_coverage:.1%}",
            f"  Perplexity Improvement: {m.perplexity_improvement:.1%}",
            f"  Adapted Terms: {m.adapted_terms}",
            "",
        ]

        # Assessment
        if m.vocabulary_overlap > 0.3 and m.ngram_coverage > 0.2:
            lines.append("Assessment: Good transfer potential")
        elif m.vocabulary_overlap > 0.1:
            lines.append("Assessment: Moderate transfer potential")
        else:
            lines.append("Assessment: Limited transfer potential - domains too different")

        return "\n".join(lines)


# Convenience functions

def create_portable_model(
    ngram_model: NGramModel,
    source_project: str = ""
) -> PortableModel:
    """
    Create a portable model from a trained n-gram model.

    Convenience function that uses default analyzer settings.

    Args:
        ngram_model: Trained model to export.
        source_project: Name of the source project.

    Returns:
        PortableModel ready for transfer.
    """
    return PortableModel.from_ngram_model(
        ngram_model,
        source_project=source_project
    )


def transfer_knowledge(
    source_model: NGramModel,
    target_model: NGramModel,
    blend_weight: float = 0.3,
    source_project: str = ""
) -> Tuple[NGramModel, TransferMetrics]:
    """
    Transfer knowledge from source to target model.

    Convenience function that handles the full transfer pipeline.

    Args:
        source_model: Model to transfer from.
        target_model: Model to transfer to.
        blend_weight: How much to weight transferred knowledge.
        source_project: Name of source project for tracking.

    Returns:
        Tuple of (adapted_model, transfer_metrics).
    """
    # Create portable model
    portable = create_portable_model(source_model, source_project)

    # Adapt to target
    adapter = TransferAdapter(portable, blend_weight=blend_weight)
    adapted = adapter.adapt(target_model, in_place=False)

    # Measure effectiveness
    metrics = adapter.measure_effectiveness(target_model)

    return adapted, metrics
