"""
Synthetic Data Generators for Repository-Native SLM.

This module provides extractors and generators to create training data
from the repository's code, documentation, and metadata.

Design Principles:
- Batch processing to avoid timeouts
- Caching for incremental updates
- Resumable operations
- Progress reporting
"""

from .code_extractor import CodeExtractor, CodePattern
from .doc_extractor import DocExtractor, DocPattern
from .meta_extractor import MetaExtractor, MetaPattern
from .pattern_generator import PatternGenerator, TrainingPattern

__all__ = [
    'CodeExtractor', 'CodePattern',
    'DocExtractor', 'DocPattern',
    'MetaExtractor', 'MetaPattern',
    'PatternGenerator', 'TrainingPattern',
]
