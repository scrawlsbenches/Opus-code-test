"""
Cortical Text Processing Package
================================

A neocortex-inspired text processing system for semantic analysis,
document retrieval, and knowledge gap detection.

Example:
    from cortical import CorticalTextProcessor
    
    processor = CorticalTextProcessor()
    processor.process_document("doc1", "Neural networks process information...")
    processor.compute_all()
    results = processor.find_documents_for_query("neural processing")
"""

from .tokenizer import Tokenizer
from .minicolumn import Minicolumn
from .layers import CorticalLayer, HierarchicalLayer
from .processor import CorticalTextProcessor

__version__ = "2.0.0"
__all__ = [
    "CorticalTextProcessor",
    "CorticalLayer",
    "HierarchicalLayer",
    "Minicolumn",
    "Tokenizer",
]
