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
from .minicolumn import Minicolumn, Edge
from .layers import CorticalLayer, HierarchicalLayer
# Import from processor package (split from monolithic processor.py)
from .processor import CorticalTextProcessor
from .config import CorticalConfig, get_default_config, VALID_RELATION_CHAINS
from .fluent import FluentProcessor
from .progress import (
    ProgressReporter,
    ConsoleProgressReporter,
    CallbackProgressReporter,
    SilentProgressReporter,
    MultiPhaseProgress,
)
from .results import (
    DocumentMatch,
    PassageMatch,
    QueryResult,
    convert_document_matches,
    convert_passage_matches
)
from .diff import (
    SemanticDiff,
    TermChange,
    RelationChange,
    ClusterChange,
    compare_processors,
    compare_documents,
    what_changed
)
# Pickle support removed - JSON-only persistence now

# MCP Server support (optional import)
try:
    from .mcp_server import CorticalMCPServer, create_mcp_server
    _has_mcp = True
except ImportError:
    _has_mcp = False
    CorticalMCPServer = None
    create_mcp_server = None

__version__ = "2.0.0"
__all__ = [
    "CorticalTextProcessor",
    "FluentProcessor",
    "CorticalConfig",
    "CorticalLayer",
    "HierarchicalLayer",
    "Minicolumn",
    "Edge",
    "Tokenizer",
    "get_default_config",
    "VALID_RELATION_CHAINS",
    "ProgressReporter",
    "ConsoleProgressReporter",
    "CallbackProgressReporter",
    "SilentProgressReporter",
    "MultiPhaseProgress",
    "DocumentMatch",
    "PassageMatch",
    "QueryResult",
    "convert_document_matches",
    "convert_passage_matches",
    # Semantic diff
    "SemanticDiff",
    "TermChange",
    "RelationChange",
    "ClusterChange",
    "compare_processors",
    "compare_documents",
    "what_changed",
]

# Add MCP exports if available
if _has_mcp:
    __all__.extend([
        "CorticalMCPServer",
        "create_mcp_server",
    ])
