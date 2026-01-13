"""
Query Executors for Unified Query Pipeline (Phase 2).

Each executor wraps a backend system and provides:
- execute(query) -> ExecutionResult
- format_result(result) -> str

Executors:
- AuditExecutor: PLN-based audit reasoning
- SemanticExecutor: Document retrieval via TF-IDF
- CodeExecutor: Code structure queries via CodeBridge
- CDGExecutor: CDG graph queries (placeholder)
"""

from .protocol import (
    QueryExecutorProtocol,
    ExecutionResult,
)
from .audit_executor import AuditExecutor
from .semantic_executor import SemanticExecutor
from .code_executor import CodeExecutor

__all__ = [
    "QueryExecutorProtocol",
    "ExecutionResult",
    "AuditExecutor",
    "SemanticExecutor",
    "CodeExecutor",
]
