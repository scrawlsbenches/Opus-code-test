"""
Query Executors for Unified Query Pipeline (Phase 2).

Each executor wraps a backend system and provides:
- execute(query) -> ExecutionResult
- format_result(result) -> str

Executors:
- AuditExecutor: PLN-based audit reasoning
- SemanticExecutor: Document retrieval via cognitive graph
- CodeExecutor: Code structure queries via CodeBridge
- CDGExecutor: SQL-like CDG graph queries
"""

from .protocol import (
    QueryExecutorProtocol,
    ExecutionResult,
)
from .audit_executor import AuditExecutor
from .semantic_executor import SemanticExecutor
from .code_executor import CodeExecutor
from .cdg_executor import CDGExecutor

__all__ = [
    "QueryExecutorProtocol",
    "ExecutionResult",
    "AuditExecutor",
    "SemanticExecutor",
    "CodeExecutor",
    "CDGExecutor",
]
