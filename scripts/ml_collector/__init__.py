"""
ML Collector Package

A modular ML data collection system split from the monolithic ml_data_collector.py.
"""

# Re-export key functions and classes for backward compatibility
from .core import ML_COLLECTION_ENABLED, GitCommandError, SchemaValidationError
from .config import (
    ML_DATA_DIR, COMMITS_DIR, CHATS_DIR, SESSIONS_DIR, ACTIONS_DIR,
    MILESTONES, validate_schema, redact_sensitive_data
)
from .data_classes import DiffHunk, CommitContext, ChatEntry, ActionEntry, TranscriptExchange
from .persistence import (
    save_commit_data, save_commit_lite, save_session_lite,
    save_chat_entry, log_chat, save_action, log_action,
    generate_chat_id, generate_session_id
)
from .session import (
    get_current_session, start_session, end_session,
    add_chat_to_session, find_chat_file, add_chat_feedback,
    list_chats_needing_feedback, generate_session_handoff
)
from .commit import (
    collect_commit_data, find_commit_file,
    update_commit_ci_result, mark_commit_reverted
)
from .transcript import parse_transcript_jsonl, process_transcript
from .orchestration import (
    extract_orchestration_from_directory, save_orchestration, save_orchestration_lite,
    extract_and_save, print_orchestration_summary,
    SubAgentExecution, OrchestrationBatch, ExtractedOrchestration,
    find_agent_transcripts, parse_agent_transcript, detect_batches,
    ORCHESTRATION_DIR, ORCHESTRATION_LITE_FILE
)
from .export import export_data
from .stats import count_data, print_stats, estimate_project_size
from .quality import analyze_data_quality, print_quality_report
from .ci import ci_autocapture
from .hooks import install_hooks

__all__ = [
    # Core
    'ML_COLLECTION_ENABLED', 'GitCommandError', 'SchemaValidationError',
    # Config
    'ML_DATA_DIR', 'COMMITS_DIR', 'CHATS_DIR', 'SESSIONS_DIR', 'ACTIONS_DIR',
    'MILESTONES', 'validate_schema', 'redact_sensitive_data',
    # Data classes
    'DiffHunk', 'CommitContext', 'ChatEntry', 'ActionEntry', 'TranscriptExchange',
    # Persistence
    'save_commit_data', 'save_commit_lite', 'save_session_lite',
    'save_chat_entry', 'log_chat', 'save_action', 'log_action',
    'generate_chat_id', 'generate_session_id',
    # Session
    'get_current_session', 'start_session', 'end_session',
    'add_chat_to_session', 'find_chat_file', 'add_chat_feedback',
    'list_chats_needing_feedback', 'generate_session_handoff',
    # Commit
    'collect_commit_data', 'find_commit_file',
    'update_commit_ci_result', 'mark_commit_reverted',
    # Transcript
    'parse_transcript_jsonl', 'process_transcript',
    # Orchestration
    'extract_orchestration_from_directory', 'save_orchestration', 'save_orchestration_lite',
    'extract_and_save', 'print_orchestration_summary',
    'SubAgentExecution', 'OrchestrationBatch', 'ExtractedOrchestration',
    'find_agent_transcripts', 'parse_agent_transcript', 'detect_batches',
    'ORCHESTRATION_DIR', 'ORCHESTRATION_LITE_FILE',
    # Export
    'export_data',
    # Stats
    'count_data', 'print_stats', 'estimate_project_size',
    # Quality
    'analyze_data_quality', 'print_quality_report',
    # CI
    'ci_autocapture',
    # Hooks
    'install_hooks',
]
