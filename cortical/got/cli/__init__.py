"""
GoT CLI modules.

This package contains CLI command handlers that can be integrated
into the main got_utils.py CLI or used standalone.

Modules:
- doc: Document registry and linking
- task: Task management (create, list, show, start, complete, etc.)
- sprint: Sprint and epic management
- handoff: Agent handoff coordination
- decision: Decision logging with rationale
- query: Graph queries, validation, statistics
- backup: Backup, restore, and sync
- shared: Common utilities and formatters
"""

# Document commands
from .doc import (
    setup_doc_parser,
    handle_doc_command,
    cmd_doc_scan,
    cmd_doc_list,
    cmd_doc_show,
    cmd_doc_link,
    cmd_doc_tasks,
    cmd_doc_docs,
)

# Task commands
from .task import (
    setup_task_parser,
    handle_task_command,
    cmd_task_create,
    cmd_task_list,
    cmd_task_show,
    cmd_task_next,
    cmd_task_start,
    cmd_task_complete,
    cmd_task_block,
    cmd_task_delete,
    cmd_task_depends,
)

# Sprint and epic commands
from .sprint import (
    setup_sprint_parser,
    setup_epic_parser,
    handle_sprint_command,
    handle_epic_command,
    cmd_sprint_create,
    cmd_sprint_list,
    cmd_sprint_status,
    cmd_sprint_start,
    cmd_sprint_complete,
    cmd_sprint_claim,
    cmd_sprint_release,
    cmd_sprint_goal_add,
    cmd_sprint_goal_list,
    cmd_sprint_goal_complete,
    cmd_sprint_link,
    cmd_sprint_unlink,
    cmd_sprint_tasks,
    cmd_sprint_suggest,
    cmd_epic_create,
    cmd_epic_list,
    cmd_epic_show,
)

# Handoff commands
from .handoff import (
    setup_handoff_parser,
    handle_handoff_command,
    cmd_handoff_initiate,
    cmd_handoff_accept,
    cmd_handoff_complete,
    cmd_handoff_list,
)

# Decision commands
from .decision import (
    setup_decision_parser,
    handle_decision_command,
    cmd_decision_log,
    cmd_decision_list,
    cmd_decision_why,
)

# Query and validation commands
from .query import (
    setup_query_parser,
    handle_query_commands,
    cmd_query,
    cmd_blocked,
    cmd_active,
    cmd_stats,
    cmd_dashboard,
    cmd_validate,
    cmd_infer,
    cmd_compact,
    cmd_export,
)

# Backup and sync commands
from .backup import (
    setup_backup_parser,
    handle_backup_command,
    handle_sync_migrate_commands,
    cmd_backup_create,
    cmd_backup_list,
    cmd_backup_verify,
    cmd_backup_restore,
    cmd_sync,
    cmd_migrate_events,
)

# Shared utilities
from .shared import (
    VALID_STATUSES,
    VALID_PRIORITIES,
    VALID_CATEGORIES,
    STATUS_PENDING,
    STATUS_IN_PROGRESS,
    STATUS_COMPLETED,
    STATUS_BLOCKED,
    STATUS_DEFERRED,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,
    PRIORITY_SCORES,
    format_task_table,
    format_sprint_status,
    format_task_details,
    truncate,
)

__all__ = [
    # Doc
    "setup_doc_parser",
    "handle_doc_command",
    "cmd_doc_scan",
    "cmd_doc_list",
    "cmd_doc_show",
    "cmd_doc_link",
    "cmd_doc_tasks",
    "cmd_doc_docs",
    # Task
    "setup_task_parser",
    "handle_task_command",
    "cmd_task_create",
    "cmd_task_list",
    "cmd_task_show",
    "cmd_task_next",
    "cmd_task_start",
    "cmd_task_complete",
    "cmd_task_block",
    "cmd_task_delete",
    "cmd_task_depends",
    # Sprint
    "setup_sprint_parser",
    "setup_epic_parser",
    "handle_sprint_command",
    "handle_epic_command",
    "cmd_sprint_create",
    "cmd_sprint_list",
    "cmd_sprint_status",
    "cmd_sprint_start",
    "cmd_sprint_complete",
    "cmd_sprint_claim",
    "cmd_sprint_release",
    "cmd_sprint_goal_add",
    "cmd_sprint_goal_list",
    "cmd_sprint_goal_complete",
    "cmd_sprint_link",
    "cmd_sprint_unlink",
    "cmd_sprint_tasks",
    "cmd_sprint_suggest",
    "cmd_epic_create",
    "cmd_epic_list",
    "cmd_epic_show",
    # Handoff
    "setup_handoff_parser",
    "handle_handoff_command",
    "cmd_handoff_initiate",
    "cmd_handoff_accept",
    "cmd_handoff_complete",
    "cmd_handoff_list",
    # Decision
    "setup_decision_parser",
    "handle_decision_command",
    "cmd_decision_log",
    "cmd_decision_list",
    "cmd_decision_why",
    # Query
    "setup_query_parser",
    "handle_query_commands",
    "cmd_query",
    "cmd_blocked",
    "cmd_active",
    "cmd_stats",
    "cmd_dashboard",
    "cmd_validate",
    "cmd_infer",
    "cmd_compact",
    "cmd_export",
    # Backup
    "setup_backup_parser",
    "handle_backup_command",
    "handle_sync_migrate_commands",
    "cmd_backup_create",
    "cmd_backup_list",
    "cmd_backup_verify",
    "cmd_backup_restore",
    "cmd_sync",
    "cmd_migrate_events",
    # Shared
    "VALID_STATUSES",
    "VALID_PRIORITIES",
    "VALID_CATEGORIES",
    "STATUS_PENDING",
    "STATUS_IN_PROGRESS",
    "STATUS_COMPLETED",
    "STATUS_BLOCKED",
    "STATUS_DEFERRED",
    "PRIORITY_CRITICAL",
    "PRIORITY_HIGH",
    "PRIORITY_MEDIUM",
    "PRIORITY_LOW",
    "PRIORITY_SCORES",
    "format_task_table",
    "format_sprint_status",
    "format_task_details",
    "truncate",
]
