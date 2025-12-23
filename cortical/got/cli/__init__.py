"""
GoT CLI modules.

This package contains CLI command handlers that can be integrated
into the main got_utils.py CLI or used standalone.
"""

from .doc import (
    setup_doc_parser,
    handle_doc_command,
    # Individual command handlers
    cmd_doc_scan,
    cmd_doc_list,
    cmd_doc_show,
    cmd_doc_link,
    cmd_doc_tasks,
    cmd_doc_docs,
)

__all__ = [
    "setup_doc_parser",
    "handle_doc_command",
    "cmd_doc_scan",
    "cmd_doc_list",
    "cmd_doc_show",
    "cmd_doc_link",
    "cmd_doc_tasks",
    "cmd_doc_docs",
]
