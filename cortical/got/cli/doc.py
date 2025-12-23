"""
Document registry and linking CLI commands for GoT system.

Provides commands for:
- Scanning and registering documentation files
- Linking documents to tasks
- Detecting stale documents
- Querying document-task relationships

This module can be integrated into got_utils.py CLI or used standalone.
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from cortical.got.api import GoTManager
from cortical.got.types import Document
from cortical.utils.id_generation import generate_document_id


# =============================================================================
# CONSTANTS
# =============================================================================

# Document type detection patterns
DOC_TYPE_PATTERNS = {
    "architecture": [
        r"architecture",
        r"design",
        r"system",
        r"overview",
    ],
    "api": [
        r"api",
        r"reference",
        r"interface",
    ],
    "guide": [
        r"guide",
        r"how-?to",
        r"tutorial",
        r"quickstart",
    ],
    "memory": [
        r"memories?",
        r"session",
        r"knowledge-transfer",
    ],
    "decision": [
        r"adr",
        r"decision",
        r"rationale",
    ],
    "research": [
        r"research",
        r"investigation",
        r"analysis",
        r"forensic",
    ],
    "knowledge-transfer": [
        r"knowledge-transfer",
        r"handoff",
        r"continuation",
    ],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_project_root() -> Path:
    """Get project root directory."""
    # Walk up from this file to find project root
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / ".got").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def detect_doc_type(path: str, title: str = "") -> str:
    """
    Detect document type from path and title.

    Args:
        path: File path
        title: Document title (from content)

    Returns:
        Detected doc_type string
    """
    combined = f"{path.lower()} {title.lower()}"

    for doc_type, patterns in DOC_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return doc_type

    return "general"


def extract_title_from_content(content: str) -> str:
    """
    Extract title from markdown content.

    Args:
        content: File content

    Returns:
        Extracted title or empty string
    """
    # Look for first heading
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line.startswith("## "):
            return line[3:].strip()

    return ""


def extract_tags_from_content(content: str) -> List[str]:
    """
    Extract tags from markdown content.

    Args:
        content: File content

    Returns:
        List of tags found
    """
    tags = []

    # Look for Tags: or **Tags:** line
    match = re.search(r"\*?\*?Tags\*?\*?:\s*(.+)", content, re.IGNORECASE)
    if match:
        tag_str = match.group(1)
        # Extract backtick-quoted tags or comma-separated
        if "`" in tag_str:
            tags = re.findall(r"`([^`]+)`", tag_str)
        else:
            tags = [t.strip() for t in tag_str.split(",")]

    return tags


def get_file_mtime(path: Path) -> str:
    """Get file modification time as ISO timestamp."""
    mtime = os.path.getmtime(path)
    return datetime.fromtimestamp(mtime, timezone.utc).isoformat()


def scan_documents(
    got_dir: Path,
    doc_dirs: List[str],
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Scan directories and register documents in GoT.

    Args:
        got_dir: Path to .got directory
        doc_dirs: List of directories to scan
        dry_run: If True, don't actually create entities
        verbose: If True, print detailed output

    Returns:
        Dictionary with scan results
    """
    manager = GoTManager(got_dir)
    project_root = _get_project_root()

    results = {
        "scanned": 0,
        "registered": 0,
        "updated": 0,
        "skipped": 0,
        "errors": [],
    }

    for doc_dir in doc_dirs:
        doc_path = project_root / doc_dir
        if not doc_path.exists():
            if verbose:
                print(f"Skipping non-existent directory: {doc_dir}")
            continue

        for md_file in doc_path.rglob("*.md"):
            results["scanned"] += 1
            rel_path = str(md_file.relative_to(project_root))

            try:
                # Read content
                content = md_file.read_text(encoding="utf-8")

                # Extract metadata
                title = extract_title_from_content(content)
                tags = extract_tags_from_content(content)
                doc_type = detect_doc_type(rel_path, title)
                file_mtime = get_file_mtime(md_file)

                # Check if document already exists
                doc_id = generate_document_id(rel_path)
                existing = manager.get_document(doc_id)

                if existing:
                    # Check if content changed
                    new_hash = existing.compute_content_hash(content)
                    if existing.content_hash == new_hash:
                        results["skipped"] += 1
                        if verbose:
                            print(f"  ⏭️  {rel_path} (unchanged)")
                        continue

                    # Update existing document
                    if not dry_run:
                        existing.update_from_file(content, file_mtime)
                        manager.update_document(doc_id, **{
                            "content_hash": existing.content_hash,
                            "line_count": existing.line_count,
                            "word_count": existing.word_count,
                            "last_file_modified": existing.last_file_modified,
                            "last_verified": existing.last_verified,
                            "is_stale": False,
                        })
                    results["updated"] += 1
                    if verbose:
                        print(f"  🔄 {rel_path} (updated)")
                else:
                    # Create new document
                    if not dry_run:
                        doc = manager.create_document(
                            path=rel_path,
                            title=title or md_file.stem,
                            doc_type=doc_type,
                            tags=tags,
                            category=doc_dir,
                        )
                        # Update with content info
                        doc.update_from_file(content, file_mtime)
                        manager.update_document(doc.id, **{
                            "content_hash": doc.content_hash,
                            "line_count": doc.line_count,
                            "word_count": doc.word_count,
                            "last_file_modified": doc.last_file_modified,
                            "last_verified": doc.last_verified,
                        })
                    results["registered"] += 1
                    if verbose:
                        print(f"  ✅ {rel_path} ({doc_type})")

            except Exception as e:
                results["errors"].append(f"{rel_path}: {e}")
                if verbose:
                    print(f"  ❌ {rel_path}: {e}")

    return results


def list_documents(
    got_dir: Path,
    doc_type: Optional[str] = None,
    stale_only: bool = False,
    tag: Optional[str] = None,
) -> List[Document]:
    """
    List registered documents.

    Args:
        got_dir: Path to .got directory
        doc_type: Filter by document type
        stale_only: Only show stale documents
        tag: Filter by tag

    Returns:
        List of Document objects
    """
    manager = GoTManager(got_dir)
    return manager.list_documents(
        doc_type=doc_type,
        tag=tag,
        is_stale=True if stale_only else None,
    )


def show_document(got_dir: Path, doc_id: str) -> Optional[Document]:
    """
    Get document details.

    Args:
        got_dir: Path to .got directory
        doc_id: Document ID or path

    Returns:
        Document object or None
    """
    manager = GoTManager(got_dir)

    # Try as direct ID first
    doc = manager.get_document(doc_id)
    if doc:
        return doc

    # Try as path
    doc = manager.get_document_by_path(doc_id)
    return doc


def link_document_to_task(
    got_dir: Path,
    doc_id: str,
    task_id: str,
    edge_type: str = "DOCUMENTED_BY",
) -> bool:
    """
    Link a document to a task.

    Args:
        got_dir: Path to .got directory
        doc_id: Document ID or path
        task_id: Task ID
        edge_type: Edge type (DOCUMENTED_BY, PRODUCES, REFERENCES)

    Returns:
        True if link created successfully
    """
    manager = GoTManager(got_dir)

    # Resolve doc_id if it's a path
    doc = manager.get_document(doc_id)
    if not doc:
        doc = manager.get_document_by_path(doc_id)

    if not doc:
        print(f"Document not found: {doc_id}")
        return False

    # Verify task exists
    task = manager.get_task(task_id)
    if not task:
        print(f"Task not found: {task_id}")
        return False

    # Create link
    manager.link_document_to_task(doc.id, task_id, edge_type)
    return True


def get_tasks_for_document(got_dir: Path, doc_id: str) -> List:
    """Get all tasks linked to a document."""
    manager = GoTManager(got_dir)

    # Resolve doc_id if it's a path
    doc = manager.get_document(doc_id)
    if not doc:
        doc = manager.get_document_by_path(doc_id)

    if not doc:
        return []

    return manager.get_tasks_for_document(doc.id)


def get_documents_for_task(got_dir: Path, task_id: str) -> List[Document]:
    """Get all documents linked to a task."""
    manager = GoTManager(got_dir)
    return manager.get_documents_for_task(task_id)


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_doc_scan(args, manager: GoTManager) -> int:
    """Handle 'got doc scan' command."""
    results = scan_documents(
        manager.got_dir,
        args.dirs,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Scan Results:")
    print(f"  Scanned:    {results['scanned']}")
    print(f"  Registered: {results['registered']}")
    print(f"  Updated:    {results['updated']}")
    print(f"  Skipped:    {results['skipped']}")
    if results["errors"]:
        print(f"  Errors:     {len(results['errors'])}")
        for err in results["errors"][:5]:
            print(f"    - {err}")

    return 0


def cmd_doc_list(args, manager: GoTManager) -> int:
    """Handle 'got doc list' command."""
    docs = list_documents(
        manager.got_dir,
        doc_type=getattr(args, 'doc_type', None),
        stale_only=getattr(args, 'stale', False),
        tag=getattr(args, 'tag', None),
    )

    if not docs:
        print("No documents found.")
        return 0

    print(f"Documents ({len(docs)}):\n")
    for doc in sorted(docs, key=lambda d: d.path):
        stale_mark = " [STALE]" if doc.is_stale else ""
        print(f"  {doc.id}")
        print(f"    Path: {doc.path}")
        print(f"    Title: {doc.title}")
        print(f"    Type: {doc.doc_type}{stale_mark}")
        if doc.tags:
            print(f"    Tags: {', '.join(doc.tags)}")
        print()

    return 0


def cmd_doc_show(args, manager: GoTManager) -> int:
    """Handle 'got doc show' command."""
    doc = show_document(manager.got_dir, args.doc_id)

    if not doc:
        print(f"Document not found: {args.doc_id}")
        return 1

    print(f"Document: {doc.id}")
    print(f"  Path:          {doc.path}")
    print(f"  Title:         {doc.title}")
    print(f"  Type:          {doc.doc_type}")
    print(f"  Category:      {doc.category}")
    print(f"  Tags:          {', '.join(doc.tags) if doc.tags else '(none)'}")
    print(f"  Lines:         {doc.line_count}")
    print(f"  Words:         {doc.word_count}")
    print(f"  Content Hash:  {doc.content_hash}")
    print(f"  Last Modified: {doc.last_file_modified}")
    print(f"  Last Verified: {doc.last_verified}")
    print(f"  Is Stale:      {doc.is_stale}")
    print(f"  Version:       {doc.version}")

    return 0


def cmd_doc_link(args, manager: GoTManager) -> int:
    """Handle 'got doc link' command."""
    success = link_document_to_task(
        manager.got_dir,
        args.doc_id,
        args.task_id,
        getattr(args, 'edge_type', 'DOCUMENTED_BY'),
    )

    if success:
        print(f"Linked: {args.doc_id} --{args.edge_type}--> {args.task_id}")
        return 0

    return 1


def cmd_doc_tasks(args, manager: GoTManager) -> int:
    """Handle 'got doc tasks' command."""
    tasks = get_tasks_for_document(manager.got_dir, args.doc_id)

    if not tasks:
        print(f"No tasks linked to document: {args.doc_id}")
        return 0

    print(f"Tasks linked to {args.doc_id}:\n")
    for task in tasks:
        print(f"  {task.id}")
        print(f"    Title:    {task.title}")
        print(f"    Status:   {task.status}")
        print(f"    Priority: {task.priority}")
        print()

    return 0


def cmd_doc_docs(args, manager: GoTManager) -> int:
    """Handle 'got doc docs' command."""
    docs = get_documents_for_task(manager.got_dir, args.task_id)

    if not docs:
        print(f"No documents linked to task: {args.task_id}")
        return 0

    print(f"Documents linked to {args.task_id}:\n")
    for doc in docs:
        print(f"  {doc.id}")
        print(f"    Path:  {doc.path}")
        print(f"    Title: {doc.title}")
        print()

    return 0


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_doc_parser(subparsers) -> None:
    """
    Set up argparse subparsers for doc commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Create doc subparser
    doc_parser = subparsers.add_parser("doc", help="Document management commands")
    doc_subparsers = doc_parser.add_subparsers(
        dest="doc_command",
        help="Document subcommands"
    )

    # scan command
    scan_parser = doc_subparsers.add_parser("scan", help="Scan and register documents")
    scan_parser.add_argument(
        "--dirs",
        nargs="+",
        default=["docs", "samples/memories"],
        help="Directories to scan (default: docs, samples/memories)",
    )
    scan_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    scan_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # list command
    list_parser = doc_subparsers.add_parser("list", help="List registered documents")
    list_parser.add_argument(
        "--type",
        dest="doc_type",
        help="Filter by document type",
    )
    list_parser.add_argument(
        "--tag",
        help="Filter by tag",
    )
    list_parser.add_argument(
        "--stale",
        action="store_true",
        help="Only show stale documents",
    )

    # show command
    show_parser = doc_subparsers.add_parser("show", help="Show document details")
    show_parser.add_argument("doc_id", help="Document ID or path")

    # link command
    link_parser = doc_subparsers.add_parser("link", help="Link document to task")
    link_parser.add_argument("doc_id", help="Document ID or path")
    link_parser.add_argument("task_id", help="Task ID")
    link_parser.add_argument(
        "--type",
        dest="edge_type",
        default="DOCUMENTED_BY",
        choices=["DOCUMENTED_BY", "PRODUCES", "REFERENCES"],
        help="Edge type (default: DOCUMENTED_BY)",
    )

    # tasks command
    tasks_parser = doc_subparsers.add_parser("tasks", help="Show tasks linked to document")
    tasks_parser.add_argument("doc_id", help="Document ID or path")

    # docs command
    docs_parser = doc_subparsers.add_parser("docs", help="Show documents linked to task")
    docs_parser.add_argument("task_id", help="Task ID")


def handle_doc_command(args, manager: GoTManager) -> int:
    """
    Route doc subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: GoTManager instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'doc_command') or args.doc_command is None:
        print("Error: No doc subcommand specified. Use 'got doc --help' for usage.")
        return 1

    command_handlers = {
        "scan": cmd_doc_scan,
        "list": cmd_doc_list,
        "show": cmd_doc_show,
        "link": cmd_doc_link,
        "tasks": cmd_doc_tasks,
        "docs": cmd_doc_docs,
    }

    handler = command_handlers.get(args.doc_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown doc subcommand: {args.doc_command}")
    return 1


# =============================================================================
# STANDALONE CLI
# =============================================================================

def main():
    """Main CLI entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Document registry and linking utilities for GoT"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Set up doc commands (without the 'doc' prefix for standalone use)
    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan and register documents")
    scan_parser.add_argument(
        "--dirs",
        nargs="+",
        default=["docs", "samples/memories"],
        help="Directories to scan (default: docs, samples/memories)",
    )
    scan_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    scan_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List registered documents")
    list_parser.add_argument(
        "--type",
        dest="doc_type",
        help="Filter by document type",
    )
    list_parser.add_argument(
        "--tag",
        help="Filter by tag",
    )
    list_parser.add_argument(
        "--stale",
        action="store_true",
        help="Only show stale documents",
    )

    # show command
    show_parser = subparsers.add_parser("show", help="Show document details")
    show_parser.add_argument("doc_id", help="Document ID or path")

    # link command
    link_parser = subparsers.add_parser("link", help="Link document to task")
    link_parser.add_argument("doc_id", help="Document ID or path")
    link_parser.add_argument("task_id", help="Task ID")
    link_parser.add_argument(
        "--type",
        dest="edge_type",
        default="DOCUMENTED_BY",
        choices=["DOCUMENTED_BY", "PRODUCES", "REFERENCES"],
        help="Edge type (default: DOCUMENTED_BY)",
    )

    # tasks command
    tasks_parser = subparsers.add_parser("tasks", help="Show tasks linked to document")
    tasks_parser.add_argument("doc_id", help="Document ID or path")

    # docs command
    docs_parser = subparsers.add_parser("docs", help="Show documents linked to task")
    docs_parser.add_argument("task_id", help="Task ID")

    args = parser.parse_args()

    # Get project root and GoT directory
    project_root = _get_project_root()
    got_dir = project_root / ".got"

    # Create manager
    manager = GoTManager(got_dir)

    # Create a pseudo doc_command attribute for standalone usage
    args.doc_command = args.command

    # Route to handler
    return handle_doc_command(args, manager)


if __name__ == "__main__":
    sys.exit(main())
