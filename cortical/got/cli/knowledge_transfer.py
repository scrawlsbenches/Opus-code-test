"""
KnowledgeTransfer CLI commands for GoT system.

Provides commands for:
- Creating knowledge transfer documents
- Appending sections to existing documents
- Linking to related entities (handoffs, tasks, decisions)
- Listing and viewing knowledge transfer documents
- Importing from markdown files

This module can be integrated into got_utils.py CLI or used standalone.
"""

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from scripts.got_utils import TransactionalGoTAdapter


# =============================================================================
# MARKDOWN PARSING
# =============================================================================

def parse_markdown_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse a knowledge transfer markdown file.

    Extracts:
    - Title from first # heading
    - Session metadata (date, ID, branch) from **Key:** lines
    - Content sections from ## headings
    - Code references from inline code blocks (optional)

    Args:
        file_path: Path to markdown file

    Returns:
        Dictionary with extracted data:
        {
            "title": "Document Title",
            "session_id": "abc123",
            "session_date": "2025-12-29",
            "branch": "feature/xyz",
            "summary": "Executive summary content...",
            "sections": {
                "Section 1": "content...",
                "Section 2": "content...",
            },
            "code_refs": ["file.py:123", "other.py:456"],
        }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    # Initialize result
    result = {
        "title": "",
        "session_id": "",
        "session_date": "",
        "branch": "",
        "summary": "",
        "sections": {},
        "code_refs": [],
    }

    # Extract title (first # heading)
    for line in lines:
        if line.startswith('# ') and not result["title"]:
            result["title"] = line[2:].strip()
            break

    # Extract session metadata
    # Patterns: **Session Date:** 2025-12-29
    metadata_pattern = r'\*\*([^:]+):\*\*\s*(.+)'
    for line in lines[:30]:  # Check first 30 lines for metadata
        match = re.match(metadata_pattern, line)
        if match:
            key = match.group(1).strip().lower().replace(' ', '_')
            value = match.group(2).strip()

            if key == "session_date":
                result["session_date"] = value
            elif key == "session_id":
                result["session_id"] = value
            elif key == "branch":
                result["branch"] = value

    # Extract sections
    current_section = None
    current_content = []
    in_summary = False

    for line in lines:
        # Check for ## section headings
        if line.startswith('## '):
            # Save previous section
            if current_section:
                result["sections"][current_section] = '\n'.join(current_content).strip()
                current_content = []
            elif in_summary:
                # Save summary content before starting new section
                result["summary"] = '\n'.join(current_content).strip()
                current_content = []

            # Start new section
            section_heading = line[3:].strip()

            # Special handling for Executive Summary
            if section_heading.lower() in ['executive summary', 'summary']:
                in_summary = True
                current_section = None  # Don't treat as regular section
            else:
                in_summary = False
                current_section = section_heading

        elif current_section or in_summary:
            # Skip horizontal rules and empty headers
            if line.strip() in ['---', '===', '***']:
                continue

            # Add to current section or summary
            current_content.append(line)

    # Save last section
    if current_section and current_content:
        result["sections"][current_section] = '\n'.join(current_content).strip()
    elif in_summary and current_content:
        result["summary"] = '\n'.join(current_content).strip()

    # Extract code references (file:line patterns)
    # Look for patterns like: cortical/got/store.py:45
    code_ref_pattern = r'[a-zA-Z0-9_/.-]+\.[a-zA-Z]+:[0-9]+'
    for line in lines:
        matches = re.findall(code_ref_pattern, line)
        for match in matches:
            if match not in result["code_refs"]:
                result["code_refs"].append(match)

    # Generate title from filename if not found
    if not result["title"]:
        result["title"] = file_path.stem.replace('-', ' ').replace('_', ' ').title()

    return result


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_kt_create(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Handle 'got knowledge create' command.

    Creates a new knowledge transfer document.
    Content can be provided via --content flag or stdin.
    """
    title = args.title

    # Read content from stdin if '-' is specified
    content = getattr(args, 'content', '')
    if content == '-':
        content = sys.stdin.read().strip()

    # Parse sections if structured content provided
    sections = {}
    if getattr(args, 'sections', None):
        # Format: "Section1:content1" "Section2:content2"
        for section_spec in args.sections:
            if ':' in section_spec:
                heading, section_content = section_spec.split(':', 1)
                sections[heading.strip()] = section_content.strip()

    # Create entity via manager
    kt_id = manager.create_knowledge_transfer(
        title=title,
        session_id=getattr(args, 'session', ''),
        summary=getattr(args, 'summary', ''),
        sections=sections if sections else {},
        tags=getattr(args, 'tags', []),
    )

    print(f"Created: {kt_id}")
    print(f"  Title: {title}")
    if args.session:
        print(f"  Session: {args.session}")

    return 0


def cmd_kt_append(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Handle 'got knowledge append' command.

    Appends a section to an existing knowledge transfer document.
    Content can be provided as argument or via stdin.
    """
    kt_id = args.kt_id
    section_heading = args.section_heading

    # Read content from stdin if '-' is specified
    content = args.content
    if content == '-':
        content = sys.stdin.read().strip()

    # Append section
    success = manager.append_kt_section(kt_id, section_heading, content)

    if not success:
        print(f"Failed to append section to: {kt_id}")
        print("  Knowledge transfer document not found or update failed")
        return 1

    print(f"Appended section to: {kt_id}")
    print(f"  Section: {section_heading}")
    content_preview = content[:60] + "..." if len(content) > 60 else content
    print(f"  Content: {content_preview}")

    return 0


def cmd_kt_link(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Handle 'got knowledge link' command.

    Links a knowledge transfer document to related entities
    (handoffs, tasks, decisions).
    """
    kt_id = args.kt_id

    # Determine what we're linking
    if getattr(args, 'handoff', None):
        success = manager.link_kt_handoff(kt_id, args.handoff)
        entity_type = "handoff"
        entity_id = args.handoff
    elif getattr(args, 'task', None):
        success = manager.link_kt_task(kt_id, args.task)
        entity_type = "task"
        entity_id = args.task
    elif getattr(args, 'decision', None):
        success = manager.link_kt_decision(kt_id, args.decision)
        entity_type = "decision"
        entity_id = args.decision
    else:
        print("Error: Must specify --handoff, --task, or --decision")
        return 1

    if not success:
        print(f"Failed to link {entity_type}: {kt_id} -> {entity_id}")
        print("  Knowledge transfer document or target entity not found")
        return 1

    print(f"Linked {entity_type}: {kt_id} -> {entity_id}")

    return 0


def cmd_kt_list(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Handle 'got knowledge list' command.

    Lists knowledge transfer documents with optional filters.
    """
    # Get filters
    status = getattr(args, 'status', None)
    tags = getattr(args, 'tags', None)

    # List knowledge transfers
    kts = manager.list_knowledge_transfers(status=status, tags=tags)

    if not kts:
        print("No knowledge transfer documents found.")
        return 0

    # Apply limit if specified
    limit = getattr(args, 'limit', None)
    if limit is not None and limit > 0:
        kts = kts[:limit]

    print(f"Knowledge Transfer Documents ({len(kts)}):\n")

    for kt in kts:
        # Format output
        kt_id = kt.get("id", "?")
        title = kt.get("title", "Untitled")
        status = kt.get("status", "published")
        session_date = kt.get("session_date", "")
        tags = kt.get("tags", [])

        # Status icon
        status_icon = {
            "draft": "📝",
            "published": "✓",
            "archived": "📦",
        }.get(status, "?")

        print(f"  {status_icon} {kt_id}")
        print(f"      {title}")
        if session_date:
            print(f"      Date: {session_date}")
        if tags:
            print(f"      Tags: {', '.join(tags)}")
        print()

    return 0


def cmd_kt_show(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Handle 'got knowledge show' command.

    Displays detailed information about a knowledge transfer document.
    """
    kt_id = args.kt_id

    # Get the knowledge transfer
    kt = manager.get_knowledge_transfer(kt_id)

    if not kt:
        print(f"Knowledge transfer not found: {kt_id}")
        return 1

    # Display full details
    print("=" * 70)
    print(f"KNOWLEDGE TRANSFER: {kt_id}")
    print("=" * 70)
    print(f"Title:        {kt.get('title', 'Untitled')}")
    print(f"Status:       {kt.get('status', 'unknown')}")

    if kt.get('session_id'):
        print(f"Session ID:   {kt['session_id']}")
    if kt.get('session_date'):
        print(f"Session Date: {kt['session_date']}")

    # Tags
    if kt.get('tags'):
        print(f"Tags:         {', '.join(kt['tags'])}")

    # Summary
    if kt.get('summary'):
        print(f"\nSummary:")
        print("-" * 70)
        print(kt['summary'])
        print("-" * 70)

    # Sections
    sections = kt.get('sections', {})
    if sections:
        print(f"\nSections ({len(sections)}):")
        for heading, content in sections.items():
            print(f"\n## {heading}")
            print("-" * 70)
            # Truncate long sections in display
            if len(content) > 500:
                print(content[:500] + "\n... (truncated)")
            else:
                print(content)
            print("-" * 70)

    # Code references
    if kt.get('code_refs'):
        print(f"\nCode References ({len(kt['code_refs'])}):")
        for ref in kt['code_refs']:
            print(f"  - {ref}")

    # Related entities
    if kt.get('related_handoffs'):
        print(f"\nRelated Handoffs:")
        for h_id in kt['related_handoffs']:
            print(f"  - {h_id}")

    if kt.get('related_tasks'):
        print(f"\nRelated Tasks:")
        for t_id in kt['related_tasks']:
            print(f"  - {t_id}")

    if kt.get('related_decisions'):
        print(f"\nRelated Decisions:")
        for d_id in kt['related_decisions']:
            print(f"  - {d_id}")

    # Timestamps
    print(f"\nTimestamps:")
    if kt.get('created_at'):
        print(f"  Created:  {kt['created_at']}")
    if kt.get('modified_at'):
        print(f"  Modified: {kt['modified_at']}")

    print("=" * 70)

    return 0


def cmd_kt_import(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Handle 'got knowledge import' command.

    Imports a knowledge transfer document from a markdown file.
    Parses the markdown to extract title, session metadata, sections, etc.
    """
    file_path = Path(args.file)

    # Check if file exists
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1

    # Parse markdown file
    try:
        parsed = parse_markdown_file(file_path)
    except Exception as e:
        print(f"Error parsing markdown file: {e}")
        return 1

    # Validate required fields
    if not parsed['title']:
        print("Error: Could not extract title from markdown file")
        print("  Ensure file has a '# Title' heading at the top")
        return 1

    # Extract tags from command line or infer from filename
    tags = getattr(args, 'tags', [])
    if not tags:
        # Try to infer tags from filename
        filename = file_path.stem.lower()
        common_tags = ['architecture', 'performance', 'testing', 'refactor',
                      'migration', 'security', 'integration', 'unification']
        tags = [tag for tag in common_tags if tag in filename]

    # Determine status (draft if filename starts with [DRAFT])
    status = 'draft' if file_path.name.startswith('[DRAFT]') else 'published'

    # Create knowledge transfer entity
    try:
        kt_id = manager.create_knowledge_transfer(
            title=parsed['title'],
            session_id=parsed['session_id'],
            session_date=parsed['session_date'],
            summary=parsed['summary'],
            sections=parsed['sections'],
            code_refs=parsed['code_refs'],
            tags=tags,
            status=status,
            source_file=str(file_path.absolute()),
        )
    except Exception as e:
        print(f"Error creating knowledge transfer: {e}")
        return 1

    # Report success
    print(f"✅ Imported: {kt_id}")
    print(f"   Title: {parsed['title']}")
    print(f"   Source: {file_path}")
    if parsed['session_date']:
        print(f"   Session Date: {parsed['session_date']}")
    if parsed['session_id']:
        print(f"   Session ID: {parsed['session_id']}")
    print(f"   Sections: {len(parsed['sections'])}")
    print(f"   Code Refs: {len(parsed['code_refs'])}")
    print(f"   Tags: {', '.join(tags) if tags else 'none'}")
    print(f"   Status: {status}")

    return 0


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_knowledge_transfer_parser(subparsers) -> None:
    """
    Set up argparse subparsers for knowledge transfer commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Create knowledge transfer subparser with aliases
    kt_parser = subparsers.add_parser(
        "knowledge",
        aliases=["kt"],
        help="Knowledge transfer operations"
    )
    kt_subparsers = kt_parser.add_subparsers(
        dest="kt_command",
        help="Knowledge transfer subcommands"
    )

    # kt create
    create_parser = kt_subparsers.add_parser(
        "create",
        help="Create a knowledge transfer document"
    )
    create_parser.add_argument("title", help="Document title")
    create_parser.add_argument(
        "--session", "-s",
        default="",
        help="Session ID"
    )
    create_parser.add_argument(
        "--summary", "-S",
        default="",
        help="Executive summary"
    )
    create_parser.add_argument(
        "--content", "-c",
        default="",
        help="Document content (use '-' to read from stdin)"
    )
    create_parser.add_argument(
        "--sections",
        nargs="*",
        help="Sections as 'Heading:content' pairs"
    )
    create_parser.add_argument(
        "--tags", "-t",
        nargs="*",
        default=[],
        help="Classification tags"
    )

    # kt append
    append_parser = kt_subparsers.add_parser(
        "append",
        help="Append a section to existing document"
    )
    append_parser.add_argument("kt_id", help="Knowledge transfer ID")
    append_parser.add_argument("section_heading", help="Section heading")
    append_parser.add_argument(
        "content",
        help="Section content (use '-' to read from stdin)"
    )

    # kt link
    link_parser = kt_subparsers.add_parser(
        "link",
        help="Link to related entities"
    )
    link_parser.add_argument("kt_id", help="Knowledge transfer ID")
    link_group = link_parser.add_mutually_exclusive_group(required=True)
    link_group.add_argument(
        "--handoff",
        help="Handoff entity ID to link"
    )
    link_group.add_argument(
        "--task",
        help="Task entity ID to link"
    )
    link_group.add_argument(
        "--decision",
        help="Decision entity ID to link"
    )

    # kt list
    list_parser = kt_subparsers.add_parser(
        "list",
        help="List knowledge transfer documents"
    )
    list_parser.add_argument(
        "--status",
        choices=["draft", "published", "archived"],
        help="Filter by status"
    )
    list_parser.add_argument(
        "--tags", "-t",
        nargs="*",
        help="Filter by tags (matches any)"
    )
    list_parser.add_argument(
        "--limit", "-n",
        type=int,
        help="Limit number of results"
    )

    # kt show
    show_parser = kt_subparsers.add_parser(
        "show",
        help="Show knowledge transfer details"
    )
    show_parser.add_argument("kt_id", help="Knowledge transfer ID to display")

    # kt import
    import_parser = kt_subparsers.add_parser(
        "import",
        help="Import from markdown file"
    )
    import_parser.add_argument("file", help="Path to markdown file")
    import_parser.add_argument(
        "--tags", "-t",
        nargs="*",
        help="Classification tags (auto-detected if not provided)"
    )


def handle_knowledge_transfer_command(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Route knowledge transfer subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: TransactionalGoTAdapter instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'kt_command') or args.kt_command is None:
        print("Error: No knowledge transfer subcommand specified. Use 'got knowledge --help' for usage.")
        return 1

    command_handlers = {
        "create": cmd_kt_create,
        "append": cmd_kt_append,
        "link": cmd_kt_link,
        "list": cmd_kt_list,
        "show": cmd_kt_show,
        "import": cmd_kt_import,
    }

    handler = command_handlers.get(args.kt_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown knowledge transfer subcommand: {args.kt_command}")
    return 1
