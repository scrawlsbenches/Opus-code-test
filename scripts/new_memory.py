#!/usr/bin/env python3
"""
Quick memory and decision record creation from command line.

Usage:
    # Create a memory entry (interactive prompts)
    python scripts/new_memory.py

    # Create with title
    python scripts/new_memory.py "dogfooding session insights"

    # Create with tags
    python scripts/new_memory.py "fuzzing discoveries" --tags "security,testing,fuzzing"

    # Create a decision record
    python scripts/new_memory.py "use microseconds in task IDs" --decision

    # Dry-run to preview
    python scripts/new_memory.py "test topic" --dry-run

Examples:
    $ python scripts/new_memory.py "learned about NaN validation" --tags "testing,validation"
    Created: samples/memories/2025-12-14_14-30-52_a1b2-nan-validation.md

    $ python scripts/new_memory.py "add microseconds to timestamps" --decision
    Created: samples/decisions/2025-12-14_14-31-15_c3d4-microseconds-timestamps.md
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from task_utils import generate_session_id


# Directories for memories and decisions
MEMORIES_DIR = Path("samples/memories")
DECISIONS_DIR = Path("samples/decisions")


def get_git_author() -> str:
    """Get git author name from config."""
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "Unknown"


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    # Simple slugification: lowercase, replace spaces with hyphens
    slug = text.lower().strip()
    slug = slug.replace(" ", "-")
    # Remove non-alphanumeric except hyphens
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    # Remove duplicate hyphens
    while "--" in slug:
        slug = slug.replace("--", "-")
    # Truncate to reasonable length
    return slug[:50]


def generate_memory_filename(title: str, is_decision: bool = False) -> str:
    """
    Generate merge-safe filename with timestamp and session ID.

    Format: YYYY-MM-DD_HH-MM-SS_XXXX-topic.md

    Args:
        title: Topic or title of the memory/decision
        is_decision: If True, generates a decision record filename

    Returns:
        Filename string
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    session_id = generate_session_id()
    slug = slugify(title)

    return f"{date_str}_{time_str}_{session_id}-{slug}.md"


def create_memory_template(title: str, tags: str = "", author: str = "") -> str:
    """
    Create a memory entry template.

    Args:
        title: Memory title/topic
        tags: Comma-separated tags
        author: Git author name

    Returns:
        Markdown template string
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Format tags
    tag_list = ""
    if tags:
        tag_items = [f"`{t.strip()}`" for t in tags.split(",")]
        tag_list = ", ".join(tag_items)

    template = f"""# Memory Entry: {date_str} {title.title()}

**Tags:** {tag_list}
**Related:** [[link-to-related-docs.md]]

---

## Context

What prompted this memory entry?

## What I Learned

### 1. Key Insight

Describe what you learned or discovered.

### 2. Additional Findings

Any other important learnings?

## Connections Made

- **Concept A → Concept B**: How are they related?
- **Pattern → Implementation**: What patterns emerged?

## Emotional State

How did this work feel? What was satisfying or challenging?

## Future Exploration

- [ ] Follow-up item 1
- [ ] Follow-up item 2

## Artifacts Created

- Files, tasks, or other outputs from this session

---

*Committed to memory at: {timestamp}*
"""
    return template


def create_decision_template(title: str, tags: str = "", author: str = "") -> str:
    """
    Create an ADR (Architecture Decision Record) template.

    Args:
        title: Decision title
        tags: Comma-separated tags
        author: Git author name

    Returns:
        Markdown template string
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    # Format tags
    tag_list = ""
    if tags:
        tag_items = [f"`{t.strip()}`" for t in tags.split(",")]
        tag_list = ", ".join(tag_items)

    # Find next ADR number (simple approach - count existing files)
    try:
        existing = list(DECISIONS_DIR.glob("adr-*.md"))
        numbers = []
        for f in existing:
            parts = f.stem.split("-")
            if len(parts) >= 2 and parts[0] == "adr" and parts[1].isdigit():
                numbers.append(int(parts[1]))
        next_num = max(numbers) + 1 if numbers else 1
    except (ValueError, IndexError, AttributeError, OSError):
        next_num = 1

    adr_id = f"ADR-{next_num:03d}"

    template = f"""# {adr_id}: {title.title()}

**Status:** Proposed
**Date:** {date_str}
**Deciders:** Development team
**Tags:** {tag_list}

---

## Context and Problem Statement

What is the problem we're trying to solve? What factors are influencing this decision?

## Decision Drivers

1. **Factor 1**: Description
2. **Factor 2**: Description
3. **Factor 3**: Description

## Considered Options

### Option 1: [Name]

**Pros:**
- Advantage 1
- Advantage 2

**Cons:**
- Disadvantage 1
- Disadvantage 2

### Option 2: [Name]

**Pros:**
- Advantage 1

**Cons:**
- Disadvantage 1

## Decision Outcome

**Chosen Option:** Option X - [Name]

**Rationale:**
Explain why this option was chosen.

## Implementation

```python
# Code example if applicable
```

## Consequences

### Positive
- Benefit 1
- Benefit 2

### Negative
- Trade-off 1
- Trade-off 2

### Neutral
- Other effects

## Validation

How will we verify this decision was correct?

## Related Decisions

- Link to related ADRs or documentation

---

*Decision recorded on: {date_str}*
"""
    return template


def create_memory(
    title: str,
    tags: str = "",
    is_decision: bool = False,
    dry_run: bool = False
) -> Path:
    """
    Create a memory entry or decision record.

    Args:
        title: Title/topic of the memory
        tags: Comma-separated tags
        is_decision: If True, creates a decision record
        dry_run: If True, only shows what would be created

    Returns:
        Path to created file (or would-be path in dry-run)
    """
    # Determine directory and filename
    target_dir = DECISIONS_DIR if is_decision else MEMORIES_DIR
    filename = generate_memory_filename(title, is_decision)
    filepath = target_dir / filename

    # Get git author
    author = get_git_author()

    # Create template
    if is_decision:
        content = create_decision_template(title, tags, author)
    else:
        content = create_memory_template(title, tags, author)

    if dry_run:
        print("=== DRY RUN ===")
        print(f"Would create: {filepath}")
        print(f"\nContent preview:\n")
        print(content[:500] + "..." if len(content) > 500 else content)
        return filepath

    # Create directory if needed
    target_dir.mkdir(parents=True, exist_ok=True)

    # Write file
    with open(filepath, "w") as f:
        f.write(content)

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Create merge-safe memory entries and decision records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "title",
        nargs="?",
        help="Memory topic or decision title"
    )
    parser.add_argument(
        "-d", "--decision",
        action="store_true",
        help="Create a decision record (ADR) instead of memory entry"
    )
    parser.add_argument(
        "-t", "--tags",
        default="",
        help="Comma-separated tags (e.g., 'security,testing,fuzzing')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without writing file"
    )

    args = parser.parse_args()

    # Ensure directories exist
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)

    if args.title:
        filepath = create_memory(
            title=args.title,
            tags=args.tags,
            is_decision=args.decision,
            dry_run=args.dry_run
        )

        if not args.dry_run:
            doc_type = "Decision record" if args.decision else "Memory entry"
            print(f"Created {doc_type}:")
            print(f"  {filepath}")
            print(f"\nEdit with: $EDITOR {filepath}")
    else:
        # Interactive mode
        doc_type = "decision record" if args.decision else "memory entry"
        print(f"Create a new {doc_type} (Ctrl+C to cancel)\n")

        title = input("Title/topic: ").strip()
        if not title:
            print("Title is required")
            return

        tags = input("Tags (comma-separated, optional): ").strip()

        filepath = create_memory(
            title=title,
            tags=tags,
            is_decision=args.decision,
            dry_run=args.dry_run
        )

        if not args.dry_run:
            print(f"\nCreated {doc_type}:")
            print(f"  {filepath}")
            print(f"\nEdit with: $EDITOR {filepath}")


if __name__ == "__main__":
    main()
