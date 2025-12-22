#!/usr/bin/env python3
"""
Migrate sprint data from CURRENT_SPRINT.md to GoT transactional backend.

Parses the existing CURRENT_SPRINT.md file and creates Sprint and Epic
entities in the GoT system with proper relationships.

Usage:
    python scripts/migrate_sprints_to_got.py --dry-run  # Preview changes
    python scripts/migrate_sprints_to_got.py            # Perform migration
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.got.api import GoTManager
from cortical.got.types import Sprint, Epic
from cortical.utils.id_generation import generate_task_id


def parse_sprint_file(filepath: Path) -> Dict[str, Any]:
    """
    Parse CURRENT_SPRINT.md and extract sprints and epics.

    Args:
        filepath: Path to CURRENT_SPRINT.md

    Returns:
        Dictionary with 'epics' and 'sprints' keys
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    epics = []
    sprints = []

    # Parse epics from "# Epics" section
    epic_section = re.search(r'^# Epics\s*$(.*?)(?=^#\s|\Z)', content, re.MULTILINE | re.DOTALL)
    if epic_section:
        epics.extend(parse_epics(epic_section.group(1)))

    # Parse standalone epic sections (e.g., "# NLU Enhancement Epic")
    # Pattern: # [Epic Name] Epic followed by ## Epic: [Name] (id)
    standalone_epic_pattern = r'^# .+Epic\s*$.*?^## Epic:\s+(.+?)\s*\((\w+)\)\s*$(.*?)(?=^#\s|\Z)'
    for match in re.finditer(standalone_epic_pattern, content, re.MULTILINE | re.DOTALL):
        epic_title = match.group(1).strip()
        epic_short_id = match.group(2).strip()
        epic_content = match.group(3).strip()

        # Extract started date
        started_match = re.search(r'\*\*Started:\*\*\s+([\d-]+)', epic_content)
        started = started_match.group(1) if started_match else ""

        # Extract status
        status_match = re.search(r'\*\*Status:\*\*\s+(\w+)', epic_content)
        status = status_match.group(1).lower() if status_match else "active"

        # Extract phases
        phases = []
        phase_pattern = r'-\s+\*\*Phase (\d+):\*\*\s+(.*?)\s+(✅|🔄|⏳|←)'
        for phase_match in re.finditer(phase_pattern, epic_content):
            phase_num = int(phase_match.group(1))
            phase_name = phase_match.group(2).strip()
            phase_status = phase_match.group(3).strip()

            # Map emoji to status
            status_map = {
                '✅': 'completed',
                '🔄': 'in_progress',
                '⏳': 'pending',
                '←': 'in_progress'
            }
            phase_data = {
                'number': phase_num,
                'name': phase_name,
                'status': status_map.get(phase_status, 'pending')
            }
            phases.append(phase_data)

        epic_data = {
            'title': epic_title,
            'short_id': epic_short_id,
            'status': status,
            'started': started,
            'phases': phases
        }
        epics.append(epic_data)

    # Parse all sprint sections (## Sprint N:)
    sprint_pattern = r'^## (Sprint \d+:.*?)$\n(.*?)(?=^##\s|^#\s|\Z)'
    for match in re.finditer(sprint_pattern, content, re.MULTILINE | re.DOTALL):
        sprint_title = match.group(1).strip()
        sprint_content = match.group(2).strip()
        sprint_data = parse_sprint(sprint_title, sprint_content)
        if sprint_data:
            sprints.append(sprint_data)

    return {
        'epics': epics,
        'sprints': sprints
    }


def parse_epics(epic_section: str) -> List[Dict[str, Any]]:
    """
    Parse epic entries from the Epics section.

    Args:
        epic_section: Content of the Epics section

    Returns:
        List of epic dictionaries
    """
    epics = []

    # Pattern: ## Active: Epic Name (id)
    # or ## Epic: Epic Name (id)
    epic_pattern = r'^##\s+(?:Active:\s+)?(?:Epic:\s+)?(.*?)\s*\((\w+)\)\s*$'

    for match in re.finditer(epic_pattern, epic_section, re.MULTILINE):
        epic_title = match.group(1).strip()
        epic_short_id = match.group(2).strip()

        # Find the content after this heading until next ## or end
        start_pos = match.end()
        next_heading = re.search(r'^##', epic_section[start_pos:], re.MULTILINE)
        if next_heading:
            content = epic_section[start_pos:start_pos + next_heading.start()]
        else:
            content = epic_section[start_pos:]

        # Extract status (default: active)
        status_match = re.search(r'\*\*Status:\*\*\s+(\w+)', content)
        status = status_match.group(1).lower() if status_match else "active"

        # Extract started date
        started_match = re.search(r'\*\*Started:\*\*\s+([\d-]+)', content)
        started = started_match.group(1) if started_match else ""

        # Extract phases
        phases = []
        phase_pattern = r'-\s+\*\*Phase (\d+):\*\*\s+(.*?)\s+(✅|🔄|⏳)'
        for phase_match in re.finditer(phase_pattern, content):
            phase_num = int(phase_match.group(1))
            phase_name = phase_match.group(2).strip()
            phase_status = phase_match.group(3).strip()

            # Map emoji to status
            status_map = {
                '✅': 'completed',
                '🔄': 'in_progress',
                '⏳': 'pending'
            }
            phase_data = {
                'number': phase_num,
                'name': phase_name,
                'status': status_map.get(phase_status, 'pending')
            }
            phases.append(phase_data)

        epic_data = {
            'title': epic_title,
            'short_id': epic_short_id,
            'status': status,
            'started': started,
            'phases': phases
        }
        epics.append(epic_data)

    return epics


def parse_sprint(title: str, content: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single sprint section.

    Args:
        title: Sprint title (e.g., "Sprint 6: TestExpert Activation ✅")
        content: Sprint section content

    Returns:
        Sprint dictionary or None if parsing fails
    """
    # Extract sprint number from title
    number_match = re.match(r'Sprint (\d+):', title)
    if not number_match:
        return None

    sprint_number = int(number_match.group(1))

    # Extract title and status emoji
    title_clean = re.sub(r'\s*[🟢🟡✅🔴]\s*$', '', title)

    # Map status emoji to status string
    status = "available"  # default

    # First check title for emoji
    if '🟢' in title:
        status = "available"
    elif '🟡' in title:
        status = "in_progress"
    elif '✅' in title:
        status = "completed"
    elif '🔴' in title:
        status = "blocked"

    # Then check **Status:** line (overrides title emoji)
    status_line_match = re.search(r'\*\*Status:\*\*\s+(.+?)$', content, re.MULTILINE)
    if status_line_match:
        status_text = status_line_match.group(1).strip().lower()
        if 'complete' in status_text or '✅' in status_text:
            status = "completed"
        elif 'progress' in status_text or '🟡' in status_text:
            status = "in_progress"
        elif 'available' in status_text or '🟢' in status_text:
            status = "available"
        elif 'blocked' in status_text or '🔴' in status_text:
            status = "blocked"

    # Extract Sprint ID
    sprint_id_match = re.search(r'\*\*Sprint ID:\*\*\s+(.+?)$', content, re.MULTILINE)
    sprint_id = sprint_id_match.group(1).strip() if sprint_id_match else f"sprint-{sprint_number:03d}"

    # Extract Epic reference
    epic_match = re.search(r'\*\*Epic:\*\*\s+(.+?)(?:\((\w+)\))?$', content, re.MULTILINE)
    epic_name = ""
    epic_short_id = ""
    if epic_match:
        epic_name = epic_match.group(1).strip()
        epic_short_id = epic_match.group(2).strip() if epic_match.group(2) else ""

    # Extract Session ID
    session_match = re.search(r'\*\*Session:\*\*\s+(.+?)$', content, re.MULTILINE)
    session_id = session_match.group(1).strip() if session_match else ""

    # Extract Isolation paths
    isolation_match = re.search(r'\*\*Isolation:\*\*\s+(.+?)(?:\n|$)', content, re.MULTILINE)
    isolation = []
    if isolation_match:
        isolation_text = isolation_match.group(1).strip()
        # Parse comma-separated paths, removing backticks
        isolation = [
            path.strip().strip('`')
            for path in isolation_text.split(',')
        ]

    # Extract Goals
    goals = []
    goals_section = re.search(r'^### Goals\s*$(.*?)(?=^###|^##|\Z)', content, re.MULTILINE | re.DOTALL)
    if goals_section:
        goal_pattern = r'^-\s+\[([ x])\]\s+(.+?)$'
        for goal_match in re.finditer(goal_pattern, goals_section.group(1), re.MULTILINE):
            completed = goal_match.group(1) == 'x'
            text = goal_match.group(2).strip()
            goals.append({
                'text': text,
                'completed': completed
            })

    # Extract Notes
    notes = []
    notes_section = re.search(r'^### Notes\s*$(.*?)(?=^###|^##|\Z)', content, re.MULTILINE | re.DOTALL)
    if notes_section:
        note_pattern = r'^-\s+(.+?)$'
        for note_match in re.finditer(note_pattern, notes_section.group(1), re.MULTILINE):
            notes.append(note_match.group(1).strip())

    return {
        'number': sprint_number,
        'title': title_clean,
        'status': status,
        'sprint_id': sprint_id,
        'epic_name': epic_name,
        'epic_short_id': epic_short_id,
        'session_id': session_id,
        'isolation': isolation,
        'goals': goals,
        'notes': notes
    }


def migrate_epics(manager: GoTManager, epics: List[Dict], dry_run: bool) -> Dict[str, str]:
    """
    Migrate epics to GoT.

    Args:
        manager: GoT manager instance
        epics: List of epic dictionaries
        dry_run: If True, don't actually create entities

    Returns:
        Dictionary mapping epic short_id to full entity ID
    """
    epic_id_map = {}

    for epic_data in epics:
        # Generate full ID (use short_id as prefix if available)
        short_id = epic_data.get('short_id', '')
        epic_id = f"EPIC-{short_id}" if short_id else generate_task_id()

        # Map status to valid Epic status
        status = epic_data.get('status', 'active')
        if status not in {'active', 'completed', 'on_hold'}:
            status = 'active'

        # Determine current phase (highest in_progress or completed phase)
        phases = epic_data.get('phases', [])
        current_phase = 1
        for phase in phases:
            if phase.get('status') in {'in_progress', 'completed'}:
                current_phase = max(current_phase, phase.get('number', 1))

        if dry_run:
            print(f"[DRY-RUN] Would create Epic: {epic_data['title']}")
            print(f"  ID: {epic_id}")
            print(f"  Status: {status}")
            print(f"  Phase: {current_phase}/{len(phases)}")
            print(f"  Phases: {len(phases)}")
        else:
            with manager.transaction() as tx:
                epic = Epic(
                    id=epic_id,
                    title=epic_data['title'],
                    status=status,
                    phase=current_phase,
                    phases=phases,
                    properties={
                        'short_id': short_id,
                        'started': epic_data.get('started', '')
                    }
                )
                tx.write(epic)
            print(f"✓ Created Epic: {epic_data['title']} ({epic_id})")

        # Map short_id to full ID for sprint references
        if short_id:
            epic_id_map[short_id] = epic_id
        # Also map by name for fuzzy matching
        epic_id_map[epic_data['title'].lower()] = epic_id

    return epic_id_map


def migrate_sprints(
    manager: GoTManager,
    sprints: List[Dict],
    epic_id_map: Dict[str, str],
    dry_run: bool
) -> int:
    """
    Migrate sprints to GoT and create PART_OF edges to epics.

    Args:
        manager: GoT manager instance
        sprints: List of sprint dictionaries
        epic_id_map: Mapping of epic identifiers to entity IDs
        dry_run: If True, don't actually create entities

    Returns:
        Number of sprints created
    """
    created_count = 0

    for sprint_data in sprints:
        # Resolve epic ID
        epic_id = ""
        if sprint_data.get('epic_short_id'):
            epic_id = epic_id_map.get(sprint_data['epic_short_id'], "")
        if not epic_id and sprint_data.get('epic_name'):
            # Try fuzzy match by name
            epic_id = epic_id_map.get(sprint_data['epic_name'].lower(), "")

        # Generate sprint ID
        sprint_id = f"S-{sprint_data['sprint_id']}"

        if dry_run:
            print(f"\n[DRY-RUN] Would create Sprint: {sprint_data['title']}")
            print(f"  ID: {sprint_id}")
            print(f"  Number: {sprint_data['number']}")
            print(f"  Status: {sprint_data['status']}")
            print(f"  Epic: {epic_id or '(none)'}")
            print(f"  Session: {sprint_data.get('session_id', '(none)')}")
            print(f"  Isolation: {len(sprint_data['isolation'])} paths")
            print(f"  Goals: {len(sprint_data['goals'])} ({sum(1 for g in sprint_data['goals'] if g['completed'])} completed)")
            print(f"  Notes: {len(sprint_data['notes'])}")
        else:
            with manager.transaction() as tx:
                sprint = Sprint(
                    id=sprint_id,
                    title=sprint_data['title'],
                    status=sprint_data['status'],
                    epic_id=epic_id,
                    number=sprint_data['number'],
                    session_id=sprint_data.get('session_id', ''),
                    isolation=sprint_data['isolation'],
                    goals=sprint_data['goals'],
                    notes=sprint_data['notes'],
                    properties={
                        'epic_name': sprint_data.get('epic_name', ''),
                        'sprint_id': sprint_data['sprint_id']
                    }
                )
                tx.write(sprint)

                # Create PART_OF edge if epic exists
                if epic_id:
                    tx.add_edge(sprint_id, epic_id, "PART_OF")

            completed = sum(1 for g in sprint_data['goals'] if g['completed'])
            total = len(sprint_data['goals'])
            print(f"✓ Created Sprint {sprint_data['number']}: {sprint_data['title']} ({completed}/{total} goals)")
            created_count += 1

    return created_count


def main():
    parser = argparse.ArgumentParser(
        description="Migrate sprints from CURRENT_SPRINT.md to GoT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be migrated
  python scripts/migrate_sprints_to_got.py --dry-run

  # Perform migration
  python scripts/migrate_sprints_to_got.py

  # Migrate from custom file
  python scripts/migrate_sprints_to_got.py --source custom_sprints.md
        """
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without creating entities"
    )
    parser.add_argument(
        "--source",
        default="tasks/CURRENT_SPRINT.md",
        help="Source markdown file (default: tasks/CURRENT_SPRINT.md)"
    )
    parser.add_argument(
        "--got-dir",
        default=".got",
        help="GoT directory (default: .got)"
    )
    args = parser.parse_args()

    # Resolve paths
    source_path = PROJECT_ROOT / args.source
    got_dir = PROJECT_ROOT / args.got_dir

    # Validate source file exists
    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}", file=sys.stderr)
        return 1

    # Parse the markdown file
    print(f"Parsing {source_path}...")
    data = parse_sprint_file(source_path)

    print(f"\nFound {len(data['epics'])} epics and {len(data['sprints'])} sprints")

    if args.dry_run:
        print("\n" + "="*70)
        print("DRY RUN MODE - No changes will be made")
        print("="*70)

    # Initialize GoT manager
    manager = GoTManager(got_dir)

    # Migrate epics first
    print("\n" + "-"*70)
    print("Migrating Epics")
    print("-"*70)
    epic_id_map = migrate_epics(manager, data['epics'], args.dry_run)

    # Migrate sprints
    print("\n" + "-"*70)
    print("Migrating Sprints")
    print("-"*70)
    sprint_count = migrate_sprints(manager, data['sprints'], epic_id_map, args.dry_run)

    # Summary
    print("\n" + "="*70)
    if args.dry_run:
        print("DRY RUN COMPLETE - No changes made")
    else:
        print("MIGRATION COMPLETE")
    print("="*70)
    print(f"Epics: {len(data['epics'])}")
    print(f"Sprints: {sprint_count if not args.dry_run else len(data['sprints'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
