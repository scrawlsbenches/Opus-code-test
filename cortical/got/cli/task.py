"""
Task CLI commands for GoT system.

Provides commands for:
- Creating tasks
- Listing tasks
- Showing task details
- Starting/completing/blocking tasks
- Managing task dependencies

This module can be integrated into got_utils.py CLI or used standalone.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .shared import (
    VALID_STATUSES,
    VALID_PRIORITIES,
    VALID_CATEGORIES,
    PRIORITY_MEDIUM,
    STATUS_IN_PROGRESS,
    format_task_table,
    format_task_details,
)

if TYPE_CHECKING:
    from cortical.got.adapter import TransactionalGoTAdapter

# Learning integration
try:
    from cortical.got.learning_integration import GoTLearningBridge
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    logging.warning("Learning integration not available - install llm_orchestration package")

logger = logging.getLogger(__name__)


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_task_create(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task create' command."""
    task_id = manager.create_task(
        title=args.title,
        priority=getattr(args, 'priority', PRIORITY_MEDIUM),
        category=getattr(args, 'category', 'feature'),
        description=getattr(args, 'description', ''),
        sprint_id=getattr(args, 'sprint', None),
        depends_on=getattr(args, 'depends', None),
        blocks=getattr(args, 'blocks', None),
    )

    # Auto-link to current sprint unless --no-sprint or explicit --sprint
    if not getattr(args, 'no_sprint', False) and not getattr(args, 'sprint', None):
        current_sprint = manager.get_current_sprint()
        if current_sprint:
            # Add CONTAINS edge from sprint to task
            try:
                manager._manager.add_edge(
                    source_id=current_sprint.id,
                    target_id=task_id,
                    edge_type="CONTAINS"
                )
                print(f"  Linked to sprint: {current_sprint.id}")
            except Exception as e:
                # Non-fatal - just warn the user
                print(f"  Warning: Could not auto-link to sprint {current_sprint.id}: {e}")

    manager.save()
    print(f"Created: {task_id}")
    return 0


def cmd_task_list(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task list' command."""
    tasks = manager.list_tasks(
        status=getattr(args, 'status', None),
        priority=getattr(args, 'priority', None),
        category=getattr(args, 'category', None),
        sprint_id=getattr(args, 'sprint', None),
        blocked_only=getattr(args, 'blocked', False),
    )

    # Apply limit if specified
    limit = getattr(args, 'limit', None)
    if limit is not None and limit > 0:
        tasks = tasks[:limit]

    if getattr(args, 'json', False):
        data = [{"id": t.id, "title": t.content, **t.properties} for t in tasks]
        print(json.dumps(data, indent=2))
    else:
        print(format_task_table(tasks))

    return 0


def cmd_task_next(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task next' command."""
    result = manager.get_next_task()

    if result is None:
        print("No pending tasks available.")
        return 0

    # Format output
    print(f"Next task: {result['id']}")
    print(f"  Title:    {result['title']}")
    print(f"  Priority: {result['priority']}")
    print(f"  Category: {result['category']}")

    # If --start flag, also start the task
    if getattr(args, 'start', False):
        task_id = result['id']
        if task_id.startswith("task:"):
            task_id = task_id[5:]
        success = manager.start_task(task_id)
        if success:
            print(f"\nStarted: {result['id']}")

    return 0


def cmd_task_show(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task show' command."""
    task_id = args.task_id

    # Try to get task (with ID normalization)
    task = manager.get_task(task_id)

    # If not found, try with/without task: prefix
    if task is None:
        if task_id.startswith("task:"):
            task = manager.get_task(task_id[5:])
        else:
            task = manager.get_task(f"task:{task_id}")

    if task is None:
        print(f"Task not found: {task_id}")
        return 1

    # Display task details
    print(format_task_details(task))

    # Show sprint membership
    sprint_info = manager.get_task_sprint(task.id)
    if sprint_info:
        print(f"\nSprint: {sprint_info['name']}")
        print(f"        {sprint_info['id']}")

    # Show dependencies
    deps = manager.get_task_dependencies(task.id)
    if deps:
        print(f"\nDepends On ({len(deps)}):")
        for dep in deps:
            print(f"  - {dep.id}: {dep.content}")

    # Show what depends on this task
    dependents = manager.what_depends_on(task.id)
    if dependents:
        print(f"\nBlocks ({len(dependents)}):")
        for dep in dependents:
            print(f"  - {dep.id}: {dep.content}")

    return 0


def cmd_task_start(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task start' command."""
    # Get task details for guidance
    task = manager.get_task(args.task_id)
    if not task:
        print(f"Task not found: {args.task_id}")
        return 1

    # Show learning guidance if requested
    show_guidance = getattr(args, 'show_guidance', False)
    if show_guidance and LEARNING_AVAILABLE:
        try:
            # Get GoT directory from manager
            got_dir = getattr(manager, 'got_dir', Path('.got'))
            bridge = GoTLearningBridge(got_dir)

            # Extract task metadata
            task_title = task.content
            task_category = task.properties.get('category', 'general')
            task_priority = task.properties.get('priority', 'medium')

            # Get guidance
            guidance = bridge.get_guidance_for_task(
                task_title=task_title,
                task_category=task_category,
                task_priority=task_priority
            )

            # Display guidance
            print("\n" + "="*70)
            print("📚 LEARNING GUIDANCE")
            print("="*70)

            if guidance['lessons']:
                print(f"\n✓ {len(guidance['lessons'])} Relevant Lessons:")
                for lesson in guidance['lessons'][:5]:  # Show top 5
                    print(f"  • {lesson.principle}")
                    if lesson.context_tags:
                        print(f"    Tags: {', '.join(list(lesson.context_tags)[:3])}")

            if guidance['recommendations']:
                print(f"\n💡 Recommendations:")
                for rec in guidance['recommendations'][:3]:  # Show top 3
                    print(f"  • {rec}")

            if guidance['warnings']:
                print(f"\n⚠️  Warnings:")
                for warn in guidance['warnings'][:3]:  # Show top 3
                    print(f"  • {warn}")

            if guidance['relevant_successes']:
                print(f"\n✅ {len(guidance['relevant_successes'])} Similar Successful Tasks")

            if guidance['relevant_failures']:
                print(f"\n❌ {len(guidance['relevant_failures'])} Similar Failed Tasks (avoid these approaches)")

            print("="*70 + "\n")

        except Exception as e:
            logger.debug(f"Failed to retrieve learning guidance: {e}")
            # Don't fail task start if learning fails

    # Start the task
    if manager.start_task(args.task_id):
        manager.save()
        print(f"Started: {args.task_id}")
        return 0
    else:
        print(f"Task not found: {args.task_id}")
        return 1


def cmd_task_complete(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task complete' command."""
    # Get task details before completing (for learning capture)
    task = manager.get_task(args.task_id)
    if not task:
        print(f"Task not found: {args.task_id}")
        return 1

    # Complete the task
    retrospective = getattr(args, 'retrospective', None)
    if manager.complete_task(args.task_id, retrospective):
        manager.save()
        print(f"Completed: {args.task_id}")

        # Capture learning experience (unless skipped)
        skip_learning = getattr(args, 'skip_learning', False)
        if not skip_learning and LEARNING_AVAILABLE:
            try:
                # Get GoT directory from manager
                got_dir = getattr(manager, 'got_dir', Path('.got'))
                bridge = GoTLearningBridge(got_dir)

                # Extract task metadata
                task_title = task.content
                task_category = task.properties.get('category', 'general')
                task_priority = task.properties.get('priority', 'medium')

                # Try to infer approach from retrospective or category
                approach = None
                if retrospective:
                    lower_retro = retrospective.lower()
                    if 'tdd' in lower_retro or 'test-first' in lower_retro:
                        approach = 'test-first'
                    elif 'refactor' in lower_retro:
                        approach = 'refactoring'
                    elif 'debug' in lower_retro:
                        approach = 'debugging'

                # Capture the experience
                experience = bridge.capture_task_completion(
                    task_id=args.task_id,
                    retrospective=retrospective or "",
                    task_title=task_title,
                    task_category=task_category,
                    task_priority=task_priority,
                    approach=approach,
                )

                logger.debug(f"Learning experience captured: {experience.id}")
                print(f"📚 Learning experience captured: {experience.id}")

            except Exception as e:
                # Don't fail task completion if learning capture fails
                logger.debug(f"Failed to capture learning experience: {e}")
                print(f"⚠️  Warning: Could not capture learning experience: {e}")

        return 0
    else:
        print(f"Failed to complete task: {args.task_id}")
        return 1


def cmd_task_block(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task block' command."""
    # Get task details before blocking (for learning capture)
    task = manager.get_task(args.task_id)
    if not task:
        print(f"Task not found: {args.task_id}")
        return 1

    # Block the task
    if manager.block_task(args.task_id, args.reason, getattr(args, 'blocker', None)):
        manager.save()
        print(f"Blocked: {args.task_id}")

        # Capture learning experience for failure
        skip_learning = getattr(args, 'skip_learning', False)
        if not skip_learning and LEARNING_AVAILABLE:
            try:
                # Get GoT directory from manager
                got_dir = getattr(manager, 'got_dir', Path('.got'))
                bridge = GoTLearningBridge(got_dir)

                # Extract task metadata
                task_title = task.content
                task_category = task.properties.get('category', 'general')
                task_priority = task.properties.get('priority', 'medium')

                # Build blockers list
                blockers = [args.reason]
                if getattr(args, 'blocker', None):
                    blockers.append(f"Blocked by: {args.blocker}")

                # Capture the failure experience
                experience = bridge.capture_task_failure(
                    task_id=args.task_id,
                    error_message=args.reason,
                    task_title=task_title,
                    task_category=task_category,
                    task_priority=task_priority,
                    blockers=blockers,
                )

                logger.debug(f"Learning experience (failure) captured: {experience.id}")
                print(f"📚 Learning experience (failure) captured: {experience.id}")

            except Exception as e:
                # Don't fail task blocking if learning capture fails
                logger.debug(f"Failed to capture learning experience: {e}")
                print(f"⚠️  Warning: Could not capture learning experience: {e}")

        return 0
    else:
        print(f"Failed to block task: {args.task_id}")
        return 1


def cmd_task_depends(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task depends' command."""
    try:
        # Use add_dependency method
        if manager.add_dependency(args.task_id, args.depends_on_id):
            manager.save()
            print(f"Created dependency: {args.task_id} depends on {args.depends_on_id}")
            return 0
        else:
            print("Failed to create dependency - check that both task IDs exist")
            return 1
    except Exception as e:
        print(f"Error creating dependency: {e}")
        return 1


def cmd_task_update(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task update' command.

    Updates task properties. Only specified fields are updated.
    """
    task_id = args.task_id

    # Get existing task
    task = manager.get_task(task_id)
    if not task:
        print(f"Task not found: {task_id}")
        return 1

    # Build updates dict from provided arguments
    updates = {}

    if getattr(args, 'title', None):
        updates['title'] = args.title
    if getattr(args, 'priority', None):
        updates['priority'] = args.priority
    if getattr(args, 'category', None):
        updates['category'] = args.category
    if getattr(args, 'description', None):
        updates['description'] = args.description
    if getattr(args, 'retrospective', None):
        updates['retrospective'] = args.retrospective

    if not updates:
        print("No updates specified. Use --title, --priority, --category, --description, or --retrospective")
        return 1

    # Apply updates
    if manager.update_task(task_id, **updates):
        manager.save()
        print(f"Updated: {task_id}")
        for key, value in updates.items():
            # Truncate long values for display
            display_value = value if len(str(value)) < 60 else str(value)[:57] + "..."
            print(f"  {key}: {display_value}")
        return 0
    else:
        print(f"Failed to update: {task_id}")
        return 1


def cmd_task_delete(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task delete' command.

    TRANSACTIONAL: Verifies pre-conditions before deletion.
    - Task must exist
    - Without --force: fails if task has dependents, blocks others, or is in progress
    - With --force: removes edges and deletes the task
    """
    task_id = args.task_id
    force = getattr(args, 'force', False)

    # Get task info before deletion for display
    task = manager.get_task(task_id)
    if not task:
        print(f"Task not found: {task_id}")
        return 1

    # Show what we're about to do
    task_title = task.content
    task_status = task.properties.get("status", "unknown")

    if not force:
        # Show warnings about what might block deletion
        dependents = manager.what_depends_on(
            task_id if task_id.startswith("task:") else f"task:{task_id}"
        )
        if dependents:
            print(f"⚠️  Cannot delete: {len(dependents)} task(s) depend on this task:")
            for d in dependents[:5]:
                print(f"    - {d.id}: {d.content}")
            if len(dependents) > 5:
                print(f"    ... and {len(dependents) - 5} more")
            print("\nUse --force to delete anyway (will orphan dependent tasks)")
            return 1

        if task_status == STATUS_IN_PROGRESS:
            print("⚠️  Cannot delete: task is in progress")
            print("Use --force to delete anyway")
            return 1

    # Attempt deletion
    if manager.delete_task(task_id, force=force):
        manager.save()
        print(f"🗑️  Deleted: {task_id}")
        print(f"   Title: {task_title}")
        if force:
            print("   (forced deletion)")
        return 0
    else:
        print(f"Failed to delete: {task_id}")
        return 1


def cmd_task_history(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task history' command.

    Shows the version history of a task, including:
    - All previous versions with timestamps
    - Changes between versions
    - Useful for auditing and recovery

    The history is stored in .got/entities/_history/{entity_id}.jsonl
    """
    from datetime import datetime

    task_id = args.task_id

    # Get the history file path
    # Support both TransactionalGoTAdapter (got_dir) and direct GoTManager (got_dir)
    got_dir = getattr(manager, 'got_dir', None)
    if got_dir is None:
        # Fallback for direct managers
        got_dir = getattr(manager, '_store_dir', Path('.got'))
    history_dir = got_dir / "entities" / "_history"
    history_file = history_dir / f"{task_id}.jsonl"

    if not history_file.exists():
        # Check if task exists but has no history (never modified)
        task = manager.get_task(task_id)
        if task:
            print(f"Task {task_id} has no modification history (never modified).")
            print(f"\nCurrent state:")
            print(format_task_details(task))
            return 0
        else:
            print(f"No history found for: {task_id}")
            print("The task may not exist or may have been deleted without history.")
            return 1

    # Read and parse history entries
    entries = []
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

    if not entries:
        print(f"History file exists but is empty for: {task_id}")
        return 1

    # Display history
    print(f"History for: {task_id}")
    print(f"Total versions: {len(entries)}")
    print("-" * 70)

    if getattr(args, 'json', False):
        # JSON output for programmatic access
        print(json.dumps(entries, indent=2))
        return 0

    # Show entries in reverse chronological order (most recent first)
    limit = getattr(args, 'limit', 10)
    entries_to_show = entries[-limit:] if limit else entries
    entries_to_show = list(reversed(entries_to_show))

    for i, entry in enumerate(entries_to_show):
        timestamp = entry.get('timestamp', 'unknown')
        global_version = entry.get('global_version', '?')
        data = entry.get('data', {})

        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except (ValueError, AttributeError):
            timestamp_str = timestamp

        print(f"\n[Version {global_version}] {timestamp_str}")
        print(f"  Title:    {data.get('title', 'N/A')}")
        print(f"  Status:   {data.get('status', 'N/A')}")
        print(f"  Priority: {data.get('priority', 'N/A')}")

        if getattr(args, 'verbose', False):
            desc = data.get('description', '')
            if desc:
                print(f"  Description: {desc[:100]}{'...' if len(desc) > 100 else ''}")
            props = data.get('properties', {})
            if props:
                print(f"  Properties: {json.dumps(props)}")

    if len(entries) > limit:
        print(f"\n... {len(entries) - limit} more entries (use --limit to see more)")

    # Show current state if task still exists
    task = manager.get_task(task_id)
    if task:
        print(f"\n{'='*70}")
        print("CURRENT STATE:")
        print(f"  Title:    {task.content}")
        print(f"  Status:   {task.properties.get('status', 'N/A')}")
        print(f"  Priority: {task.properties.get('priority', 'N/A')}")
    else:
        print(f"\n{'='*70}")
        print("⚠️  TASK DELETED - above history shows state before deletion")

    return 0


def cmd_task_import(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got task import' command.

    Imports tasks from a YAML or JSON file.

    Expected file format:
    ```yaml
    tasks:
      - title: "Task one"
        priority: high
        category: feature
        description: "Optional description"
      - title: "Task two"
        priority: medium
        category: bugfix
    ```
    """
    file_path = Path(args.file)

    # Check if file exists
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1

    # Read and parse file
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Detect format and parse
        if file_path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
            except ImportError:
                print("Error: pyyaml is required for YAML files. Install with: pip install pyyaml")
                return 1
            data = yaml.safe_load(content)
        elif file_path.suffix == '.json':
            data = json.loads(content)
        else:
            # Try to detect by content
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                try:
                    import yaml
                except ImportError:
                    print("Error: pyyaml is required for YAML files. Install with: pip install pyyaml")
                    return 1
                data = yaml.safe_load(content)
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1

    # Validate structure
    if not isinstance(data, dict) or 'tasks' not in data:
        print("Error: File must contain a 'tasks' key with a list of tasks")
        return 1

    tasks_data = data['tasks']
    if not isinstance(tasks_data, list):
        print("Error: 'tasks' must be a list")
        return 1

    # Validate and create tasks
    created_ids = []
    errors = []

    for i, task_data in enumerate(tasks_data, 1):
        # Validate required fields
        if not isinstance(task_data, dict):
            errors.append(f"Task {i}: Must be a dictionary")
            continue

        if 'title' not in task_data:
            errors.append(f"Task {i}: Missing required field 'title'")
            continue

        if 'priority' not in task_data:
            errors.append(f"Task {i}: Missing required field 'priority'")
            continue

        # Validate priority
        if task_data['priority'] not in VALID_PRIORITIES:
            errors.append(f"Task {i}: Invalid priority '{task_data['priority']}'. Must be one of: {', '.join(VALID_PRIORITIES)}")
            continue

        # Validate category if provided
        category = task_data.get('category', 'feature')
        if category not in VALID_CATEGORIES:
            errors.append(f"Task {i}: Invalid category '{category}'. Must be one of: {', '.join(VALID_CATEGORIES)}")
            continue

        # Create task
        try:
            task_id = manager.create_task(
                title=task_data['title'],
                priority=task_data['priority'],
                category=category,
                description=task_data.get('description', ''),
                sprint_id=getattr(args, 'sprint', None),
            )
            created_ids.append(task_id)
        except Exception as e:
            errors.append(f"Task {i} ('{task_data['title']}'): {e}")

    # Save if any tasks were created
    if created_ids:
        manager.save()

    # Report results
    if errors:
        print(f"\n⚠️  Errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")

    if created_ids:
        print(f"\n✅ Created {len(created_ids)} task(s):")
        for task_id in created_ids:
            print(f"  - {task_id}")
        return 0
    else:
        print("\n❌ No tasks were created")
        return 1


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_task_parser(subparsers) -> None:
    """
    Set up argparse subparsers for task commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Create task subparser
    task_parser = subparsers.add_parser("task", help="Task operations")
    task_subparsers = task_parser.add_subparsers(
        dest="task_command",
        help="Task subcommands"
    )

    # task create
    create_parser = task_subparsers.add_parser("create", help="Create a task")
    create_parser.add_argument("title", help="Task title")
    create_parser.add_argument(
        "--priority", "-p",
        choices=VALID_PRIORITIES,
        default=PRIORITY_MEDIUM
    )
    create_parser.add_argument(
        "--category", "-c",
        choices=VALID_CATEGORIES,
        default="feature"
    )
    create_parser.add_argument("--description", "--notes", "-d", default="")
    create_parser.add_argument("--sprint", "-s", help="Sprint ID")
    create_parser.add_argument(
        "--no-sprint",
        action="store_true",
        help="Skip auto-linking to current sprint"
    )
    create_parser.add_argument(
        "--depends-on", "--depends",
        nargs="+",
        dest="depends",
        help="Task IDs this task depends on"
    )
    create_parser.add_argument(
        "--blocks",
        nargs="+",
        help="Task IDs this task blocks"
    )

    # task list
    list_parser = task_subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--status", choices=VALID_STATUSES)
    list_parser.add_argument("--priority", choices=VALID_PRIORITIES)
    list_parser.add_argument("--category", choices=VALID_CATEGORIES)
    list_parser.add_argument("--sprint", help="Filter by sprint")
    list_parser.add_argument("--blocked", action="store_true", help="Show only blocked")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    list_parser.add_argument(
        "--limit", "-n",
        type=int,
        help="Limit number of results"
    )

    # task show
    show_parser = task_subparsers.add_parser("show", help="Show task details")
    show_parser.add_argument("task_id", help="Task ID to display")

    # task next
    next_parser = task_subparsers.add_parser("next", help="Get the next task to work on")
    next_parser.add_argument(
        "--start", "-s",
        action="store_true",
        help="Also start the task after selecting it"
    )

    # task start
    start_parser = task_subparsers.add_parser("start", help="Start a task")
    start_parser.add_argument("task_id", help="Task ID")
    start_parser.add_argument(
        "--show-guidance", "-g",
        action="store_true",
        help="Show relevant learning guidance before starting"
    )

    # task complete
    complete_parser = task_subparsers.add_parser("complete", help="Complete a task")
    complete_parser.add_argument("task_id", help="Task ID")
    complete_parser.add_argument(
        "--retrospective", "--notes", "-r", "-n",
        dest="retrospective",
        help="Retrospective notes"
    )
    complete_parser.add_argument(
        "--skip-learning",
        action="store_true",
        help="Skip capturing learning experience (not recommended)"
    )

    # task block
    block_parser = task_subparsers.add_parser("block", help="Block a task")
    block_parser.add_argument("task_id", help="Task ID")
    block_parser.add_argument("--reason", "-r", required=True, help="Block reason")
    block_parser.add_argument("--blocker", "-b", help="Blocking task ID")
    block_parser.add_argument(
        "--skip-learning",
        action="store_true",
        help="Skip capturing learning experience (not recommended)"
    )

    # task update
    update_parser = task_subparsers.add_parser("update", help="Update a task's properties")
    update_parser.add_argument("task_id", help="Task ID to update")
    update_parser.add_argument("--title", "-t", help="New title")
    update_parser.add_argument(
        "--priority", "-p",
        choices=VALID_PRIORITIES,
        help="New priority"
    )
    update_parser.add_argument(
        "--category", "-c",
        choices=VALID_CATEGORIES,
        help="New category"
    )
    update_parser.add_argument("--description", "-d", help="New description")
    update_parser.add_argument(
        "--retrospective", "--notes", "-r", "-n",
        dest="retrospective",
        help="Retrospective notes"
    )

    # task delete
    delete_parser = task_subparsers.add_parser("delete", help="Delete a task (transactional)")
    delete_parser.add_argument("task_id", help="Task ID to delete")
    delete_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force delete even if task has dependencies or is in progress"
    )

    # task history - view modification history for audit/recovery
    history_parser = task_subparsers.add_parser("history", help="View task modification history")
    history_parser.add_argument("task_id", help="Task ID to view history for")
    history_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Number of history entries to show (default: 10)"
    )
    history_parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output history as JSON for programmatic access"
    )
    history_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full details including description and properties"
    )

    # task depends
    depends_parser = task_subparsers.add_parser("depends", help="Create task dependency")
    depends_parser.add_argument("task_id", help="Task that depends on another")
    depends_parser.add_argument(
        "--on",
        dest="depends_on_id",
        required=True,
        help="Task ID to depend on"
    )

    # task import
    import_parser = task_subparsers.add_parser("import", help="Import tasks from YAML or JSON file")
    import_parser.add_argument("file", help="Path to YAML or JSON file")
    import_parser.add_argument(
        "--sprint", "-s",
        help="Sprint ID to add all imported tasks to"
    )


def handle_task_command(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Route task subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: GoTProjectManager instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'task_command') or args.task_command is None:
        print("Error: No task subcommand specified. Use 'got task --help' for usage.")
        return 1

    command_handlers = {
        "create": cmd_task_create,
        "list": cmd_task_list,
        "show": cmd_task_show,
        "next": cmd_task_next,
        "start": cmd_task_start,
        "complete": cmd_task_complete,
        "block": cmd_task_block,
        "update": cmd_task_update,
        "delete": cmd_task_delete,
        "history": cmd_task_history,
        "depends": cmd_task_depends,
        "import": cmd_task_import,
    }

    handler = command_handlers.get(args.task_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown task subcommand: {args.task_command}")
    return 1
