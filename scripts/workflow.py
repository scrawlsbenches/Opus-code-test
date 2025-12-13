#!/usr/bin/env python3
"""
Workflow template engine for task creation.

Spawns multiple linked tasks from YAML workflow templates.

Usage:
    # List available workflows
    python scripts/workflow.py list

    # Run a workflow
    python scripts/workflow.py run bugfix --bug_title "Login crashes on special chars"

    # Run with all options
    python scripts/workflow.py run feature \\
        --feature_name "Dark mode" \\
        --priority high \\
        --effort large

    # Dry run (show tasks without creating)
    python scripts/workflow.py run bugfix --bug_title "Test" --dry-run
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import yaml, fall back to basic parsing if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from task_utils import TaskSession, Task


# Default workflows directory
WORKFLOWS_DIR = Path(__file__).parent.parent / ".claude" / "workflows"


@dataclass
class WorkflowVariable:
    """A variable in a workflow template."""
    name: str
    description: str
    required: bool = True
    default: Optional[str] = None
    choices: Optional[List[str]] = None


@dataclass
class WorkflowTask:
    """A task template in a workflow."""
    id: str
    title: str
    category: str = "general"
    priority: str = "medium"
    effort: str = "medium"
    description: str = ""
    depends_on: List[str] = None

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


@dataclass
class Workflow:
    """A workflow template that creates multiple tasks."""
    name: str
    description: str
    category: str
    variables: List[WorkflowVariable]
    tasks: List[WorkflowTask]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        """Create workflow from parsed YAML dict."""
        variables = [
            WorkflowVariable(
                name=v['name'],
                description=v.get('description', ''),
                required=v.get('required', True),
                default=v.get('default'),
                choices=v.get('choices')
            )
            for v in data.get('variables', [])
        ]

        tasks = [
            WorkflowTask(
                id=t['id'],
                title=t['title'],
                category=t.get('category', 'general'),
                priority=t.get('priority', 'medium'),
                effort=t.get('effort', 'medium'),
                description=t.get('description', ''),
                depends_on=t.get('depends_on', [])
            )
            for t in data.get('tasks', [])
        ]

        return cls(
            name=data['name'],
            description=data.get('description', ''),
            category=data.get('category', 'general'),
            variables=variables,
            tasks=tasks
        )

    @classmethod
    def load(cls, filepath: Path) -> 'Workflow':
        """Load workflow from YAML file."""
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for workflow templates. "
                "Install with: pip install pyyaml"
            )

        with open(filepath) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)


def list_workflows(workflows_dir: Path = WORKFLOWS_DIR) -> List[Workflow]:
    """List all available workflow templates."""
    workflows = []

    if not workflows_dir.exists():
        return workflows

    for filepath in sorted(workflows_dir.glob("*.yaml")):
        try:
            workflow = Workflow.load(filepath)
            workflows.append(workflow)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")

    return workflows


def substitute_variables(text: str, variables: Dict[str, str]) -> str:
    """Substitute {variable} placeholders in text."""
    result = text
    for name, value in variables.items():
        result = result.replace(f"{{{name}}}", value)
    return result


def run_workflow(
    workflow: Workflow,
    variables: Dict[str, str],
    tasks_dir: str = "tasks",
    dry_run: bool = False
) -> List[Task]:
    """
    Execute a workflow template, creating all tasks.

    Args:
        workflow: The workflow to execute
        variables: Variable values to substitute
        tasks_dir: Directory to save tasks
        dry_run: If True, show tasks but don't create

    Returns:
        List of created Task objects
    """
    # Validate required variables
    for var in workflow.variables:
        if var.required and var.name not in variables:
            if var.default:
                variables[var.name] = var.default
            else:
                raise ValueError(f"Missing required variable: {var.name}")

        # Validate choices
        if var.choices and var.name in variables:
            if variables[var.name] not in var.choices:
                raise ValueError(
                    f"Invalid value for {var.name}: {variables[var.name]}. "
                    f"Must be one of: {var.choices}"
                )

    # Create session
    session = TaskSession()

    # Map workflow task IDs to actual task IDs
    id_mapping: Dict[str, str] = {}

    # Create tasks in order
    created_tasks = []
    for wf_task in workflow.tasks:
        # Substitute variables
        title = substitute_variables(wf_task.title, variables)
        description = substitute_variables(wf_task.description, variables)
        priority = substitute_variables(wf_task.priority, variables)
        effort = substitute_variables(wf_task.effort, variables)

        # Resolve dependencies to actual task IDs
        depends_on = [
            id_mapping[dep_id]
            for dep_id in wf_task.depends_on
            if dep_id in id_mapping
        ]

        # Create task
        task = session.create_task(
            title=title,
            category=wf_task.category,
            priority=priority,
            effort=effort,
            description=description,
            depends_on=depends_on
        )

        id_mapping[wf_task.id] = task.id
        created_tasks.append(task)

    if dry_run:
        print(f"\n[Dry Run] Would create {len(created_tasks)} tasks:\n")
        for task in created_tasks:
            deps = f" (depends on: {len(task.depends_on)})" if task.depends_on else ""
            print(f"  [{task.priority.upper()}] {task.title}{deps}")
            if task.description:
                # Show first line of description
                first_line = task.description.strip().split('\n')[0]
                print(f"           {first_line[:60]}...")
        return created_tasks

    # Save
    filepath = session.save(tasks_dir)
    print(f"\nCreated {len(created_tasks)} tasks from '{workflow.name}' workflow")
    print(f"Saved to: {filepath}\n")

    for task in created_tasks:
        deps = f" (depends on: {len(task.depends_on)})" if task.depends_on else ""
        print(f"  {task.id}: {task.title}{deps}")

    return created_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Workflow template engine for task creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List available workflows")
    list_parser.add_argument(
        "--dir", default=str(WORKFLOWS_DIR),
        help="Workflows directory"
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a workflow")
    run_parser.add_argument("workflow", help="Workflow name (e.g., 'bugfix', 'feature')")
    run_parser.add_argument("--dry-run", action="store_true", help="Show tasks without creating")
    run_parser.add_argument("--tasks-dir", default="tasks", help="Tasks directory")

    # Parse known args first to get workflow name
    args, remaining = parser.parse_known_args()

    if args.command == "list":
        workflows = list_workflows(Path(args.dir))
        if not workflows:
            print("No workflows found.")
            print(f"Add .yaml files to: {args.dir}")
            return

        print("\nAvailable Workflows:\n")
        for wf in workflows:
            print(f"  {wf.name.lower().replace(' ', '_')}")
            print(f"    {wf.description}")
            print(f"    Tasks: {len(wf.tasks)}")
            if wf.variables:
                var_names = [v.name for v in wf.variables]
                print(f"    Variables: {', '.join(var_names)}")
            print()

    elif args.command == "run":
        # Find workflow file
        workflow_name = args.workflow.lower().replace(' ', '_')
        workflow_path = WORKFLOWS_DIR / f"{workflow_name}.yaml"

        if not workflow_path.exists():
            print(f"Workflow not found: {workflow_name}")
            print(f"Available: {', '.join(p.stem for p in WORKFLOWS_DIR.glob('*.yaml'))}")
            return

        workflow = Workflow.load(workflow_path)

        # Add workflow-specific arguments dynamically
        for var in workflow.variables:
            arg_name = f"--{var.name}"
            help_text = var.description
            if var.default:
                help_text += f" (default: {var.default})"
            if var.choices:
                help_text += f" (choices: {', '.join(var.choices)})"

            run_parser.add_argument(
                arg_name,
                default=var.default,
                required=var.required and not var.default,
                help=help_text
            )

        # Re-parse with dynamic arguments
        args = parser.parse_args()

        # Collect variables
        variables = {}
        for var in workflow.variables:
            value = getattr(args, var.name, None)
            if value:
                variables[var.name] = value

        # Run workflow
        try:
            run_workflow(
                workflow,
                variables,
                tasks_dir=args.tasks_dir,
                dry_run=args.dry_run
            )
        except ValueError as e:
            print(f"Error: {e}")
            return

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
