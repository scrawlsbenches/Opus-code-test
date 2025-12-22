#!/usr/bin/env python3
"""
Sprint Reasoning Graph Runner

Connects the Graph of Thought (GoT) task system with the Reasoning Graph
to provide structured reasoning during sprint execution.

Usage:
    # Basic reasoning session
    python scripts/run_sprint_reasoning.py --sprint sprint-020-forensic-remediation

    # Generate sub-agent configs
    python scripts/run_sprint_reasoning.py --sprint sprint-020-forensic-remediation --spawn-agents

    # Export configs for later use
    python scripts/run_sprint_reasoning.py --sprint sprint-020-forensic-remediation --export-configs

    # Run a single task
    python scripts/run_sprint_reasoning.py --sprint sprint-020-forensic-remediation --task 1

    # Check sprint progress
    python scripts/run_sprint_reasoning.py --sprint sprint-020-forensic-remediation --status

The script:
1. Loads sprint tasks from GoT or CURRENT_SPRINT.md
2. Creates a ThoughtGraph representing the sprint goals
3. Generates rich context for each task (relevant files, patterns)
4. Tracks progress through QAPV phases (Question, Answer, Produce, Verify)
5. Supports sub-agent spawning with work boundaries
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.reasoning import (
    ThoughtGraph,
    NodeType,
    EdgeType,
    ReasoningWorkflow,
    CognitiveLoop,
    LoopPhase,
    create_feature_graph,
    # Sub-agent spawning and coordination
    ClaudeCodeSpawner,
    TaskToolConfig,
    generate_parallel_task_calls,
    # Communication primitives
    PubSubBroker,
    ContextPool,
    CollaborationManager,
    ParallelWorkBoundary,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SPRINT_FILE = PROJECT_ROOT / "tasks" / "CURRENT_SPRINT.md"
PROGRESS_DIR = PROJECT_ROOT / ".sprint-progress"
CONFIGS_DIR = PROJECT_ROOT / ".sprint-configs"

# Task context - maps task patterns to relevant files and search patterns
TASK_CONTEXT_MAP = {
    "id generation": {
        "files": [
            "cortical/utils/id_generation.py",
            "scripts/orchestration_utils.py",
            "scripts/new_memory.py",
        ],
        "search_patterns": ["uuid.uuid4", "secrets.token_hex", "generate.*id"],
        "description": "Migrate UUID-based ID generation to use canonical secrets-based module",
    },
    "wal": {
        "files": [
            "cortical/wal.py",
            "cortical/got/wal.py",
            "cortical/reasoning/graph_persistence.py",
        ],
        "search_patterns": ["WALWriter", "WALReader", "write_ahead_log"],
        "description": "Consolidate WAL implementations into a single module with adapters",
    },
    "checksum": {
        "files": [
            "cortical/wal.py",
            "cortical/got/checksums.py",
            "cortical/reasoning/graph_persistence.py",
        ],
        "search_patterns": ["sha256", "hashlib", "checksum"],
        "description": "Extract checksum utilities to cortical/utils/checksums.py",
    },
    "query": {
        "files": [
            "cortical/query/search.py",
            "cortical/query/ranking.py",
            "cortical/query/chunking.py",
        ],
        "search_patterns": ["tf.*idf", "score.*doc", "rank"],
        "description": "Create query/utils.py with shared TF-IDF scoring helper",
    },
    "atomic": {
        "files": [
            "scripts/task_utils.py",
            "scripts/orchestration_utils.py",
        ],
        "search_patterns": ["atomic.*save", "temp.*file", "os.rename"],
        "description": "Extract atomic save pattern to cortical/utils/persistence.py",
    },
    "slugify": {
        "files": [
            "scripts/task_utils.py",
            "scripts/new_memory.py",
        ],
        "search_patterns": ["slugify", "re.sub.*[^a-z]"],
        "description": "Extract slugify utility to cortical/utils/text.py",
    },
}


# =============================================================================
# SPRINT LOADING
# =============================================================================

def load_sprint_from_markdown(sprint_id: str) -> Optional[Dict[str, Any]]:
    """Load sprint details from CURRENT_SPRINT.md."""
    if not SPRINT_FILE.exists():
        print(f"Error: Sprint file not found: {SPRINT_FILE}")
        return None

    content = SPRINT_FILE.read_text()

    # Find the sprint section
    pattern = rf"## Sprint \d+:.*?\n\*\*Sprint ID:\*\* {re.escape(sprint_id)}.*?(?=\n---|\n## Sprint|\Z)"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print(f"Error: Sprint '{sprint_id}' not found in CURRENT_SPRINT.md")
        return None

    sprint_section = match.group(0)

    # Extract sprint details
    sprint = {
        "id": sprint_id,
        "section": sprint_section,
        "goals": [],
        "got_tasks": [],
        "key_files_new": [],
        "key_files_modify": [],
    }

    # Extract goals (checkbox items)
    goal_pattern = r"- \[([ x])\] (.+)"
    for match in re.finditer(goal_pattern, sprint_section):
        completed = match.group(1) == "x"
        goal_text = match.group(2).strip()
        sprint["goals"].append({
            "text": goal_text,
            "completed": completed,
        })

    # Extract GoT task IDs with their descriptions
    task_pattern = r"- (T-\d{8}-\d{6}-[a-f0-9]{8}): (.+)"
    sprint["got_tasks"] = []
    for match in re.finditer(task_pattern, sprint_section):
        task_id, title = match.groups()
        sprint["got_tasks"].append({
            "id": task_id,
            "title": title.strip(),
        })

    return sprint


def load_got_tasks(task_ids: List[str]) -> List[Dict[str, Any]]:
    """Load task details from GoT system."""
    tasks = []

    try:
        from scripts.got_utils import GoTBackendFactory
        manager = GoTBackendFactory.create()

        for task_id in task_ids:
            try:
                task = manager.get_task(task_id)
                if task:
                    tasks.append({
                        "id": task_id,
                        "title": task.title if hasattr(task, 'title') else str(task),
                        "status": task.status if hasattr(task, 'status') else "unknown",
                        "priority": task.priority if hasattr(task, 'priority') else "medium",
                    })
            except Exception as e:
                tasks.append({
                    "id": task_id,
                    "title": f"Task {task_id}",
                    "status": "unknown",
                    "priority": "medium",
                })
    except ImportError:
        for task_id in task_ids:
            tasks.append({
                "id": task_id,
                "title": f"Task {task_id}",
                "status": "unknown",
                "priority": "medium",
            })

    return tasks


# =============================================================================
# TASK CONTEXT ENRICHMENT
# =============================================================================

def get_task_context(task: Dict[str, Any]) -> Dict[str, Any]:
    """Get rich context for a task including relevant files and patterns."""
    title_lower = task["title"].lower()
    context = {
        "files": [],
        "search_patterns": [],
        "description": "",
        "existing_code": [],
        "related_tests": [],
    }

    # Match against known patterns
    for pattern_key, pattern_context in TASK_CONTEXT_MAP.items():
        if pattern_key in title_lower:
            context["files"].extend(pattern_context["files"])
            context["search_patterns"].extend(pattern_context["search_patterns"])
            context["description"] = pattern_context["description"]
            break

    # Find existing code snippets
    for filepath in context["files"]:
        full_path = PROJECT_ROOT / filepath
        if full_path.exists():
            context["existing_code"].append({
                "file": filepath,
                "exists": True,
                "size": full_path.stat().st_size,
            })
        else:
            context["existing_code"].append({
                "file": filepath,
                "exists": False,
                "note": "File to be created",
            })

    # Find related test files
    for filepath in context["files"]:
        test_name = f"test_{Path(filepath).stem}.py"
        test_paths = list(PROJECT_ROOT.glob(f"tests/**/{test_name}"))
        context["related_tests"].extend([str(p.relative_to(PROJECT_ROOT)) for p in test_paths])

    return context


def search_codebase(pattern: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search codebase for a pattern and return matches."""
    results = []
    try:
        cmd = ["grep", "-rn", "--include=*.py", pattern, str(PROJECT_ROOT / "cortical"), str(PROJECT_ROOT / "scripts")]
        output = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        for line in output.stdout.strip().split("\n")[:max_results]:
            if line:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    results.append({
                        "file": parts[0].replace(str(PROJECT_ROOT) + "/", ""),
                        "line": int(parts[1]),
                        "content": parts[2].strip()[:100],
                    })
    except Exception:
        pass
    return results


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

def get_progress_file(sprint_id: str) -> Path:
    """Get the progress file path for a sprint."""
    PROGRESS_DIR.mkdir(exist_ok=True)
    return PROGRESS_DIR / f"{sprint_id}.json"


def load_progress(sprint_id: str) -> Dict[str, Any]:
    """Load progress for a sprint."""
    progress_file = get_progress_file(sprint_id)
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {
        "sprint_id": sprint_id,
        "started_at": None,
        "tasks": {},
        "completed_count": 0,
        "total_count": 0,
    }


def save_progress(sprint_id: str, progress: Dict[str, Any]) -> None:
    """Save progress for a sprint."""
    progress_file = get_progress_file(sprint_id)
    progress["updated_at"] = datetime.now().isoformat()
    progress_file.write_text(json.dumps(progress, indent=2))


def update_task_progress(sprint_id: str, task_id: str, status: str, notes: str = "") -> None:
    """Update progress for a specific task."""
    progress = load_progress(sprint_id)
    progress["tasks"][task_id] = {
        "status": status,
        "notes": notes,
        "updated_at": datetime.now().isoformat(),
    }
    progress["completed_count"] = sum(1 for t in progress["tasks"].values() if t["status"] == "completed")
    save_progress(sprint_id, progress)


# =============================================================================
# THOUGHT GRAPH
# =============================================================================

def create_sprint_graph(sprint: Dict[str, Any], tasks: List[Dict[str, Any]]) -> ThoughtGraph:
    """Create a ThoughtGraph representing the sprint structure."""
    graph = ThoughtGraph()

    # Create root node for the sprint
    sprint_node_id = f"sprint-{sprint['id']}"
    graph.add_node(
        node_id=sprint_node_id,
        node_type=NodeType.GOAL,
        content=f"Sprint: {sprint['id']}",
        properties={"sprint_id": sprint["id"]},
    )

    # Create nodes for each goal
    for i, goal in enumerate(sprint["goals"]):
        node_type = NodeType.DECISION if goal["completed"] else NodeType.QUESTION
        goal_node_id = f"goal-{i}"
        graph.add_node(
            node_id=goal_node_id,
            node_type=node_type,
            content=goal["text"],
            properties={"goal_index": i, "completed": goal["completed"]},
        )
        graph.add_edge(
            from_id=sprint_node_id,
            to_id=goal_node_id,
            edge_type=EdgeType.DEPENDS_ON,
        )

    # Create nodes for each task
    task_node_ids = []
    for task in tasks:
        task_node_id = f"task-{task['id']}"
        graph.add_node(
            node_id=task_node_id,
            node_type=NodeType.HYPOTHESIS,
            content=task["title"],
            properties={
                "task_id": task["id"],
                "status": task["status"],
                "priority": task["priority"],
            },
        )
        task_node_ids.append(task_node_id)
        graph.add_edge(
            from_id=sprint_node_id,
            to_id=task_node_id,
            edge_type=EdgeType.SUPPORTS,
        )

    # Create dependency edges between tasks
    if len(task_node_ids) > 1:
        for i in range(len(task_node_ids) - 1):
            graph.add_edge(
                from_id=task_node_ids[i],
                to_id=task_node_ids[i + 1],
                edge_type=EdgeType.PRECEDES,
            )

    return graph


# =============================================================================
# SUB-AGENT CONFIGURATION
# =============================================================================

def setup_agent_communication() -> Dict[str, Any]:
    """Set up inter-agent communication infrastructure."""
    broker = PubSubBroker()
    pool = ContextPool()
    collab = CollaborationManager()

    topics = [
        "task.started",
        "task.completed",
        "task.blocked",
        "discovery.code",
        "discovery.issue",
        "discovery.pattern",
    ]

    return {
        "broker": broker,
        "pool": pool,
        "collaboration": collab,
        "topics": topics,
    }


def create_work_boundaries(tasks: List[Dict[str, Any]]) -> List[ParallelWorkBoundary]:
    """Create work boundaries to prevent agent conflicts."""
    boundaries = []

    for task in tasks:
        context = get_task_context(task)
        if context["files"]:
            boundary = ParallelWorkBoundary(
                agent_id=f"agent-{task['id']}",
                scope_description=task["title"],
                files_owned=set(context["files"]),
            )
            boundaries.append(boundary)

    return boundaries


def generate_rich_prompt(task: Dict[str, Any], sprint: Dict[str, Any], position: int, total: int) -> str:
    """Generate a rich, context-aware prompt for a task."""
    context = get_task_context(task)

    prompt = f"""# Sprint Task: {task['title']}

## Task Identity
- **Task ID:** {task['id']}
- **Sprint:** {sprint['id']}
- **Position:** {position} of {total}
- **Priority:** {task.get('priority', 'medium')}

## Objective
{context['description'] or task['title']}

## Relevant Files
"""

    for file_info in context["existing_code"]:
        if file_info["exists"]:
            prompt += f"- `{file_info['file']}` ({file_info['size']} bytes) - Modify\n"
        else:
            prompt += f"- `{file_info['file']}` - Create new\n"

    if context["related_tests"]:
        prompt += "\n## Related Tests\n"
        for test_file in context["related_tests"]:
            prompt += f"- `{test_file}`\n"

    if context["search_patterns"]:
        prompt += "\n## Code Patterns to Look For\n"
        for pattern in context["search_patterns"]:
            prompt += f"- `{pattern}`\n"

    prompt += f"""
## Implementation Steps
1. Read the relevant files to understand current implementation
2. Implement the changes described above
3. Update or add tests as needed
4. Run tests: `python -m pytest tests/ -q`
5. Verify no regressions

## Completion
When done, run:
```bash
python scripts/got_utils.py task complete {task['id']}
```

## Notes
- Follow existing code patterns and style
- Add type hints to new code
- Keep changes focused on this task only
- Commit with message: `refactor: {task['title'].lower()}`
"""

    return prompt


def generate_agent_configs(
    sprint: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    graph: ThoughtGraph,
) -> List[TaskToolConfig]:
    """Generate Task tool configurations for sub-agents."""
    configs = []
    boundaries = create_work_boundaries(tasks)
    boundary_map = {b.agent_id: b for b in boundaries}

    for i, task in enumerate(tasks):
        agent_id = f"agent-{task['id']}"
        prompt = generate_rich_prompt(task, sprint, i + 1, len(tasks))

        config = TaskToolConfig(
            agent_id=agent_id,
            description=f"{task['title'][:40]}",
            prompt=prompt,
            subagent_type="general-purpose",
            boundary=boundary_map.get(agent_id),
        )
        configs.append(config)

    return configs


def export_configs(configs: List[TaskToolConfig], sprint_id: str) -> Path:
    """Export agent configs to a JSON file."""
    CONFIGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = CONFIGS_DIR / f"{sprint_id}-{timestamp}.json"

    export_data = {
        "sprint_id": sprint_id,
        "generated_at": datetime.now().isoformat(),
        "agents": [],
    }

    for config in configs:
        export_data["agents"].append({
            "agent_id": config.agent_id,
            "description": config.description,
            "subagent_type": config.subagent_type,
            "prompt": config.prompt,
            "boundary": {
                "files_owned": list(config.boundary.files_owned) if config.boundary else [],
            } if config.boundary else None,
        })

    output_file.write_text(json.dumps(export_data, indent=2))
    return output_file


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_status(sprint: Dict[str, Any], tasks: List[Dict[str, Any]]) -> None:
    """Display sprint status with progress tracking."""
    progress = load_progress(sprint["id"])

    print("\n" + "=" * 60)
    print(f"SPRINT STATUS: {sprint['id']}")
    print("=" * 60)

    # Goals
    completed_goals = sum(1 for g in sprint["goals"] if g["completed"])
    print(f"\nGoals: {completed_goals}/{len(sprint['goals'])} complete")
    for goal in sprint["goals"]:
        status = "✅" if goal["completed"] else "⬜"
        print(f"  {status} {goal['text']}")

    # Tasks
    print(f"\nTasks: {progress.get('completed_count', 0)}/{len(tasks)} complete")
    for task in tasks:
        task_progress = progress.get("tasks", {}).get(task["id"], {})
        status = task_progress.get("status", "pending")
        emoji = {"completed": "✅", "in_progress": "🔄", "blocked": "🚫"}.get(status, "⬜")
        print(f"  {emoji} {task['title'][:50]}")
        if task_progress.get("notes"):
            print(f"      Note: {task_progress['notes'][:60]}")

    print()


def display_agent_configs(configs: List[TaskToolConfig], verbose: bool = False) -> None:
    """Display generated agent configurations."""
    print("\n" + "=" * 60)
    print("SUB-AGENT CONFIGURATIONS")
    print("=" * 60)

    print(f"\nGenerated {len(configs)} agent configurations.\n")

    for i, config in enumerate(configs, 1):
        print(f"--- Agent {i}: {config.description} ---")
        print(f"  ID: {config.agent_id}")
        print(f"  Type: {config.subagent_type}")
        if config.boundary:
            print(f"  Files: {', '.join(sorted(config.boundary.files_owned))}")

        if verbose:
            print(f"\n  Prompt:\n{config.prompt[:500]}...")
        print()


def run_reasoning_session(sprint: Dict[str, Any], graph: ThoughtGraph) -> None:
    """Run a reasoning session for the sprint using QAPV cycle."""
    print("\n" + "=" * 60)
    print("REASONING SESSION")
    print("=" * 60)

    workflow = ReasoningWorkflow()
    ctx = workflow.start_session(f"Execute sprint: {sprint['id']}")
    print(f"\nSession started: {ctx.session_id if hasattr(ctx, 'session_id') else 'active'}")

    # QUESTION PHASE
    print("\n--- QUESTION PHASE ---")
    incomplete_goals = [g for g in sprint["goals"] if not g["completed"]]

    if not incomplete_goals:
        print("All goals are complete! Sprint finished.")
        return

    for goal in incomplete_goals:
        print(f"  ❓ {goal['text']}")

    # ANSWER PHASE
    print("\n--- ANSWER PHASE ---")
    task_count = len(sprint.get("got_tasks", []))
    print(f"  📋 {task_count} tasks identified")
    print(f"  📊 {len(incomplete_goals)} goals remaining")

    print("\n  Recommended execution order:")
    for i, goal in enumerate(incomplete_goals, 1):
        print(f"    {i}. {goal['text']}")

    # PRODUCE PHASE
    print("\n--- PRODUCE PHASE ---")
    print("  Options:")
    print("    1. Run with --spawn-agents to generate sub-agent configs")
    print("    2. Run with --task N to execute a single task")
    print("    3. Run with --export-configs to save configs for later")

    # VERIFY PHASE
    print("\n--- VERIFY PHASE ---")
    print("  Verification checklist:")
    print("    □ python -m pytest tests/ -q")
    print("    □ python scripts/got_utils.py validate")
    print("    □ All goals marked complete in CURRENT_SPRINT.md")


def show_graph_structure(graph: ThoughtGraph) -> None:
    """Display the graph structure."""
    print("\n" + "=" * 60)
    print("THOUGHT GRAPH STRUCTURE")
    print("=" * 60)

    nodes = list(graph.nodes.values()) if hasattr(graph, 'nodes') else []
    edges = graph.edges if hasattr(graph, 'edges') else []

    print(f"\nNodes: {len(nodes)}")
    for node in nodes[:15]:
        node_type = node.node_type.name if hasattr(node.node_type, 'name') else str(node.node_type)
        content = node.content[:40] if len(node.content) > 40 else node.content
        print(f"  [{node_type}] {content}")

    print(f"\nEdges: {len(edges)}")
    for edge in edges[:10]:
        edge_type = edge.edge_type.name if hasattr(edge.edge_type, 'name') else str(edge.edge_type)
        print(f"  {edge.source_id[:12]} --{edge_type}--> {edge.target_id[:12]}")


def list_available_sprints() -> None:
    """List all available sprints."""
    if not SPRINT_FILE.exists():
        print(f"Error: Sprint file not found: {SPRINT_FILE}")
        return

    content = SPRINT_FILE.read_text()
    pattern = r"## Sprint (\d+): (.+)\n\*\*Sprint ID:\*\* (sprint-\d+-[a-z-]+)\n.*?\*\*Status:\*\* (.+)"

    print("\n" + "=" * 60)
    print("AVAILABLE SPRINTS")
    print("=" * 60)

    for match in re.finditer(pattern, content):
        number, name, sprint_id, status = match.groups()
        emoji = "✅" if "Complete" in status else "🟢" if "Available" in status else "🟡"
        print(f"\n  {emoji} Sprint {number}: {name}")
        print(f"     ID: {sprint_id}")
        print(f"     Status: {status.strip()}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sprint Reasoning Graph Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--sprint", help="Sprint ID to execute")
    parser.add_argument("--list-sprints", action="store_true", help="List available sprints")
    parser.add_argument("--status", action="store_true", help="Show sprint status")
    parser.add_argument("--show-graph", action="store_true", help="Show thought graph structure")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--spawn-agents", action="store_true", help="Generate sub-agent configs")
    parser.add_argument("--show-boundaries", action="store_true", help="Show work boundaries")
    parser.add_argument("--export-configs", action="store_true", help="Export configs to file")
    parser.add_argument("--task", type=int, help="Show config for specific task (1-indexed)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.list_sprints:
        list_available_sprints()
        return

    if not args.sprint:
        parser.print_help()
        print("\nError: --sprint is required unless using --list-sprints")
        sys.exit(1)

    # Load sprint
    print(f"Loading sprint: {args.sprint}")
    sprint = load_sprint_from_markdown(args.sprint)

    if not sprint:
        sys.exit(1)

    # Load tasks
    sprint_tasks = sprint.get("got_tasks", [])
    if sprint_tasks and isinstance(sprint_tasks[0], dict):
        tasks = [
            {"id": t["id"], "title": t["title"], "status": "pending", "priority": "medium"}
            for t in sprint_tasks
        ]
        print(f"Loaded {len(tasks)} tasks from sprint file")
    else:
        tasks = load_got_tasks(sprint_tasks)
        print(f"Loaded {len(tasks)} tasks from GoT")

    # Create thought graph
    graph = create_sprint_graph(sprint, tasks)

    # Handle different modes
    if args.status:
        display_status(sprint, tasks)

    elif args.show_graph:
        show_graph_structure(graph)

    elif args.task:
        if 1 <= args.task <= len(tasks):
            task = tasks[args.task - 1]
            prompt = generate_rich_prompt(task, sprint, args.task, len(tasks))
            print(f"\n{'='*60}")
            print(f"TASK {args.task}: {task['title']}")
            print("=" * 60)
            print(prompt)
        else:
            print(f"Error: Task {args.task} not found (valid: 1-{len(tasks)})")
            sys.exit(1)

    elif args.spawn_agents or args.export_configs:
        print("\n" + "=" * 60)
        print("SETTING UP SUB-AGENT COORDINATION")
        print("=" * 60)

        comm = setup_agent_communication()
        print(f"\nCommunication: {', '.join(comm['topics'])}")

        configs = generate_agent_configs(sprint, tasks, graph)

        if args.show_boundaries:
            boundaries = create_work_boundaries(tasks)
            print(f"\n--- Work Boundaries ({len(boundaries)}) ---")
            for b in boundaries:
                print(f"  {b.agent_id}: {', '.join(sorted(b.files_owned))}")

        display_agent_configs(configs, verbose=args.verbose)

        if args.export_configs:
            output_file = export_configs(configs, sprint["id"])
            print(f"\nConfigs exported to: {output_file}")

    elif args.json:
        output = {
            "sprint": sprint,
            "tasks": tasks,
            "graph": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
        }
        print(json.dumps(output, indent=2, default=str))

    else:
        run_reasoning_session(sprint, graph)

    print("\n" + "=" * 60)
    print("SESSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
