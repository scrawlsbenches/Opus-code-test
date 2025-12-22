#!/usr/bin/env python3
"""
Sprint Reasoning Graph Runner

Connects the Graph of Thought (GoT) task system with the Reasoning Graph
to provide structured reasoning during sprint execution.

Usage:
    python scripts/run_sprint_reasoning.py --sprint sprint-020-forensic-remediation
    python scripts/run_sprint_reasoning.py --list-sprints
    python scripts/run_sprint_reasoning.py --show-graph

The script:
1. Loads sprint tasks from GoT
2. Creates a ThoughtGraph representing the sprint goals
3. Uses ReasoningWorkflow to structure execution
4. Tracks progress through QAPV phases (Question, Answer, Produce, Verify)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.reasoning import (
    ThoughtGraph,
    NodeType,
    EdgeType,
    ReasoningWorkflow,
    CognitiveLoop,
    LoopPhase,
    create_feature_graph,
)


def load_sprint_from_markdown(sprint_id: str) -> Optional[Dict[str, Any]]:
    """Load sprint details from CURRENT_SPRINT.md."""
    sprint_file = Path(__file__).parent.parent / "tasks" / "CURRENT_SPRINT.md"

    if not sprint_file.exists():
        print(f"Error: Sprint file not found: {sprint_file}")
        return None

    content = sprint_file.read_text()

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

    # Extract GoT task IDs
    task_pattern = r"(T-\d{8}-\d{6}-[a-f0-9]{8})"
    sprint["got_tasks"] = re.findall(task_pattern, sprint_section)

    return sprint


def load_got_tasks(task_ids: List[str]) -> List[Dict[str, Any]]:
    """Load task details from GoT system."""
    tasks = []

    # Import GoT utilities
    try:
        from scripts.got_utils import create_manager
        manager = create_manager()

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
                print(f"Warning: Could not load task {task_id}: {e}")
                tasks.append({
                    "id": task_id,
                    "title": f"Task {task_id}",
                    "status": "unknown",
                    "priority": "medium",
                })
    except ImportError as e:
        print(f"Warning: Could not import GoT manager: {e}")
        # Fallback: just use IDs
        for task_id in task_ids:
            tasks.append({
                "id": task_id,
                "title": f"Task {task_id}",
                "status": "unknown",
                "priority": "medium",
            })

    return tasks


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
    goal_node_ids = []
    for i, goal in enumerate(sprint["goals"]):
        node_type = NodeType.DECISION if goal["completed"] else NodeType.QUESTION
        goal_node_id = f"goal-{i}"
        graph.add_node(
            node_id=goal_node_id,
            node_type=node_type,
            content=goal["text"],
            properties={
                "goal_index": i,
                "completed": goal["completed"],
            },
        )
        goal_node_ids.append(goal_node_id)

        # Connect goal to sprint
        graph.add_edge(
            from_id=sprint_node_id,
            to_id=goal_node_id,
            edge_type=EdgeType.DEPENDS_ON,
        )

    # Create nodes for each GoT task
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

        # Connect task to sprint
        graph.add_edge(
            from_id=sprint_node_id,
            to_id=task_node_id,
            edge_type=EdgeType.SUPPORTS,
        )

    # Create dependency edges between related tasks
    # (This is a simple heuristic - could be enhanced)
    if len(task_node_ids) > 1:
        for i in range(len(task_node_ids) - 1):
            # Sequential dependency assumption
            graph.add_edge(
                from_id=task_node_ids[i],
                to_id=task_node_ids[i + 1],
                edge_type=EdgeType.PRECEDES,
            )

    return graph


def run_reasoning_session(sprint: Dict[str, Any], graph: ThoughtGraph) -> None:
    """Run a reasoning session for the sprint using QAPV cycle."""
    print("\n" + "=" * 60)
    print("REASONING SESSION")
    print("=" * 60)

    workflow = ReasoningWorkflow()

    # Start session
    ctx = workflow.start_session(f"Execute sprint: {sprint['id']}")
    print(f"\nSession started: {ctx.session_id if hasattr(ctx, 'session_id') else 'active'}")

    # QUESTION PHASE: Identify what needs to be done
    print("\n--- QUESTION PHASE ---")
    incomplete_goals = [g for g in sprint["goals"] if not g["completed"]]

    if not incomplete_goals:
        print("All goals are complete! Sprint finished.")
        return

    for goal in incomplete_goals:
        print(f"  ❓ {goal['text']}")

    # ANSWER PHASE: Plan the approach
    print("\n--- ANSWER PHASE ---")
    print("  Analysis: Breaking down goals into actionable tasks")

    task_count = len(sprint.get("got_tasks", []))
    print(f"  📋 {task_count} tasks identified in GoT")
    print(f"  📊 {len(incomplete_goals)} goals remaining")

    # Show recommended execution order
    print("\n  Recommended execution order:")
    priorities = {"high": 1, "medium": 2, "low": 3}

    # Just show the incomplete goals as the plan
    for i, goal in enumerate(incomplete_goals, 1):
        print(f"    {i}. {goal['text']}")

    # PRODUCE PHASE: Show what actions to take
    print("\n--- PRODUCE PHASE ---")
    print("  Ready to execute. Run the following to start work:")
    print(f"\n  1. Switch to sprint branch:")
    print(f"     git checkout -b claude/{sprint['id']}-$(date +%s)")
    print(f"\n  2. Start first task:")
    if sprint.get("got_tasks"):
        first_task = sprint["got_tasks"][0]
        print(f"     python scripts/got_utils.py task start {first_task}")

    # VERIFY PHASE: Define success criteria
    print("\n--- VERIFY PHASE ---")
    print("  Verification checklist:")
    print("    □ All goals marked complete in CURRENT_SPRINT.md")
    print("    □ All GoT tasks marked complete")
    print("    □ Tests pass: python -m pytest tests/ -q")
    print("    □ Coverage maintained: 88%+")
    print("    □ GoT validation healthy: python scripts/got_utils.py validate")


def show_graph_structure(graph: ThoughtGraph) -> None:
    """Display the graph structure in ASCII format."""
    print("\n" + "=" * 60)
    print("THOUGHT GRAPH STRUCTURE")
    print("=" * 60)

    # Get all nodes using .nodes property (dict) and .edges (list)
    nodes = list(graph.nodes.values()) if hasattr(graph, 'nodes') else []
    edges = graph.edges if hasattr(graph, 'edges') else []

    print(f"\nNodes: {len(nodes)}")
    for node in nodes[:20]:  # Limit output
        node_type = node.node_type.name if hasattr(node.node_type, 'name') else str(node.node_type)
        content = node.content[:50] if len(node.content) > 50 else node.content
        print(f"  [{node_type}] {content}")

    if len(nodes) > 20:
        print(f"  ... and {len(nodes) - 20} more nodes")

    print(f"\nEdges: {len(edges)}")
    for edge in edges[:15]:  # Limit output
        edge_type = edge.edge_type.name if hasattr(edge.edge_type, 'name') else str(edge.edge_type)
        source = edge.source_id[:15] if len(edge.source_id) > 15 else edge.source_id
        target = edge.target_id[:15] if len(edge.target_id) > 15 else edge.target_id
        print(f"  {source} --{edge_type}--> {target}")

    if len(edges) > 15:
        print(f"  ... and {len(edges) - 15} more edges")

    # Show ASCII representation if available
    if hasattr(graph, 'to_ascii') and len(nodes) > 0:
        print("\n--- ASCII Graph ---")
        try:
            print(graph.to_ascii())
        except Exception as e:
            print(f"  (Could not generate ASCII: {e})")


def list_available_sprints() -> None:
    """List all available sprints from CURRENT_SPRINT.md."""
    sprint_file = Path(__file__).parent.parent / "tasks" / "CURRENT_SPRINT.md"

    if not sprint_file.exists():
        print(f"Error: Sprint file not found: {sprint_file}")
        return

    content = sprint_file.read_text()

    # Find all sprint sections
    pattern = r"## Sprint (\d+): (.+)\n\*\*Sprint ID:\*\* (sprint-\d+-[a-z-]+)\n.*?\*\*Status:\*\* (.+)"

    print("\n" + "=" * 60)
    print("AVAILABLE SPRINTS")
    print("=" * 60)

    for match in re.finditer(pattern, content):
        number, name, sprint_id, status = match.groups()
        status_emoji = "✅" if "Complete" in status else "🟢" if "Available" in status else "🟡"
        print(f"  {status_emoji} Sprint {number}: {name}")
        print(f"     ID: {sprint_id}")
        print(f"     Status: {status.strip()}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run reasoning graph for a sprint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--sprint",
        help="Sprint ID to execute (e.g., sprint-020-forensic-remediation)",
    )
    parser.add_argument(
        "--list-sprints",
        action="store_true",
        help="List all available sprints",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Show the thought graph structure",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

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

    # Load GoT tasks
    tasks = load_got_tasks(sprint.get("got_tasks", []))
    print(f"Loaded {len(tasks)} tasks from GoT")

    # Create thought graph
    graph = create_sprint_graph(sprint, tasks)

    if args.show_graph:
        show_graph_structure(graph)

    if args.json:
        output = {
            "sprint": sprint,
            "tasks": tasks,
            "graph": {
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
            },
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        # Run reasoning session
        run_reasoning_session(sprint, graph)

    print("\n" + "=" * 60)
    print("SESSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
