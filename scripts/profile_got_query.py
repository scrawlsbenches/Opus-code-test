#!/usr/bin/env python3
"""
Profile the GoT Query API components.

Usage:
    python scripts/profile_got_query.py              # Quick profile (default)
    python scripts/profile_got_query.py --full       # Full profile with larger graphs
    python scripts/profile_got_query.py --component query  # Profile specific component
"""

import cProfile
import pstats
import io
import time
import random
import argparse
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.got import GoTManager
from cortical.got.query_builder import Query
from cortical.got.graph_walker import GraphWalker
from cortical.got.path_finder import PathFinder
from cortical.got.pattern_matcher import Pattern, PatternMatcher


# =============================================================================
# Profiling Infrastructure
# =============================================================================

@dataclass
class ProfileResult:
    """Result from a single profiling run."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    ops_per_sec: float

    def __str__(self) -> str:
        return (
            f"{self.name:40} | "
            f"{self.avg_time_ms:8.3f}ms avg | "
            f"{self.min_time_ms:8.3f}ms min | "
            f"{self.max_time_ms:8.3f}ms max | "
            f"{self.ops_per_sec:8.1f} ops/s"
        )


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    times = {"elapsed": 0}
    yield times
    times["elapsed"] = (time.perf_counter() - start) * 1000  # ms


def profile_function(
    func: Callable,
    iterations: int = 100,
    warmup: int = 5,
    name: str = None
) -> ProfileResult:
    """Profile a function over multiple iterations."""
    name = name or func.__name__

    # Warmup runs (not counted)
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(iterations):
        with timer() as t:
            func()
        times.append(t["elapsed"])

    total = sum(times)
    avg = total / iterations

    return ProfileResult(
        name=name,
        iterations=iterations,
        total_time_ms=total,
        avg_time_ms=avg,
        min_time_ms=min(times),
        max_time_ms=max(times),
        ops_per_sec=1000 / avg if avg > 0 else float('inf')
    )


def detailed_profile(func: Callable, name: str = None) -> str:
    """Run cProfile on a function and return formatted stats."""
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    return stream.getvalue()


# =============================================================================
# Test Graph Generation
# =============================================================================

def create_test_graph(
    manager: GoTManager,
    num_tasks: int = 100,
    num_decisions: int = 20,
    edge_density: float = 0.1
) -> Dict[str, List[str]]:
    """
    Create a synthetic test graph for profiling.

    Args:
        manager: GoT manager to populate
        num_tasks: Number of tasks to create
        num_decisions: Number of decisions to create
        edge_density: Probability of edge between any two nodes (0-1)

    Returns:
        Dict with 'tasks' and 'decisions' lists of IDs
    """
    statuses = ["pending", "in_progress", "completed", "blocked"]
    priorities = ["low", "medium", "high", "critical"]
    edge_types = ["DEPENDS_ON", "BLOCKS", "RELATES_TO", "CONTAINS"]

    task_ids = []
    decision_ids = []

    # Create tasks
    for i in range(num_tasks):
        status = random.choice(statuses)
        task = manager.create_task(
            title=f"Task {i}: {random.choice(['Implement', 'Fix', 'Test', 'Review', 'Document'])} feature {i}",
            priority=random.choice(priorities),
            status=status,
            description=f"Description for task {i} with some content to search through."
        )
        task_ids.append(task.id)

    # Create decisions
    for i in range(num_decisions):
        decision = manager.log_decision(
            title=f"Decision {i}: Choose approach {random.choice(['A', 'B', 'C'])}",
            rationale=f"Rationale for decision {i}",
        )
        decision_ids.append(decision.id)

    # Create edges based on density
    all_ids = task_ids + decision_ids
    for i, source_id in enumerate(all_ids):
        for target_id in all_ids[i+1:]:
            if random.random() < edge_density:
                edge_type = random.choice(edge_types)
                try:
                    manager.add_edge(source_id, target_id, edge_type)
                except Exception:
                    pass  # Skip if edge already exists or invalid

    return {"tasks": task_ids, "decisions": decision_ids}


# =============================================================================
# Query API Profiling
# =============================================================================

def profile_query_api(manager: GoTManager, graph_info: Dict) -> List[ProfileResult]:
    """Profile various Query API operations."""
    results = []

    # Basic query - all tasks
    results.append(profile_function(
        lambda: list(Query(manager).tasks().execute()),
        iterations=100,
        name="Query: All tasks"
    ))

    # Filtered query - by status
    results.append(profile_function(
        lambda: list(Query(manager).tasks().where(status="completed").execute()),
        iterations=100,
        name="Query: Filter by status"
    ))

    # Filtered query - by priority
    results.append(profile_function(
        lambda: list(Query(manager).tasks().where(priority="high").execute()),
        iterations=100,
        name="Query: Filter by priority"
    ))

    # OR conditions
    results.append(profile_function(
        lambda: list(Query(manager).tasks()
            .where(status="pending")
            .or_where(status="in_progress")
            .execute()),
        iterations=100,
        name="Query: OR conditions"
    ))

    # Sorted query
    results.append(profile_function(
        lambda: list(Query(manager).tasks().order_by("priority").execute()),
        iterations=100,
        name="Query: Order by priority"
    ))

    # Paginated query
    results.append(profile_function(
        lambda: list(Query(manager).tasks().limit(10).offset(5).execute()),
        iterations=100,
        name="Query: Pagination (limit 10)"
    ))

    # Count aggregation
    results.append(profile_function(
        lambda: Query(manager).tasks().count(),
        iterations=100,
        name="Query: Count"
    ))

    # Group by with count
    results.append(profile_function(
        lambda: Query(manager).tasks().group_by("status").count().execute(),
        iterations=100,
        name="Query: Group by + count"
    ))

    # Complex query chain
    results.append(profile_function(
        lambda: list(Query(manager).tasks()
            .where(priority="high")
            .or_where(priority="critical")
            .order_by("created_at", desc=True)
            .limit(20)
            .execute()),
        iterations=100,
        name="Query: Complex chain"
    ))

    return results


# =============================================================================
# GraphWalker Profiling
# =============================================================================

def profile_graph_walker(manager: GoTManager, graph_info: Dict) -> List[ProfileResult]:
    """Profile GraphWalker operations."""
    results = []
    task_ids = graph_info["tasks"]

    if not task_ids:
        return results

    start_id = task_ids[0]

    # Collector visitor: accumulates all node IDs
    def collect_ids(node, acc):
        acc.append(node.id)
        return acc

    # BFS traversal
    results.append(profile_function(
        lambda: GraphWalker(manager).starting_from(start_id).bfs()
            .visit(collect_ids, initial=[]).run(),
        iterations=50,
        name="Walker: BFS traversal"
    ))

    # DFS traversal
    results.append(profile_function(
        lambda: GraphWalker(manager).starting_from(start_id).dfs()
            .visit(collect_ids, initial=[]).run(),
        iterations=50,
        name="Walker: DFS traversal"
    ))

    # Limited depth
    results.append(profile_function(
        lambda: GraphWalker(manager).starting_from(start_id).max_depth(2)
            .visit(collect_ids, initial=[]).run(),
        iterations=50,
        name="Walker: Max depth 2"
    ))

    # With filter - count high priority tasks
    def count_high_priority(node, acc):
        if getattr(node, 'priority', None) == 'high':
            return acc + 1
        return acc

    results.append(profile_function(
        lambda: GraphWalker(manager).starting_from(start_id).bfs()
            .visit(count_high_priority, initial=0).run(),
        iterations=50,
        name="Walker: Count with filter"
    ))

    # Visitor pattern - collect titles
    def collect_titles(node, acc):
        title = getattr(node, 'title', str(node.id))
        acc.append(title)
        return acc

    results.append(profile_function(
        lambda: GraphWalker(manager).starting_from(start_id).bfs()
            .visit(collect_titles, initial=[]).run(),
        iterations=50,
        name="Walker: Collect titles"
    ))

    return results


# =============================================================================
# PathFinder Profiling
# =============================================================================

def profile_path_finder(manager: GoTManager, graph_info: Dict) -> List[ProfileResult]:
    """Profile PathFinder operations."""
    results = []
    task_ids = graph_info["tasks"]

    if len(task_ids) < 2:
        return results

    # Pick two tasks that are likely connected
    start_id = task_ids[0]
    end_id = task_ids[min(10, len(task_ids) - 1)]  # Pick one not too far

    finder = PathFinder(manager)

    # Shortest path
    results.append(profile_function(
        lambda: finder.shortest_path(start_id, end_id),
        iterations=50,
        name="PathFinder: Shortest path"
    ))

    # NOTE: all_paths() is O(2^n) on connected graphs - SKIP for profiling
    # It's designed for sparse graphs or when you need exhaustive path enumeration

    # Connected components
    results.append(profile_function(
        lambda: finder.connected_components(),
        iterations=20,
        name="PathFinder: Connected components"
    ))

    # Reachable nodes
    results.append(profile_function(
        lambda: finder.reachable_from(start_id),
        iterations=50,
        name="PathFinder: Reachable from"
    ))

    return results


# =============================================================================
# PatternMatcher Profiling
# =============================================================================

def profile_pattern_matcher(manager: GoTManager, graph_info: Dict) -> List[ProfileResult]:
    """Profile PatternMatcher operations."""
    results = []

    matcher = PatternMatcher(manager)

    # Simple 2-node pattern: a --DEPENDS_ON--> b
    pattern1 = (Pattern()
        .node("a", type="task")
        .edge("DEPENDS_ON", direction="outgoing")
        .node("b", type="task"))

    results.append(profile_function(
        lambda: list(matcher.find(pattern1)),
        iterations=20,
        name="Pattern: 2-node dependency"
    ))

    # Pattern with constraints
    pattern2 = (Pattern()
        .node("a", type="task", status="pending")
        .edge("DEPENDS_ON", direction="outgoing")
        .node("b", type="task", status="completed"))

    results.append(profile_function(
        lambda: list(matcher.find(pattern2)),
        iterations=20,
        name="Pattern: With constraints"
    ))

    # 3-node chain pattern: a -> b -> c
    pattern3 = (Pattern()
        .node("a", type="task")
        .edge("DEPENDS_ON", direction="outgoing")
        .node("b", type="task")
        .edge("DEPENDS_ON", direction="outgoing")
        .node("c", type="task"))

    results.append(profile_function(
        lambda: list(matcher.find(pattern3)),
        iterations=10,  # 3-node patterns are expensive
        name="Pattern: 3-node chain"
    ))

    # Count matches using count() method
    results.append(profile_function(
        lambda: matcher.count(pattern1),
        iterations=20,
        name="Pattern: Count matches"
    ))

    return results


# =============================================================================
# Main Profiling Driver
# =============================================================================

def run_profiling(
    graph_sizes: List[tuple] = None,
    components: List[str] = None,
    detailed: bool = False
):
    """
    Run profiling on all components.

    Args:
        graph_sizes: List of (num_tasks, num_decisions, edge_density) tuples
        components: List of components to profile ('query', 'walker', 'pathfinder', 'pattern')
        detailed: If True, also run cProfile for detailed analysis
    """
    if graph_sizes is None:
        graph_sizes = [(50, 10, 0.05)]  # Quick default

    if components is None:
        components = ['query', 'walker', 'pathfinder', 'pattern']

    print("=" * 80)
    print("GoT QUERY API PROFILING")
    print("=" * 80)

    for num_tasks, num_decisions, edge_density in graph_sizes:
        print(f"\n{'─' * 80}")
        print(f"Graph: {num_tasks} tasks, {num_decisions} decisions, {edge_density:.0%} edge density")
        print(f"{'─' * 80}")

        # Create isolated test environment
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GoTManager(got_dir=tmpdir)

            # Generate test graph
            print("\nGenerating test graph...", end=" ", flush=True)
            with timer() as t:
                graph_info = create_test_graph(
                    manager, num_tasks, num_decisions, edge_density
                )
            print(f"done ({t['elapsed']:.1f}ms)")

            # Count actual edges
            edges = list(manager.list_edges())
            print(f"Created: {len(graph_info['tasks'])} tasks, "
                  f"{len(graph_info['decisions'])} decisions, "
                  f"{len(edges)} edges")

            all_results = []

            # Profile each component
            if 'query' in components:
                print(f"\n{'─' * 40}")
                print("QUERY API")
                print(f"{'─' * 40}")
                results = profile_query_api(manager, graph_info)
                for r in results:
                    print(r)
                all_results.extend(results)

                if detailed:
                    print("\n[Detailed cProfile for complex query]")
                    print(detailed_profile(
                        lambda: list(Query(manager).tasks()
                            .where(priority="high")
                            .order_by("created_at")
                            .execute())
                    ))

            if 'walker' in components:
                print(f"\n{'─' * 40}")
                print("GRAPH WALKER")
                print(f"{'─' * 40}")
                results = profile_graph_walker(manager, graph_info)
                for r in results:
                    print(r)
                all_results.extend(results)

            if 'pathfinder' in components:
                print(f"\n{'─' * 40}")
                print("PATH FINDER")
                print(f"{'─' * 40}")
                results = profile_path_finder(manager, graph_info)
                for r in results:
                    print(r)
                all_results.extend(results)

            if 'pattern' in components:
                print(f"\n{'─' * 40}")
                print("PATTERN MATCHER")
                print(f"{'─' * 40}")
                results = profile_pattern_matcher(manager, graph_info)
                for r in results:
                    print(r)
                all_results.extend(results)

            # Summary
            if all_results:
                print(f"\n{'=' * 80}")
                print("SUMMARY")
                print(f"{'=' * 80}")

                # Sort by avg time (slowest first)
                sorted_results = sorted(all_results, key=lambda r: r.avg_time_ms, reverse=True)

                print("\nSlowest operations:")
                for r in sorted_results[:5]:
                    print(f"  {r}")

                print("\nFastest operations:")
                for r in sorted_results[-3:]:
                    print(f"  {r}")

                # Performance thresholds
                print("\nPerformance assessment:")
                slow_ops = [r for r in all_results if r.avg_time_ms > 10]
                medium_ops = [r for r in all_results if 1 < r.avg_time_ms <= 10]
                fast_ops = [r for r in all_results if r.avg_time_ms <= 1]

                print(f"  🔴 Slow (>10ms):     {len(slow_ops)} operations")
                print(f"  🟡 Medium (1-10ms):  {len(medium_ops)} operations")
                print(f"  🟢 Fast (<1ms):      {len(fast_ops)} operations")


def main():
    parser = argparse.ArgumentParser(description="Profile GoT Query API")
    parser.add_argument("--full", action="store_true", help="Run full profiling with larger graphs")
    parser.add_argument("--component", choices=['query', 'walker', 'pathfinder', 'pattern'],
                        help="Profile specific component only")
    parser.add_argument("--detailed", action="store_true", help="Include cProfile detailed output")
    parser.add_argument("--size", type=int, default=50, help="Number of tasks to create")

    args = parser.parse_args()

    if args.full:
        # Test with multiple graph sizes
        graph_sizes = [
            (50, 10, 0.05),   # Small
            (100, 20, 0.05),  # Medium
            (200, 40, 0.03),  # Large (lower density)
        ]
    else:
        graph_sizes = [(args.size, args.size // 5, 0.05)]

    components = [args.component] if args.component else None

    run_profiling(graph_sizes, components, args.detailed)


if __name__ == "__main__":
    main()
