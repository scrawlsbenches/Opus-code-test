"""
Demonstration of NestedLoopExecutor for hierarchical QAPV cycles.

This example shows how to use the NestedLoopExecutor to manage
hierarchical goal decomposition with parent-child loop relationships.

Run:
    python examples/nested_loop_demo.py
"""

from cortical.reasoning import NestedLoopExecutor, LoopPhase


def print_separator(title: str = ""):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)
    else:
        print('-'*60)


def demo_basic_nesting():
    """Demonstrate basic nested loop creation and advancement."""
    print_separator("Demo 1: Basic Nested Loops")

    executor = NestedLoopExecutor(max_depth=5)

    # Create root loop
    root = executor.start_root("Build web application")
    print(f"✓ Created root loop: {root}")
    print(f"  Goal: Build web application")
    print(f"  Starting phase: {executor.get_loop(root).current_phase.value}")

    # Advance through a few phases
    print("\n  Advancing through QAPV phases...")
    phase = executor.advance(root)
    print(f"  → {phase.value}")
    phase = executor.advance(root)
    print(f"  → {phase.value}")

    # Spawn a child loop
    backend = executor.spawn_child(root, "Implement backend API")
    print(f"\n✓ Spawned child loop: {backend}")
    print(f"  Goal: Implement backend API")
    print(f"  Depth: {executor.get_context(backend).depth}")
    print(f"  Parent status: {executor.get_loop(root).status.name}")

    # Work on the child
    executor.record_answer(backend, "Using Flask framework")
    executor.record_answer(backend, "PostgreSQL for database")

    # Complete the child
    result = {"framework": "Flask", "db": "PostgreSQL"}
    parent_id = executor.complete(backend, result)
    print(f"\n✓ Completed child loop")
    print(f"  Returned parent: {parent_id}")
    print(f"  Parent status: {executor.get_loop(root).status.name}")

    # Check result aggregation
    root_context = executor.get_context(root)
    print(f"\n  Parent received child result:")
    print(f"  {root_context.child_results[backend]}")


def demo_multi_level_nesting():
    """Demonstrate multiple levels of nesting."""
    print_separator("Demo 2: Multi-Level Nesting")

    executor = NestedLoopExecutor(max_depth=5)

    # Create hierarchy
    root = executor.start_root("Build e-commerce platform")
    print(f"Level 0 (root): {root}")

    backend = executor.spawn_child(root, "Build backend services")
    print(f"Level 1: {backend}")

    auth = executor.spawn_child(backend, "Implement authentication")
    print(f"Level 2: {auth}")

    jwt = executor.spawn_child(auth, "Setup JWT tokens")
    print(f"Level 3: {jwt}")

    # Show hierarchy
    hierarchy = executor.get_loop_hierarchy(jwt)
    print(f"\nHierarchy path to deepest loop:")
    for i, loop_id in enumerate(hierarchy):
        loop = executor.get_loop(loop_id)
        indent = "  " * i
        print(f"{indent}└─ {loop.goal}")

    # Show depths
    print(f"\nDepth verification:")
    for loop_id in hierarchy:
        ctx = executor.get_context(loop_id)
        print(f"  {loop_id}: depth={ctx.depth}")


def demo_result_aggregation():
    """Demonstrate result aggregation from multiple children."""
    print_separator("Demo 3: Result Aggregation from Multiple Children")

    executor = NestedLoopExecutor(max_depth=3)

    # Create parent
    parent = executor.start_root("Develop mobile app")
    print(f"Parent task: {executor.get_loop(parent).goal}")

    # Spawn multiple children
    children = []
    child_tasks = [
        ("Design UI mockups", {"screens": 12, "status": "done"}),
        ("Setup CI/CD pipeline", {"platform": "GitHub Actions", "status": "done"}),
        ("Write API client", {"endpoints": 15, "status": "done"}),
    ]

    print("\nSpawning and completing children:")
    for task, result in child_tasks:
        child = executor.spawn_child(parent, task)
        children.append(child)
        print(f"  ✓ Spawned: {task}")

        executor.record_answer(child, f"Completed {task}")
        executor.complete(child, result)
        print(f"    → Completed with result: {result}")

    # Show aggregated results
    parent_context = executor.get_context(parent)
    print(f"\nParent now has {len(parent_context.child_results)} child results:")
    for child_id, result in parent_context.child_results.items():
        loop = executor.get_loop(child_id)
        print(f"  • {loop.goal}: {result}")


def demo_early_termination():
    """Demonstrate early termination with break_loop."""
    print_separator("Demo 4: Early Termination")

    executor = NestedLoopExecutor(max_depth=3)

    # Create loops
    root = executor.start_root("Research new technology")
    investigation = executor.spawn_child(root, "Investigate framework X")

    print(f"Created investigation loop: {investigation}")
    print(f"Parent paused: {executor.get_loop(root).status.name}")

    # Realize we don't need it
    executor.break_loop(investigation, "Framework X is deprecated")

    print(f"\n✓ Loop broken early")
    print(f"  Investigation status: {executor.get_loop(investigation).status.name}")
    print(f"  Parent resumed: {executor.get_loop(root).status.name}")


def demo_depth_limiting():
    """Demonstrate max depth enforcement."""
    print_separator("Demo 5: Depth Limiting")

    executor = NestedLoopExecutor(max_depth=3)
    print(f"Max depth limit: {executor._max_depth}")

    # Create chain up to limit
    root = executor.start_root("Level 0")
    child1 = executor.spawn_child(root, "Level 1")
    child2 = executor.spawn_child(child1, "Level 2")

    print(f"\nCreated chain:")
    print(f"  Root (depth 0): {root}")
    print(f"  Child 1 (depth 1): {child1}")
    print(f"  Child 2 (depth 2): {child2}")

    # Try to exceed limit
    print(f"\nAttempting to spawn at depth 3 (would exceed limit)...")
    try:
        child3 = executor.spawn_child(child2, "Level 3")
        print("  ✗ Should have raised RecursionError!")
    except RecursionError as e:
        print(f"  ✓ Correctly raised RecursionError:")
        print(f"    {e}")


def demo_summary():
    """Demonstrate executor summary and statistics."""
    print_separator("Demo 6: Executor Summary")

    executor = NestedLoopExecutor(max_depth=5)

    # Create some loops in various states
    root = executor.start_root("Build system")
    child1 = executor.spawn_child(root, "Component A")
    child2 = executor.spawn_child(child1, "Subcomponent A1")

    # Complete the nested children (this resumes the parent)
    executor.complete(child2, {"done": True})
    executor.complete(child1, {"component_a": "done"})

    # Now root is active again, spawn another child to break
    child3 = executor.spawn_child(root, "Component B")
    executor.break_loop(child3, "Not needed")

    # Show summary
    summary = executor.get_summary()
    print("Executor summary:")
    print(f"  Total loops: {summary['total_loops']}")
    print(f"  Active loops: {summary['active_loops']}")
    print(f"  Max depth limit: {summary['max_depth_limit']}")
    print(f"  Max depth reached: {summary['max_depth_reached']}")
    print(f"\n  Status breakdown:")
    for status, count in summary['status_counts'].items():
        print(f"    {status}: {count}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("  NestedLoopExecutor Demonstration")
    print("  Hierarchical QAPV Cycle Management")
    print("="*60)

    demo_basic_nesting()
    demo_multi_level_nesting()
    demo_result_aggregation()
    demo_early_termination()
    demo_depth_limiting()
    demo_summary()

    print("\n" + "="*60)
    print("  All demonstrations complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
