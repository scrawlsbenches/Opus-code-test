"""
Demo: GitAutoCommitter for automatic git commits.

This example demonstrates how to use GitAutoCommitter to automatically
commit graph changes to git with configurable safety features.

Run with:
    python examples/git_auto_committer_demo.py
"""

from pathlib import Path
from cortical.reasoning import ThoughtGraph, NodeType, EdgeType
from cortical.reasoning.graph_persistence import GitAutoCommitter


def demo_immediate_mode():
    """Demo: Immediate commit mode."""
    print("=" * 60)
    print("DEMO: Immediate Commit Mode")
    print("=" * 60)

    # Create committer in immediate mode
    committer = GitAutoCommitter(
        mode='immediate',
        auto_push=False  # Don't auto-push
    )

    # Create a simple graph
    graph = ThoughtGraph()
    graph.add_node('Q1', NodeType.QUESTION, 'What is the best approach?')
    graph.add_node('H1', NodeType.HYPOTHESIS, 'Use approach A')
    graph.add_edge('Q1', 'H1', EdgeType.EXPLORES)

    # Simulate saving and committing
    print("\n1. Saving graph...")
    graph_path = '/tmp/test_graph.json'
    print(f"   Graph path: {graph_path}")

    print("\n2. Triggering auto-commit (immediate mode)...")
    committer.commit_on_save(
        graph_path=graph_path,
        graph=graph,
        message='demo: Add initial question and hypothesis'
    )

    print("\n✓ Demo complete\n")


def demo_debounced_mode():
    """Demo: Debounced commit mode."""
    print("=" * 60)
    print("DEMO: Debounced Commit Mode")
    print("=" * 60)

    # Create committer in debounced mode
    committer = GitAutoCommitter(
        mode='debounced',
        debounce_seconds=2,  # Wait 2 seconds
        auto_push=False
    )

    # Create a simple graph
    graph = ThoughtGraph()
    graph.add_node('Q1', NodeType.QUESTION, 'How to optimize performance?')

    print("\n1. First save...")
    committer.commit_on_save(
        graph_path='/tmp/graph_v1.json',
        graph=graph
    )
    print("   ⏱️  Timer started (2 seconds)")

    print("\n2. Second save (resets timer)...")
    graph.add_node('H1', NodeType.HYPOTHESIS, 'Cache frequently used data')
    committer.commit_on_save(
        graph_path='/tmp/graph_v2.json',
        graph=graph
    )
    print("   ⏱️  Timer reset (2 seconds)")

    print("\n3. Waiting for commit...")
    print("   (In real usage, commit happens after 2 seconds of inactivity)")

    # Cleanup
    committer.cleanup()
    print("\n✓ Demo complete (timer cancelled for demo)\n")


def demo_validation():
    """Demo: Pre-commit validation."""
    print("=" * 60)
    print("DEMO: Pre-commit Validation")
    print("=" * 60)

    committer = GitAutoCommitter(mode='manual')  # Manual mode for demo

    # Test 1: Empty graph (invalid)
    print("\n1. Testing empty graph...")
    empty_graph = ThoughtGraph()
    valid, error = committer.validate_before_commit(empty_graph)
    print(f"   Valid: {valid}")
    print(f"   Error: {error}")

    # Test 2: All orphans (invalid)
    print("\n2. Testing all orphaned nodes...")
    orphan_graph = ThoughtGraph()
    orphan_graph.add_node('N1', NodeType.CONCEPT, 'Isolated concept 1')
    orphan_graph.add_node('N2', NodeType.CONCEPT, 'Isolated concept 2')
    valid, error = committer.validate_before_commit(orphan_graph)
    print(f"   Valid: {valid}")
    print(f"   Error: {error}")

    # Test 3: Valid graph with connections
    print("\n3. Testing valid connected graph...")
    valid_graph = ThoughtGraph()
    valid_graph.add_node('Q1', NodeType.QUESTION, 'What to implement?')
    valid_graph.add_node('D1', NodeType.DECISION, 'Implementation choice')
    valid_graph.add_edge('Q1', 'D1', EdgeType.RAISES)
    valid, error = committer.validate_before_commit(valid_graph)
    print(f"   Valid: {valid}")
    print(f"   Error: {error}")

    print("\n✓ Demo complete\n")


def demo_protected_branches():
    """Demo: Protected branch safety."""
    print("=" * 60)
    print("DEMO: Protected Branch Safety")
    print("=" * 60)

    # Create committer with custom protected branches
    committer = GitAutoCommitter(
        protected_branches=['main', 'master', 'prod', 'release']
    )

    print("\n1. Checking protected branches...")
    test_branches = ['main', 'master', 'prod', 'feature/test', 'dev']
    for branch in test_branches:
        is_protected = committer.is_protected_branch(branch)
        status = "🔒 PROTECTED" if is_protected else "✓ OK to push"
        print(f"   {branch:20s} → {status}")

    print("\n2. Getting current branch...")
    current = committer.get_current_branch()
    if current:
        print(f"   Current branch: {current}")
        is_protected = committer.is_protected_branch(current)
        print(f"   Protected: {is_protected}")
    else:
        print("   Not in a git repository or detached HEAD")

    print("\n✓ Demo complete\n")


def demo_backup_branch():
    """Demo: Creating backup branches."""
    print("=" * 60)
    print("DEMO: Backup Branch Creation")
    print("=" * 60)

    committer = GitAutoCommitter()

    print("\n1. Creating backup branch...")
    print("   This creates a timestamped backup of current work")
    print("   Format: backup/{current-branch}/{timestamp}")

    backup = committer.create_backup_branch(prefix='backup')
    if backup:
        print(f"   ✓ Created: {backup}")
    else:
        print("   ✗ Failed (not in git repo or detached HEAD)")

    print("\n✓ Demo complete\n")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "GitAutoCommitter Demo Suite" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    demos = [
        ("Immediate Mode", demo_immediate_mode),
        ("Debounced Mode", demo_debounced_mode),
        ("Validation", demo_validation),
        ("Protected Branches", demo_protected_branches),
        ("Backup Branch", demo_backup_branch),
    ]

    for i, (name, func) in enumerate(demos, 1):
        print(f"\n[{i}/{len(demos)}] Running: {name}")
        print("-" * 60)
        func()

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
