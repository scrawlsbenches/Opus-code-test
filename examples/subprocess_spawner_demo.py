"""
Demonstration of SubprocessClaudeCodeSpawner - Production Agent Spawning.

This example shows how to use the SubprocessClaudeCodeSpawner to spawn
actual Claude Code CLI subprocesses with proper isolation, timeout handling,
and metrics tracking.

NOTE: This is a demo showing the API usage. In production, you would have
the claude-code CLI installed and properly configured.
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.reasoning import (
    SubprocessClaudeCodeSpawner,
    ParallelWorkBoundary,
    ParallelCoordinator,
    AgentStatus,
)


def demo_basic_usage():
    """Demo: Basic synchronous spawning."""
    print("=" * 70)
    print("DEMO 1: Basic Synchronous Spawning")
    print("=" * 70)

    try:
        # Initialize spawner
        spawner = SubprocessClaudeCodeSpawner(
            max_concurrent=3,
            default_timeout=60.0,
            working_dir=Path.cwd(),
            claude_code_path="echo",  # Mock for demo - use echo instead of claude-code
            branch="feature-branch",
        )

        # Define work boundary
        boundary = ParallelWorkBoundary(
            agent_id="auth-agent",
            scope_description="Implement authentication module",
            files_owned={"src/auth.py", "src/middleware.py"},
            files_read_only={"config.py", "requirements.txt"},
        )

        # Spawn agent (synchronous - blocks until completion)
        print("\n📝 Spawning agent to implement authentication...")
        agent_id = spawner.spawn(
            task="Implement user authentication with JWT tokens",
            boundary=boundary,
            timeout_seconds=30,
        )

        print(f"✅ Agent spawned: {agent_id}")

        # Get result
        result = spawner.get_result(agent_id)
        if result:
            print(f"\n📊 Result:")
            print(f"  Status: {result.status.name}")
            print(f"  Duration: {result.duration_seconds:.2f}s")
            print(f"  Files modified: {result.files_modified}")
            print(f"  Files created: {result.files_created}")

        # Get metrics
        metrics = spawner.get_metrics()
        print(f"\n📈 Metrics:")
        print(f"  Total spawned: {metrics['total_spawned']}")
        print(f"  Completed: {metrics['completed']}")
        print(f"  Success rate: {metrics['success_rate']:.1%}")

        # Cleanup
        spawner.cleanup()

    except RuntimeError as e:
        print(f"\n⚠️  Note: {e}")
        print("   This is expected in the demo (claude-code CLI not installed)")


def demo_async_spawning():
    """Demo: Asynchronous spawning with parallel execution."""
    print("\n" + "=" * 70)
    print("DEMO 2: Asynchronous Spawning (Parallel Execution)")
    print("=" * 70)

    try:
        spawner = SubprocessClaudeCodeSpawner(
            max_concurrent=5,
            default_timeout=60.0,
            working_dir=Path.cwd(),
            claude_code_path="echo",  # Mock for demo
        )

        # Define multiple boundaries for parallel work
        boundaries = [
            ParallelWorkBoundary(
                agent_id="agent-1",
                scope_description="Auth module",
                files_owned={"src/auth.py"},
            ),
            ParallelWorkBoundary(
                agent_id="agent-2",
                scope_description="API module",
                files_owned={"src/api.py"},
            ),
            ParallelWorkBoundary(
                agent_id="agent-3",
                scope_description="Database module",
                files_owned={"src/database.py"},
            ),
        ]

        tasks = [
            "Implement authentication",
            "Create REST API endpoints",
            "Set up database models",
        ]

        # Spawn all agents asynchronously
        print("\n📝 Spawning 3 agents in parallel...")
        handles = []
        for task, boundary in zip(tasks, boundaries):
            agent_id, handle = spawner.spawn_async(task, boundary, timeout_seconds=30)
            handles.append((agent_id, handle))
            print(f"  ✓ Spawned: {agent_id}")

        # Poll for completion
        print("\n⏳ Waiting for agents to complete...")
        for agent_id, handle in handles:
            try:
                result = handle.wait(timeout_seconds=30)
                print(f"  ✅ {agent_id}: {result.exit_code}")
            except Exception as e:
                print(f"  ❌ {agent_id}: {e}")

        # Show final metrics
        metrics = spawner.get_metrics()
        print(f"\n📈 Final Metrics:")
        print(f"  Total spawned: {metrics['total_spawned']}")
        print(f"  Peak concurrent: {metrics['peak_concurrent']}")
        print(f"  Avg duration: {metrics['avg_duration_seconds']:.2f}s")

        spawner.cleanup()

    except RuntimeError as e:
        print(f"\n⚠️  Note: {e}")


def demo_with_coordinator():
    """Demo: Integration with ParallelCoordinator."""
    print("\n" + "=" * 70)
    print("DEMO 3: Integration with ParallelCoordinator")
    print("=" * 70)

    try:
        # Create spawner
        spawner = SubprocessClaudeCodeSpawner(
            max_concurrent=3,
            default_timeout=60.0,
            working_dir=Path.cwd(),
            claude_code_path="echo",  # Mock for demo
        )

        # Create coordinator
        coordinator = ParallelCoordinator(spawner)

        # Define boundaries
        boundaries = [
            ParallelWorkBoundary("a1", "Auth", {"auth.py"}),
            ParallelWorkBoundary("a2", "API", {"api.py"}),
        ]

        # Check if tasks can run in parallel
        can_spawn, issues = coordinator.can_spawn(boundaries)
        print(f"\n✅ Can spawn in parallel: {can_spawn}")
        if issues:
            print(f"⚠️  Issues: {issues}")

        # Spawn agents via coordinator
        print("\n📝 Spawning agents via coordinator...")
        agent_ids = coordinator.spawn_agents(
            ["Implement auth", "Implement API"],
            boundaries,
            timeout_seconds=30,
        )

        print(f"  Spawned {len(agent_ids)} agents")

        # Note: In real usage, we'd wait for completion here
        # For demo, we just show the API

        spawner.cleanup()

    except RuntimeError as e:
        print(f"\n⚠️  Note: {e}")


def demo_error_handling():
    """Demo: Timeout and error handling."""
    print("\n" + "=" * 70)
    print("DEMO 4: Timeout and Error Handling")
    print("=" * 70)

    try:
        spawner = SubprocessClaudeCodeSpawner(
            max_concurrent=2,
            default_timeout=5.0,  # Short timeout
            working_dir=Path.cwd(),
            claude_code_path="sleep",  # Mock - will timeout
        )

        boundary = ParallelWorkBoundary("a1", "Test", {"test.py"})

        print("\n📝 Spawning agent with short timeout...")
        agent_id, handle = spawner.spawn_async(
            "Long-running task",
            boundary,
            timeout_seconds=2,  # 2 second timeout
        )

        try:
            result = handle.wait(timeout_seconds=2)
            print(f"✅ Completed: {result.success}")
        except Exception as e:
            print(f"⏱️  Timeout detected (expected): {type(e).__name__}")

            # Check status
            status = spawner.get_status(agent_id)
            print(f"   Agent status: {status.name}")

        # Show timeout metrics
        metrics = spawner.get_metrics()
        print(f"\n📈 Metrics:")
        print(f"  Timed out: {metrics['timed_out']}")
        print(f"  Failed: {metrics['failed']}")

        spawner.cleanup()

    except RuntimeError as e:
        print(f"\n⚠️  Note: {e}")


def demo_metrics_tracking():
    """Demo: Performance metrics tracking."""
    print("\n" + "=" * 70)
    print("DEMO 5: Performance Metrics Tracking")
    print("=" * 70)

    print("""
The SubprocessClaudeCodeSpawner tracks comprehensive metrics:

📊 Metrics Available:
  - total_spawned: Total number of agents spawned
  - completed: Successfully completed agents
  - failed: Failed agents (including timeouts)
  - timed_out: Agents that exceeded timeout
  - success_rate: Completion success rate (0.0-1.0)
  - avg_duration_seconds: Average agent execution time
  - total_duration_seconds: Cumulative execution time
  - peak_concurrent: Maximum concurrent agents
  - current_active: Currently running agents

📈 Usage:
    spawner = SubprocessClaudeCodeSpawner(...)

    # ... spawn and run agents ...

    metrics = spawner.get_metrics()
    print(f"Success rate: {metrics['success_rate']:.1%}")
    print(f"Avg duration: {metrics['avg_duration_seconds']:.2f}s")
    """)


def main():
    """Run all demos."""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║     SubprocessClaudeCodeSpawner - Production Agent Spawning      ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    demos = [
        demo_basic_usage,
        demo_async_spawning,
        demo_with_coordinator,
        demo_error_handling,
        demo_metrics_tracking,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n❌ Demo error: {e}")

    print("\n" + "=" * 70)
    print("✅ All demos completed!")
    print("=" * 70)
    print("\nFor production use:")
    print("  1. Install claude-code CLI")
    print("  2. Configure authentication")
    print("  3. Set working_dir to your repository")
    print("  4. Adjust max_concurrent and timeout based on your needs")
    print()


if __name__ == "__main__":
    main()
