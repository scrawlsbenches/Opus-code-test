# SubprocessClaudeCodeSpawner - Production Agent Coordination

## Overview

The `SubprocessClaudeCodeSpawner` is a production-ready implementation of the `AgentSpawner` interface that spawns actual Claude Code CLI subprocesses for parallel agent workflows. It provides robust process management, timeout handling, and performance tracking.

## Features

### 1. Subprocess Spawning with Isolation
- Spawns actual `claude-code` CLI processes
- Each agent runs in complete isolation
- Proper process lifecycle management
- Automatic cleanup on termination

### 2. Context Passing
- **Temp Files**: Prompts written to temporary files
- **Command-line Arguments**: Context files passed via CLI args
- **Working Directory**: Each agent runs in specified working directory
- **Environment Isolation**: Clean subprocess environment

### 3. Output Capture
- **stdout**: Full output capture with line-by-line buffering
- **stderr**: Separate error stream capture
- **Exit Codes**: Proper exit code tracking
- **Output Parsing**: Automatic parsing of file changes and status

### 4. Timeout Handling
- **Per-agent Timeouts**: Individual timeout configuration
- **Graceful Termination**: SIGTERM followed by SIGKILL
- **Grace Period**: Configurable grace period before force kill (default 5s)
- **Timeout Tracking**: Metrics for timed-out agents

### 5. Concurrency Control
- **Semaphore-based Limiting**: Thread-safe concurrency control
- **Max Concurrent**: Configurable maximum concurrent agents
- **Queue Management**: Automatic queuing when limit reached
- **Rate Limiting**: Built-in rate limiting via semaphore

### 6. Performance Metrics
- **Success Rate**: Track completion vs failure rates
- **Duration Stats**: Average and total execution time
- **Concurrency Stats**: Peak and current concurrent agents
- **Timeout Stats**: Track timeout occurrences

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  SubprocessClaudeCodeSpawner                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │  Agent 1   │  │  Agent 2   │  │  Agent 3   │                │
│  │ (Process)  │  │ (Process)  │  │ (Process)  │                │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                │
│        │               │               │                        │
│        │               │               │                        │
│  ┌─────▼───────────────▼───────────────▼──────┐                │
│  │         Semaphore (max_concurrent)         │                │
│  └────────────────────────────────────────────┘                │
│                                                                  │
│  ┌────────────────────────────────────────────┐                │
│  │           Metrics Tracking                 │                │
│  │  - Success rate                            │                │
│  │  - Duration stats                          │                │
│  │  - Concurrency stats                       │                │
│  └────────────────────────────────────────────┘                │
│                                                                  │
│  ┌────────────────────────────────────────────┐                │
│  │           Temp File Management             │                │
│  │  - Create prompt files                     │                │
│  │  - Cleanup on completion                   │                │
│  └────────────────────────────────────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## API Reference

### Classes

#### `SubprocessClaudeCodeSpawner`

Main spawner class implementing the `AgentSpawner` interface.

**Constructor:**
```python
SubprocessClaudeCodeSpawner(
    max_concurrent: int = 5,
    default_timeout: float = 300.0,
    working_dir: Optional[Path] = None,
    claude_code_path: Optional[str] = None,
    branch: str = "main",
)
```

**Parameters:**
- `max_concurrent`: Maximum number of concurrent subprocesses (default: 5)
- `default_timeout`: Default timeout in seconds (default: 300.0)
- `working_dir`: Working directory for spawned processes (default: current directory)
- `claude_code_path`: Path to claude-code CLI (auto-detected if None)
- `branch`: Git branch agents should work on (default: "main")

**Methods:**

##### `spawn()` - Synchronous Spawning
```python
def spawn(
    self,
    task: str,
    boundary: ParallelWorkBoundary,
    timeout_seconds: int = 300,
    context_files: Optional[List[Path]] = None,
) -> str:
    """Spawn and wait for completion (blocking)."""
```

##### `spawn_async()` - Asynchronous Spawning
```python
def spawn_async(
    self,
    task: str,
    boundary: ParallelWorkBoundary,
    timeout_seconds: int = 300,
    context_files: Optional[List[Path]] = None,
) -> Tuple[str, SpawnHandle]:
    """Spawn without blocking, return handle."""
```

##### `get_status()` - Status Check
```python
def get_status(self, agent_id: str) -> AgentStatus:
    """Get current agent status."""
```

##### `get_result()` - Retrieve Result
```python
def get_result(self, agent_id: str) -> Optional[AgentResult]:
    """Get result if completed, None otherwise."""
```

##### `wait_for()` - Wait for Completion
```python
def wait_for(self, agent_id: str, timeout_seconds: int = 300) -> AgentResult:
    """Wait for agent to complete."""
```

##### `get_metrics()` - Performance Metrics
```python
def get_metrics(self) -> Dict[str, Any]:
    """Get spawn success rates, durations, etc."""
```

Returns:
```python
{
    "total_spawned": int,        # Total agents spawned
    "completed": int,            # Successfully completed
    "failed": int,               # Failed (including timeouts)
    "timed_out": int,            # Timed out specifically
    "success_rate": float,       # Success rate (0.0-1.0)
    "avg_duration_seconds": float,  # Average duration
    "total_duration_seconds": float, # Cumulative duration
    "peak_concurrent": int,      # Peak concurrent agents
    "current_active": int,       # Currently running
}
```

##### `cleanup()` - Resource Cleanup
```python
def cleanup(self) -> None:
    """Terminate all processes and clean up temp files."""
```

#### `SpawnHandle`

Handle for asynchronously spawned agents.

**Methods:**

##### `is_running()` - Check Running Status
```python
def is_running(self) -> bool:
    """Check if process is still running."""
```

##### `poll()` - Non-blocking Status Check
```python
def poll(self) -> Optional[SpawnResult]:
    """Poll for completion without blocking."""
```

##### `wait()` - Wait for Completion
```python
def wait(self, timeout_seconds: Optional[float] = None) -> SpawnResult:
    """Wait for completion with optional timeout."""
```

##### `terminate()` - Force Termination
```python
def terminate(self, grace_period: float = 5.0) -> None:
    """Gracefully terminate the process."""
```

#### `SpawnResult`

Result from a spawned subprocess execution.

**Attributes:**
- `success: bool` - Whether execution succeeded
- `output: str` - Captured stdout
- `error: Optional[str]` - Captured stderr
- `duration_seconds: float` - Execution duration
- `exit_code: int` - Process exit code

#### `SpawnMetrics`

Metrics tracking for spawned agents.

**Attributes:**
- `total_spawned: int` - Total agents spawned
- `completed: int` - Successfully completed
- `failed: int` - Failed agents
- `timed_out: int` - Timed out agents
- `success_rate: float` - Success rate (0.0-1.0)
- `avg_duration_seconds: float` - Average duration
- `peak_concurrent: int` - Peak concurrent agents
- `current_active: int` - Currently running

## Usage Examples

### Basic Synchronous Spawning

```python
from pathlib import Path
from cortical.reasoning import (
    SubprocessClaudeCodeSpawner,
    ParallelWorkBoundary,
)

# Initialize spawner
spawner = SubprocessClaudeCodeSpawner(
    max_concurrent=5,
    default_timeout=300.0,
    working_dir=Path("/path/to/repo"),
)

# Define work boundary
boundary = ParallelWorkBoundary(
    agent_id="auth-agent",
    scope_description="Implement authentication",
    files_owned={"src/auth.py", "tests/test_auth.py"},
    files_read_only={"config.py"},
)

# Spawn agent (blocks until completion)
agent_id = spawner.spawn(
    task="Implement JWT-based authentication",
    boundary=boundary,
    timeout_seconds=300,
)

# Get result
result = spawner.get_result(agent_id)
print(f"Status: {result.status.name}")
print(f"Duration: {result.duration_seconds:.2f}s")
print(f"Files modified: {result.files_modified}")
```

### Asynchronous Parallel Spawning

```python
# Spawn multiple agents in parallel
agents = []
for i, task in enumerate(tasks):
    agent_id, handle = spawner.spawn_async(
        task=task,
        boundary=boundaries[i],
        timeout_seconds=300,
    )
    agents.append((agent_id, handle))

# Wait for all to complete
results = []
for agent_id, handle in agents:
    try:
        result = handle.wait(timeout_seconds=300)
        results.append(result)
    except subprocess.TimeoutExpired:
        print(f"Agent {agent_id} timed out")
```

### With ParallelCoordinator

```python
from cortical.reasoning import ParallelCoordinator

# Create coordinator
coordinator = ParallelCoordinator(spawner)

# Define boundaries
boundaries = [
    ParallelWorkBoundary("a1", "Auth", {"auth.py"}),
    ParallelWorkBoundary("a2", "API", {"api.py"}),
]

# Check for conflicts
can_spawn, issues = coordinator.can_spawn(boundaries)
if not can_spawn:
    print(f"Cannot spawn: {issues}")
    exit(1)

# Spawn agents
agent_ids = coordinator.spawn_agents(
    tasks=["Implement auth", "Implement API"],
    boundaries=boundaries,
    timeout_seconds=300,
)

# Collect results
results = coordinator.collect_results(agent_ids, timeout_seconds=300)

# Detect conflicts
conflicts = coordinator.detect_conflicts(results)
if conflicts:
    print(f"Conflicts detected: {len(conflicts)}")
```

### Monitoring Metrics

```python
import time

# Spawn agents
for task, boundary in zip(tasks, boundaries):
    spawner.spawn_async(task, boundary)

# Monitor progress
while spawner.get_metrics()["current_active"] > 0:
    metrics = spawner.get_metrics()
    print(f"Active: {metrics['current_active']}")
    print(f"Completed: {metrics['completed']}")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    time.sleep(5)

# Final metrics
final_metrics = spawner.get_metrics()
print(f"\nFinal Results:")
print(f"  Total spawned: {final_metrics['total_spawned']}")
print(f"  Success rate: {final_metrics['success_rate']:.1%}")
print(f"  Avg duration: {final_metrics['avg_duration_seconds']:.2f}s")
```

## Integration with ParallelCoordinator

The `SubprocessClaudeCodeSpawner` implements the `AgentSpawner` interface, making it fully compatible with `ParallelCoordinator` for advanced parallel workflows:

```python
from cortical.reasoning import (
    SubprocessClaudeCodeSpawner,
    ParallelCoordinator,
    ParallelWorkBoundary,
)

# Create spawner
spawner = SubprocessClaudeCodeSpawner(
    max_concurrent=5,
    default_timeout=300.0,
)

# Create coordinator
coordinator = ParallelCoordinator(spawner)

# Use coordinator's advanced features
# - Boundary conflict detection
# - Parallel execution orchestration
# - Conflict detection in results
# - Summary and reporting
```

## Error Handling

### Timeout Handling

```python
try:
    result = spawner.wait_for(agent_id, timeout_seconds=60)
except subprocess.TimeoutExpired:
    # Agent timed out
    status = spawner.get_status(agent_id)
    print(f"Agent timed out, status: {status.name}")

    # Result is still available with partial output
    result = spawner.get_result(agent_id)
    if result:
        print(f"Partial output: {result.output[:200]}")
```

### Process Failures

```python
result = spawner.wait_for(agent_id)
if result.status == AgentStatus.FAILED:
    print(f"Agent failed: {result.error}")
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
```

### Boundary Violations

```python
result = spawner.wait_for(agent_id)
if result.error and "Boundary violations" in result.error:
    print(f"Agent violated its boundary!")
    print(f"Violations: {result.error}")
```

## Best Practices

### 1. Always Cleanup

```python
try:
    # ... spawn and run agents ...
finally:
    spawner.cleanup()
```

### 2. Use Context Managers (if needed)

```python
class SpawnerContext:
    def __init__(self, spawner):
        self.spawner = spawner

    def __enter__(self):
        return self.spawner

    def __exit__(self, *args):
        self.spawner.cleanup()

with SpawnerContext(spawner) as s:
    # ... use spawner ...
    pass
# Automatic cleanup
```

### 3. Set Appropriate Timeouts

```python
# Short-lived tasks
spawner = SubprocessClaudeCodeSpawner(default_timeout=60.0)

# Long-running analysis
spawner = SubprocessClaudeCodeSpawner(default_timeout=600.0)
```

### 4. Monitor Metrics

```python
# Check success rate periodically
metrics = spawner.get_metrics()
if metrics['completed'] > 10 and metrics['success_rate'] < 0.5:
    print("Warning: Low success rate detected!")
```

### 5. Handle Failures Gracefully

```python
for agent_id, handle in agents:
    try:
        result = handle.wait()
        if not result.success:
            # Log failure but continue
            log_failure(agent_id, result.error)
    except Exception as e:
        # Handle unexpected errors
        log_exception(agent_id, e)
```

## Testing

See `examples/subprocess_spawner_demo.py` for comprehensive usage examples.

For unit testing with mocked subprocesses, see `tests/unit/test_claude_code_spawner.py`.

## Comparison: ClaudeCodeSpawner vs SubprocessClaudeCodeSpawner

| Feature | ClaudeCodeSpawner | SubprocessClaudeCodeSpawner |
|---------|-------------------|----------------------------|
| **Execution** | Task tool configs | Actual subprocesses |
| **Use Case** | Claude Code sessions | Production automation |
| **Output Capture** | Manual recording | Automatic |
| **Timeout Handling** | Manual tracking | Built-in |
| **Concurrency** | Via Task tool | Via semaphore |
| **Metrics** | Basic tracking | Comprehensive |
| **Cleanup** | Manual | Automatic |
| **Integration** | ParallelCoordinator | ParallelCoordinator |

## See Also

- [Graph of Thought Documentation](graph-of-thought.md)
- [Collaboration Patterns](collaboration-patterns.md)
- [ParallelCoordinator API](parallel-coordinator.md)
