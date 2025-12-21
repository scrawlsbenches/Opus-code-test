# GitAutoCommitter: Automatic Git Commits for Graph Persistence

## Overview

The `GitAutoCommitter` class provides safe, configurable automatic commits when thought graphs are saved. It includes built-in validation, protected branch detection, and debouncing to prevent commit spam.

## Location

- **Implementation:** `cortical/reasoning/graph_persistence.py`
- **Tests:** `tests/unit/test_graph_persistence.py`
- **Demo:** `examples/git_auto_committer_demo.py`

## Features

### Commit Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `immediate` | Commits right away | Quick iterations, single developer |
| `debounced` | Waits for inactivity | Frequent saves, reduce commit noise |
| `manual` | No auto-commit | Full control, validation only |

### Safety Features

1. **Protected Branch Detection**
   - Default protected: `main`, `master`
   - Customizable via `protected_branches` parameter
   - Detached HEAD is always protected
   - Push blocked unless explicitly overridden

2. **Pre-Commit Validation**
   - Empty graph detection
   - Orphaned node detection
   - Graph integrity checks

3. **No Force Operations**
   - Never uses `--force` push
   - Never uses `--amend` without explicit request
   - Always uses standard `git commit` and `git push`

4. **Backup Branch Creation**
   - Creates timestamped backups before risky operations
   - Format: `backup/{current-branch}/{timestamp}`

## Basic Usage

### Immediate Mode

```python
from cortical.reasoning.graph_persistence import GitAutoCommitter
from cortical.reasoning import ThoughtGraph, NodeType

# Create committer
committer = GitAutoCommitter(mode='immediate')

# Build graph
graph = ThoughtGraph()
graph.add_node('Q1', NodeType.QUESTION, 'What is the best approach?')

# Save and auto-commit
committer.commit_on_save(
    graph_path='/path/to/graph.json',
    graph=graph,
    message='Add initial question'
)
```

### Debounced Mode

```python
# Debounce: wait 5 seconds of inactivity before committing
committer = GitAutoCommitter(
    mode='debounced',
    debounce_seconds=5
)

# Multiple saves within 5 seconds will only commit once
committer.commit_on_save('/path/to/graph_v1.json', graph)
# ... more changes ...
committer.commit_on_save('/path/to/graph_v2.json', graph)
# ... 5 seconds later, single commit happens

# Cleanup when done
committer.cleanup()
```

### Manual Validation Only

```python
# Manual mode: validate but don't commit
committer = GitAutoCommitter(mode='manual')

# Check if graph is valid before manual commit
valid, error = committer.validate_before_commit(graph)
if valid:
    # Manually commit yourself
    pass
else:
    print(f"Invalid graph: {error}")
```

## Configuration

### Parameters

```python
GitAutoCommitter(
    mode='immediate',              # 'immediate', 'debounced', or 'manual'
    debounce_seconds=5,            # Wait time for debounced mode
    auto_push=False,               # Auto-push after commit?
    protected_branches=['main'],   # Branches to never auto-push
    repo_path='/path/to/repo'      # Git repository path
)
```

### Protected Branches

```python
# Custom protected branches
committer = GitAutoCommitter(
    protected_branches=['main', 'master', 'prod', 'release']
)

# Check if a branch is protected
if committer.is_protected_branch('main'):
    print("Cannot auto-push to main")

# Push with override (not recommended!)
committer.push_if_safe(force_protected=True)
```

## API Reference

### Methods

#### `auto_commit(message, files, validate_graph=None)`

Commit specified files with a message.

**Parameters:**
- `message` (str): Commit message
- `files` (List[str]): Paths to files to commit
- `validate_graph` (ThoughtGraph, optional): Graph to validate before commit

**Returns:** `bool` - True if commit succeeded

#### `commit_on_save(graph_path, graph=None, message=None)`

Called after a graph save completes. Behavior depends on mode.

**Parameters:**
- `graph_path` (str): Path to saved graph file
- `graph` (ThoughtGraph, optional): Graph for validation
- `message` (str, optional): Custom commit message

#### `push_if_safe(remote='origin', branch=None, force_protected=False)`

Push to remote if safe (not a protected branch).

**Parameters:**
- `remote` (str): Remote name (default: 'origin')
- `branch` (str, optional): Branch to push (default: current)
- `force_protected` (bool): Override protected branch check

**Returns:** `bool` - True if push succeeded or was safely skipped

#### `is_protected_branch(branch=None)`

Check if a branch is protected.

**Parameters:**
- `branch` (str, optional): Branch to check (default: current)

**Returns:** `bool` - True if protected

#### `get_current_branch()`

Get the current git branch.

**Returns:** `str` or `None` - Branch name or None if detached HEAD

#### `validate_before_commit(graph)`

Validate graph before committing.

**Parameters:**
- `graph` (ThoughtGraph): Graph to validate

**Returns:** `Tuple[bool, Optional[str]]` - (is_valid, error_message)

#### `create_backup_branch(prefix='backup')`

Create a backup branch before risky operations.

**Parameters:**
- `prefix` (str): Prefix for backup branch name

**Returns:** `str` or `None` - Backup branch name or None on failure

#### `cleanup()`

Clean up resources (cancel pending timers). Call before destroying instance.

## Validation Rules

### Empty Graph

```python
graph = ThoughtGraph()
valid, error = committer.validate_before_commit(graph)
# valid = False, error = "Cannot commit empty graph"
```

### All Orphans

```python
graph = ThoughtGraph()
graph.add_node('N1', NodeType.CONCEPT, 'Isolated 1')
graph.add_node('N2', NodeType.CONCEPT, 'Isolated 2')
# No edges - all orphans

valid, error = committer.validate_before_commit(graph)
# valid = False, error = "All 2 nodes are orphaned (no edges)"
```

### Valid Graph

```python
graph = ThoughtGraph()
graph.add_node('Q1', NodeType.QUESTION, 'Question')
graph.add_node('A1', NodeType.FACT, 'Answer')
graph.add_edge('Q1', 'A1', EdgeType.ANSWERS)

valid, error = committer.validate_before_commit(graph)
# valid = True, error = None
```

## Examples

### Example 1: Safe Development Workflow

```python
# Create committer with protection
committer = GitAutoCommitter(
    mode='debounced',
    debounce_seconds=10,
    protected_branches=['main', 'master', 'prod']
)

# Build graph incrementally
graph = ThoughtGraph()
graph.add_node('Q1', NodeType.QUESTION, 'How to implement auth?')
committer.commit_on_save('/path/to/graph.json', graph)

# More changes...
graph.add_node('H1', NodeType.HYPOTHESIS, 'Use JWT')
graph.add_edge('Q1', 'H1', EdgeType.EXPLORES)
committer.commit_on_save('/path/to/graph.json', graph)

# After 10 seconds of inactivity, single commit happens
# Protected branches are safe from accidental push

committer.cleanup()
```

### Example 2: Pre-Commit Validation

```python
committer = GitAutoCommitter(mode='manual')

# Validate before manual operations
def safe_commit(graph, message):
    valid, error = committer.validate_before_commit(graph)
    if not valid:
        print(f"❌ Invalid: {error}")
        return False

    # Create backup
    backup = committer.create_backup_branch()
    if backup:
        print(f"✓ Backup: {backup}")

    # Manual commit (using subprocess or git library)
    # ...
    return True
```

### Example 3: Auto-Push Feature Branch

```python
# Auto-push to feature branches only
committer = GitAutoCommitter(
    mode='immediate',
    auto_push=True,  # Auto-push after commit
    protected_branches=['main', 'master']
)

# This will commit and push (if on feature branch)
committer.commit_on_save('/path/to/graph.json', graph)
```

## Testing

Run the test suite:

```bash
# All GitAutoCommitter tests
python -m pytest tests/unit/test_graph_persistence.py::TestGitAutoCommitter* -v

# Specific test class
python -m pytest tests/unit/test_graph_persistence.py::TestGraphValidation -v

# Run demo
PYTHONPATH=/home/user/Opus-code-test python examples/git_auto_committer_demo.py
```

## Integration with GraphWAL

GitAutoCommitter works alongside GraphWAL (Write-Ahead Log) for complete persistence:

```python
from cortical.reasoning.graph_persistence import GraphWAL, GitAutoCommitter

# WAL for crash recovery
wal = GraphWAL('reasoning_wal')

# Git for version control
committer = GitAutoCommitter(mode='debounced')

# Log operation to WAL
wal.log_add_node('Q1', NodeType.QUESTION, 'Test question')

# Apply to graph
graph = ThoughtGraph()
wal.replay(graph)

# Commit to git
committer.commit_on_save('/path/to/graph.json', graph)
```

## Best Practices

1. **Use debounced mode for frequent saves**
   - Prevents commit spam
   - Reduces git history noise
   - Improves performance

2. **Always validate before important commits**
   - Use manual mode for critical operations
   - Check graph integrity first
   - Create backups before risky operations

3. **Configure protected branches correctly**
   - Always include `main` and `master`
   - Add production/release branches
   - Override only when absolutely necessary

4. **Clean up debounced committers**
   - Call `cleanup()` before destroying
   - Prevents timer leaks
   - Ensures pending commits complete or cancel

5. **Use auto_push sparingly**
   - Enable only for feature branches
   - Never on protected branches
   - Consider manual review before pushing

## Troubleshooting

### Commits Not Happening

**Problem:** Debounced commits not triggering

**Solution:**
- Wait for full debounce period
- Check timer isn't cancelled prematurely
- Call `cleanup()` to force pending commit cancellation

### Validation Failing

**Problem:** "All nodes are orphaned"

**Solution:**
- Add edges between nodes
- Ensure graph has meaningful connections
- Use manual mode if intentional

### Push Blocked

**Problem:** Auto-push not working

**Solution:**
- Check if branch is protected
- Verify `auto_push=True` is set
- Use `force_protected=True` if intentional (not recommended)

## Implementation Notes

- **Thread-safe timers:** Uses `threading.Timer` for debouncing
- **Subprocess calls:** All git operations use `subprocess.run()`
- **Timeout protection:** Git commands have 5-30 second timeouts
- **Exception handling:** Catches `CalledProcessError`, `TimeoutExpired`, `FileNotFoundError`
- **No external dependencies:** Uses only Python stdlib

## Future Enhancements

- [ ] Git hook integration (pre-commit, post-commit)
- [ ] Commit message templates
- [ ] Automatic squashing of debounced commits
- [ ] Remote branch tracking
- [ ] Conflict detection and resolution
- [ ] Integration with GitHub/GitLab APIs

## See Also

- [graph_persistence.py](../cortical/reasoning/graph_persistence.py) - Full implementation
- [test_graph_persistence.py](../tests/unit/test_graph_persistence.py) - Comprehensive tests
- [git_auto_committer_demo.py](../examples/git_auto_committer_demo.py) - Interactive demo
- [thought_graph.py](../cortical/reasoning/thought_graph.py) - ThoughtGraph class
