# Knowledge Transfer: GoT ID Normalization Fix

**Date:** 2025-12-21
**Branch:** claude/test-task-workflow-gRXq7
**Related Commits:** 6964017e

## Summary

Fixed a critical bug causing 35+ "Cannot update non-existent node" errors during GoT event replay. The root cause was an ID format mismatch between `node.create` and `node.update` events.

## Problem Diagnosed

During GoT validation, many warnings appeared:
```
Event 795: Cannot update non-existent node T-20251221-020047-0c52
```

### Root Cause Analysis

1. **node.create events** stored IDs with prefix: `task:T-20251220-231154-d6ec`
2. **node.update events** stored IDs without prefix: `T-20251220-231154-d6ec`
3. When `rebuild_graph_from_events()` tried to update nodes, the lookup failed because the IDs didn't match

## Solution Implemented

Added ID normalization to both `node.update` and `node.delete` handlers in `scripts/got_utils.py`:

```python
elif event_type == "node.update":
    node_id = event["id"]
    changes = event.get("changes", {})

    # Normalize ID - try with and without task: prefix
    actual_id = node_id
    if node_id not in graph.nodes:
        # Try with task: prefix if it looks like a task ID
        if node_id.startswith("T-") and f"task:{node_id}" in graph.nodes:
            actual_id = f"task:{node_id}"
        # Try without task: prefix
        elif node_id.startswith("task:") and node_id[5:] in graph.nodes:
            actual_id = node_id[5:]

    if actual_id in graph.nodes:
        for key, value in changes.items():
            graph.nodes[actual_id].properties[key] = value
    else:
        logger.warning(f"Event {event_num}: Cannot update non-existent node {node_id}")
```

Same logic applied to `node.delete` handler.

## Results

- **Before fix:** 35+ "Cannot update non-existent node" errors
- **After fix:** 0 node update errors
- **Remaining issue:** 1 edge creation error with malformed node ID (comma-concatenated IDs)

## Related Discoveries

### Already Implemented Components

The user requested implementation of three "critical tasks" but research revealed they're already complete:

1. **ContextPool** (`cortical/reasoning/context_pool.py`) - 396 lines
   - Publish/query/subscribe pattern
   - Conflict detection with multiple strategies
   - TTL-based expiration
   - Persistence support

2. **NestedLoopExecutor** (`cortical/reasoning/nested_loop.py`) - 487 lines
   - Hierarchical QAPV cycle management
   - Depth limiting, result aggregation
   - Parent-child relationship tracking

3. **ClaudeCodeSpawner** (`cortical/reasoning/claude_code_spawner.py`) - 700+ lines
   - Task tool configuration generation
   - Subprocess spawning with metrics
   - Result parsing and aggregation

All three have corresponding tests in `tests/unit/` with 53 tests passing.

## Bugs to Avoid (Critical Knowledge)

1. **Edge rebuild:** Use `from_id`/`to_id`, NOT `source_id`/`target_id` in `add_edge()`
2. **EdgeType lookup:** Use `EdgeType[name]` with try/except, NOT `hasattr()`
3. **ID normalization:** Always try both `task:T-XXX` and `T-XXX` formats when looking up nodes
4. **Priority executor:** Skip query echo line when parsing blockers

## GoT Current State

```
Nodes: 409
Tasks: 273
Edges: 330
Edge density: 0.81 edges/node
Orphan nodes: 147 (35.9%)
Edge rebuild rate: 100%
```

## Next Steps for Future Sessions

1. Fix remaining edge creation issue (malformed comma-concatenated node IDs)
2. Reduce orphan node rate (currently 35.9%)
3. Consider adding ID normalization to `edge.create` handler for consistency
4. Add GoT unit tests in `tests/unit/test_got.py`
