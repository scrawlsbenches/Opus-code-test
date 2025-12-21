# GoT Stress Test Scenario: Parallel Documentation Sprint

## Purpose

Push the Graph of Thought system to its limits to identify:
1. Performance bottlenecks with many nodes/edges
2. Handoff primitive reliability under load
3. Edge inference accuracy
4. Multi-agent coordination gaps
5. Query language limitations

## Scenario: "Document Everything"

A sprint where 5 parallel sub-agents document all undocumented features while:
- Logging every decision to GoT
- Creating handoffs between agents
- Inferring edges from commits
- Querying dependencies continuously

### Phase 1: Initialization (Director)

```bash
# Create sprint with 10 tasks
python scripts/got_utils.py sprint create "Stress Test Sprint" \
  --goal "Document all undocumented GoT features" \
  --tasks 10

# Log initial decision
python scripts/got_utils.py decision log \
  "Run parallel documentation sprint with 5 agents" \
  --rationale "Test GoT under load" \
  --affects task:T-sprint-1,task:T-sprint-2,task:T-sprint-3
```

### Phase 2: Parallel Agent Spawn (5 Agents)

Spawn 5 agents simultaneously, each owning specific docs:

| Agent | Responsibility | Files |
|-------|---------------|-------|
| docs-agent-1 | Reasoning Trace Logger | `docs/got-decision-logging.md` |
| docs-agent-2 | Auto-Edge Inference | `docs/got-auto-edge-inference.md` |
| docs-agent-3 | Handoff Primitives | `docs/got-handoff-guide.md` |
| docs-agent-4 | Query Language | Update `docs/got-query-language.md` |
| docs-agent-5 | Parallel Coordination | `docs/got-parallel-agents.md` |

Each agent:
1. Logs a decision: "Taking ownership of X documentation"
2. Creates edges: decision MOTIVATES task
3. Works for ~5 minutes
4. Initiates handoff to verification agent
5. Completes handoff with results

### Phase 3: Handoff Storm

All 5 agents complete simultaneously, creating 5 handoffs:

```bash
# Agent 1 initiates
python scripts/got_utils.py handoff initiate task:T-doc-1 \
  --target verifier --instructions "Verify decision logging docs"

# Verifier accepts all 5
for i in 1 2 3 4 5; do
  python scripts/got_utils.py handoff accept handoff:H-$i --agent verifier
done

# Verifier completes all 5
for i in 1 2 3 4 5; do
  python scripts/got_utils.py handoff complete handoff:H-$i \
    --agent verifier --result '{"status": "verified"}'
done
```

### Phase 4: Edge Inference Load

After agents commit, infer edges from all recent commits:

```bash
# Infer edges from last 20 commits
python scripts/got_utils.py infer --commits 20

# Check edge count increased
python scripts/got_utils.py stats
```

Expected: 15-30 new edges from commit message parsing.

### Phase 5: Query Stress

Run complex queries against the now-dense graph:

```bash
# Find all blocked tasks
python scripts/got_utils.py query "blocked tasks"

# Find path between distant nodes
python scripts/got_utils.py query "path from decision:D-001 to task:T-doc-5"

# Get all relationships for a task
python scripts/got_utils.py query "relationships task:T-doc-1"

# Find what depends on a decision
python scripts/got_utils.py query "what depends on decision:D-001"
```

### Phase 6: Compaction Under Load

With ~300+ events, test compaction:

```bash
# Count events before
wc -l .got/events/*.jsonl

# Compact preserving 7 days
python scripts/got_utils.py compact --preserve-days 7

# Count events after
wc -l .got/events/*.jsonl

# Verify state preserved
python scripts/got_utils.py stats
```

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Total nodes | 250+ | `got_utils.py stats` |
| Total edges | 50+ | `got_utils.py stats` |
| Handoffs completed | 10+ | Count in events |
| Edge inference accuracy | >80% | Manual review |
| Query response time | <2s | Time queries |
| Compaction preserves state | 100% | Compare before/after |

## Expected Bottlenecks

1. **Event file I/O**: Many small writes may slow down
2. **Graph rebuild time**: Full replay from events
3. **Query traversal**: Deep graph traversal for complex queries
4. **Handoff serialization**: JSON encoding/decoding overhead

## Improvement Areas to Identify

1. **Need caching?** If queries are slow, add query result cache
2. **Need indexing?** If node lookup is slow, add ID index
3. **Need batching?** If many small events, add batch write
4. **Need streaming?** If graph too large for memory, add streaming replay

## How to Run

```bash
# Full stress test (takes ~15 minutes with agents)
python scripts/run_got_stress_test.py

# Quick stress test (synthetic events only)
python scripts/run_got_stress_test.py --quick

# Check results
python scripts/got_utils.py stats
cat .got/stress-test-results.json
```

## Post-Test Analysis

After running, analyze:

1. **Event log size**: How much storage used?
2. **Replay time**: How long to rebuild from events?
3. **Query performance**: Which queries are slow?
4. **Edge density**: Ratio of edges to nodes
5. **Handoff reliability**: Any failed handoffs?

## Cleanup

```bash
# Archive stress test data
mv .got/events .got/events-stress-test-$(date +%Y%m%d)

# Create fresh state
python scripts/got_utils.py init --force
```
