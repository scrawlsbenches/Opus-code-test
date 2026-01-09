# Director Sprint Completion Plan

**Created:** 2025-12-28
**Branch:** `claude/plan-future-work-SMHv5`
**Objective:** Complete 4 near-finished sprints using parallel sub-agent delegation

---

## Sprint Status Overview

| Sprint | Progress | Pending Tasks | Priority |
|--------|----------|---------------|----------|
| **S-025: Index Safety & Testing** | 75% (9/12) | 3 tasks | 1st (Safety) |
| **S-026: Schema Validation** | 86% (6/7) | 1 task | 2nd (Validation) |
| **S-022: Benchmarks & Evaluation** | 86% (6/7) | 1 task | 3rd (Metrics) |
| **S-021: Training Pipeline** | 95% (19/20) | 1 task | 4th (Enhancement) |

**Total remaining:** 6 tasks

---

## Part 1: S-025 Index Safety & Testing

**Theme:** *"Make the index bulletproof"*

### Task 1.1: Schema Validation Tests
- **ID:** `T-20251226-112830-8c315485`
- **Priority:** Medium
- **Complexity:** Low (test-only, no production code changes)

**Sub-Agent Delegation:**
```
## Task: Add schema validation tests for QueryIndexManager

### Files to modify
`tests/unit/test_got_query_index.py` (or create if needed)

### Tests to add
1. Test invalid status values (e.g., "bogus_status") → should raise ValueError
2. Test invalid priority values (e.g., "super_high") → should raise ValueError
3. Test missing required fields (e.g., no title) → should raise ValueError
4. Test type checking (e.g., priority as int instead of string)
5. Test enum validation for EdgeType

### Reference
Look at `cortical/got/api.py` for the QueryIndexManager class
Look at existing tests in `tests/unit/test_got*.py` for patterns

### Verification
pytest tests/unit/test_got_query_index.py -v

### DO NOT
- Modify production code (validation should already exist)
- Create new files outside tests/
```

---

### Task 1.2: Performance Tests for Index
- **ID:** `T-20251226-112824-d50defff`
- **Priority:** Medium
- **Complexity:** Medium (timing measurements)

**Sub-Agent Delegation:**
```
## Task: Add performance tests for index operations

### Files to modify
`tests/performance/test_got_index_perf.py` (create)

### Tests to add
1. Test index build time with 100/500/1000 tasks → should be < 1s/2s/5s
2. Test query response time → should be < 100ms for typical queries
3. Test cache hit performance → should be 10x faster than uncached
4. Test concurrent access performance under load

### Reference
Look at `tests/performance/` for existing performance test patterns
Use `@pytest.mark.slow` marker

### Verification
pytest tests/performance/test_got_index_perf.py -v --timeout=60

### DO NOT
- Modify production code
- Make tests that take > 30s individually
```

---

### Task 1.3: Observability/Profiling Tests
- **ID:** `T-20251226-112837-e0b24627`
- **Priority:** Low
- **Complexity:** Low

**Sub-Agent Delegation:**
```
## Task: Add observability and profiling tests for index

### Files to modify
`tests/unit/test_got_observability.py` (create)

### Tests to add
1. Test that timing metrics are collected during indexing
2. Test that cache hit/miss ratios are tracked
3. Test that query execution plans can be explained
4. Test log output for debug/info levels

### Reference
Look at `cortical/observability.py` for metrics patterns

### Verification
pytest tests/unit/test_got_observability.py -v

### DO NOT
- Add new production observability (just test existing)
```

---

## Part 2: S-026 Schema Validation Hardening

**Theme:** *"Consistent identifiers everywhere"*

### Task 2.1: Consolidate Sprint ID Format
- **ID:** `T-20251226-141441-08b2860f`
- **Priority:** Low
- **Complexity:** Low (documentation/consistency task)

**Sub-Agent Delegation:**
```
## Task: Consolidate Sprint ID format for consistency

### Current state
Sprint IDs have mixed formats:
- Legacy: S-sprint-NNN-slug (e.g., S-sprint-017-spark-slm)
- New: S-YYYYMMDD-HHMMSS-hash (e.g., S-20251227-211213-ae934eab)

### Action needed
1. Document the canonical format in CLAUDE.md
2. Add validation to ensure new sprints use consistent format
3. Create migration guide for legacy format (don't force migration)

### Files to modify
- `CLAUDE.md` - Add Sprint ID format documentation
- `cortical/got/api.py` - Add format validation to create_sprint()

### Verification
python -m cortical.got sprint list  # Verify both formats work
pytest tests/unit/test_got_api.py -v -k sprint

### DO NOT
- Force migration of existing sprint IDs
- Break backwards compatibility
```

---

## Part 3: S-022 Benchmarks & Evaluation

**Theme:** *"Better developer experience"*

### Task 3.1: Decision Logging UX Improvement
- **ID:** `T-20251224-184728-21e04fff`
- **Priority:** Low
- **Complexity:** Low (CLI enhancement)

**Sub-Agent Delegation:**
```
## Task: Improve decision logging UX with task linkage prompt

### Current behavior
`python -m cortical.got decision log "Decision"` creates orphan decision

### Desired behavior
After logging decision, prompt:
"Link this decision to a task? (Enter task ID or press Enter to skip): "
If task ID provided, create JUSTIFIES edge automatically

### Files to modify
`scripts/got_utils.py` - cmd_decision_log function

### Implementation
1. After decision is created, show recent in_progress tasks
2. Prompt for task ID (optional)
3. If provided, call manager.add_edge(decision_id, task_id, EdgeType.JUSTIFIES)

### Verification
python -m cortical.got decision log "Test decision" --rationale "Testing"
# Should prompt for task linkage

### DO NOT
- Make linkage mandatory
- Change decision creation logic
```

---

## Part 4: S-021 Training Pipeline

**Theme:** *"Scale the training data"*

### Task 4.1: Streaming/Pagination for Large Results
- **ID:** `T-20251224-212849-9d283fe8`
- **Priority:** Medium
- **Complexity:** Medium (API enhancement)

**Sub-Agent Delegation:**
```
## Task: Add streaming/pagination for large result sets

### Current behavior
Query results return all matches at once, which can be slow for large result sets

### Desired behavior
Add pagination support:
- `query.limit(100).offset(0)` for first page
- `query.limit(100).offset(100)` for second page
- Optionally: streaming iterator for memory efficiency

### Files to modify
- `cortical/got/query_builder.py` - Add limit() and offset() methods
- `cortical/got/api.py` - Update execute() to respect limits

### Implementation pattern
```python
# In Query class
def limit(self, n: int) -> "Query":
    self._limit = n
    return self

def offset(self, n: int) -> "Query":
    self._offset = n
    return self
```

### Verification
python -c "
from cortical.got import GoTManager
from cortical.got.query_builder import Query
m = GoTManager('.got')
results = Query(m).tasks().limit(10).execute()
print(f'Got {len(results)} results')
"

### DO NOT
- Break existing query API
- Remove support for full result sets
```

---

## Execution Strategy

### Batch 1: S-025 Tasks (Parallel - 3 agents)
All three S-025 tasks are independent test additions:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BATCH 1: INDEX SAFETY (Parallel)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Agent A: Schema validation tests (T-...-8c315485)                   │
│  Agent B: Performance tests (T-...-d50defff)                         │
│  Agent C: Observability tests (T-...-e0b24627)                       │
│                                                                       │
│  Wait for all → Run pytest to verify                                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Estimated time:** 15-20 minutes

### Batch 2: S-026 + S-022 Tasks (Parallel - 2 agents)
These are independent UX/documentation improvements:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BATCH 2: VALIDATION + UX (Parallel)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Agent D: Sprint ID consolidation (T-...-08b2860f)                   │
│  Agent E: Decision logging UX (T-...-21e04fff)                       │
│                                                                       │
│  Wait for all → Verify manually                                      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Estimated time:** 10-15 minutes

### Batch 3: S-021 Task (Single agent)
This requires careful API changes:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BATCH 3: TRAINING PIPELINE (Sequential)           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Agent F: Streaming/pagination (T-...-9d283fe8)                      │
│                                                                       │
│  After completion → Run full test suite                              │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Estimated time:** 15-20 minutes

---

## Verification Checklist

After all batches complete:

```bash
# 1. Run full test suite
python -m pytest tests/ -v --tb=short

# 2. Verify S-025 completion
python -m cortical.got sprint status S-025

# 3. Verify S-026 completion
python -m cortical.got sprint status S-026

# 4. Verify S-022 completion
python -m cortical.got sprint status S-022

# 5. Verify S-021 completion
python -m cortical.got sprint status S-021

# 6. Mark tasks complete
python -m cortical.got task complete T-20251226-112830-8c315485 --notes "Added schema validation tests"
python -m cortical.got task complete T-20251226-112824-d50defff --notes "Added performance tests"
python -m cortical.got task complete T-20251226-112837-e0b24627 --notes "Added observability tests"
python -m cortical.got task complete T-20251226-141441-08b2860f --notes "Documented Sprint ID format"
python -m cortical.got task complete T-20251224-184728-21e04fff --notes "Added task linkage prompt"
python -m cortical.got task complete T-20251224-212849-9d283fe8 --notes "Added limit/offset pagination"
```

---

## Risk Assessment

| Task | Risk | Mitigation |
|------|------|------------|
| Schema validation tests | Low | Test-only, no production impact |
| Performance tests | Low | May need timing adjustments |
| Observability tests | Low | May need to add observability first |
| Sprint ID consolidation | Low | Documentation + optional validation |
| Decision logging UX | Low | Additive change, backwards compatible |
| Streaming/pagination | Medium | API addition, needs careful testing |

---

## Success Criteria

- [ ] All 6 pending tasks completed
- [ ] Test suite passes (10,000+ tests)
- [ ] No regressions in existing functionality
- [ ] Sprint completion reaches 100% for all 4 sprints
- [ ] Documentation updated where needed

---

*Ready for Director orchestration. Invoke with `/director` command.*
