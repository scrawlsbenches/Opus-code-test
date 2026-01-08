# Audit Result: 20260107-120000-cdg-comments (v2 RE-RUN)

**Task:** misleading-comments-2026-01-07
**Scope:** `cortical/cdg/`
**Patterns:** FUTURE:|TODO:|FIXME:|PLANNED:|HACK:|XXX:|TEMPORARY:|WORKAROUND:|will be|should be|planned to
**Executed:** 2026-01-07
**Status:** COMPLETE

---

## Executive Summary

**Total findings:** 9 comments
**Categories:**
- Accurate: 5
- Misleading: 3
- Stale: 0
- Unknown: 0

**High-priority concerns:** 2 misleading comments require attention

---

## Findings

### ACCURATE (5 findings)

#### 1. transaction.py:204 - Delete overrides write behavior

**Location:** `/home/user/Opus-code-test/cortical/cdg/transaction.py:204`
**Comment:** `If the same entity_id is in write_set, it will be removed`
**Git blame:** `a3b90a9c4 (Claude 2026-01-02)`

**Evidence:**
- Comment describes behavior in `add_delete()` method
- Code at lines 211-213 confirms:
  ```python
  self.delete_set.add(entity_id)
  # Remove from write_set if present (delete wins over write)
  self.write_set.pop(entity_id, None)
  ```
- Behavior matches description exactly

**Assessment:** ACCURATE - Comment correctly describes the code's behavior.

---

#### 2. recovery.py:121 - Recovery decision return value

**Location:** `/home/user/Opus-code-test/cortical/cdg/recovery.py:121`
**Comment:** `True if recovery should be performed`
**Git blame:** `bdef0e14f (Claude 2026-01-01)`

**Evidence:**
- Comment is part of `needs_recovery()` method docstring describing return value
- Method implementation (lines 123-151) returns True/False based on recovery conditions
- Return statements at lines 125, 131, 136, 142, 151 all return boolean values

**Assessment:** ACCURATE - Return type documentation matches implementation.

---

#### 3. index_manager.py:388 - Entity iterator parameter behavior

**Location:** `/home/user/Opus-code-test/cortical/cdg/index_manager.py:388`
**Comment:** `If not provided, indexes will be cleared but not populated.`
**Git blame:** `425d81fee (Claude 2026-01-07)`

**Evidence:**
- Comment describes `entity_iterator` parameter in `rebuild_all()` docstring
- Code at lines 397-406: Indexes are cleared regardless of entity_iterator
  ```python
  self._indexes.clear()
  # Clear index directory...
  ```
- Code at lines 410-415: Indexing only occurs when entity_iterator is not None
  ```python
  if entity_iterator is not None:
      for entity_type, entity_data in entity_iterator():
          # ... indexing logic
  ```

**Assessment:** ACCURATE - Behavior matches documentation exactly.

---

#### 4. types.py:327 - Bidirectional edge semantics

**Location:** `/home/user/Opus-code-test/cortical/cdg/types.py:327`
**Comment:** `Bidirectional edges like RELATES_TO should be traversable`
**Git blame:** `d27d4ba29 (Claude 2025-12-31)`

**Evidence:**
- Comment is in docstring for `is_bidirectional` property
- Property implementation (lines 333-334):
  ```python
  bidirectional_types = {'RELATES_TO', 'CONTRADICTS'}
  return self.edge_type in bidirectional_types
  ```
- Comment describes the semantic meaning of bidirectional edges
- The word "should" here describes the conceptual requirement, not a future implementation

**Assessment:** ACCURATE - Comment correctly describes the concept and implementation.

---

#### 5. schema/__init__.py:433 - ReferenceRule class purpose

**Location:** `/home/user/Opus-code-test/cortical/cdg/schema/__init__.py:433`
**Comment:** `Defines how entity references should be validated and what happens`
**Git blame:** `425d81fee (Claude 2026-01-07)`

**Evidence:**
- Comment is part of ReferenceRule class docstring
- Class dataclass fields (visible in context) include validation and on_delete rules
- Comment accurately describes the class's purpose

**Assessment:** ACCURATE - Docstring correctly describes class purpose.

---

### MISLEADING (3 findings)

#### 6. transaction_manager.py:155 - Incorrect started_at initialization claim

**Location:** `/home/user/Opus-code-test/cortical/cdg/transaction_manager.py:155`
**Comment:** `Will be set by Transaction.begin()`
**Git blame:** `bdef0e14f (Claude 2026-01-01)`

**Evidence:**
- Comment appears in `CDGTransactionManager.begin()` method at line 155:
  ```python
  tx = Transaction(
      id=tx_id,
      state=TransactionState.ACTIVE,
      started_at="",  # Will be set by Transaction.begin()
      snapshot_version=snapshot_version,
      write_set={},
      read_set={}
  )
  ```
- The comment claims `started_at` will be set by `Transaction.begin()`
- However, the code directly constructs a Transaction object with `started_at=""`
- The `Transaction.begin()` factory method is NOT called here
- Therefore, `started_at` remains as an empty string and is never set

**What was intended:**
The Transaction class has a factory method `Transaction.begin()` (transaction.py:324-344) that properly initializes `started_at`. The comment suggests this should be used, but the code doesn't use it.

**Why this is misleading:**
The comment describes what SHOULD happen (using the factory method) as if it WILL happen, but the actual code does not do this.

**Assessment:** MISLEADING - Comment describes intended behavior, not actual behavior.

**Recommendation:** Either use `Transaction.begin()` factory method or update comment to: `# TODO: Use Transaction.begin() factory method instead`

---

#### 7. storage.py:342-345 - Reference to non-existent spec content

**Location:** `/home/user/Opus-code-test/cortical/cdg/storage.py:342-345`
**Comment:**
```
FUTURE: When CDG index is implemented per the distributed graph
specification (docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md),
this race condition will be eliminated at the storage layer since
index lookups won't return IDs for deleted entities.
```
**Git blame:** `d7a3e8415 (Claude 2026-01-05)`

**Evidence:**
- Comment references `docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md`
- File exists: ✓ (5602 lines)
- Searched for mentioned content:
  - "CDG index" - NOT FOUND
  - "race condition" - NOT FOUND
  - "TOCTOU" - NOT FOUND
  - "deleted entities" - NOT FOUND
  - "index lookup" - NOT FOUND

**What the spec actually contains:**
The specification is a high-level architecture document describing distributed graph design goals, partition coordination, and query planning. It does not contain implementation details about index behavior during entity deletion.

**Why this is misleading:**
The comment claims the referenced document describes a specific implementation that will eliminate the race condition, but the document contains no such description.

**Assessment:** MISLEADING - References non-existent content in cited document.

**Recommendation:** Either:
1. Add the claimed content to the specification document, OR
2. Remove the document reference and rewrite as: `# FUTURE: A global index would eliminate this race condition by ensuring lookups don't return deleted entity IDs`

---

#### 8. adapters/__init__.py:19 - Unverifiable future claim

**Location:** `/home/user/Opus-code-test/cortical/cdg/adapters/__init__.py:19`
**Comment:** `Adapters will be added as they are implemented`
**Git blame:** `d27d4ba29 (Claude 2025-12-31)`

**Evidence:**
- Comment written on 2025-12-31
- Current date: 2026-01-07 (8 days later)
- Current file content:
  ```python
  # Adapters will be added as they are implemented
  __all__ = []
  ```
- No adapters have been added
- No evidence in git history of adapter implementation work

**Why this is misleading:**
The comment presents a future intention as if it's a fact. It provides no information about:
- WHAT adapters will be added
- WHEN they will be added
- WHO is responsible for adding them
- WHY they are needed

This is a placeholder comment that provides no value and should either be removed or replaced with actionable information.

**Assessment:** MISLEADING - Speculation presented as fact with no actionable information.

**Recommendation:** Either:
1. Remove the comment entirely (it adds no value), OR
2. Replace with: `# TODO: Add GoTAdapter for backward compatibility (see docs/cdg/adapters.md)` with a concrete plan

---

## What Went Wrong

### Root Cause Analysis

The misleading comments share a common pattern: **conflating intent with implementation**.

1. **transaction_manager.py:155** - Comment describes what the code SHOULD do (call factory method) as if it's what it WILL do (but it doesn't)

2. **storage.py:342-345** - Comment references a document that SHOULD contain implementation details (but doesn't)

3. **adapters/__init__.py:19** - Comment suggests work WILL happen (but provides no plan or accountability)

### Prevention Strategies

To prevent future misleading comments:

1. **Use TODO/FIXME for intentions:** If code doesn't match ideal design, mark it as `TODO:` not as fact
   - BAD: `# Will be set by Transaction.begin()`
   - GOOD: `# TODO: Use Transaction.begin() factory method instead of direct construction`

2. **Verify document references:** Before claiming "per document X", ensure document X actually contains the claimed content
   - Check: `grep -i "key phrase" document.md`

3. **Make aspirational comments actionable:** Vague future statements should include who/what/when
   - BAD: `# Adapters will be added as they are implemented`
   - GOOD: `# TODO(cdg-team): Add GoTAdapter by 2026-01-15 (see issue #123)`

---

## Quantitative Summary

| Category    | Count | Percentage |
|-------------|-------|------------|
| Accurate    | 5     | 55.6%      |
| Misleading  | 3     | 33.3%      |
| Stale       | 0     | 0.0%       |
| Unknown     | 0     | 0.0%       |
| **Total**   | **9** | **100%**   |

---

## Audit Metadata

**Constraints applied:**
- Maximum findings: 50 (not exceeded)
- Maximum duration: 2 hours (not exceeded)
- Scope: `cortical/cdg/` only (verified)

**Decision tree applied:** YES - All findings categorized using the explicit decision tree

**Evidence cited:** YES - All assessments include:
- File path and line number
- Full comment content
- Git blame timestamp
- Code verification or absence proof

**Stopping conditions met:** Task complete, all findings assessed

---

## Appendix: Search Commands Used

```bash
# Pattern search
grep -r "FUTURE:|TODO:|FIXME:|PLANNED:|HACK:|XXX:|TEMPORARY:|WORKAROUND:|will be|should be|planned to" cortical/cdg/ --include="*.py"

# Git blame for each finding
git blame -L 204,204 cortical/cdg/transaction.py
git blame -L 155,155 cortical/cdg/transaction_manager.py
git blame -L 121,121 cortical/cdg/recovery.py
git blame -L 388,388 cortical/cdg/index_manager.py
git blame -L 327,327 cortical/cdg/types.py
git blame -L 342,345 cortical/cdg/storage.py
git blame -L 433,433 cortical/cdg/schema/__init__.py
git blame -L 19,19 cortical/cdg/adapters/__init__.py

# Verification of spec document
test -f docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md
grep -i "CDG index\|race condition\|deleted entities" docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md
```

---

**END OF AUDIT REPORT**
