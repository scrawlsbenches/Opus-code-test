# ML Data Collection for Ephemeral Environments

## Problem Statement

Claude Code Web runs in ephemeral environments that are destroyed after session ends. The current ML data collection stores files in `.git-ml/chats/` and `.git-ml/sessions/` directories that are gitignored, causing data loss when the environment is destroyed.

**Current Status:**
- Commits: 576 collected (persisted via JSONL)
- Sessions: ~4% of target (local files lost)
- Chats: ~225% of target (via transcript processing)

## Architecture Decision

### Chosen Solution: Git-Tracked JSONL Append (Solution B)

Store all chat and session data in append-only JSONL files in `.git-ml/tracked/`:

```
.git-ml/tracked/           [GIT-TRACKED]
├── commits.jsonl          # Lightweight commit summaries (existing)
├── sessions.jsonl         # Full session data (new - migrate from sessions/)
├── chats.jsonl           # Full chat exchanges (new - migrate from chats/)
├── orchestration.jsonl    # Orchestration patterns (existing)
└── actions.jsonl         # Tool usage summaries (new - migrate from actions/)
```

### Why JSONL Append-Only?

1. **Merge-conflict-free** - Append-only means parallel agents never conflict
2. **Git-efficient** - Only new lines added, small diffs
3. **Ephemeral-safe** - Git-tracked files persist automatically
4. **Query-friendly** - Can grep, filter, process line-by-line
5. **Compactable** - Periodic deduplication via `--compact` flag

### Migration Path

#### Phase 1: Add JSONL Writers (Now)
- Add `append_chat_to_jsonl()` function alongside existing `save_chat_entry()`
- Add `append_session_to_jsonl()` alongside existing `end_session()`
- Stop hook calls both (dual-write) for backward compatibility

#### Phase 2: Update Gitignore (After Verification)
- Remove `.git-ml/chats/` and `.git-ml/sessions/` from gitignore
- Or: Keep them gitignored as "local cache" and use JSONL as source of truth

#### Phase 3: Deprecate File-Based Storage (Future)
- Remove individual file writes once JSONL proven stable
- Keep file-based format for local development (faster reads)

## Implementation Details

### Chat Entry Format (chats.jsonl)

```json
{"id": "chat-20251216-150332-abc123", "timestamp": "2025-12-16T15:03:32Z", "session_id": "13a84f73", "input": "...", "output": "...", "tools": ["Read", "Edit"], "files": ["cortical/processor.py"]}
```

### Session Entry Format (sessions.jsonl)

```json
{"id": "2025-12-16_13a84f73", "started_at": "...", "ended_at": "...", "summary": "...", "chat_count": 5, "commits": ["abc123"]}
```

### Stop Hook Changes

```bash
# In ml-session-capture-hook.sh, after process_transcript():
python scripts/ml_data_collector.py session sync-to-jsonl
```

### New CLI Command

```bash
# Sync local files to JSONL (idempotent, safe to run multiple times)
python scripts/ml_data_collector.py session sync-to-jsonl

# Compact JSONL files (dedup, sort, compress)
python scripts/ml_data_collector.py tracked compact
```

## File Size Projections

| Data Type | Current Size | Projected at 5000 chats | Notes |
|-----------|--------------|------------------------|-------|
| commits.jsonl | 497K | ~1M | Already working |
| sessions.jsonl | ~5K | ~50K | Small, minimal growth |
| chats.jsonl | 0 | ~5M | Main growth area |
| actions.jsonl | 0 | ~500K | Tool usage patterns |
| **Total tracked/** | ~500K | ~6.5M | Acceptable for training repo |

## Alternatives Considered

### A. Commit During Stop Hook
- **Pro**: Simple conceptual model
- **Con**: Git operations in hook can timeout, race conditions

### C. Transactional Commit
- **Pro**: Most robust error handling
- **Con**: Complex implementation, overkill for current needs

### D. Manual /ml-log
- **Pro**: Works immediately
- **Con**: Doesn't scale, requires manual intervention

## Success Criteria

1. Sessions persist across ephemeral environment restarts
2. Chats captured even when environment destroyed during session
3. No merge conflicts from parallel agents
4. Stop hook completes in <5 seconds
5. Data queryable via grep/jq

## Implementation Checklist

- [ ] Add `append_to_tracked_jsonl()` utility function
- [ ] Update `end_session()` to also write to sessions.jsonl
- [ ] Update Stop hook to sync chats after transcript processing
- [ ] Add `sync-to-jsonl` subcommand
- [ ] Add `compact` subcommand for JSONL maintenance
- [ ] Update CLAUDE.md with new architecture
- [ ] Test in ephemeral environment simulation

## References

- Task T-20251215-145621-16f3-001: Investigate ML data collection for ephemeral environment
- Task T-20251215-145630-16f3-003: Design session capture strategy (completed)
- docs/ml-milestone-thresholds.md: Why 500 commits needed for training
