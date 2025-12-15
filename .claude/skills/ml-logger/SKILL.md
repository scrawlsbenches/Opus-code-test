# ML Data Logger Skill

Log chat exchanges and actions for training a project-specific micro-model.

## When to Use

Use this skill when you want to:
- Log a significant query/response exchange for ML training
- Start or end a development session
- Check data collection progress

## Quick Commands

### Log Current Exchange

After completing a significant task, log it:

```bash
python .claude/hooks/session_logger.py \
    --query "USER_QUERY_HERE" \
    --response "SUMMARY_OF_RESPONSE" \
    --files-read FILE1 FILE2 \
    --files-modified FILE3 \
    --tools Read,Edit,Bash
```

### Session Management

```bash
# Start session at beginning of work
python .claude/hooks/session_logger.py --start-session

# End session when done
python .claude/hooks/session_logger.py --end-session --summary "What was accomplished"
```

### Check Progress

```bash
python scripts/ml_data_collector.py stats
python scripts/ml_data_collector.py estimate
```

### Generate Session Handoff

Create a markdown summary of the current session for context handoff:

```bash
python scripts/ml_data_collector.py handoff
```

This generates a document with:
- Session summary (ID, duration, exchanges)
- Key work done (summarized from queries)
- Files touched (modified and referenced)
- Related commits from the session
- Suggested next steps (based on patterns)

## What Gets Collected

| Data Type | Contents | Use |
|-----------|----------|-----|
| **Query** | User's question/request | Input for generation |
| **Response** | Assistant's answer summary | Target for generation |
| **Files Read** | Which files were examined | Context prediction |
| **Files Modified** | Which files were changed | Change prediction |
| **Tools Used** | Which tools were invoked | Workflow prediction |
| **Feedback** | User satisfaction | Quality filtering |

## Integration Pattern

After completing significant work:

1. Summarize what the user asked
2. Summarize what was done
3. List files touched
4. Log the exchange

## Example

```bash
# After fixing a bug
python .claude/hooks/session_logger.py \
    --query "Fix the timeout issue in compute_all" \
    --response "Increased timeout from 10s to 30s in processor.py, added retry logic" \
    --files-read cortical/processor.py tests/test_processor.py \
    --files-modified cortical/processor.py \
    --tools Read,Edit,Bash \
    --feedback positive
```

## Why This Matters

This data trains a micro-model that learns:
- YOUR project's patterns
- YOUR coding style
- YOUR common workflows
- YOUR file relationships

The model becomes a personalized assistant that understands THIS codebase deeply.

## Disabling Collection

To temporarily disable ML data collection:

```bash
export ML_COLLECTION_ENABLED=0
```

Stats and validation commands still work when disabled. Only collection (commit, chat, action) is blocked.

## CI Integration

Record CI results for commits:

```bash
# After CI run
python scripts/ml_data_collector.py ci set \
    --commit $(git rev-parse HEAD) \
    --result pass \
    --coverage 89.5 \
    --tests-passed 150
```

This enables the model to learn which code changes pass/fail CI.
