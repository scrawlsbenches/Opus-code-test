# Log Chat Exchange for ML Training

Log significant chat exchanges to train a project-specific micro-model.

## When to Use

After completing significant work (bug fix, feature implementation, debugging session), log the exchange:

```bash
python .claude/hooks/session_logger.py \
    --query "USER_QUERY_HERE" \
    --response "BRIEF_SUMMARY_OF_RESPONSE" \
    --files-read file1.py file2.py \
    --files-modified file3.py \
    --tools Read,Edit,Bash,Grep \
    --feedback positive
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `--query` | The user's original question (2-3 sentences) |
| `--response` | Summary of what was accomplished (3-5 sentences) |
| `--files-read` | Files that were examined |
| `--files-modified` | Files that were changed |
| `--tools` | Tools used: Read, Edit, Write, Bash, Grep, Glob |
| `--feedback` | positive, negative, or neutral |

## Session Management

```bash
# Start session (optional - auto-starts on first log)
python scripts/ml_data_collector.py session start

# End session with summary
python scripts/ml_data_collector.py session end --summary "What was accomplished"

# Check session status
python scripts/ml_data_collector.py session status
```

## Why This Matters

This data trains a micro-model specific to THIS project that learns your coding patterns, common workflows, and project-specific terminology.
