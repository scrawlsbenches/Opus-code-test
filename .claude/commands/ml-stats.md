---
description: Show ML data collection statistics and training estimates
---
# ML Collection Statistics

Show ML data collection progress and training viability estimates.

## Instructions

Run these commands to check ML data collection status:

```bash
# Show collection statistics
python scripts/ml_data_collector.py stats

# Show training estimates and timeline
python scripts/ml_data_collector.py estimate
```

## What This Shows

- **Data counts**: Commits, chats, actions, sessions collected
- **Data sizes**: Storage used by each data type
- **Training milestones**: Progress toward file prediction, commit messages, code suggestions
- **Time estimates**: How long until training becomes viable

## Disabling Collection

To disable ML data collection:

```bash
export ML_COLLECTION_ENABLED=0
```

This stops collection but still allows viewing stats.
