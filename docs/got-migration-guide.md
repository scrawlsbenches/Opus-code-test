# GoT Migration Guide: Event-Sourced to Transactional Backend

This guide walks you through migrating from the event-sourced GoT backend (`.got/`) to the transactional backend (`.got-tx/`).

---

## Overview

| Aspect | Event-Sourced | Transactional |
|--------|---------------|---------------|
| Storage | Append-only event log | Entity JSON files |
| Performance | O(n) replay on startup | O(1) direct access |
| ACID | Eventual consistency | Full ACID guarantees |
| Recovery | Event replay | WAL + snapshots |
| Directory | `.got/` | `.got-tx/` |

---

## Prerequisites

1. **Backup your data** (the migration script does this automatically)
2. **Python 3.8+** with the project dependencies installed
3. **No active GoT operations** during migration

---

## Quick Start

```bash
# Step 1: Analyze (dry run - no changes made)
python scripts/migrate_got.py --dry-run

# Step 2: Migrate
python scripts/migrate_got.py

# Step 3: Verify
python scripts/migrate_got.py --verify
```

---

## Detailed Steps

### Step 1: Analyze Your Data

Before migrating, analyze your current data:

```bash
python scripts/migrate_got.py --dry-run
```

**Expected output:**
```
=== GoT Migration Analysis ===
Source: .got/
Target: .got-tx/

Entities found:
  Tasks: 279
  Decisions: 5
  Edges: 12

DRY RUN - No changes made
```

### Step 2: Run the Migration

```bash
python scripts/migrate_got.py
```

**What happens:**
1. Creates backup at `.got-backup-YYYYMMDD-HHMMSS/`
2. Parses all events from `.got/events/`
3. Replays events to build current state
4. Writes entities to `.got-tx/entities/`
5. Creates verification checksums

**Expected output:**
```
=== GoT Migration ===
Source: .got/
Target: .got-tx/
Backup: .got-backup-20251221-174624/

Parsing events... done (1,234 events)
Building state... done
Writing entities... done

Migration complete:
  Tasks: 279
  Decisions: 5
  Edges: 12

Run with --verify to confirm migration
```

### Step 3: Verify Migration

```bash
python scripts/migrate_got.py --verify
```

**Expected output:**
```
=== Migration Verification ===
Source: .got/
Target: .got-tx/

Checking entity counts... ✓
Checking data integrity... ✓
Checking edge references... ✓

Verification PASSED
```

---

## Using the New Backend

### Automatic Detection

After migration, the CLI automatically detects the transactional backend:

```bash
# Auto-detects .got-tx/ and uses transactional backend
python scripts/got_utils.py task list
```

### Explicit Backend Selection

Force a specific backend:

```bash
# Use transactional backend explicitly
python scripts/got_utils.py --backend transactional task list

# Use event-sourced backend explicitly (if .got/ still exists)
python scripts/got_utils.py --backend event-sourced task list
```

### Environment Variable

```bash
# Set default backend via environment
export GOT_BACKEND=transactional
python scripts/got_utils.py task list
```

---

## ID Format Change

The transactional backend uses unprefixed IDs:

| Backend | ID Format |
|---------|-----------|
| Event-sourced | `task:T-20251221-030434-3529a1b2` |
| Transactional | `T-20251221-030434-3529a1b2` |

**Backward compatibility:** The CLI accepts both formats. Old prefixed IDs are automatically stripped.

---

## Rollback Procedure

If you need to revert to the event-sourced backend:

```bash
# Step 1: Remove transactional directory
mv .got-tx .got-tx-backup

# Step 2: Restore event-sourced directory
mv .got-backup-YYYYMMDD-HHMMSS .got

# Step 3: Force event-sourced backend
export GOT_BACKEND=event-sourced
python scripts/got_utils.py task list
```

---

## Troubleshooting

### Migration Fails with "Permission denied"

```bash
# Check directory permissions
ls -la .got/ .got-tx/

# Ensure write access
chmod -R u+w .got-tx/
```

### Migration Fails with "Corrupted event"

The migration script skips corrupted events with a warning:

```
Warning: Skipping corrupted event at line 1234: Invalid JSON
```

This is safe - the migration continues with valid events.

### Verification Fails with "Missing entity"

This indicates the migration was incomplete. Re-run:

```bash
# Remove partial migration
rm -rf .got-tx/

# Re-run migration
python scripts/migrate_got.py
```

### CLI Shows "0 tasks" After Migration

Ensure auto-detection is working:

```bash
# Check which backend is being used
GOT_DEBUG=1 python scripts/got_utils.py task list

# Force transactional
python scripts/got_utils.py --backend transactional task list
```

---

## Performance Comparison

| Operation | Event-Sourced | Transactional | Improvement |
|-----------|---------------|---------------|-------------|
| Startup (1000 tasks) | ~2.5s | ~0.1s | 25x faster |
| Create task | ~50ms | ~5ms | 10x faster |
| Query task | ~100ms | ~1ms | 100x faster |
| List all tasks | ~500ms | ~50ms | 10x faster |

---

## Migration Script Options

```bash
python scripts/migrate_got.py [OPTIONS]

Options:
  --dry-run       Analyze without making changes
  --verify        Verify existing migration
  --source PATH   Source directory (default: .got/)
  --target PATH   Target directory (default: .got-tx/)
  --no-backup     Skip backup creation (not recommended)
  --verbose       Show detailed progress
```

---

## FAQ

### Can I run both backends simultaneously?

No. Use one backend at a time. The CLI auto-detects which to use based on:
1. `--backend` CLI flag
2. `GOT_BACKEND` environment variable
3. Presence of `.got-tx/entities/` directory

### Is the migration reversible?

Yes. The original data is backed up and can be restored (see Rollback Procedure).

### What about edge references?

Edges are migrated with updated ID references. The migration handles the `task:` prefix stripping automatically.

### Can I migrate incrementally?

No. The migration is a one-time full conversion. For ongoing sync, use git with the transactional backend.

---

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Run with `--verbose` for detailed logs
3. Open an issue with the error message and your Python version

