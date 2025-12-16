# Refactoring Backup Plan

**Orchestration ID:** OP-20251215-214001-3dc2c38f
**Created:** 2024-12-15 21:40 UTC
**Branch:** claude/refactor-large-files-Om7nP

## Backup Files

| Original | Backup | Size |
|----------|--------|------|
| scripts/ml_data_collector.py | .refactor-backup/ml_data_collector.py | 151KB |
| tests/unit/test_processor_core.py | .refactor-backup/test_processor_core.py | 138KB |
| cortical/analysis.py | .refactor-backup/analysis.py | 95KB |
| tests/unit/test_analysis.py | .refactor-backup/test_analysis.py | 91KB |

## Rollback Procedures

### Quick Rollback (Single File)
```bash
# Restore ml_data_collector.py
cp .refactor-backup/ml_data_collector.py scripts/

# Restore test_processor_core.py
cp .refactor-backup/test_processor_core.py tests/unit/

# Restore analysis.py
cp .refactor-backup/analysis.py cortical/

# Restore test_analysis.py
cp .refactor-backup/test_analysis.py tests/unit/
```

### Full Rollback (All Changes)
```bash
# Restore all backups
cp .refactor-backup/*.py scripts/ 2>/dev/null
cp .refactor-backup/test_*.py tests/unit/
cp .refactor-backup/analysis.py cortical/

# Remove new package directories if created
rm -rf scripts/ml_collector/
rm -rf cortical/analysis/

# Remove split test files
rm -f tests/unit/test_processor_init.py
rm -f tests/unit/test_processor_documents.py
rm -f tests/unit/test_processor_metadata.py
rm -f tests/unit/test_processor_staleness.py
rm -f tests/unit/test_processor_layers.py
rm -f tests/unit/test_processor_config.py
rm -f tests/unit/test_processor_compute.py
rm -f tests/unit/test_processor_query.py
rm -f tests/unit/test_processor_wrappers.py
rm -f tests/unit/test_processor_coverage.py
rm -f tests/unit/test_processor_search.py
```

### Git Rollback
```bash
# Discard all uncommitted changes
git checkout -- .

# Or reset to specific commit
git reset --hard HEAD~N  # where N is number of commits to undo
```

## Verification Commands

```bash
# Quick test (smoke)
python -m pytest tests/smoke/ -v

# Unit tests
python -m pytest tests/unit/ -v

# Full suite
python -m pytest tests/ -x

# Check imports work
python -c "from scripts.ml_data_collector import *"
python -c "from cortical.analysis import *"
```

## Success Criteria

- [ ] All tests pass after each batch
- [ ] No import errors in refactored packages
- [ ] CLI entry points still work
- [ ] Token counts under 25000 for all new files
- [ ] No functionality regression
