# Audit Reasoning Recovery Plan

**Date:** 2026-01-13
**Status:** COMPLETED
**Tests:** 118 passed, 0 failed

## Context

We recovered lost functionality from `scripts/audit_reasoning.py` that was lost during a crash/command migration. The tests served as specifications for what needed to be implemented.

## Completed

1. **CLI flags added to `reason.py`:**
   - `--show-rules`, `--show-state`, `--clear-state`
   - `--file-history`, `--add-rule`, `--aggregate`, `--no-save`

2. **Methods recovered in `AuditReasoner`:**
   - `get_stats()` - returns facts, rules, aggregate_strategy
   - `collect_rent()` - applies attention decay
   - `get_importance_trend()` - analyzes history for trend direction
   - `focus_on_high_risk_files()` - focuses attention on high-risk files
   - `query_with_aggregation()` - queries with multiple aggregation strategies

3. **Functions recovered:**
   - `analyze_with_reasoning()` - programmatic API for audit reasoning
   - `generate_reasoning_report()` - updated with `verbose` parameter

4. **Fixes applied:**
   - Added `use_importance` parameter alias to `query_file_risk()`
   - Fixed report header format to include "AUDIT REASONING"
   - Fixed STI calculation for traits (bug_prone, critical, multiple patterns)
   - Set VLTI automatically for "critical" trait in `assert_file_facts()`
   - Fixed test monkeypatches to use `cortical.audits.reasoning` module path

5. **Test updates:**
   - `TestCLIMain` - updated to use `reason.run(args)` instead of old `main()`
   - `TestAnalyzeWithReasoning` - updated to use correct imports
   - `TestMoreBranchCoverage` - fixed monkeypatches for module path
   - `test_focus_on_high_risk_files` - fixed to assert facts before querying
   - `test_query_file_risk_without_attention_and_importance` - fixed patterns

## Key Lesson Learned

**Tests are specifications, not obstacles.** When tests fail because code is missing:
- Read the tests to understand expected behavior
- Implement the missing functionality
- DO NOT delete tests to make them pass

## Files Modified

- `cortical/audits/reasoning.py` - Core implementation
- `cortical/cli/audit/reason.py` - CLI command
- `tests/unit/test_audit_reasoning_comprehensive.py` - Test suite
