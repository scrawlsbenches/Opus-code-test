---
title: "Correctness Lessons"
generated: "2025-12-17T00:01:46.525867Z"
generator: "lessons"
source_files:
  - "git log --grep=correctness"
tags:
  - lessons
  - correctness
  - experience
---

# Correctness Lessons

*Bugs we encountered and how we fixed them.*

---

## Overview

This chapter captures **24 lessons** from correctness work. Each entry shows the problem, the solution, and the principle we extracted.

### Archive ML session after transcript processing (T-003 16f3)

**Commit:** `59072c8`  
**Date:** 2025-12-16  
**Files Changed:** 1  
  - `scripts/ml_data_collector.py`
**Changes:** +12/-0 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Archive ML session after transcript processing (T-003 16f3)

### Update CSV truncation test for new defaults (input=500, output=2000)

**Commit:** `ca94a01`  
**Date:** 2025-12-16  
**Files Changed:** 4  
  - `.git-ml/chats/2025-12-16/chat-20251216-125311-0ce6d9.json`
  - `.git-ml/chats/2025-12-16/chat-20251216-132048-ba08bf.json`
  - `.git-ml/tracked/commits.jsonl`
  - *(and 1 more)*

**The Lesson:** Verify assumptions with tests. The wisdom: Update CSV truncation test for new defaults (input=500, output=2000)

### Fix ML data collection milestone counting and add session/action capture

**Commit:** `273baef`  
**Date:** 2025-12-16  
**Files Changed:** 11  
  - `.git-ml/chats/2025-12-15/chat-20251216-121720-30c3c1.json`
  - `.git-ml/chats/2025-12-16/chat-20251216-121720-01077d.json`
  - `.git-ml/chats/2025-12-16/chat-20251216-121720-306450.json`
  - *(and 8 more)*
**Changes:** +95/-29 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Fix ML data collection milestone counting and add session/action capture

### Address critical ML data collection and prediction issues

**Commit:** `fead1c1`  
**Date:** 2025-12-16  
**Files Changed:** 9  
  - `.git-ml/chats/2025-12-15/chat-20251216-115057-b5bb48.json`
  - `.git-ml/chats/2025-12-16/chat-20251216-115057-3617f9.json`
  - `.git-ml/chats/2025-12-16/chat-20251216-115057-9502fd.json`
  - *(and 6 more)*
**Changes:** +148/-17 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Address critical ML data collection and prediction issues

### Fix(proto): Make protobuf loading lazy to fix CI smoke test failures

**Commit:** `a93518f`  
**Date:** 2025-12-16  
**Files Changed:** 2  
  - `cortical/proto/__init__.py`
  - `cortical/proto/serialization.py`
**Changes:** +53/-19 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: fix(proto): Make protobuf loading lazy to fix CI smoke test failures

### Add missing imports in validate command

**Commit:** `172ad8f`  
**Date:** 2025-12-16  
**Files Changed:** 1  
  - `scripts/ml_data_collector.py`
**Changes:** +5/-0 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Add missing imports in validate command

### Clean up gitignore pattern for .git-ml/commits/

**Commit:** `a65d54f`  
**Date:** 2025-12-16  
**Files Changed:** 1  
  - `.gitignore`
**Changes:** +2/-1 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Clean up gitignore pattern for .git-ml/commits/

### Prevent infinite commit loop in ML data collection hooks

**Commit:** `66ad656`  
**Date:** 2025-12-16  
**Files Changed:** 3  
  - `.git-ml/chats/2025-12-16/chat-20251216-004054-78b531.json`
  - `.git-ml/tracked/commits.jsonl`
  - `scripts/ml_data_collector.py`
**Changes:** +9/-1 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Prevent infinite commit loop in ML data collection hooks

### Correct hook format in settings.local.json

**Commit:** `19ac02a`  
**Date:** 2025-12-16  
**Files Changed:** 1  
  - `.claude/settings.local.json`
**Changes:** +14/-4 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Correct hook format in settings.local.json

### Use filename-based sorting for deterministic session ordering

**Commit:** `61d502d`  
**Date:** 2025-12-15  
**Files Changed:** 2  
  - `scripts/session_context.py`
  - `tests/unit/test_session_context.py`

**The Lesson:** Verify assumptions with tests. The wisdom: Use filename-based sorting for deterministic session ordering

### Increase ID suffix length to prevent collisions

**Commit:** `8ac4b6b`  
**Date:** 2025-12-15  
**Files Changed:** 2  
  - `scripts/orchestration_utils.py`
  - `tests/unit/test_orchestration_utils.py`

**The Lesson:** Verify assumptions with tests. The wisdom: Increase ID suffix length to prevent collisions

### Add import guards for optional test dependencies

**Commit:** `91ffb04`  
**Date:** 2025-12-15  
**Files Changed:** 2  
  - `tests/security/test_fuzzing.py`
  - `tests/test_mcp_server.py`

**The Lesson:** Verify assumptions with tests. The wisdom: Add import guards for optional test dependencies

### Make session file sorting stable for deterministic ordering

**Commit:** `7433b36`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `scripts/session_context.py`

**The Lesson:** Verify assumptions with tests. The wisdom: Make session file sorting stable for deterministic ordering

### Feat(LEGACY-130): Expand customer service corpus and fix xfailed tests

**Commit:** `7f9664d`  
**Date:** 2025-12-15  
**Files Changed:** 6  
  - `samples/customer_service/complaint_escalation_procedures.txt`
  - `samples/customer_service/empathy_and_active_listening.txt`
  - `samples/customer_service/refund_request_handling.txt`
  - *(and 3 more)*

**The Lesson:** Verify assumptions with tests. The wisdom: feat(LEGACY-130): Expand customer service corpus and fix xfailed tests

### Cap query expansion weights to prevent term domination

**Commit:** `fecd6dc`  
**Date:** 2025-12-15  
**Files Changed:** 3  
  - `cortical/query/expansion.py`
  - `tests/behavioral/test_customer_service_quality.py`
  - `tests/unit/test_query_expansion.py`

**The Lesson:** Verify assumptions with tests. The wisdom: Cap query expansion weights to prevent term domination

### Add YAML frontmatter to slash commands for discovery

**Commit:** `5b52da2`  
**Date:** 2025-12-15  
**Files Changed:** 7  
  - `.claude/commands/delegate.md`
  - `.claude/commands/director.md`
  - `.claude/commands/knowledge-transfer.md`
  - *(and 4 more)*

**The Lesson:** Verify assumptions with tests. The wisdom: Add YAML frontmatter to slash commands for discovery

### Stop tracking ML commit data files (too large for GitHub)

**Commit:** `a6f39e0`  
**Date:** 2025-12-15  
**Files Changed:** 472  
  - `.git-ml/commits/0039ad5b_2025-12-11_24b1b10a.json`
  - `.git-ml/commits/00f88d48_2025-12-14_8749d448.json`
  - `.git-ml/commits/051d2002_2025-12-13_7896a312.json`
  - *(and 469 more)*
**Changes:** +4/-2263268 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Stop tracking ML commit data files (too large for GitHub)

### Increase ML data retention to 2 years for training milestones

**Commit:** `95e9f06`  
**Date:** 2025-12-15  
**Files Changed:** 2  
  - `README.md`
  - `scripts/ml_data_collector.py`
**Changes:** +7/-5 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Increase ML data retention to 2 years for training milestones

### Update tests for BM25 default and stop word tokenization

**Commit:** `9dc7268`  
**Date:** 2025-12-15  
**Files Changed:** 2  
  - `tests/unit/test_processor_core.py`
  - `tests/unit/test_query_search.py`
**Changes:** +23/-10 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Update tests for BM25 default and stop word tokenization

### Address audit findings and add documentation

**Commit:** `36be3a1`  
**Date:** 2025-12-15  
**Files Changed:** 4  
  - `.claude/commands/ml-log.md`
  - `.claude/commands/ml-stats.md`
  - `CLAUDE.md`
  - *(and 1 more)*
**Changes:** +201/-15 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Address audit findings and add documentation

### Harden ML data collector with critical fixes

**Commit:** `4438d60`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `scripts/ml_data_collector.py`
**Changes:** +151/-54 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Harden ML data collector with critical fixes

### Correct line number assertions in pattern detection tests

**Commit:** `1b9901d`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `tests/unit/test_patterns.py`
**Changes:** +5/-5 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Correct line number assertions in pattern detection tests

### Add test file penalty and code stop word filtering to search

**Commit:** `1fafc8b`  
**Date:** 2025-12-14  
**Files Changed:** 3  
  - `cortical/processor/query_api.py`
  - `cortical/query/passages.py`
  - `cortical/query/search.py`
**Changes:** +51/-9 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Add test file penalty and code stop word filtering to search

### Replace external action with native Python link checker

**Commit:** `901a181`  
**Date:** 2025-12-14  
**Files Changed:** 5  
  - `.github/workflows/ci.yml`
  - `.markdown-link-check.json`
  - `scripts/resolve_wiki_links.py`
  - *(and 2 more)*
**Changes:** +172/-34 lines  

**The Lesson:** Validate inputs early. The lesson? Replace external action with native Python link checker

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*
