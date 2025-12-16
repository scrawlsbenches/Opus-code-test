---
title: "Bug Fixes and Lessons"
generated: "2025-12-16T20:01:28.127474Z"
generator: "evolution"
source_files:
  - "git log --grep=fix:"
tags:
  - bugs
  - fixes
  - lessons-learned
---

# Bug Fixes and Lessons

*What broke, how we fixed it, and what we learned.*

---

## Overview

**13 bugs** have been identified and resolved. Each fix taught us something about the system.

## Bug Fix History

### Archive ML session after transcript processing (T-003 16f3)

**Commit:** `59072c8`  
**Date:** 2025-12-16  
**Files Changed:** scripts/ml_data_collector.py  

### Update CSV truncation test for new defaults (input=500, output=2000)

**Commit:** `ca94a01`  
**Date:** 2025-12-16  

### Fix ML data collection milestone counting and add session/action capture

**Commit:** `273baef`  
**Date:** 2025-12-16  
**Files Changed:** .git-ml/chats/2025-12-15/chat-20251216-121720-30c3c1.json, .git-ml/chats/2025-12-16/chat-20251216-121720-01077d.json, .git-ml/chats/2025-12-16/chat-20251216-121720-306450.json, .git-ml/chats/2025-12-16/chat-20251216-121720-5ef95b.json, .git-ml/chats/2025-12-16/chat-20251216-121720-8a1e7b.json  
*(and 6 more)*  

### Address critical ML data collection and prediction issues

**Commit:** `fead1c1`  
**Date:** 2025-12-16  
**Files Changed:** .git-ml/chats/2025-12-15/chat-20251216-115057-b5bb48.json, .git-ml/chats/2025-12-16/chat-20251216-115057-3617f9.json, .git-ml/chats/2025-12-16/chat-20251216-115057-9502fd.json, .git-ml/chats/2025-12-16/chat-20251216-115057-cbbe64.json, .git-ml/chats/2025-12-16/chat-20251216-115057-f65b7a.json  
*(and 4 more)*  

### Add missing imports in validate command

**Commit:** `172ad8f`  
**Date:** 2025-12-16  
**Files Changed:** scripts/ml_data_collector.py  

### Clean up gitignore pattern for .git-ml/commits/

**Commit:** `a65d54f`  
**Date:** 2025-12-16  
**Files Changed:** .gitignore  

### Prevent infinite commit loop in ML data collection hooks

**Commit:** `66ad656`  
**Date:** 2025-12-16  
**Files Changed:** .git-ml/chats/2025-12-16/chat-20251216-004054-78b531.json, .git-ml/tracked/commits.jsonl, scripts/ml_data_collector.py  

### Correct hook format in settings.local.json

**Commit:** `19ac02a`  
**Date:** 2025-12-16  
**Files Changed:** .claude/settings.local.json  

### Use filename-based sorting for deterministic session ordering

**Commit:** `61d502d`  
**Date:** 2025-12-15  

### Increase ID suffix length to prevent collisions

**Commit:** `8ac4b6b`  
**Date:** 2025-12-15  

### Add import guards for optional test dependencies

**Commit:** `91ffb04`  
**Date:** 2025-12-15  

### Make session file sorting stable for deterministic ordering

**Commit:** `7433b36`  
**Date:** 2025-12-15  

### Cap query expansion weights to prevent term domination

**Commit:** `fecd6dc`  
**Date:** 2025-12-15  

