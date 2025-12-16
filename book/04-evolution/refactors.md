---
title: "Refactorings and Architecture Evolution"
generated: "2025-12-16T17:26:23.884089Z"
generator: "evolution"
source_files:
  - "git log --grep=refactor:"
tags:
  - refactoring
  - architecture
  - design
---

# Refactorings and Architecture Evolution

*How the codebase structure improved over time.*

---

## Overview

The codebase has undergone **3 refactorings**. Each improved code quality, maintainability, or performance.

## Refactoring History

### Remove unused protobuf serialization (T-013 f0ff)

**Commit:** `d7a98ae`  
**Date:** 2025-12-16  
**Changes:** +100/-1460 lines  
**Scope:** 6 files affected  

### Split large files exceeding 25000 token limit

**Commit:** `21ec5ea`  
**Date:** 2025-12-15  

### Consolidate ML data to single JSONL files

**Commit:** `205fe34`  
**Date:** 2025-12-15  
**Changes:** +658/-12208 lines  
**Scope:** 486 files affected  

