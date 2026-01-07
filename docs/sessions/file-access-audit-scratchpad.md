# CDG Layer Migration - Working Scratchpad

*Session: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*

---

## CURRENT FOCUS: Configuration Consolidation

Working through configuration architecture decisions one at a time.

---

## CONFIGURATION ISSUES FOUND

1. **Two GoTConfigs** - different files, different purposes, same name
2. **Two DurabilityMode enums** - incompatible (RELAXED vs FAST)
3. **CDGConfig** - 23 fields with factory methods hiding defaults
4. **Path flow** - actually well-structured, bootstrap is entry point

---

## DECISIONS TO MAKE (one at a time)

- [ ] **Q1:** CDGConfig's 23 fields - keep all, trim, or split?
- [ ] **Q2:** DurabilityMode naming - RELAXED or FAST?
- [ ] **Q3:** `use_memory` - config field or separate concern?
- [ ] **Q4:** Paths in config - only base_dir, all explicit, or path factory?

---

## USER DIRECTION

- ONE config object (test version + production version)
- NO `.default()` methods - explicit defaults visible in bootstrap
- All paths visible in container setup
- IoC is first-class citizen
- Pass classes into constructors, not primitives

---

## COMPLETED THIS SESSION

1. Fixed `use_memory=True` test isolation bug
2. Extracted CDG durability tests from GoT test file
3. Analyzed CDGConfig (23 fields) and GoTConfig (2 versions)
4. Mapped all path configurations

---

## ARCHITECTURE

- **CDG owns:** storage, transactions, WAL, recovery, indexes
- **GoT is:** thin domain layer, NO file I/O
- **FileSystem abstraction:** InMemoryFileSystem for tests, RealFileSystem for production
