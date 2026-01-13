# Context Window Communication File

*Created: 2026-01-13*

---

# ARCHIVE

*Old context. Reference only when needed.*

## Initial Setup (2026-01-13)

- User proposed file-based communication to reduce misalignment
- Claude analyzed: works as supplement, not replacement
- Key issues identified: file grows, git noise, overhead
- User suggested: append-only + read-from-end
- Claude agreed: this solves growth problem

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Use append-only format | Prevents file growth from being a problem |
| Claude reads from end | Recent context is what matters |
| Archive old content | Keep history but don't process it |

---

# RECENT

*Active working context. Claude reads this section.*

---

## [001] Claude restructured file

Format is now:
- **ARCHIVE** (top): Old context, rarely read
- **RECENT** (bottom): Active context, always read

Each entry is numbered. When RECENT gets long (~30 entries), older entries move to ARCHIVE.

---

## [002] Waiting for direction

What do you want to work on?

- This codebase is a Graph of Thought / cognitive text processor
- I can explore it, fix bugs, add features, or continue this experiment
- Your call

---

