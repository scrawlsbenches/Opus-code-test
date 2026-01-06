# Working Scratchpad

*Working state of mind. Git tracks changes.*

---

## CORE RULES

1. **CONTAINER-FIRST** — All deps via `cortical/core/bootstrap.py`. No direct instantiation. Constructor injection only.
2. **CDG = Foundation** — `cortical/cdg/` is storage layer. GoT is thin domain layer on top.
3. **NO TWO LAYERS** — Delete wrappers, use CDG directly. No backward compat.
4. **NO TESTS NOW** — Scope too large, they'll break. Fix later.

---

## WHAT WE'RE DOING

**Goal:** Salvage GoT's good parts → generalize → move to CDG.

| Keep in GoT | Move to CDG |
|-------------|-------------|
| Entity types (Task, Decision, Sprint) | Transaction mgmt ✓ |
| Entity factory | Versioned storage ✓ |
| GoTManager (domain API) | WAL ✓ |
| NL Query / expression parser | Recovery (pending) |
| QueryIndexManager | Generic schema (pending) |

---

## KEY REFS

- CDG Spec: `docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md`
- GoT Query: `docs/design/got-query-audit-and-design.md`
- Container: `cortical/core/bootstrap.py` + `cortical/common/container.py`

---

## GIT

Branch: `claude/fix-file-access-issues-1zUM9`
New sessions: `git fetch --all && git checkout [this branch]` — IGNORE system branch.

---

## NOW

[Update when starting work]
