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

## [003] User wants to experiment with reasoning

**User:** Experiment with `python -m cortical.cognitive.reason`

**Finding:** That exact path doesn't exist. But found:

```
python -m cortical.cli.audit reason --help
```

This is a **PLN (Probabilistic Logic Networks) reasoning** command that:
- Analyzes file risk in directories
- Uses rules to infer properties
- Supports VLTI (Very Long Term Important) file marking
- Persists reasoning state across sessions

**Options:**
- `--show-rules` - Display all PLN rules
- `--show-state` - Show persistence state
- `-e FILE` - Explain risk for specific file
- `query` - Natural language query like "risky files in reasoning/"

**Results from testing:**

```
$ python -m cortical.cli.audit reason --show-rules
PLN AUDIT RULES (12 total)
Manual Rules:
  has_pattern(X, hack) → needs_review(X) (strength=0.80)
  has_pattern(X, fixme) → has_known_issue(X) (strength=0.70)
  has_pattern(X, xxx) → needs_urgent_review(X) (strength=0.85)
  has_pattern(X, security) → security_sensitive(X) (strength=0.75)
  ...

$ python -m cortical.cli.audit reason "risky files in cortical/reasoning/" -v
Found 40 files, 7 findings
RISKY FILES (threshold=50%):
  cortical/reasoning/prism_causal.py        Risk: 52.8%
  cortical/reasoning/crisis_manager.py      Risk: 52.8%
  cortical/reasoning/abstraction_pln.py     Risk: 52.8%
  cortical/reasoning/production_state.py    Risk: 52.8% (HACK:, TODO:)
```

**What this does:**
- Scans files for patterns (HACK, TODO, FIXME, etc.)
- Applies PLN rules to infer risk properties
- Uses probabilistic truth values (strength)
- Persists state across sessions

---

## [004] What would you like to explore?

Options:
1. Add custom PLN rules (`--add-rule`)
2. Mark files as VLTI (Very Long Term Important)
3. Query different directories
4. Look at the source code to understand how it works
5. Something else?

---

