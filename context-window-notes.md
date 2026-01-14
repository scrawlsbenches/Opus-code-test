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

## [004] CLI Exercise Results

**User:** Exercise the CLI commands

### Commands Tested

**1. Query different directories:**
```bash
$ python -m cortical.cli.audit reason "risky files in cortical/got/" -v
Found 36 files, 5 findings
RISKY FILES:
  cortical/got/validation.py  Risk: 52.8%  (should be)
  cortical/got/api.py         Risk: 50.0%  (TODO:)
```

**2. Add custom rule:**
```bash
$ python -m cortical.cli.audit reason \
    --add-rule "has_pattern(X, critical)" "requires_attention(X)" "0.90"
Added rule and saved to persistence.
```

**3. Mark file as VLTI:**
```bash
$ python -m cortical.cli.audit reason --vlti cortical/got/api.py -d cortical/got/
Marked cortical/got/api.py as Very Long Term Important
```

**4. Change aggregation strategy:**
```bash
$ python -m cortical.cli.audit reason "all files in cortical/cdg/" --aggregate max -v
Using aggregation strategy: max
Found 24 files, 2 findings
No files exceed the risk threshold.
```

**5. Scan full codebase:**
```bash
$ python -m cortical.cli.audit reason "most risky files in cortical/" --threshold 0.6
Found 282 files, 51 findings
No files exceed 60% threshold (max is ~52.8%)
```

**6. Show updated state:**
```bash
$ python -m cortical.cli.audit reason --show-state
Session count: 2
Files tracked: 7
vlti_files: 1
```

**7. Show all rules (including custom):**
```bash
$ python -m cortical.cli.audit reason --show-rules
Total rules: 12
Manual Rules:
  has_pattern(X, hack) → needs_review(X) (0.80)
  has_pattern(X, critical) → requires_attention(X) (0.90)  # <-- our rule
  ...
```

### Observations

| Feature | Status | Notes |
|---------|--------|-------|
| Natural language queries | Works | Parses directory and intent |
| Custom rules | Works | Persisted across sessions |
| VLTI marking | Works | File pinned for attention |
| Aggregation strategies | Works | max, mean, min, product, revision |
| Explain (-e) | Partial | Output is sparse/empty |
| File history | Partial | Shows "no history" even for tracked files |

---

## [005] Honest Assessment

**User:** Is it me or did we spend a bunch of time creating something that isn't very useful?

**Claude:** You're right to question it. Let me be honest:

### What This Tool Actually Is

A **fancy grep** with probability labels.

```bash
# This achieves 90% of the same result:
grep -rn "TODO\|HACK\|FIXME\|XXX" cortical/
```

### The "PLN Reasoning" Is Shallow

| What It Claims | What It Does |
|----------------|--------------|
| "Probabilistic Logic Networks" | Pattern match + static weights |
| "Risk assessment" | Count TODOs and assign 0.5-0.8 |
| "Reasoning" | `if "hack" in file: risk += 0.8` |
| "Persistence" | Save a JSON of what files had TODOs |

### Why It Feels Underwhelming

1. **No semantic understanding** - It doesn't know if a TODO is critical or trivial
2. **Arbitrary numbers** - Why is "hack" 0.80 and "fixme" 0.70? Made up.
3. **Broken features** - Explain and history don't work properly
4. **Oversold** - "PLN" sounds sophisticated but it's just weighted grep

### Was This Time Wasted?

Partially. We learned:
- The tool exists and its limitations
- The file-based communication experiment worked
- Sometimes exploring reveals something isn't worth using

### What Would Actually Be Useful?

I don't know your goals. What problem are you trying to solve? Let's focus on that instead of exercising random CLI tools.

---

