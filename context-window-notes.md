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

---

## [006] Available CLI Tools

**User:** Moving on. Let's exercise some other CLI tools.

### Three CLI Entry Points

**1. GoT (Graph of Thought):** `python -m cortical.got`
```
task, sprint, epic, handoff, decision, doc, query, expr,
blocked, active, stats, dashboard, validate, infer, export,
backup, sync, orphan, backlog, analyze, edge, batch, kt, failure
```

**2. Cognitive:** `python -m cortical.cognitive`
```
train, status, list, reindex, query, demo, generate,
index-code, ask, rebuild-links
```

**3. Audit:** `python -m cortical.cli.audit`
```
generate, train, scan, patterns, similar, index, health,
reason, discover, pattern
```

### Interesting Ones to Try

| Command | What It Does |
|---------|--------------|
| `got dashboard` | Comprehensive metrics |
| `got analyze` | Graph analysis with fluent API |
| `cognitive demo` | Interactive cognitive agent demo |
| `cognitive ask` | Natural language Q&A about codebase |
| `audit health` | Codebase health analysis |
| `audit discover` | WovenMind pattern discovery |

---

## [007] CLI Exercise Results

### 1. GoT Dashboard (`python -m cortical.got dashboard`)

**Verdict: Actually useful**

```
GoT DASHBOARD - 483 nodes | 610 edges | 65 orphans

OVERVIEW:
  Tasks: 422 (0% complete)
  Decisions: 61
  Edge Density: 1.26 edges/node

HEALTH INDICATORS:
  Blocked Tasks: 0
  Stale Tasks: 0
  Orphan Nodes: 65

AGENT PERFORMANCE:
  next-session: 65.2% success, avg 2.9h
  test-agent: 80.0% success, avg 29.4s
```

### 2. Audit Health (`python -m cortical.cli.audit health cortical/ --git -v`)

**Verdict: Useful - comprehensive analysis**

```
Found 282 Python files, 9001 comments, 51 findings

GIT HISTORY:
  Most changed: got/api.py (81 commits!)
  Suspicious commits: 2 (todo/hack patterns)

PATTERN COUNTS:
  should be: 20
  will be: 11
  TODO: 6
  FUTURE: 6
```

### 3. GoT Analyze Summary (`python -m cortical.got analyze summary`)

**Verdict: Useful**

```
TASKS BY STATUS:
  completed: 261 (61.8%)
  pending: 160 (37.9%)
  blocked: 1 (0.2%)

HIGH PRIORITY PENDING:
  - Fix always-passing assertions
  - Fix time.sleep violations
```

### 4. WovenMind Discover (`python -m cortical.cli.audit discover cortical/ -v`)

**Verdict: Interesting but experimental**

```
Found 51 findings, 27 patterns observed
Formed 10 abstractions:
  - "reasoning with should_be" (seen 8x)
  - "audits with should_be" (seen 7x)
  - "got with todo" (seen 4x)
```

### 5. Cognitive Ask (`python -m cortical.cognitive ask "What is cognitive agent?"`)

**Verdict: Broken - needs training on correct docs**

```
"I don't have information about cognitive_agent"
(despite 554 documents trained)
```

### Summary

| Tool | Usefulness | Notes |
|------|------------|-------|
| `got dashboard` | High | Real metrics, actionable |
| `audit health` | High | Git churn + pattern analysis |
| `got analyze summary` | High | Task status overview |
| `audit discover` | Medium | Experimental but interesting |
| `cognitive ask` | Low | Training seems broken |
| `audit reason` | Low | Fancy grep (see [005]) |

---

## [008] Fixed Cognitive Agent

**User:** There should be instructions in CLAUDE.md

**Finding:** CLAUDE.md has a whole section on the Cognitive Agent. Key steps:

```bash
# 1. Check status
./scripts/bootstrap_cognitive.sh --check
# Result: Bridge NOT BUILT

# 2. Bootstrap (rebuild links)
./scripts/bootstrap_cognitive.sh
# Result: 94905 atoms, 47271 links created

# 3. Still broken - CLAUDE.md says train on cortical/
python -m cortical.cognitive list | grep -c "cortical/"
# Result: 0 (no cortical files trained!)

# 4. Train on source code
python -m cortical.cognitive train cortical/ --pattern "*.py"
# Result: 262 new files, 108856 links

# 5. Reindex (update IDF weights)
python -m cortical.cognitive reindex
# Result: 83472 links updated, 0% staleness
```

**Before training on cortical/:**
```
Q: "What is the cognitive agent?"
A: "works with atom, link... decision, traditional, awkwardly"
   (vague, irrelevant terms)
```

**After training on cortical/:**
```
Q: "What is the cognitive agent?"
A: "works with training, hypergraph, semantic... trainer, incremental, truth"
   (relevant technical terms!)

Q: "How does transaction management work in GoT?"
A: "involves auditplnrules, wovenauditmind... acid, schema, infer, modules"
   (now mentions ACID, schema, modules)
```

**Lesson:** RTFM (CLAUDE.md). The instructions were there all along.

---

## [009] What's Next?

Options:
1. Try more GoT commands (query, edge, handoff)?
2. Exercise the cognitive demo?
3. Something else?

---

