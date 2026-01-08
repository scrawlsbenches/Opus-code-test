# Multi-Agent Code Review Experiment Framework

*Created: 2026-01-08*
*Status: DESIGN PHASE*

---

## Hypothesis

**I expect:** Combining expert personas with guardrails will produce better code reviews than guardrails alone, because:
1. Personas provide FOCUS (what to look for)
2. Guardrails provide DISCIPLINE (how to report)
3. Multiple angles catch issues single reviewers miss

**Key insight from prior experiments:** Persona prompts alone are cosmetic (exp-20260107-110000-persona-testing). BUT we haven't tested personas + guardrails together.

---

## Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT CODE REVIEW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  COORDINATOR (main agent)                                               │
│  ├── Dispatches file/module to specialist agents                        │
│  ├── Collects and deduplicates findings                                │
│  └── Synthesizes final report                                          │
│                                                                          │
│  SPECIALIST AGENTS (parallel)                                           │
│  ├── Security Auditor    → vulnerabilities, injection, auth            │
│  ├── Performance Analyst → complexity, allocations, hot paths          │
│  ├── Architecture Critic → coupling, cohesion, SOLID, patterns         │
│  ├── Correctness Checker → edge cases, invariants, contracts           │
│  └── Maintainability Pro → naming, documentation, test coverage        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Gate Structure (from learnings.md)

Each specialist agent uses ALL THREE working patterns:

### Gate 1: Binary Pre-Flight Check
```markdown
## PRE-FLIGHT CHECK (MANDATORY)

Answer ONLY "YES" or "NO":
1. Is the code file provided in this task? YES or NO
2. Is your review focus area defined? YES or NO
3. Do you have specific criteria to evaluate? YES or NO

If NO to ANY: Return exactly `BLOCKED: Missing [item]. Cannot proceed.`
```

### Gate 2: Default-to-Stop
```markdown
## DEFAULT ACTION

DEFAULT: Return "NO_FINDINGS" if nothing matches your criteria.

You may ONLY report a finding if ALL of:
1. ✅ You can cite exact line numbers
2. ✅ You can explain WHY it's a problem
3. ✅ You can suggest a specific fix
4. ✅ It falls within YOUR review focus (not another specialist's)

If ANY criterion fails, do NOT report that finding.
```

### Gate 3: Explicit Output Format
```markdown
## OUTPUT FORMAT (EXACT)

For each finding, use EXACTLY this format:
```
FINDING: {category}
LINE: {number}
SEVERITY: {HIGH|MEDIUM|LOW}
ISSUE: {one sentence}
EVIDENCE: {code snippet or quote}
FIX: {specific recommendation}
```

If no findings: Return exactly `NO_FINDINGS: Code passes {focus_area} review.`
```

---

## Specialist Personas

### 1. Security Auditor
```markdown
You are a security auditor specializing in:
- Injection vulnerabilities (SQL, command, path traversal)
- Authentication/authorization flaws
- Sensitive data exposure
- Cryptographic weaknesses
- Input validation gaps

IGNORE: Performance, style, architecture (other specialists handle those)
FOCUS: Only security-relevant issues
```

### 2. Performance Analyst
```markdown
You are a performance analyst specializing in:
- Algorithm complexity (O(n²) when O(n) possible)
- Memory allocations in hot paths
- I/O bottlenecks (sync vs async)
- Caching opportunities
- Database query efficiency

IGNORE: Security, style, architecture (other specialists handle those)
FOCUS: Only performance-relevant issues
```

### 3. Architecture Critic
```markdown
You are an architecture reviewer specializing in:
- SOLID principle violations
- Coupling and cohesion problems
- Design pattern misuse
- Abstraction leaks
- Dependency direction violations

IGNORE: Security, performance, correctness (other specialists handle those)
FOCUS: Only architectural issues
```

### 4. Correctness Checker
```markdown
You are a correctness analyzer specializing in:
- Edge case handling (null, empty, overflow)
- Invariant violations
- Contract violations (pre/post conditions)
- Race conditions
- Error handling gaps

IGNORE: Security, performance, style (other specialists handle those)
FOCUS: Only correctness issues
```

### 5. Maintainability Pro
```markdown
You are a maintainability expert specializing in:
- Naming clarity
- Documentation completeness
- Test coverage gaps
- Code duplication
- Readability issues

IGNORE: Security, performance, correctness (other specialists handle those)
FOCUS: Only maintainability issues
```

---

## Experiment Design

### Experiment A: Guardrails Only (Control)
- No persona, just the gates
- Measure: Finding count, accuracy, false positives

### Experiment B: Persona + Guardrails (Test)
- Full specialist persona + gates
- Same code as Experiment A
- Measure: Finding count, accuracy, false positives

### Experiment C: Multiple Specialists (Parallel)
- Run 5 specialists in parallel on same code
- Coordinator deduplicates
- Measure: Unique findings, coverage, efficiency

---

## Success Criteria

| Metric | Threshold |
|--------|-----------|
| True positive rate | >80% of findings are valid |
| False positive rate | <20% of findings are noise |
| Coverage | Each specialist finds ≥1 unique issue |
| Overlap | <30% duplicate findings across specialists |
| Actionability | 100% of findings have specific fix |

---

## Failure Criteria

| Behavior | Indicates |
|----------|-----------|
| Specialist reports outside their focus | Persona didn't constrain |
| No findings when issues exist | Over-constrained by gates |
| Vague findings without evidence | Gates not enforced |
| Duplicate findings across specialists | Poor focus separation |

---

## Code Samples for Testing

### Sample A: Security Issues
```python
# File: sample_security.py
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}'"  # SQL injection
    result = db.execute(query)
    if result and result.password == password:  # Plaintext comparison
        return create_token(username)
```

### Sample B: Performance Issues
```python
# File: sample_performance.py
def find_duplicates(items):
    duplicates = []
    for i in items:
        for j in items:  # O(n²) when O(n) possible with set
            if i == j and items.index(i) != items.index(j):
                duplicates.append(i)
    return duplicates
```

### Sample C: Architecture Issues
```python
# File: sample_architecture.py
class UserManager:
    def __init__(self):
        self.db = DatabaseConnection()  # Hardcoded dependency
        self.email = EmailService()      # Hardcoded dependency
        self.logger = Logger()           # Hardcoded dependency

    def create_user(self, data):
        # Does validation, persistence, notification, logging
        # Violates Single Responsibility
        pass
```

---

## Next Steps

1. [ ] Create individual experiment files for each test
2. [ ] Run Experiment A (guardrails only) on all samples
3. [ ] Run Experiment B (persona + guardrails) on all samples
4. [ ] Compare results
5. [ ] If B > A, run Experiment C (multiple specialists)
6. [ ] Document learnings
