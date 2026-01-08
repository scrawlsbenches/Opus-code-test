# Experiment: 20260108-100000-persona-plus-guardrails

*Date: 2026-01-08*
*Coordinator: claude/fix-scratchpad-focus-SUJkx*
*Builds on: exp-20260107-110000-persona-testing (REJECTED hypothesis)*

---

## Hypothesis

**I expect:** Combining a security expert persona WITH guardrails (gates) will produce MORE FOCUSED findings than guardrails alone.

**Because:** Prior experiment showed personas alone don't change behavior. But personas + guardrails might:
1. Persona provides FOCUS (what to look for)
2. Guardrails enforce DISCIPLINE (evidence requirements)
3. Together they constrain AND direct

**This differs from prior persona test because:** Prior test had no guardrails - just persona + task. This test combines persona + gates.

---

## Test Design

**Code under review:**
```python
# sample_security.py
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}'"
    result = db.execute(query)
    if result and result.password == password:
        return create_token(username)
    return None
```

**Known issues:**
1. SQL injection (f-string query)
2. Plaintext password comparison
3. No input validation
4. Token creation without expiry consideration

---

### Agent A: Guardrails Only (Control)

```markdown
## PRE-FLIGHT CHECK (MANDATORY)

Answer ONLY "YES" or "NO":
1. Is code provided in this task? YES or NO
2. Is your task clearly defined? YES or NO

If NO to ANY: Return exactly `BLOCKED: Missing [item]. Cannot proceed.`

---

## DEFAULT ACTION

DEFAULT: Return "NO_FINDINGS" if nothing notable.

You may ONLY report a finding if ALL of:
1. ✅ You can cite exact line numbers
2. ✅ You can explain WHY it's a problem
3. ✅ You can suggest a specific fix

---

## TASK

Review this code and identify issues:

```python
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}'"
    result = db.execute(query)
    if result and result.password == password:
        return create_token(username)
    return None
```

## OUTPUT FORMAT

For each finding:
```
FINDING: {category}
LINE: {number}
ISSUE: {description}
FIX: {recommendation}
```

Max 5 findings. Be specific.
```

---

### Agent B: Persona + Guardrails (Test)

```markdown
## PRE-FLIGHT CHECK (MANDATORY)

Answer ONLY "YES" or "NO":
1. Is code provided in this task? YES or NO
2. Is your review focus (security) defined? YES or NO
3. Do you have specific criteria to evaluate? YES or NO

If NO to ANY: Return exactly `BLOCKED: Missing [item]. Cannot proceed.`

---

## YOUR ROLE

You are a SECURITY AUDITOR specializing in:
- Injection vulnerabilities (SQL, command, XSS)
- Authentication/authorization flaws
- Sensitive data exposure
- Cryptographic weaknesses

IGNORE: Performance, style, architecture (not your focus)
ONLY REPORT: Security-relevant issues

---

## DEFAULT ACTION

DEFAULT: Return "NO_SECURITY_FINDINGS" if code is secure.

You may ONLY report a finding if ALL of:
1. ✅ It is a SECURITY issue (not performance/style)
2. ✅ You can cite exact line numbers
3. ✅ You can explain the ATTACK VECTOR
4. ✅ You can suggest a specific SECURE fix

If ANY criterion fails for a finding, do NOT report it.

---

## TASK

Security review this code:

```python
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}'"
    result = db.execute(query)
    if result and result.password == password:
        return create_token(username)
    return None
```

## OUTPUT FORMAT (EXACT)

For each SECURITY finding:
```
FINDING: {vulnerability_type}
LINE: {number}
SEVERITY: {CRITICAL|HIGH|MEDIUM|LOW}
ATTACK: {how an attacker exploits this}
FIX: {secure alternative}
```

If no security issues: Return exactly `NO_SECURITY_FINDINGS: Code passes security review.`
```

---

## Success Criteria

| Metric | Agent A (Control) | Agent B (Test) |
|--------|-------------------|----------------|
| Finds SQL injection | ? | ? |
| Finds password issue | ? | ? |
| Reports non-security issues | ? (expected: yes) | ? (expected: no) |
| Includes attack vectors | ? (expected: no) | ? (expected: yes) |
| Findings are actionable | ? | ? |

---

## Predictions

**Agent A (guardrails only):**
- Will find SQL injection ✓
- Will find password issue ✓
- May report non-security issues (style, naming)
- Will NOT include attack vectors (not asked)

**Agent B (persona + guardrails):**
- Will find SQL injection ✓
- Will find password issue ✓
- Will NOT report non-security issues (constrained by persona)
- WILL include attack vectors (required by format)

**Predicted difference:** Agent B's findings will be MORE FOCUSED (security only) and MORE ACTIONABLE (attack vectors + severity)

---

## Actual Results

### Agent A Output
```
[TO BE FILLED AFTER RUNNING]
```

### Agent B Output
```
[TO BE FILLED AFTER RUNNING]
```

---

## Analysis

**Agent A findings:**
- Security issues found: ?
- Non-security issues found: ?
- Attack vectors included: ?

**Agent B findings:**
- Security issues found: ?
- Non-security issues found: ?
- Attack vectors included: ?

**Hypothesis result:** [CONFIRMED / REJECTED / PARTIAL]

---

## Learning

**If CONFIRMED:** Persona + guardrails IS better than guardrails alone. Update task templates to include specialist personas.

**If REJECTED:** Personas add no value even with guardrails. Stick to guardrails-only approach.

**If PARTIAL:** Document what worked and what didn't.
