# Experiment: 20260108-110000-multi-agent-parallel-review

*Date: 2026-01-08*
*Coordinator: claude/fix-scratchpad-focus-SUJkx*
*Depends on: exp-20260108-100000-persona-plus-guardrails (must pass first)*

---

## Hypothesis

**I expect:** Running multiple specialist agents IN PARALLEL on the same code will:
1. Find MORE issues than any single agent
2. Have MINIMAL overlap (<30% duplicate findings)
3. Each specialist finds ≥1 UNIQUE issue in their domain

**Because:** Different perspectives catch different issues. A security expert sees injection; a performance analyst sees O(n²).

---

## Test Design

**Code under review:** Complex module with multiple issue types

```python
# File: user_service.py
import hashlib

class UserService:
    def __init__(self):
        self.db = DatabaseConnection()  # Hardcoded dependency
        self.cache = {}

    def authenticate(self, username, password):
        # Check cache first
        cache_key = f"{username}:{password}"  # Password in cache key!
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Query database
        query = f"SELECT * FROM users WHERE username='{username}'"
        user = self.db.execute(query)

        if user:
            # Verify password
            hashed = hashlib.md5(password.encode()).hexdigest()  # MD5!
            if user.password_hash == hashed:
                self.cache[cache_key] = user
                return user
        return None

    def find_users_by_role(self, role, all_users):
        # Find users with matching role
        result = []
        for user in all_users:
            for r in user.roles:
                if r.lower() == role.lower():
                    result.append(user)
        return result

    def export_user_data(self, user_id):
        user = self.get_user(user_id)
        return {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "password_hash": user.password_hash,  # Exposing hash!
            "ssn": user.ssn,  # PII exposure!
            "created": str(user.created_at)
        }
```

---

## Known Issues by Domain

| Domain | Issue | Line |
|--------|-------|------|
| **Security** | SQL injection | 16 |
| **Security** | MD5 for passwords | 20 |
| **Security** | Password in cache key | 10 |
| **Security** | PII exposure (SSN, hash) | 30-32 |
| **Performance** | O(n×m) in find_users_by_role | 26-29 |
| **Performance** | No cache eviction | 21 |
| **Architecture** | Hardcoded dependency | 6 |
| **Architecture** | Mixed responsibilities | class |
| **Correctness** | No null check on user | 19 |
| **Maintainability** | Magic string "username" | 16 |

**Total known issues:** 10

---

## Specialist Agent Prompts

### Agent 1: Security Auditor

```markdown
## PRE-FLIGHT CHECK
Answer YES or NO:
1. Is code provided? YES or NO
2. Is security your focus? YES or NO
If NO to ANY: Return `BLOCKED: Missing [item]`

## YOUR ROLE
You are a SECURITY AUDITOR. Report ONLY:
- Injection vulnerabilities
- Authentication flaws
- Sensitive data exposure
- Cryptographic weaknesses

IGNORE: Performance, architecture, style

## DEFAULT
Return "NO_SECURITY_FINDINGS" if code is secure.
Report finding ONLY if you can cite line + attack vector + fix.

## OUTPUT FORMAT
FINDING: {vuln_type}
LINE: {n}
SEVERITY: {CRITICAL|HIGH|MEDIUM|LOW}
ATTACK: {exploitation method}
FIX: {secure alternative}

## CODE
[paste user_service.py]
```

### Agent 2: Performance Analyst

```markdown
## PRE-FLIGHT CHECK
Answer YES or NO:
1. Is code provided? YES or NO
2. Is performance your focus? YES or NO
If NO to ANY: Return `BLOCKED: Missing [item]`

## YOUR ROLE
You are a PERFORMANCE ANALYST. Report ONLY:
- Algorithm complexity issues
- Memory/allocation problems
- I/O bottlenecks
- Caching issues

IGNORE: Security, architecture, correctness

## DEFAULT
Return "NO_PERFORMANCE_FINDINGS" if code is performant.
Report finding ONLY if you can cite line + complexity + fix.

## OUTPUT FORMAT
FINDING: {issue_type}
LINE: {n}
IMPACT: {complexity or resource cost}
FIX: {optimized alternative}

## CODE
[paste user_service.py]
```

### Agent 3: Architecture Critic

```markdown
## PRE-FLIGHT CHECK
Answer YES or NO:
1. Is code provided? YES or NO
2. Is architecture your focus? YES or NO
If NO to ANY: Return `BLOCKED: Missing [item]`

## YOUR ROLE
You are an ARCHITECTURE CRITIC. Report ONLY:
- SOLID violations
- Coupling problems
- Cohesion issues
- Pattern misuse

IGNORE: Security, performance, correctness

## DEFAULT
Return "NO_ARCHITECTURE_FINDINGS" if design is sound.
Report finding ONLY if you can cite location + principle + fix.

## OUTPUT FORMAT
FINDING: {principle_violated}
LOCATION: {class/method}
PROBLEM: {why this is bad}
FIX: {refactoring suggestion}

## CODE
[paste user_service.py]
```

### Agent 4: Correctness Checker

```markdown
## PRE-FLIGHT CHECK
Answer YES or NO:
1. Is code provided? YES or NO
2. Is correctness your focus? YES or NO
If NO to ANY: Return `BLOCKED: Missing [item]`

## YOUR ROLE
You are a CORRECTNESS CHECKER. Report ONLY:
- Edge case failures
- Null/empty handling
- Invariant violations
- Race conditions

IGNORE: Security, performance, style

## DEFAULT
Return "NO_CORRECTNESS_FINDINGS" if logic is correct.
Report finding ONLY if you can cite line + failure case + fix.

## OUTPUT FORMAT
FINDING: {issue_type}
LINE: {n}
FAILURE_CASE: {input that breaks it}
FIX: {defensive code}

## CODE
[paste user_service.py]
```

---

## Success Criteria

| Metric | Threshold |
|--------|-----------|
| Total unique findings | ≥8 of 10 known issues |
| Each specialist finds ≥1 | All 4 specialists report |
| Overlap rate | <30% duplicate across specialists |
| False positive rate | <20% |
| Findings are actionable | 100% have specific fix |

---

## Predictions

| Specialist | Expected Findings |
|------------|-------------------|
| Security | 4 (injection, MD5, cache key, PII) |
| Performance | 2 (O(n×m), no eviction) |
| Architecture | 2 (hardcoded dep, mixed responsibility) |
| Correctness | 1 (null check) |

**Expected total:** 9 unique + maybe 1 overlap
**Expected overlap:** Security might also mention PII (architecture concern)

---

## Execution Plan

1. Spawn 4 sub-agents in PARALLEL with Task tool
2. Collect all findings
3. Deduplicate by line number
4. Score against known issues
5. Calculate metrics

---

## Actual Results

### Agent 1: Security Auditor
```
[TO BE FILLED]
```

### Agent 2: Performance Analyst
```
[TO BE FILLED]
```

### Agent 3: Architecture Critic
```
[TO BE FILLED]
```

### Agent 4: Correctness Checker
```
[TO BE FILLED]
```

---

## Analysis

| Metric | Actual | Target | Pass? |
|--------|--------|--------|-------|
| Total unique | ? | ≥8 | ? |
| All specialists report | ? | 4/4 | ? |
| Overlap rate | ? | <30% | ? |
| False positive rate | ? | <20% | ? |
| Actionable | ? | 100% | ? |

---

## Learning

**What worked:**
- ?

**What didn't work:**
- ?

**Recommendations for production use:**
- ?
