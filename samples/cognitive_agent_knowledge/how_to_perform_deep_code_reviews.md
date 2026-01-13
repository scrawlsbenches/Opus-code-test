# How to Perform Deep Code Reviews

A deep code review goes beyond surface-level syntax checking to understand architecture, design decisions, and whether the code is doing the right thing correctly.

## Deep Code Review vs Surface Review

| Surface Review | Deep Code Review |
|----------------|------------------|
| "Does it compile?" | "Is this the right approach?" |
| "Are there syntax errors?" | "Does this match the architecture?" |
| "Does the test pass?" | "Does the test test the right thing?" |
| "Is the code formatted?" | "Is the code maintainable?" |

## Deep Code Review Process

### Step 1: Understand the Context

Before reviewing code, understand:
- What problem is being solved?
- What was the previous approach?
- What constraints exist?
- Who will maintain this?

```bash
# Check recent changes
git log --oneline -10

# Check what changed
git diff main...HEAD

# Understand the file's history
git log -p --follow path/to/file.py
```

### Step 2: Review Architecture

Ask these questions:
1. **Does this belong here?** - Is this the right module/class/function?
2. **Does it follow existing patterns?** - Or does it introduce inconsistency?
3. **Does it duplicate existing code?** - Search before approving
4. **Are dependencies appropriate?** - Circular imports? Wrong direction?

```bash
# Search for similar implementations
grep -r "class.*Store" cortical/
grep -r "def.*validate" cortical/

# Check imports
grep "^from\|^import" path/to/file.py
```

### Step 3: Review the Implementation

Check for:
1. **Correctness** - Does it do what it claims?
2. **Edge cases** - What happens with empty input? None? Large data?
3. **Error handling** - Are errors caught and handled appropriately?
4. **Performance** - Are there O(n²) loops? Unnecessary allocations?
5. **Security** - Input validation? SQL injection? Path traversal?

### Step 4: Review the Tests

Tests are as important as implementation:
1. **Do tests exist?** - No tests = no confidence
2. **Do tests test the right thing?** - Not just "it runs"
3. **Are edge cases covered?** - Empty, None, boundary values
4. **Can tests fail?** - A test that can't fail is useless

### Step 5: Review Documentation

Check that:
1. Docstrings explain the "why", not just the "what"
2. Complex logic has inline comments
3. Public APIs have clear documentation
4. Changes are reflected in relevant docs

## Code Review Checklist

### Architecture
- [ ] Code is in the right location
- [ ] Follows existing patterns
- [ ] No unnecessary duplication
- [ ] Dependencies flow in correct direction
- [ ] No circular imports

### Implementation
- [ ] Logic is correct
- [ ] Edge cases handled
- [ ] Errors handled appropriately
- [ ] No obvious performance issues
- [ ] No security vulnerabilities

### Testing
- [ ] Tests exist
- [ ] Tests are meaningful
- [ ] Edge cases tested
- [ ] Tests can fail

### Documentation
- [ ] Docstrings present
- [ ] Complex logic commented
- [ ] Public API documented

## Using Tools for Deep Review

### Static Analysis
```bash
# Find code health issues
python -m cortical.cli.audit health cortical/

# Check for TODOs, FIXMEs
grep -r "TODO\|FIXME\|HACK" cortical/
```

### Understanding APIs
```python
import inspect

# Check function signature
sig = inspect.signature(SomeClass.method)
print(sig)

# List all methods
methods = [m for m in dir(SomeClass) if not m.startswith('_')]
print(methods)

# Get inheritance chain
print(inspect.getmro(SomeClass))
```

### Checking for Duplication
```bash
# Find similar patterns
grep -r "def validate" cortical/
grep -r "class.*Manager" cortical/

# Check if utility exists
grep -r "split_identifier\|split_camel" cortical/
```

## Example Deep Code Review

**Code Under Review**: New tokenizer normalization using regex

**Step 1: Context**
- Problem: CamelCase identifiers not being split
- Previous: No splitting, compound words isolated

**Step 2: Architecture Review**
```bash
# Check for existing implementations
grep -r "split_identifier\|camel\|snake" cortical/
# Found: cortical/tokenizer.py has split_identifier!
```

**Finding**: There's already a well-tested `split_identifier` function. The new regex duplicates this functionality but is inferior (doesn't handle acronyms).

**Step 3: Implementation Review**
```python
# Test the existing function
from cortical.tokenizer import split_identifier
split_identifier("XMLParser")  # ['xml', 'parser'] - handles acronyms!

# Test the new regex approach
# XMLParser -> ['parser'] - loses the acronym!
```

**Finding**: The existing `split_identifier` is better. Use it instead.

**Step 4: Recommendation**
- Don't reinvent the wheel
- Import and use `split_identifier` from `cortical.tokenizer`
- Delete the duplicate regex implementation

## Key Principles

1. **Search before approving** - Does this duplicate existing code?
2. **Understand the architecture** - Does this fit?
3. **Think about the future** - Will this be maintainable?
4. **Test the tests** - Are they meaningful?
5. **Question assumptions** - Why this approach?

## Red Flags to Watch For

| Red Flag | Why It's Concerning |
|----------|---------------------|
| No tests | Can't verify correctness |
| Duplicate code | Technical debt |
| Magic numbers | Unclear intent |
| Catch-all exception handlers | Hides bugs |
| Complex nested logic | Hard to maintain |
| Missing error handling | Silent failures |
| Hardcoded paths/values | Not configurable |
| No documentation | Knowledge silos |
