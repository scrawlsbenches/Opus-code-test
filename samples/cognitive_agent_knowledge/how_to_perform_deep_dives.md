# How to Perform Deep Dives

A deep dive is a thorough investigation into a system, codebase, or problem to understand its true nature before making changes.

## When to Deep Dive

- Before fixing a bug that seems simple but might have hidden complexity
- When encountering unfamiliar code or architecture
- When a quick fix didn't work and you need to understand why
- Before making significant changes to core systems
- When symptoms don't match your initial hypothesis

## Deep Dive Process

### Phase 1: Orient Without Acting

1. **Read before writing** - Never change code you haven't read
2. **Use inspect API** - Understand function signatures, class hierarchies
3. **Trace the data flow** - Follow inputs to outputs
4. **Check existing tools** - Search for utilities that already exist

```python
import inspect

# Understand a function's signature
sig = inspect.signature(SomeClass.method)
print(sig)

# Find where a class is defined
file_path = inspect.getfile(SomeClass)

# Get source code
source = inspect.getsource(SomeClass.method)
```

### Phase 2: Form and Test Hypotheses

1. **State your hypothesis clearly** - "I believe X causes Y because Z"
2. **Find evidence** - Don't assume, verify
3. **Test incrementally** - Small experiments, not big changes
4. **Document what you learn** - Future sessions will thank you

### Phase 3: Verify Understanding

1. **Can you explain it simply?** - If not, you don't understand it
2. **Can you predict behavior?** - Test your predictions
3. **Do the tests pass?** - Run the test suite
4. **Does it match documentation?** - Or is the docs wrong?

## Deep Dive Anti-Patterns

| Anti-Pattern | Why It's Bad | Better Approach |
|--------------|--------------|-----------------|
| Guessing at fixes | Creates new bugs | Understand first, then fix |
| Reading only the error | Misses root cause | Trace the full path |
| Skipping existing code | Reinvents the wheel | Search before creating |
| Changing without tests | Can't verify fix | Write test first |
| Not documenting findings | Knowledge lost | Write it down |

## Tools for Deep Dives

### Code Analysis
- `grep` / `Grep tool` - Find patterns across codebase
- `inspect` module - Understand APIs programmatically
- `ast` module - Parse Python structure
- Test suite - Understand expected behavior

### Cognitive Agent
- `python -m cortical.cognitive query "concept"` - Find related concepts
- `python -m cortical.cognitive ask "question"` - Natural language queries
- `python -m cortical.cli.audit health` - Find code quality issues

### Git Archaeology
- `git log --oneline -20` - Recent changes
- `git blame file.py` - Who changed what
- `git log -p --follow file.py` - Full history of a file

## Example Deep Dive

**Problem**: Cognitive agent can't answer "What is TextToAtomsBridge?"

**Phase 1: Orient**
```python
# Check if word exists in vocabulary
python -m cortical.cognitive query "texttoatomsbridge"
# Result: No associations found

# Check the tokenizer
import inspect
from cortical.cognitive.text_bridge import BPETokenizer
print(inspect.getsource(BPETokenizer._normalize))
```

**Phase 2: Hypothesis**
"The tokenizer converts TextToAtomsBridge to a single token 'texttoatomsbridge' which has no semantic links because it's rare."

**Phase 3: Verify**
```python
# Test tokenization
tok = BPETokenizer()
print(tok.tokenize("TextToAtomsBridge"))
# Result: ['texttoatomsbridge'] - single isolated token!

# Check if component words work
python -m cortical.cognitive query "bridge"
# Result: Has associations! The components work.
```

**Conclusion**: The tokenizer needs to split CamelCase identifiers into components.

## Key Insight

Deep dives are about understanding before acting. The time spent understanding saves more time than it costs because:
- You fix the right thing the first time
- You don't introduce new bugs
- You discover related issues
- You learn the codebase
- You can explain your fix to others
