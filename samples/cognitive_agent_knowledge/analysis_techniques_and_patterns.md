# Analysis Techniques and Patterns

This document describes techniques for analyzing code, systems, and problems effectively.

## The Analysis Mindset

Analysis is about understanding before acting. Key principles:

1. **Observe before hypothesizing** - Gather data first
2. **Question assumptions** - Including your own
3. **Follow the evidence** - Not your preferences
4. **Document as you go** - Memory is unreliable
5. **Verify your conclusions** - Test your understanding

## Root Cause Analysis

### The Five Whys

Keep asking "why" until you reach the root cause:

1. Why did the query fail? → The word had no associations
2. Why did it have no associations? → It was an isolated token
3. Why was it isolated? → No other words co-occurred with it
4. Why didn't words co-occur? → The compound word wasn't split
5. Why wasn't it split? → The tokenizer didn't handle CamelCase

**Root cause**: Tokenizer doesn't split CamelCase identifiers

### Fault Tree Analysis

Work backwards from the failure:

```
Query returns empty
├── Word not in vocabulary
│   └── Training didn't include it
└── Word has no links
    ├── Word is rare (appears once)
    └── Word is compound (never co-occurs)
        └── Tokenizer doesn't split identifiers ← ROOT CAUSE
```

## Code Analysis Techniques

### Static Analysis

Analyze code without running it:

```bash
# Find all classes
grep -r "^class " cortical/

# Find all functions
grep -r "^def \|^    def " cortical/

# Find imports
grep -r "^from\|^import" cortical/

# Find patterns
grep -r "TODO\|FIXME\|HACK" cortical/
```

### Dynamic Analysis

Analyze code by running it:

```python
import inspect

# Understand a function
sig = inspect.signature(function)
source = inspect.getsource(function)

# Trace execution
import sys
def trace(frame, event, arg):
    print(f"{event}: {frame.f_code.co_name}")
    return trace
sys.settrace(trace)
```

### Dependency Analysis

Understand how components relate:

```python
# Check what a module imports
import ast
tree = ast.parse(open('module.py').read())
imports = [node for node in ast.walk(tree)
           if isinstance(node, (ast.Import, ast.ImportFrom))]

# Check who imports a module
grep -r "from cortical.tokenizer import" cortical/
grep -r "import cortical.tokenizer" cortical/
```

## Pattern Recognition

### Code Smells

| Smell | Symptom | Likely Problem |
|-------|---------|----------------|
| Long functions | > 50 lines | Doing too much |
| Deep nesting | > 3 levels | Complex logic |
| Many parameters | > 5 params | Needs refactoring |
| Duplicate code | Similar blocks | Missing abstraction |
| Feature envy | Uses other class's data | Wrong location |
| God class | Does everything | Needs splitting |

### Architecture Patterns

| Pattern | When to Use | Example |
|---------|-------------|---------|
| Repository | Data access abstraction | StorageBackend |
| Factory | Object creation | create_container() |
| Strategy | Interchangeable algorithms | TokenizerStrategy |
| Observer | Event notification | EventEmitter |
| Decorator | Adding behavior | @cached |

### Anti-Patterns

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| Reinventing the wheel | Duplicates existing code | Search first |
| Not invented here | Rejects good solutions | Use what works |
| Golden hammer | One solution for everything | Right tool for job |
| Premature optimization | Optimizes before needed | Measure first |
| Copy-paste programming | Duplicates bugs too | Extract function |

## Investigation Workflow

### Step 1: Reproduce

Before analyzing, confirm you can reproduce:

```bash
# Run the failing case
python -m cortical.cognitive query "texttoatomsbridge"
# Confirm: "No associations found"
```

### Step 2: Isolate

Narrow down the problem:

```python
# Is the word in vocabulary?
# Is the word in the bridge?
# Does the word have links?
# What does the tokenizer produce?
```

### Step 3: Trace

Follow the execution path:

```python
# Trace tokenization
tok = BPETokenizer()
text = "TextToAtomsBridge"
normalized = tok._normalize(text)
print(f"Normalized: {normalized}")
words = tok._split_words(normalized)
print(f"Words: {words}")
```

### Step 4: Compare

Compare working vs non-working cases:

```python
# Working case
tok.tokenize("bridge")  # ['bridge'] - single word, has links

# Non-working case
tok.tokenize("TextToAtomsBridge")  # ['texttoatomsbridge'] - compound, isolated
```

### Step 5: Hypothesize and Test

Form a hypothesis and test it:

```python
# Hypothesis: Splitting CamelCase would help
"TextToAtomsBridge" → "text to atoms bridge" → ['text', 'to', 'atoms', 'bridge']
# Each word has associations! Hypothesis confirmed.
```

## Documentation During Analysis

### What to Record

1. **Initial state** - What did you observe?
2. **Hypothesis** - What did you think caused it?
3. **Evidence** - What did you find?
4. **Conclusion** - What was the root cause?
5. **Solution** - How did you fix it?

### Where to Record

- `samples/cognitive_agent_knowledge/` - For future sessions to query
- `docs/sessions/` - Session-specific findings
- Code comments - For future maintainers
- CLAUDE.md - For operational guidance

## Tools Reference

### Search Tools
- `grep -r "pattern" path/` - Search file contents
- `find path/ -name "*.py"` - Find files by name
- `git log --oneline -20` - Recent changes
- `git blame file.py` - Who changed what

### Understanding Tools
- `inspect.signature()` - Function parameters
- `inspect.getsource()` - Source code
- `inspect.getmro()` - Class hierarchy
- `ast.parse()` - Parse structure

### Cognitive Agent
- `python -m cortical.cognitive query "word"` - Find associations
- `python -m cortical.cognitive ask "question"` - Natural language
- `python -m cortical.cli.audit health path/` - Code health

## Key Insight

Good analysis is systematic:
1. **Observe** - What's actually happening?
2. **Hypothesize** - What might cause it?
3. **Test** - Is your hypothesis correct?
4. **Conclude** - What's the root cause?
5. **Document** - What did you learn?

The goal is understanding, not just fixing. A fix without understanding is a fix that might break again.
