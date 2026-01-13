# Debugging Complex Systems

Effective debugging is not trial and error. It is a systematic, scientific process of forming hypotheses, designing experiments, and drawing conclusions from evidence.

## The Scientific Debugging Method

Debugging is empirical science applied to code. Follow the scientific method:

### The Cycle

```
1. OBSERVE     → What is actually happening?
2. HYPOTHESIZE → What could cause this behavior?
3. PREDICT     → If my hypothesis is correct, what else should I see?
4. EXPERIMENT  → Design a test to verify or falsify the hypothesis
5. CONCLUDE    → Was the hypothesis correct? What did I learn?
6. REPEAT      → If wrong, form a new hypothesis based on new evidence
```

### Example

**Observation**: `transaction.commit()` fails intermittently with "file locked"

**Hypothesis 1**: Another process holds the lock

**Prediction**: If another process holds the lock, I should see it in `lsof`

**Experiment**:
```bash
lsof +D /path/to/.got/
# Result: Only this process has the file open
```

**Conclusion**: Hypothesis falsified. Another process is not the cause.

**Hypothesis 2**: The same process acquires the lock twice (deadlock)

**Prediction**: If deadlocking, I should see recursive lock acquisition in the call stack

**Experiment**:
```python
import traceback
try:
    transaction.commit()
except Exception:
    traceback.print_exc()
# Result: Shows lock acquired in both commit() and nested flush()
```

**Conclusion**: Hypothesis confirmed. Fix: Use reentrant lock (RLock).

## Reproducing Bugs Reliably

A bug you cannot reproduce is a bug you cannot fix with confidence.

### Reproduction Requirements

1. **Exact inputs** - What data triggered the bug?
2. **Exact state** - What was the system state before?
3. **Exact sequence** - What operations led to the failure?
4. **Exact environment** - Python version, OS, dependencies?

### Reproduction Strategies

| Strategy | When to Use | Example |
|----------|-------------|---------|
| Minimal reproducer | Complex scenarios | Strip away unrelated code until bug remains |
| Seed capture | Random behavior | Log the random seed, replay with same seed |
| State snapshot | Complex state | Serialize state before failure, restore in test |
| Request logging | API bugs | Log full request/response, replay |
| Event recording | UI/async bugs | Record event sequence, replay |

### Creating a Minimal Reproducer

```python
# Start with the failing scenario
processor.process_document("doc1", "long complex text...")
processor.process_document("doc2", "more complex text...")
processor.compute_all()
result = processor.query("specific query")  # FAILS

# Remove unrelated parts
processor.process_document("doc1", "minimal text")  # Simpler
# processor.process_document("doc2", ...)  # Remove if not needed
processor.compute_all()
result = processor.query("specific query")  # Still FAILS

# You now have a minimal reproducer for the test suite
```

### Write the Test First

Once you can reproduce, write a failing test BEFORE fixing:

```python
def test_regression_issue_123():
    """
    Regression test for issue #123.

    Bug: Query fails when document contains only stopwords.
    Root cause: Empty token list caused division by zero.
    """
    processor = CorticalTextProcessor()
    processor.process_document("doc1", "the and or")
    processor.compute_all()
    # This used to raise ZeroDivisionError
    result = processor.find_documents_for_query("test")
    assert result == []  # Empty result, not an exception
```

## Binary Search for Bug Location

When you know the bug exists but not where, use binary search to narrow down.

### Code Binary Search

If a large function fails, bisect it:

```python
def complex_operation(data):
    # 100 lines of code
    step1_result = step1(data)
    print(f"After step1: {step1_result}")  # Checkpoint 1

    step2_result = step2(step1_result)
    print(f"After step2: {step2_result}")  # Checkpoint 2

    step3_result = step3(step2_result)
    print(f"After step3: {step3_result}")  # Checkpoint 3

    # ... continue bisecting until you find the bad step
```

### Input Binary Search

If certain inputs fail, bisect the input:

```python
# Large input fails
large_corpus = load_corpus()  # 1000 documents
result = process(large_corpus)  # FAILS

# Try first half
first_half = large_corpus[:500]
result = process(first_half)  # PASSES

# Try third quarter (500-750)
third_quarter = large_corpus[:750]
result = process(third_quarter)  # FAILS

# The bug is triggered by documents 500-750
# Continue bisecting...
```

### Time Binary Search (git bisect)

See the dedicated section below.

## Reading Error Messages Carefully

The error message is your first and most important clue. Read it completely.

### Anatomy of a Python Traceback

```
Traceback (most recent call last):           ← Start here (oldest frame)
  File "cortical/got/api.py", line 42, in create_task
    entity = self.store.create(task_data)
  File "cortical/got/versioned_store.py", line 156, in create
    self._validate(data)
  File "cortical/got/versioned_store.py", line 203, in _validate
    raise ValueError(f"Missing required field: {field}")
ValueError: Missing required field: priority  ← End here (actual error)
```

### What to Extract

| Part | Information | Example |
|------|-------------|---------|
| Exception type | Category of error | `ValueError` = bad input |
| Exception message | Specific problem | `Missing required field: priority` |
| Bottom frame | Where error was raised | `_validate` line 203 |
| Top frame | Entry point | `create_task` line 42 |
| Frame chain | Call path | api.py -> versioned_store.py |

### Common Exception Types and What They Mean

| Exception | Common Cause | Debugging Approach |
|-----------|--------------|-------------------|
| `AttributeError: 'NoneType' has no attribute 'x'` | Function returned None unexpectedly | Check return values upstream |
| `KeyError: 'key'` | Dictionary missing expected key | Print dict contents, check key spelling |
| `IndexError: list index out of range` | Empty list or wrong index | Print list length and index |
| `TypeError: cannot unpack non-iterable NoneType` | Function returned None instead of tuple | Check function return statement |
| `FileNotFoundError` | Wrong path or file not created | Print absolute path, check directory |
| `RecursionError` | Infinite recursion | Check base case, print recursion depth |

### Reading the Full Message

```python
# BAD: Skimming the error
"Some kind of ValueError"

# GOOD: Reading every word
"ValueError: Missing required field: priority"
# This tells you exactly what's wrong: the 'priority' field is missing
```

## Git Bisect for Regression Hunting

When code worked before and now it doesn't, use git bisect to find the breaking commit.

### Basic Workflow

```bash
# Start bisect
git bisect start

# Mark current (broken) commit as bad
git bisect bad

# Mark a known-good commit (when it worked)
git bisect good abc123

# Git checks out a middle commit
# Test it manually
python -m pytest tests/unit/test_feature.py
# Result: fails

# Mark as bad
git bisect bad

# Git checks out another commit
# Test again
python -m pytest tests/unit/test_feature.py
# Result: passes

# Mark as good
git bisect good

# Continue until git finds the first bad commit
# "abc456 is the first bad commit"

# When done, return to original state
git bisect reset
```

### Automated Bisect

Write a script that returns 0 for good, 1 for bad:

```bash
#!/bin/bash
# test_script.sh
python -m pytest tests/unit/test_specific.py -x
exit $?
```

Run automated bisect:

```bash
git bisect start
git bisect bad HEAD
git bisect good v1.0.0
git bisect run ./test_script.sh
# Git automatically finds the first bad commit
git bisect reset
```

### Bisect Tips

| Situation | Solution |
|-----------|----------|
| Test requires build step | Include build in test script |
| Some commits don't compile | Use `git bisect skip` |
| Need to preserve changes | Stash before bisect, pop after |
| Bisect takes too long | Use larger steps with `git bisect visualize` |

## Debugging Concurrent and Async Code

Concurrency bugs are notoriously difficult because they are timing-dependent.

### Race Condition Signatures

| Symptom | Likely Cause |
|---------|--------------|
| Works sometimes, fails sometimes | Race condition |
| Fails under load, works when quiet | Resource contention |
| Works in debugger, fails normally | Timing-dependent bug |
| "Impossible" state observed | Interleaved operations |

### Debugging Strategies

**1. Add Strategic Logging**

```python
import threading
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def critical_section():
    thread_id = threading.current_thread().name
    logger.debug(f"[{thread_id}] Entering critical section")
    # ... operation ...
    logger.debug(f"[{thread_id}] Exiting critical section")
```

**2. Make It Deterministic**

```python
# Force sequential execution to confirm it's a concurrency bug
with lock:
    operation_a()
    operation_b()
# If this works but parallel fails, you have a race condition
```

**3. Add Sleep to Expose Races**

```python
# Temporarily add sleep to widen race window
def operation_a():
    read_value()
    time.sleep(0.1)  # Widen window for race
    write_value()
```

**4. Use Thread Sanitizers**

```bash
# Run with thread safety checks
python -X dev your_script.py
```

### Async Debugging

```python
import asyncio

async def debug_coroutine():
    print(f"Current task: {asyncio.current_task().get_name()}")
    print(f"All tasks: {[t.get_name() for t in asyncio.all_tasks()]}")

# Add timeouts to catch deadlocks
async def with_timeout():
    try:
        await asyncio.wait_for(potentially_hanging_op(), timeout=5.0)
    except asyncio.TimeoutError:
        print("Operation timed out - possible deadlock")
```

### Common Concurrency Bugs

| Bug | Symptom | Fix |
|-----|---------|-----|
| Race condition | Intermittent wrong results | Add proper locking |
| Deadlock | System hangs | Use lock ordering or timeout |
| Livelock | System busy but no progress | Add backoff |
| Thread starvation | Some threads never run | Use fair scheduling |
| Resource leak | Memory/handles grow | Ensure cleanup in finally |

## Common Bug Patterns and Signatures

Learn to recognize these patterns immediately.

### Off-by-One Errors

**Signature**: Works for most inputs, fails at boundaries

```python
# BUG: misses last element
for i in range(len(items) - 1):  # Should be len(items)
    process(items[i])

# BUG: index out of bounds
items[len(items)]  # Should be len(items) - 1
```

### Null/None Reference

**Signature**: `AttributeError: 'NoneType' has no attribute`

```python
# BUG: doesn't handle None return
result = find_item(query)
result.process()  # Crashes if find_item returns None

# FIX: explicit None check
result = find_item(query)
if result is not None:
    result.process()
```

### Mutation During Iteration

**Signature**: `RuntimeError: dictionary changed size during iteration`

```python
# BUG: modifying while iterating
for key in dictionary:
    if should_remove(key):
        del dictionary[key]  # Crashes

# FIX: iterate over copy
for key in list(dictionary.keys()):
    if should_remove(key):
        del dictionary[key]
```

### Shallow vs Deep Copy

**Signature**: Changes to copy affect original

```python
# BUG: shallow copy shares nested objects
config = default_config.copy()
config['nested']['value'] = 'new'  # Modifies default_config too!

# FIX: deep copy
import copy
config = copy.deepcopy(default_config)
```

### Resource Leaks

**Signature**: Handles/memory grow over time

```python
# BUG: file never closed
f = open('file.txt')
data = f.read()
# Missing f.close()

# FIX: use context manager
with open('file.txt') as f:
    data = f.read()
# Automatically closed
```

### Type Coercion Surprises

**Signature**: Unexpected behavior with types

```python
# BUG: string vs int comparison
user_input = "5"
if user_input > 10:  # String comparison, not numeric!
    print("Large")  # "5" > "10" is True (string comparison)

# FIX: explicit conversion
if int(user_input) > 10:
    print("Large")
```

### Floating Point Precision

**Signature**: Math seems wrong

```python
# BUG: floating point comparison
if 0.1 + 0.2 == 0.3:  # False!
    print("Equal")

# FIX: use approximate comparison
import math
if math.isclose(0.1 + 0.2, 0.3):
    print("Equal")  # True
```

## Debugging Tools

### Python Debugger (pdb)

```python
# Drop into debugger at specific point
import pdb; pdb.set_trace()

# Or use breakpoint() in Python 3.7+
breakpoint()

# Key commands:
# n - next line
# s - step into function
# c - continue execution
# p expr - print expression
# l - list source code
# w - print stack trace
# q - quit debugger
```

### Logging

```python
import logging

# Configure once at startup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

# Use throughout code
logger.debug("Detailed info for debugging")
logger.info("General information")
logger.warning("Something unexpected but handled")
logger.error("Something failed")
logger.exception("Exception with traceback")
```

### Profilers

```python
# CPU profiling
import cProfile
cProfile.run('slow_function()', sort='cumtime')

# Line-by-line profiling
# pip install line_profiler
# @profile decorator, then: kernprof -l script.py

# Memory profiling
# pip install memory_profiler
# @profile decorator, then: python -m memory_profiler script.py
```

### The inspect Module

```python
import inspect

# Get function signature
sig = inspect.signature(some_function)
print(f"Parameters: {sig}")

# Get source code
source = inspect.getsource(SomeClass.method)
print(source)

# Get file location
location = inspect.getfile(SomeClass)
print(f"Defined in: {location}")

# Get call stack
for frame_info in inspect.stack():
    print(f"{frame_info.filename}:{frame_info.lineno} in {frame_info.function}")

# Get current frame locals
frame = inspect.currentframe()
print(f"Local variables: {frame.f_locals}")
```

### Print Debugging (When Appropriate)

```python
# Strategic prints for quick debugging
print(f"DEBUG: value = {value!r}")  # !r shows repr for strings
print(f"DEBUG: type = {type(value)}")
print(f"DEBUG: len = {len(value) if hasattr(value, '__len__') else 'N/A'}")

# Remember to remove before committing!
```

## Debugging Anti-Patterns

| Anti-Pattern | Why It's Bad | Better Approach |
|--------------|--------------|-----------------|
| Random changes | Creates new bugs | Form hypothesis first |
| Debugging in production | Risky and slow | Reproduce locally |
| Ignoring warnings | Warnings often predict bugs | Fix warnings |
| Debugging without version control | Can't undo mistakes | Commit before debugging |
| Not reading the error | Misses obvious clues | Read the full message |
| Assuming the bug is elsewhere | Delays finding real cause | Start with your code |
| Fixing symptoms not causes | Bug will return | Find root cause |

## The Debugging Mindset

### Rules for Effective Debugging

1. **The bug is in your code** - Assume the bug is in your code until proven otherwise
2. **The computer is not lying** - If behavior seems impossible, your mental model is wrong
3. **Recent changes are suspect** - Most bugs are in recently changed code
4. **Simplify to isolate** - Remove complexity until the bug disappears
5. **One change at a time** - Change one thing, test, repeat
6. **Read before writing** - Understand the code before modifying it
7. **Take breaks** - Fresh eyes find bugs faster

### When You're Stuck

1. **Explain the bug to someone** - Rubber duck debugging works
2. **Step away** - Take a walk, sleep on it
3. **Question your assumptions** - What are you sure of that might be wrong?
4. **Read the documentation** - Maybe the API doesn't work how you think
5. **Search for similar bugs** - Someone else may have hit this
6. **Ask for help** - Fresh perspective finds blind spots

## Key Insight

Debugging is not about cleverness or intuition. It is about discipline and method. The developer who methodically gathers evidence, forms hypotheses, and tests them will outperform the developer who makes random changes hoping something works.

The goal of debugging is not just to fix the bug, but to understand why it occurred and prevent similar bugs in the future. A fix without understanding is a temporary fix.

**Remember**: Every bug is an opportunity to improve the test suite. If a bug reached production, the test suite has a gap. Fill it.
