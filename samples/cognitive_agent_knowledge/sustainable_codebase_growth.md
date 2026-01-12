# Sustainable Codebase Growth

This document teaches AI agents how to grow a codebase sustainably over time. The principles here apply whether you're adding a single function or designing a new subsystem.

## What Is Sustainable Growth?

Sustainable growth means the codebase can continue to evolve without becoming progressively harder to work with. Each change should leave the code at least as healthy as before.

**Sustainable growth looks like:**
- New features integrate cleanly with existing code
- Changes require understanding local context, not the entire system
- Bugs stay contained to their origin, not rippling everywhere
- New developers can become productive within days, not months
- The team moves faster as the codebase matures (patterns emerge)

**Unsustainable growth looks like:**
- Each feature takes longer than the last
- Simple changes require touching many files
- Fixes in one area break unrelated areas
- Only the original author can safely modify certain code
- The team moves slower as the codebase grows

## Technical Debt: The Hidden Cost

Technical debt is the gap between how code should be written and how it actually is. Like financial debt, it accumulates interest.

### How Debt Accumulates

| Action | Immediate Benefit | Long-Term Cost |
|--------|-------------------|----------------|
| Skip tests | Ship faster today | Debug longer tomorrow |
| Copy-paste code | Avoid refactoring | Fix same bug in 5 places |
| Hardcode values | Works for this case | Breaks for the next case |
| Skip documentation | Save 10 minutes | Waste 2 hours later |
| "Temporary" workaround | Unblocks today | Becomes permanent |

### The Compound Interest Problem

Technical debt compounds. A small shortcut creates friction that encourages more shortcuts:

```
Day 1: "Just hardcode this path for now"
Day 30: "Another hardcoded path, same as before"
Day 90: "Paths are everywhere, let's add a config file"
Day 180: "Config file has grown to 500 lines, nobody understands it"
Day 360: "We need to rewrite the configuration system"
```

The rewrite at Day 360 is 10x the effort of doing it right on Day 1.

### Sustainable Alternative

```
Day 1: "Let me create a Config class with one path"
Day 30: "I'll add the new path to the Config class"
Day 90: "Config class has 10 paths, still clear"
Day 180: "Config splits into StorageConfig and NetworkConfig"
Day 360: "Configuration is clean, extensible, documented"
```

Small investments compound positively too.

## Signs of Healthy vs Unhealthy Growth

### Healthy Growth Indicators

1. **Decreasing time per feature** - Patterns are established, abstractions work
2. **Localized changes** - Most features touch 1-3 files
3. **Stable test suite** - Tests rarely need modification when features change
4. **Clear module boundaries** - You know where code belongs
5. **Consistent style** - New code looks like existing code
6. **Growing documentation** - Knowledge accumulates, not just code

### Unhealthy Growth Indicators

1. **Increasing time per feature** - Each change fights the existing structure
2. **Shotgun surgery** - Simple changes touch 10+ files
3. **Fragile tests** - Tests break for unrelated reasons
4. **Ambiguous ownership** - "This could go in three places"
5. **Inconsistent patterns** - Every module does things differently
6. **Stale documentation** - Docs describe how it worked 6 months ago

### Self-Assessment Questions

Ask yourself after each significant change:

- Did this make the codebase easier or harder to work with?
- Would a new team member understand this code?
- If I returned in 6 months, would I understand my own changes?
- Did I leave the code better than I found it?

If the answer to any is "no," consider whether you've introduced debt.

## Core Principles

### Principle 1: Modularity

Modularity means organizing code into discrete units with clear responsibilities.

**Good modularity:**
```
cortical/
  tokenizer/        # Knows only about text -> tokens
  storage/          # Knows only about persistence
  reasoning/        # Knows only about inference

Each module:
- Has a single, clear purpose
- Exposes a clean interface
- Hides implementation details
- Can be tested in isolation
```

**Poor modularity:**
```
cortical/
  utils.py          # 3000 lines of "various utilities"
  helpers.py        # "Helper functions" - for what?
  core.py           # The entire application in one file
```

**Why it matters:** Modular code limits the blast radius of changes. When you modify `tokenizer/`, you don't break `reasoning/`.

### Principle 2: Clear Boundaries

Boundaries define what each component knows about and depends on.

**Strong boundaries:**
```python
# storage/backend.py
class StorageBackend:
    def save(self, key: str, data: bytes) -> None: ...
    def load(self, key: str) -> bytes: ...

# storage/ doesn't know about tokens, documents, or reasoning
# Other modules only interact through this interface
```

**Weak boundaries:**
```python
# storage/backend.py
class StorageBackend:
    def save_tokenized_document(self, doc: TokenizedDoc) -> None:
        # Storage knows about tokenization
        for token in doc.tokens:
            # Storage manipulates tokens directly
            ...
```

**The test:** Can you describe what a module does in one sentence without using "and"?

- Good: "The tokenizer converts text into tokens."
- Bad: "The tokenizer converts text into tokens and stores them and indexes them."

### Principle 3: Incremental Improvement

Never attempt to "fix everything at once." Improve incrementally.

**The incremental approach:**
```
Week 1: Extract one function from the 500-line method
Week 2: Extract another function, add tests
Week 3: Identify common patterns, extract base class
Week 4: Document the emerging architecture
```

**Why this works:**
- Each step is small and reversible
- Tests catch regressions early
- You learn the code while improving it
- Progress is visible and motivating

**Why "big bang" rewrites fail:**
- Months of work with no visible progress
- Old bugs return, new bugs appear
- Context is lost during the rewrite
- Rarely completed before being abandoned

### Principle 4: Design for Change

Assume requirements will change. Write code that accommodates change.

**Designing for change:**
```python
# Easy to extend with new formats
class Serializer:
    def serialize(self, data: Any) -> bytes: ...

class JsonSerializer(Serializer): ...
class ProtobufSerializer(Serializer): ...

# Adding YAML is one new class, no changes to existing code
class YamlSerializer(Serializer): ...
```

**Resisting change:**
```python
def serialize(data: Any, format: str) -> bytes:
    if format == "json":
        ...
    elif format == "protobuf":
        ...
    elif format == "yaml":  # Must modify this function for every new format
        ...
```

**The Open-Closed Principle:** Code should be open for extension but closed for modification. New features should add code, not change existing code.

## When to Refactor vs Leave Alone

Not all imperfect code needs fixing. Refactoring has costs and risks.

### Refactor When

| Situation | Why Refactor |
|-----------|--------------|
| You're already changing the area | Minimal additional cost |
| Code is frequently modified | Investment pays off repeatedly |
| Code blocks new features | Removing the blocker is necessary |
| Code causes recurring bugs | Root cause, not symptoms |
| Team can't understand the code | Understanding is required anyway |

### Leave Alone When

| Situation | Why Leave It |
|-----------|--------------|
| Code is stable and working | Don't introduce risk without need |
| Code is rarely modified | Investment won't pay off |
| Change is purely aesthetic | Preferences differ, bugs don't |
| You don't fully understand it | You might break hidden invariants |
| Deadline pressure is high | Rushed refactoring creates debt |

### The Rule of Three

Before creating an abstraction, wait until you have three concrete examples:

```
1st occurrence: Write the code directly
2nd occurrence: Note the duplication, accept it temporarily
3rd occurrence: Now extract the abstraction
```

**Why wait?** Premature abstraction creates wrong abstractions. With three examples, the pattern is clear.

### Boy Scout Rule

"Leave the code better than you found it."

This doesn't mean rewrite everything. It means:

- Fix the typo in the comment while you're here
- Rename the confusing variable you just figured out
- Extract the repeated code block you copied twice
- Add the test for the edge case you just discovered

Small improvements compound into large ones.

## Managing Complexity at Scale

As systems grow, complexity management becomes critical.

### Complexity Budgets

Treat complexity like a resource with limits:

```
Simple component (< 100 lines): Can be understood in one sitting
Moderate component (100-500 lines): Needs documentation
Complex component (500+ lines): Must be split or heavily documented
```

When a component exceeds its budget, split it before adding more.

### Layered Architecture

Organize code into layers with clear dependencies:

```
┌─────────────────────────────────────┐
│         User Interface              │  Depends on everything below
├─────────────────────────────────────┤
│         Application Logic           │  Coordinates components
├─────────────────────────────────────┤
│         Domain Logic                │  Business rules
├─────────────────────────────────────┤
│         Infrastructure              │  Storage, network, etc.
└─────────────────────────────────────┘
```

**Rule:** Dependencies flow downward only. Infrastructure never imports Application Logic.

### Dependency Injection

Don't hardcode dependencies. Inject them:

```python
# Hardcoded - inflexible, untestable
class DocumentProcessor:
    def __init__(self):
        self.storage = FileStorage("/var/data")  # Fixed!

# Injected - flexible, testable
class DocumentProcessor:
    def __init__(self, storage: Storage):
        self.storage = storage  # Caller decides
```

Dependency injection enables:
- Testing with mocks
- Swapping implementations
- Configuration at runtime
- Clear dependency graphs

### Information Hiding

Hide implementation details. Expose only what's necessary.

```python
# Too much exposure
class TokenCache:
    cache: Dict[str, List[Token]]  # Implementation detail exposed

    def add(self, key: str, tokens: List[Token]): ...
    def get(self, key: str) -> List[Token]: ...
    def rebuild_index(self): ...  # Why does caller need this?

# Proper hiding
class TokenCache:
    def add(self, key: str, tokens: List[Token]): ...
    def get(self, key: str) -> List[Token]: ...
    # Cache manages its own index internally
```

**Why hiding matters:** Exposed details become dependencies. If users access `cache`, you can never change how caching works.

## Documentation as Investment

Documentation is not overhead. It's infrastructure.

### Documentation Pays Dividends

| Documentation Type | Initial Cost | Ongoing Benefit |
|-------------------|--------------|-----------------|
| API docstrings | 5 minutes | Hours saved per user |
| Architecture guide | 2 hours | Days saved onboarding |
| Decision records | 10 minutes | Weeks saved debugging "why" |
| README | 30 minutes | Immediate productivity for newcomers |

### What to Document

**Always document:**
- Public APIs (what they do, parameters, return values)
- Non-obvious decisions (why this approach, not that one)
- Configuration (what each option does)
- Setup and installation (how to get started)

**Often document:**
- Architecture (how components relate)
- Performance characteristics (what's fast, what's slow)
- Error handling (what can go wrong, what to do)

**Sometimes document:**
- Internal implementation (only if complex)
- History (only if it affects current behavior)

### Self-Documenting Code

Good code reduces documentation needs:

```python
# Needs documentation
def proc(d, f):
    """Process data with filter."""  # Still unclear
    ...

# Self-documenting
def process_documents_matching_filter(
    documents: List[Document],
    content_filter: ContentFilter
) -> List[ProcessedDocument]:
    ...  # The signature tells the story
```

Self-documenting means:
- Descriptive names
- Clear types
- Logical structure
- Consistent patterns

Documentation then covers the "why," not the "what."

### Living Documentation

Dead documentation is worse than no documentation. It misleads.

**Keep documentation alive:**
- Review docs when changing related code
- Delete outdated information immediately
- Automate what can be automated (API docs from code)
- Date time-sensitive information

## Test Coverage as Growth Enabler

Tests are not overhead. They're the foundation that enables confident change.

### Why Tests Enable Growth

Without tests:
```
Developer wants to change X
 → Doesn't know what might break
  → Manually tests everything (slow)
   → Misses edge cases
    → Introduces bugs
     → Becomes afraid to change X
      → X accumulates debt
```

With tests:
```
Developer wants to change X
 → Changes X
  → Runs tests
   → Tests catch regressions
    → Fixes regressions
     → Confidently ships
      → X stays healthy
```

### The Test Pyramid

```
         /\
        /  \        E2E Tests (few)
       /────\       - Slow, expensive
      /      \      - Test user journeys
     /────────\
    /          \    Integration Tests
   /────────────\   - Medium speed/cost
  /              \  - Test component interactions
 /────────────────\
/                  \  Unit Tests (many)
 ──────────────────   - Fast, cheap
                      - Test single functions
```

**Balance:**
- Many unit tests (fast feedback)
- Some integration tests (verify connections)
- Few E2E tests (verify critical paths)

### What to Test

**Always test:**
- Public APIs
- Business logic
- Edge cases
- Error handling
- Regressions (bugs that were fixed)

**Usually test:**
- Complex internal logic
- Data transformations
- State transitions

**Rarely test:**
- Simple getters/setters
- Framework code (it's already tested)
- Trivial wrappers

### Test Quality

Bad tests are worse than no tests. They create maintenance burden without catching bugs.

**Good tests:**
- Test behavior, not implementation
- Are independent (can run in any order)
- Are deterministic (same result every time)
- Are fast (< 1 second per test)
- Have clear assertions (know what they're checking)

**Bad tests:**
- Test internal implementation details
- Depend on global state or order
- Flake randomly
- Take minutes to run
- Assert everything (unclear what matters)

### Coverage Targets

Coverage is a metric, not a goal. High coverage with bad tests is worse than moderate coverage with good tests.

**Reasonable targets:**
- Core business logic: 90%+
- Application code: 80%+
- Infrastructure/glue: 60%+
- Overall: 75-85%

**The 100% trap:** Chasing 100% coverage leads to testing trivial code and writing tests for test's sake. Aim for meaningful coverage.

## Practical Guidelines

### Before Writing Code

1. Understand the existing patterns
2. Identify where new code belongs
3. Consider how it will be tested
4. Plan the interface before implementation

### While Writing Code

1. Follow existing conventions
2. Write tests alongside code
3. Keep functions small and focused
4. Name things clearly

### After Writing Code

1. Review your own changes
2. Run the full test suite
3. Update relevant documentation
4. Consider: "Did I leave this better?"

### When Inheriting Code

1. Read before changing
2. Add tests before refactoring
3. Improve incrementally
4. Document what you learn

## Key Takeaways

1. **Sustainable growth compounds positively** - Small investments in quality yield large returns over time.

2. **Technical debt compounds negatively** - Small shortcuts accumulate into large problems.

3. **Modularity contains complexity** - Clear boundaries let you think about one thing at a time.

4. **Refactor strategically** - Not all imperfect code needs fixing. Prioritize high-impact areas.

5. **Documentation is investment** - Time spent documenting saves multiples of that time later.

6. **Tests enable confidence** - Without tests, change becomes risky. With tests, change becomes routine.

7. **Think long-term** - The code you write today will be maintained for years. Write for your future self.

The goal is not perfect code. The goal is code that can continue to evolve without fighting itself. Sustainable growth is about making that evolution possible.
