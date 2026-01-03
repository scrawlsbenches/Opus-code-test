# GoT Learning Integration

Integration between the Graph of Thought (GoT) task system and the LearningCycle experience capture system.

## Overview

The `GoTLearningBridge` connects GoT task management with machine learning experience tracking, enabling:

- **Experience Capture**: Convert completed tasks into structured learning experiences
- **Failure Tracking**: Record failed attempts with error context and blockers
- **Guidance Retrieval**: Get relevant lessons and past experiences when planning new tasks
- **Pattern Extraction**: Automatically identify successful strategies and anti-patterns
- **Context Awareness**: Map task metadata (category, priority) to learning contexts

## Architecture

```
GoT Task System                    LearningCycle System
┌─────────────┐                   ┌──────────────────┐
│   Tasks     │                   │   Experiences    │
│   Edges     │ ──────────────▶   │   Patterns       │
│  Metadata   │  GoTLearning-     │   Lessons        │
└─────────────┘    Bridge          └──────────────────┘
       │                                    │
       │                                    │
       ▼                                    ▼
  .got/entities/                    .got/learning/
  - Tasks                           - experiences/
  - Decisions                       - patterns/
  - Edges                           - lessons/
```

## Storage Structure

Learning data is stored under `.got/learning/`:

```
.got/
└── learning/
    ├── experiences/
    │   ├── exp_20260103_123456_0001.json
    │   ├── exp_20260103_123457_0002.json
    │   └── ...
    ├── patterns/
    │   ├── seq_abc123.json
    │   ├── strat_def456.json
    │   └── ...
    └── lessons/
        ├── lesson_seq_abc123.json
        ├── lesson_strat_def456.json
        └── ...
```

## Usage

### 1. Initialize the Bridge

```python
from pathlib import Path
from cortical.got.learning_integration import GoTLearningBridge

got_dir = Path("/path/to/.got")
bridge = GoTLearningBridge(got_dir)
```

### 2. Capture Task Completion

```python
experience = bridge.capture_task_completion(
    task_id="T-20260103-123456-abc123",
    task_title="Implement user authentication",
    task_category="feature",
    task_priority="high",
    approach="test-first",
    retrospective=(
        "TDD approach worked well. "
        "Tests caught edge cases early. "
        "Would use same approach for similar features."
    ),
    files_changed=[
        "api/auth.py",
        "api/middleware.py",
        "tests/test_auth.py"
    ],
    duration_seconds=7200  # 2 hours
)
```

### 3. Capture Task Failure

```python
experience = bridge.capture_task_failure(
    task_id="T-20260103-234567-def456",
    task_title="Integrate payment API",
    task_category="feature",
    task_priority="critical",
    attempted_approach="direct-integration",
    error_message="API authentication failed - missing credentials",
    files_attempted=["payment/gateway.py"],
    blockers=[
        "Need API credentials from vendor",
        "Documentation incomplete"
    ]
)
```

### 4. Get Guidance for New Task

```python
guidance = bridge.get_guidance_for_task(
    task_title="Implement OAuth2 authentication",
    task_category="feature",
    task_priority="high",
    files_to_modify=["api/oauth.py", "api/tokens.py"]
)

# Use guidance
for lesson in guidance['lessons']:
    print(f"Lesson: {lesson.title}")
    for rec in lesson.recommendations:
        print(f"  - {rec}")
    for warn in lesson.warnings:
        print(f"  ⚠ {warn}")

# Review past successes
for exp in guidance['relevant_successes']:
    print(f"Success: {exp.intent}")
    for item in exp.what_worked:
        print(f"  ✓ {item}")

# Avoid past failures
for exp in guidance['relevant_failures']:
    print(f"Failure: {exp.intent}")
    print(f"  Error: {exp.outcome.error_message}")
```

### 5. Link Tasks to Experiences

```python
related = bridge.link_task_to_experiences(
    task_id="T-20260103-345678-ghi789",
    task_category="feature",
    task_title="New feature development"
)

for exp in related:
    print(f"Related: {exp.intent} ({exp.outcome.outcome_type.name})")
```

### 6. Extract Patterns and Lessons

```python
# Run periodically (e.g., every 10 task completions)
results = bridge.extract_patterns_and_lessons()

print(f"Extracted {results['sequence_patterns']} sequence patterns")
print(f"Extracted {results['strategy_patterns']} strategy patterns")
print(f"Distilled {results['lessons']} lessons")
```

### 7. Get Learning Statistics

```python
stats = bridge.get_learning_stats()

print(f"Total experiences: {stats['total_experiences']}")
print(f"Total patterns: {stats['total_patterns']}")
print(f"Total lessons: {stats['total_lessons']}")
print(f"High confidence lessons: {stats['high_confidence_lessons']}")
```

## Mapping Rules

### Task Category → Goal Type

| Task Category | Goal Type       |
|---------------|-----------------|
| feature       | implementation  |
| bugfix        | debugging       |
| refactor      | refactoring     |
| docs          | documentation   |
| test          | testing         |
| chore         | maintenance     |
| *other*       | general         |

### Task Priority → Complexity

| Priority  | Complexity |
|-----------|------------|
| critical  | complex    |
| high      | complex    |
| medium    | moderate   |
| low       | simple     |

## Auto-Tagging

Experiences are automatically tagged with:

- `task:{task_id}` - The originating task ID
- `category:{category}` - Task category
- `priority:{priority}` - Task priority
- `approach:{approach}` - Strategy used (if provided)
- `{goal_type}` - Derived goal type (e.g., "implementation")
- `{domain}` - Inferred domain from file paths
- `success` or `failure` - Outcome type

## Efficiency Scoring

Task duration is converted to an efficiency score:

- **< 1 hour**: 1.0 (very efficient)
- **1-4 hours**: 0.8 (efficient)
- **4-8 hours**: 0.6 (moderate)
- **> 8 hours**: 0.4 (slow)

This helps identify which approaches lead to faster completions.

## Retrospective Parsing

The bridge attempts to parse retrospectives into structured reflection:

- **What worked**: Sentences with "worked", "successful", "good", "effective"
- **What didn't work**: Sentences with "failed", "problem", "issue", "difficult"
- **Would do differently**: Sentences with "next time", "should have", "could have"

Example:
```
"TDD worked well. Had issues with mocks. Next time, use fixtures."
```

Becomes:
```python
{
    'worked': ["TDD worked well"],
    'didnt_work': ["Had issues with mocks"],
    'different': ["Next time, use fixtures"]
}
```

## Integration with got_utils.py

The bridge can be integrated into `scripts/got_utils.py` for CLI usage:

```python
# In got_utils.py
from cortical.got.learning_integration import GoTLearningBridge

@task.command()
@click.argument('task_id')
@click.option('--retrospective', help='Completion notes')
def complete(task_id, retrospective):
    """Complete a task and capture learning."""
    # ... existing completion logic ...

    # Capture learning
    bridge = GoTLearningBridge(got_dir)
    bridge.capture_task_completion(
        task_id=task_id,
        retrospective=retrospective,
        # ... other fields from task ...
    )
```

## Pattern Types

The system extracts three types of patterns:

1. **Sequence Patterns**: Action sequences that correlate with success
   - Example: "write_test → write_code → refactor" succeeds 85% of the time

2. **Strategy Patterns**: Strategies that work for specific goal types
   - Example: "test-first" succeeds 90% for "implementation" goals

3. **Anti-Patterns**: Combinations that frequently fail
   - Example: "direct-integration" fails for "complex" external APIs

## Lessons

Patterns are distilled into actionable lessons:

- **Recommendations**: What to do in similar situations
- **Warnings**: What to avoid
- **Confidence**: Statistical strength (0.0-1.0)
- **Applicability**: Context conditions where lesson applies

## Demo

Run the demo to see the integration in action:

```bash
PYTHONPATH=/home/user/Opus-code-test python examples/got_learning_demo.py
```

## Tests

Run the test suite:

```bash
python -m pytest tests/unit/test_got_learning_integration.py -v
```

Coverage:
- Experience capture (success and failure)
- Guidance retrieval
- Experience linking
- Pattern extraction
- Helper method validation
- Error handling

## Future Enhancements

Potential improvements:

1. **Automatic Capture**: Hook into GoT task completion events
2. **CLI Integration**: Add learning commands to `got_utils.py`
3. **Lesson Suggestions**: Proactively suggest lessons during task creation
4. **Team Learning**: Aggregate experiences across multiple agents/sessions
5. **Lesson Validation**: Track whether applied lessons actually helped
6. **Cross-Project Learning**: Share lessons across different projects

## See Also

- `llm_orchestration/learning.py` - Core learning system
- `cortical/got/api.py` - GoT task management API
- `examples/got_learning_demo.py` - Usage demonstration
- `tests/unit/test_got_learning_integration.py` - Test suite
