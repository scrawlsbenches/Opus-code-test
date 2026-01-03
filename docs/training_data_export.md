# Training Data Export

Export Graph of Thought (GoT) data in formats suitable for AI training.

## Quick Start

```bash
# Show statistics about available training data
python scripts/training_data_exporter.py stats

# Export all training data to a directory
python scripts/training_data_exporter.py export --output ./training_data/

# Export specific data types
python scripts/training_data_exporter.py export-decisions --output ./decisions.jsonl
python scripts/training_data_exporter.py export-retrospectives --output ./retrospectives.jsonl
python scripts/training_data_exporter.py export-handoffs --output ./handoffs.jsonl
python scripts/training_data_exporter.py export-kts --output ./knowledge_transfers.jsonl
python scripts/training_data_exporter.py export-edges --output ./edges.jsonl
```

## Data Types

### 1. Decision Rationales (High Value)
**Format:**
```json
{
  "context": "Decision: WAL-first architecture for ACID guarantees",
  "decision": "WAL-first architecture for ACID guarantees",
  "rationale": "Current WAL logs COMMIT after applying writes...",
  "affects": [],
  "outcome": "documented",
  "quality_score": 0.98,
  "metadata": {
    "decision_id": "D-20260102-231952-62e237a5",
    "created_at": "2026-01-02T23:19:52.521353+00:00"
  }
}
```

**Training Use:** Decision-making, architectural reasoning, tradeoff analysis

### 2. Task Retrospectives (High Value)
**Format:**
```json
{
  "task": "Implement fingerprinting benchmarks",
  "description": "Benchmarks: fingerprint_generation...",
  "approach": "feature",
  "retrospective": "Implemented 3 FINGERPRINT benchmarks...",
  "success": true,
  "priority": "medium",
  "quality_score": 1.0,
  "metadata": {
    "task_id": "T-20251229-101620-e8cb4859",
    "created_at": "2025-12-29T10:16:20.571879+00:00",
    "completed_at": "2025-12-29T13:26:23.146214+00:00"
  }
}
```

**Training Use:** Learning from successes/failures, task decomposition, reflection

### 3. Handoff Instructions (Procedural Knowledge)
**Format:**
```json
{
  "task": "Remove deprecated EventLog classes",
  "handoff_to": "next-session",
  "instructions": "## CLI Migration Complete...",
  "result": {},
  "success": false,
  "quality_score": 0.94,
  "metadata": {
    "handoff_id": "H-20251224-070920-ab3027eb",
    "task_id": "T-20251222-145525-445df343",
    "created_at": "2025-12-24T07:09:20.910173+00:00"
  }
}
```

**Training Use:** Delegation, context transfer, work continuity

### 4. Knowledge Transfers (Synthesis Knowledge)
**Format:**
```json
{
  "topic": "WAL Entity Rollback Implementation",
  "summary": "Implemented WAL entity reconstruction...",
  "key_points": ["Full entity state now stored in WAL..."],
  "related_tasks": [],
  "related_decisions": [],
  "status": "published",
  "quality_score": 0.63,
  "metadata": {
    "kt_id": "KT-20260103-013737",
    "created_at": "2026-01-03T01:37:37.548129"
  }
}
```

**Training Use:** Session synthesis, knowledge consolidation, learning transfer

### 5. Edge Relationships (Graph Knowledge)
**Format:**
```json
{
  "from": "epic:Cognitive NLU/NLG: Meta-Learning Architecture",
  "relationship": "CONTAINS",
  "to": "task:Review and continue LLM Cognitive Architecture",
  "context": "EPIC-cognitive-nlu-nlg CONTAINS T-20251228-083905-2aa7980d",
  "weight": 1.0,
  "confidence": 1.0,
  "metadata": {
    "edge_id": "E-EPIC-cognitive-nlu-nlg-T-20251228-083905-2aa7980d-CONTAINS",
    "source_id": "EPIC-cognitive-nlu-nlg",
    "target_id": "T-20251228-083905-2aa7980d",
    "created_at": "2025-12-28T08:56:42.059070+00:00"
  }
}
```

**Training Use:** Relationship learning, dependency understanding, graph reasoning

## Output Formats

### JSONL (Machine Learning)
- One JSON object per line
- Easy to stream and process
- Compatible with most ML frameworks
- Default format for all exports

### Markdown (Human Review)
- Generated automatically with `export` command
- Includes summary statistics
- Shows quality distribution
- Lists top examples by quality score

## Quality Filtering

The exporter automatically filters training data by quality:

1. **Completeness**: Text length and structure
2. **Information Density**: Unique words vs total words
3. **Readability**: Sentence structure

**Quality Score** (0.0-1.0):
- 0.8+ : Excellent, detailed content
- 0.5-0.8 : Good, useful content
- 0.2-0.5 : Minimal, basic content
- 0.0-0.2 : Filtered out (too sparse)

## Statistics

Current repository contains:
- **56 decisions** with quality rationales (100% quality rate)
- **100 task retrospectives** (30.8% of all tasks)
- **31 successful handoffs** (75.6% success rate)
- **18 published knowledge transfers** (94.7% publish rate)
- **431 edge relationships** (7 types)

**Total training data points: 636**

## Integration with ML Pipelines

### Example: Load Decisions for Training
```python
import json

decisions = []
with open('training_data/decisions.jsonl', 'r') as f:
    for line in f:
        decisions.append(json.loads(line))

# Filter by quality
high_quality = [d for d in decisions if d['quality_score'] >= 0.8]

# Extract training pairs
training_pairs = [
    (d['context'], d['rationale'])
    for d in high_quality
]
```

### Example: Load Retrospectives for Few-Shot Learning
```python
import json

retrospectives = []
with open('training_data/retrospectives.jsonl', 'r') as f:
    for line in f:
        retrospectives.append(json.loads(line))

# Get successful task examples
successful = [r for r in retrospectives if r['success']]

# Create few-shot examples
few_shot_examples = [
    f"Task: {r['task']}\nApproach: {r['approach']}\nRetrospective: {r['retrospective']}"
    for r in successful[:5]
]
```

## Future Enhancements

Potential improvements for the exporter:

1. **Time-based filtering**: Export data from specific date ranges
2. **Category filtering**: Export only specific task categories or decision types
3. **Quality thresholds**: Configurable minimum quality scores
4. **Deduplication**: Remove similar/duplicate examples
5. **Augmentation**: Generate variations of high-quality examples
6. **Embedding generation**: Pre-compute embeddings for similarity search
7. **Cross-validation splits**: Generate train/val/test splits
8. **Format converters**: Export to CSV, Parquet, Arrow, etc.

## See Also

- `/home/user/Opus-code-test/scripts/training_data_exporter.py` - Source code
- `/home/user/Opus-code-test/llm_orchestration/learning.py` - Experience/Pattern/Lesson classes
- `/home/user/Opus-code-test/scripts/ml_file_prediction.py` - ML training example format
