# Vocabulary Management Design

> **STATUS: DESIGN IN PROGRESS**
>
> Do not implement yet. Edge cases and design decisions are still under discussion.
> This document is a working draft for collaborative design review.

## Problem Statement

The current experiment CLI builds vocabulary per-document, creating a rigid coupling between trained model weights and the specific tokens seen during training. This prevents:
- Fine-tuning on new documents
- Iterative training across multiple documents
- Graceful handling of vocabulary changes over time

## Design Goals

1. **Extendable vocabulary** - Start with corpus, grow as needed
2. **Strict by default** - Error on OOV, don't silently degrade
3. **Transparent `<UNK>` handling** - If used, track when and how
4. **Model-vocabulary coupling** - Model knows its vocab, detects changes
5. **Future-proof** - Support code patterns and unseen tokens eventually

## Architecture

### Vocabulary File Format

```json
{
  "version": 1,
  "created_at": "2026-01-15T12:00:00Z",
  "source_files": [
    "samples/unix_evolution.txt",
    "samples/other_doc.txt"
  ],
  "config": {
    "min_freq": 1,
    "lowercase": true,
    "embedding_scale": 0.35
  },
  "tokens": {
    "token_to_id": {
      "unix": 0,
      "bell": 1,
      "labs": 2
    },
    "id_to_token": ["unix", "bell", "labs"]
  },
  "special_tokens": {
    "<PAD>": -1,
    "<UNK>": -2
  },
  "extensions": [
    {
      "added_at": "2026-01-16T10:00:00Z",
      "tokens": ["linux", "kernel"],
      "reason": "New document: samples/linux_history.txt"
    }
  ],
  "unk_usage_log": []
}
```

### Checkpoint-Vocabulary Relationship

```
checkpoint.pkl
├── parameters: {...}
├── optimizer_state: {...}
├── scheduler_state: {...}
├── vocab_reference:
│   ├── path: "experiments/vocab/corpus_v1.json"
│   ├── hash: "sha256:abc123..."
│   └── token_count: 76
└── epoch: 500
```

On resume:
1. Load checkpoint
2. Verify vocab file exists at referenced path
3. Verify hash matches (detect modifications)
4. If mismatch: ERROR with clear explanation

### OOV Handling Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Token Encountered                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ In vocabulary?  │
                    └─────────────────┘
                       │           │
                      YES          NO
                       │           │
                       ▼           ▼
                 ┌─────────┐  ┌──────────────────┐
                 │ Use ID  │  │ --extend-vocab?  │
                 └─────────┘  └──────────────────┘
                                  │           │
                                 YES          NO
                                  │           │
                                  ▼           ▼
                           ┌───────────┐  ┌─────────────────┐
                           │ Add token │  │ --allow-unk?    │
                           │ + random  │  └─────────────────┘
                           │ embedding │      │           │
                           └───────────┘     YES          NO
                                              │           │
                                              ▼           ▼
                                        ┌──────────┐  ┌───────┐
                                        │ Map to   │  │ ERROR │
                                        │ <UNK> +  │  │ List  │
                                        │ LOG it   │  │ OOV   │
                                        └──────────┘  └───────┘
```

Default behavior: **ERROR** with list of OOV tokens
User must explicitly opt-in to `--extend-vocab` or `--allow-unk`

### `<UNK>` Transparency

When `--allow-unk` is used:
- Log every `<UNK>` substitution to `unk_usage_log` in vocab file
- Print summary after training: "X tokens mapped to <UNK>"
- Store in checkpoint for auditability

```json
"unk_usage_log": [
  {
    "timestamp": "2026-01-16T10:30:00Z",
    "document": "samples/new_doc.txt",
    "tokens": ["kubernetes", "containerization"],
    "count": 15
  }
]
```

## CLI Interface

### Creating Vocabulary

```bash
# From single file
python -m cortical.experiments.cli vocab create \
    --from samples/unix_evolution.txt \
    --output experiments/vocab/unix.json

# From directory (all .txt files)
python -m cortical.experiments.cli vocab create \
    --from samples/ \
    --output experiments/vocab/corpus.json \
    --min-freq 1

# Inspect vocabulary
python -m cortical.experiments.cli vocab inspect \
    experiments/vocab/corpus.json

# Extend existing vocabulary
python -m cortical.experiments.cli vocab extend \
    --vocab experiments/vocab/corpus.json \
    --from samples/new_doc.txt
```

### Training with Vocabulary

```bash
# Initial training
python -m cortical.experiments.cli run \
    --name my-experiment \
    --input samples/doc1.txt \
    --vocab experiments/vocab/corpus.json \
    --epochs 500

# Resume on same document (vocab unchanged)
python -m cortical.experiments.cli run \
    --resume checkpoint.pkl \
    --epochs 1000

# Train on different document (same vocab)
python -m cortical.experiments.cli run \
    --resume checkpoint.pkl \
    --input samples/doc2.txt \
    --epochs 500

# New document with OOV - will ERROR by default
python -m cortical.experiments.cli run \
    --resume checkpoint.pkl \
    --input samples/new_doc.txt
# ERROR: 3 out-of-vocabulary tokens found: ['linux', 'kernel', 'gnu']
# Use --extend-vocab to add them, or --allow-unk to map to <UNK>

# Extend vocab and continue
python -m cortical.experiments.cli run \
    --resume checkpoint.pkl \
    --input samples/new_doc.txt \
    --extend-vocab
# WARNING: Extended vocabulary with 3 new tokens (random embeddings)
# Vocabulary saved to: experiments/vocab/corpus.json
```

## Iterative Training Workflow

### Scenario: New Document Added

```bash
# 1. Check what's new
python -m cortical.experiments.cli vocab diff \
    --vocab experiments/vocab/corpus.json \
    --document samples/new_doc.txt
# New tokens: ['linux', 'kernel', 'gnu']
# Missing from document: ['pdp', 'minicomputer']

# 2. Decision point:
#    a) Extend vocab (recommended if new terms are meaningful)
#    b) Use <UNK> (if terms are noise/typos)
#    c) Edit document to use existing vocab

# 3. Extend and train
python -m cortical.experiments.cli vocab extend \
    --vocab experiments/vocab/corpus.json \
    --from samples/new_doc.txt

python -m cortical.experiments.cli run \
    --resume checkpoint.pkl \
    --input samples/new_doc.txt
```

### Scenario: Document Modified

```bash
# Check impact
python -m cortical.experiments.cli vocab diff \
    --vocab experiments/vocab/corpus.json \
    --document samples/modified_doc.txt

# If new tokens: extend vocab first
# If only removed tokens: no action needed (embeddings stay, unused)
```

### Scenario: Vocab Becomes Stale

```bash
# Audit vocabulary usage across corpus
python -m cortical.experiments.cli vocab audit \
    --vocab experiments/vocab/corpus.json \
    --corpus samples/

# Output:
# Tokens in vocab: 156
# Tokens used in corpus: 142
# Unused tokens: 14 (wasting 14 embedding slots)
# Tokens in corpus but not vocab: 0

# Optional: rebuild vocab (requires retraining from scratch)
python -m cortical.experiments.cli vocab rebuild \
    --from samples/ \
    --output experiments/vocab/corpus_v2.json
```

## Implementation Plan

### Phase 1: Core Vocabulary Management
- [ ] `Vocabulary` class with save/load/extend methods
- [ ] `vocab create` command
- [ ] `vocab inspect` command
- [ ] Modify `run` to accept `--vocab` parameter
- [ ] Store vocab reference in checkpoint

### Phase 2: OOV Handling
- [ ] OOV detection before training starts
- [ ] `--extend-vocab` flag with embedding expansion
- [ ] `--allow-unk` flag with logging
- [ ] `vocab diff` command

### Phase 3: Iterative Training
- [ ] `vocab extend` command
- [ ] `vocab audit` command
- [ ] Checkpoint-vocab hash verification
- [ ] Clear error messages for vocab mismatches

### Phase 4: Future Enhancements
- [ ] BPE integration (optional, for code/mixed content)
- [ ] Vocabulary compression (merge rare tokens)
- [ ] Pre-trained embeddings import

## Open Questions

1. **Embedding initialization for new tokens**: Random with same scale, or something smarter?
   - Could average embeddings of similar existing tokens
   - Could use character-level features
   - Start simple: random with `EMBEDDING_INIT_SCALE`

2. **Vocabulary versioning**: Extend in-place or create new file?
   - Proposal: Extend in-place but track history in `extensions` array
   - Hash changes, so checkpoints know vocab was modified

3. **Maximum vocab size**: Should there be a limit?
   - Larger vocab = larger embedding matrix = more memory
   - Could warn if vocab exceeds threshold (e.g., 10,000 tokens)

4. **Case sensitivity**: Currently lowercase by default
   - Code will need case sensitivity
   - Make configurable in vocab config

## Hybrid Tokenization Strategy

### Existing Infrastructure

| Component | Location | What It Does |
|-----------|----------|--------------|
| Word tokenizer | `experiments/tokenizer.py` | Split text, build vocab, UNK support |
| BPE tokenizer | `cognitive/text_bridge.py` | Word-pair merging, underscore_style splitting |
| Sharded storage | `cognitive/tokenizer_storage.py` | Conflict-free vocab storage |

### Design: Word-First with Subword Fallback

```
┌─────────────────────────────────────────────────────────────────┐
│                     Input Token: "get_user_data"                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 1: Direct Lookup                                           │
│   Is "get_user_data" in word_vocab?                             │
│   YES → Return ID                                               │
│   NO  → Continue to Tier 2                                      │
└─────────────────────────────────────────────────────────────────┘
                                │ NO
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 2: Identifier Splitting                                    │
│   Split by underscore_style: ["get", "user", "data"]            │
│   All parts in vocab? YES → Return [ID, ID, ID]                 │
│   Some missing? → Continue to Tier 3                            │
└─────────────────────────────────────────────────────────────────┘
                                │ SOME MISSING
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 3: Fallback Strategy (configurable)                        │
│   --strict (default): ERROR with list of missing tokens         │
│   --extend-vocab: Add missing + random embeddings               │
│   --allow-unk: Map to <UNK> + log for transparency              │
└─────────────────────────────────────────────────────────────────┘
```

### Vocabulary Structure

```json
{
  "version": 2,
  "tokenizer_type": "hybrid_word_subword",
  "config": {
    "min_freq": 2,
    "lowercase": true,
    "split_identifiers": true,
    "embedding_scale": 0.35
  },
  "word_vocab": {
    "token_to_id": {
      "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
      "the": 4, "unix": 5, "system": 6, "get": 7, "user": 8, "data": 9
    },
    "id_to_token": ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "the", "unix", ...]
  },
  "compound_tokens": {
    "get_user_data": [7, 8, 9],
    "unix_system": [5, 6]
  },
  "statistics": {
    "total_tokens": 25000,
    "unique_words": 11202,
    "after_min_freq_filter": 6340,
    "hapax_filtered": 4862
  }
}
```

### Token Count Reduction Strategy

Based on repository analysis:
- **Raw vocabulary**: 137,680 tokens
- **54% are hapax** (appear once): UUIDs, hashes, rare identifiers

| Strategy | Vocab Size | Coverage | Notes |
|----------|-----------|----------|-------|
| All tokens | 137,680 | 100% | Wasteful, most unused |
| min_freq ≥ 2 | ~63,000 | ~99% | Removes hapax |
| min_freq ≥ 5 | ~36,000 | ~97% | Good balance |
| min_freq ≥ 10 | ~25,000 | ~95% | Recommended |
| + identifier splitting | ~20,000 | ~95% | Reuses subwords |

**Recommendation**: `min_freq=10` + identifier splitting → ~20K vocab

### Identifier Splitting Rules

Reuse logic from `cognitive/text_bridge.py:_normalize()`:

```python
def split_identifier(token: str) -> List[str]:
    """
    Split identifier into component words.

    Examples:
        "getUserData"     → ["get", "user", "data"]
        "get_user_data"   → ["get", "user", "data"]
        "XMLParser"       → ["xml", "parser"]
        "parseHTTPResponse" → ["parse", "http", "response"]
        "word2vec"        → ["word", "2", "vec"]
    """
    # Handle underscore_style
    if '_' in token:
        parts = token.split('_')
        return [p.lower() for p in parts if p]

    # Handle CamelCase/PascalCase
    # Split on transitions: lowercase→uppercase, letter→digit
    parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+', token)
    return [p.lower() for p in parts if p]
```

### Incremental Training Flow

```
┌──────────────────────────────────────────────────────────────┐
│ DAY 1: Initial Training                                      │
├──────────────────────────────────────────────────────────────┤
│ 1. vocab create --from samples/ --min-freq 10                │
│    → Creates vocab.json with ~20K words                      │
│                                                              │
│ 2. run --vocab vocab.json --input samples/doc1.txt           │
│    → Trains model, saves checkpoint with vocab reference     │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ DAY 2: New Document                                          │
├──────────────────────────────────────────────────────────────┤
│ 1. vocab diff --vocab vocab.json --document new_doc.txt      │
│    → Shows: 5 new tokens, 3 can be split into known words    │
│                                                              │
│ 2. Option A: vocab extend (if new terms important)           │
│    → Adds 2 truly-new tokens, extends embedding matrix       │
│                                                              │
│ 3. run --resume checkpoint.pkl --input new_doc.txt           │
│    → Continues training with extended vocab                  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ DAY N: Periodic Audit                                        │
├──────────────────────────────────────────────────────────────┤
│ 1. vocab audit --vocab vocab.json --corpus samples/          │
│    → Shows: 500 unused tokens, 50 high-freq tokens missing   │
│                                                              │
│ 2. Decision: rebuild vocab or continue extending?            │
│    → Rebuild requires retraining, but cleaner                │
│    → Extend preserves weights, but accumulates cruft         │
└──────────────────────────────────────────────────────────────┘
```

### Compromises Acknowledged

1. **No character-level BPE**: Truly novel tokens (new abbreviations, typos)
   will map to `<UNK>` or require vocab extension. This is acceptable because:
   - Most OOV can be split into known subwords
   - Explicit extension is safer than silent degradation
   - Character-level BPE can be added later if needed

2. **Embedding matrix can only grow**: Once a token is added, removing it
   would invalidate trained weights. Mitigation:
   - Use `min_freq` filter to avoid adding rare tokens
   - Periodic vocab rebuild (requires retraining)

3. **Split tokens vs whole tokens**: "get_user_data" as [get, user, data]
   loses some information vs a single compound token. Mitigation:
   - Keep high-frequency compounds as single tokens
   - Store `compound_tokens` mapping for reconstruction

## Future Consideration: Morphological Variants

### Problem

"word" and "words" are semantically similar but consume two vocab slots.
Repository analysis shows many such variants wasting embedding capacity.

### Options Under Consideration

| Approach | Example | Pros | Cons |
|----------|---------|------|------|
| **Stemming** | "words"→"word" | Smaller vocab | Loses tense/plurality |
| **Lemmatization** | "better"→"good" | More accurate | Requires dictionary |
| **Suffix tokens** | "words"→["word","+s"] | Preserves info, reuses stems | More tokens per word |
| **Shared embeddings** | "word","words"→same ID | Simple | Loses distinction entirely |

### Proposed: Suffix-Aware Tokenization (Tier 2.5)

Insert between identifier splitting and fallback:

```
Token: "containerizations"
    │
TIER 2.5: Suffix stripping
    │
    ├─ Try: "containerization" + "+s"
    │       └─ "containerization" not in vocab
    │
    └─ Recursive: "container" + "+ization" + "+s"
                  └─ All found → [ID, ID, ID] ✓
```

### Common Suffixes (~20 tokens)

```python
SUFFIXES = [
    # Plurality/possession
    ("'s", "+poss"), ("s", "+s"),

    # Verb forms
    ("ing", "+ing"), ("ed", "+ed"), ("er", "+er"), ("est", "+est"),

    # Noun forms
    ("tion", "+tion"), ("ation", "+ation"), ("ment", "+ment"),
    ("ness", "+ness"), ("ity", "+ity"),

    # Adjective/adverb forms
    ("ly", "+ly"), ("ful", "+ful"), ("less", "+less"),
    ("able", "+able"), ("ible", "+ible"),
]
```

### Impact Estimate

- Adds ~20 suffix tokens to vocab
- Could reduce word vocab by 10-20% through stem reuse
- Preserves morphological information via suffix tokens

### Decision Needed

1. Implement suffix stripping or defer to character-level BPE later?
2. If suffix stripping: apply during vocab build or only at tokenization?
3. Should suffixes share embedding space with words or be separate?

## Notes

- BPE exists in `cortical/cognitive/text_bridge.py` (word-level, not character-level)
- Sharded storage in `cortical/cognitive/tokenizer_storage.py` for conflict-free updates
- Current tokenizer is simple word-based (`cortical/experiments/tokenizer.py`)
- `EMBEDDING_INIT_SCALE = 0.35` must be consistent across vocab extensions
