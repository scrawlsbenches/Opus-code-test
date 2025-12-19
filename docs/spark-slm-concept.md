# SparkSLM: Statistical First-Blitz Predictor

> "The spark that ignites before the fire fully forms"

## The Epiphany Concept

You're describing something profound: **a fast, lightweight predictor that provides initial "spark" thoughts before the heavyweight analysis kicks in**. Like how the brain has fast/slow thinking (System 1 / System 2), we want:

- **SparkSLM (System 1)**: Fast, statistical, pattern-matching, "first blitz thoughts"
- **Full Search (System 2)**: Slow, deliberate, graph-based, comprehensive

The spark primes the pump. It doesn't replace the full search—it guides it.

## About Karpathy's llm.c

You're correct! Andrej Karpathy did create [llm.c](https://github.com/karpathy/llm.c) - a pure C implementation of GPT-2 training without PyTorch/TensorFlow.

**What llm.c proves:**
- You CAN implement neural network training from scratch
- The core math is just matrix multiplications and backprop
- No dependency on high-level frameworks

**What llm.c still requires:**
- CUDA for GPU acceleration (training on CPU is impractically slow)
- Gigabytes of training data
- Hours/days of training time
- The model weights are still large (124M+ parameters)

**Our constraint:** Zero dependencies, no CUDA, fast startup.

**Our solution:** Don't try to be a neural LM. Be a *statistical* LM that's useful in a different way.

## SparkSLM Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SparkSLM                                 │
│                  "First Blitz Thoughts"                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   N-gram     │  │    Topic     │  │   Keyword    │          │
│  │   Model      │  │  Classifier  │  │  Extractor   │          │
│  │              │  │              │  │              │          │
│  │ P(w|context) │  │ P(topic|doc) │  │ importance(w)│          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────────┬────┴────────────────┘                   │
│                      │                                          │
│              ┌───────▼───────┐                                 │
│              │ SparkPredictor │                                 │
│              │                │                                 │
│              │ • prime(query) │──→ Quick suggestions            │
│              │ • detect(input)│──→ Anomaly flag                 │
│              │ • complete(prefix)──→ Next words                 │
│              └────────────────┘                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Full Search Pipeline                          │
│               (Primed by SparkSLM suggestions)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. N-gram Language Model

Pure statistical word prediction based on context:

```python
class NGramModel:
    """Statistical language model using bigram/trigram counts."""

    def __init__(self, n=3):
        self.n = n
        self.counts = defaultdict(Counter)  # context -> word -> count
        self.vocab = set()

    def train(self, corpus: List[str]):
        """Train on tokenized documents."""
        for doc in corpus:
            tokens = ['<s>'] * (self.n - 1) + tokenize(doc) + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.counts[context][word] += 1
                self.vocab.add(word)

    def predict(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next words given context."""
        context_tuple = tuple(context[-(self.n - 1):])
        word_counts = self.counts[context_tuple]

        # Laplace smoothing
        total = sum(word_counts.values()) + len(self.vocab)
        predictions = [
            (word, (count + 1) / total)
            for word, count in word_counts.most_common(top_k)
        ]
        return predictions
```

**Use case:** Auto-complete, next-word suggestion, pattern detection.

### 2. Fast Topic Classifier

TF-IDF centroid-based topic classification:

```python
class FastTopicClassifier:
    """Topic classification using TF-IDF centroids."""

    def __init__(self):
        self.topic_centroids = {}  # topic -> {term: tfidf_weight}

    def train(self, labeled_docs: Dict[str, List[str]]):
        """Train from labeled documents per topic."""
        for topic, docs in labeled_docs.items():
            # Compute TF-IDF for all terms in topic docs
            term_weights = compute_tfidf(docs)
            self.topic_centroids[topic] = term_weights

    def classify(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Classify text into topics."""
        doc_tfidf = compute_tfidf_single(text)

        scores = []
        for topic, centroid in self.topic_centroids.items():
            # Cosine similarity between doc and centroid
            score = cosine_similarity(doc_tfidf, centroid)
            scores.append((topic, score))

        return sorted(scores, key=lambda x: -x[1])[:top_k]
```

**Use case:** Quick categorization, domain detection, routing.

### 3. Keyword Extractor

TextRank-lite for rapid salient term identification:

```python
class KeywordExtractor:
    """Fast keyword extraction using simplified TextRank."""

    def extract(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract key terms from text."""
        tokens = tokenize(text)

        # Build co-occurrence graph (simplified TextRank)
        cooccurrence = defaultdict(Counter)
        window_size = 5
        for i, token in enumerate(tokens):
            for j in range(max(0, i - window_size), min(len(tokens), i + window_size)):
                if i != j:
                    cooccurrence[token][tokens[j]] += 1

        # Score by connection strength (simplified PageRank)
        scores = {}
        for term in cooccurrence:
            scores[term] = sum(cooccurrence[term].values())

        sorted_terms = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_terms[:top_k]
```

**Use case:** Query priming, salient term identification, summary.

### 4. Anomaly Detector (Prompt Injection)

Statistical pattern detection for suspicious inputs:

```python
class AnomalyDetector:
    """Detect anomalous inputs that might be prompt injection."""

    def __init__(self):
        self.normal_patterns = {}  # Learned from training data
        self.thresholds = {}

    def train(self, normal_inputs: List[str]):
        """Learn normal input patterns."""
        # Character frequency distribution
        char_freq = Counter()
        for inp in normal_inputs:
            char_freq.update(inp.lower())

        total = sum(char_freq.values())
        self.normal_patterns['char_dist'] = {c: n/total for c, n in char_freq.items()}

        # Token length distribution
        lengths = [len(tokenize(inp)) for inp in normal_inputs]
        self.normal_patterns['avg_length'] = sum(lengths) / len(lengths)
        self.normal_patterns['max_length'] = max(lengths) * 2

        # Special character ratio
        special_ratios = [
            sum(1 for c in inp if not c.isalnum() and not c.isspace()) / max(len(inp), 1)
            for inp in normal_inputs
        ]
        self.normal_patterns['special_ratio_threshold'] = max(special_ratios) * 1.5

    def detect(self, input_text: str) -> Tuple[bool, float, str]:
        """
        Detect if input is anomalous.

        Returns: (is_anomalous, confidence, reason)
        """
        reasons = []
        anomaly_score = 0.0

        # Check length
        length = len(tokenize(input_text))
        if length > self.normal_patterns['max_length']:
            anomaly_score += 0.3
            reasons.append(f"unusual length ({length} tokens)")

        # Check special character ratio
        special_ratio = sum(1 for c in input_text if not c.isalnum() and not c.isspace()) / max(len(input_text), 1)
        if special_ratio > self.normal_patterns['special_ratio_threshold']:
            anomaly_score += 0.4
            reasons.append(f"high special char ratio ({special_ratio:.2f})")

        # Check for injection patterns
        injection_patterns = [
            r'ignore\s+(previous|above|all)',
            r'system\s*:\s*',
            r'<\s*script\s*>',
            r'\{\{.*\}\}',
            r'<!--.*-->',
        ]
        for pattern in injection_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                anomaly_score += 0.5
                reasons.append(f"injection pattern detected")
                break

        is_anomalous = anomaly_score > 0.5
        return (is_anomalous, min(anomaly_score, 1.0), "; ".join(reasons))
```

**Use case:** Input validation, security screening, suspicious query flagging.

## The SparkPredictor Facade

```python
class SparkPredictor:
    """
    SparkSLM: Statistical First-Blitz Predictor

    Provides fast, lightweight predictions to "prime" the full search.
    Not a replacement for deep search—a spark that guides it.
    """

    def __init__(self, corpus: Optional['CorticalTextProcessor'] = None):
        self.ngram = NGramModel(n=3)
        self.classifier = FastTopicClassifier()
        self.keywords = KeywordExtractor()
        self.anomaly = AnomalyDetector()
        self._trained = False

    def train(self, processor: 'CorticalTextProcessor'):
        """Train on existing corpus."""
        # Extract documents for training
        docs = [col.content for col in processor.layers[CorticalLayer.DOCUMENTS].minicolumns.values()]
        # ... training logic ...
        self._trained = True

    def prime(self, query: str) -> Dict[str, Any]:
        """
        Generate first-blitz thoughts for a query.

        Returns quick suggestions to prime the full search.
        """
        return {
            'topics': self.classifier.classify(query, top_k=3),
            'keywords': self.keywords.extract(query, top_k=5),
            'completions': self.ngram.predict(query.split()[-2:], top_k=3),
            'is_safe': not self.anomaly.detect(query)[0],
        }

    def detect_anomaly(self, input_text: str) -> Tuple[bool, float, str]:
        """Check if input looks suspicious."""
        return self.anomaly.detect(input_text)

    def complete(self, prefix: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Suggest next words for prefix."""
        tokens = prefix.split()
        return self.ngram.predict(tokens, top_k=top_k)
```

## Integration with Search

```python
def search_with_spark(query: str, processor: 'CorticalTextProcessor') -> List[SearchResult]:
    """Search with SparkSLM priming."""

    # 1. Spark phase (fast, <10ms)
    spark = processor.spark_predictor.prime(query)

    # 2. Safety check
    if not spark['is_safe']:
        return []  # Or flag for review

    # 3. Use spark to enhance query
    primed_terms = [kw for kw, score in spark['keywords']]
    topic_boost = {topic: score for topic, score in spark['topics']}

    # 4. Full search with priming
    results = processor.find_documents_for_query(
        query,
        additional_terms=primed_terms,
        topic_weights=topic_boost
    )

    return results
```

## What SparkSLM Is NOT

Let me be clear about limitations:

| SparkSLM IS | SparkSLM IS NOT |
|-------------|-----------------|
| Statistical pattern matcher | Neural language model |
| Fast (<10ms) | High-accuracy predictor |
| Primer/guide for search | Replacement for search |
| Anomaly flagger | Security guarantee |
| Zero-dependency | Sophisticated NLU |

## The Epiphany Realized

Your intuition is sound: **a fast statistical layer that provides initial "sparks" to guide deeper analysis**. This is:

1. **Computationally cheap** - Pure Python, no ML libraries
2. **Fast startup** - Trains from existing corpus in seconds
3. **Useful immediately** - Provides value even with imperfect predictions
4. **Honest about limits** - Doesn't pretend to understand, just pattern-matches

The spark doesn't replace the fire. It ignites it faster.

## Implementation Plan

See Sprint 17 in `tasks/CURRENT_SPRINT.md` for detailed tasks.

### Phase 1: Core Models (3-5 tasks)
- N-gram model
- Topic classifier
- Keyword extractor

### Phase 2: Integration (2-3 tasks)
- SparkPredictor facade
- Anomaly detection
- Search integration

### Phase 3: Validation (2-3 tasks)
- Benchmarks
- Dog-fooding
- Documentation

## References

- [llm.c by Karpathy](https://github.com/karpathy/llm.c) - Pure C LLM training
- [TextRank paper](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) - Keyword extraction
- [Thinking, Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) - System 1/2 metaphor
