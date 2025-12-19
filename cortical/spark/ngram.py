"""
N-gram Language Model
=====================

Pure Python implementation of n-gram language models for word prediction.
No dependencies beyond Python stdlib.

Theory:
    P(word | context) = count(context + word) / count(context)

    With Laplace smoothing:
    P(word | context) = (count + 1) / (total + vocab_size)

Example:
    >>> model = NGramModel(n=3)  # Trigram
    >>> model.train(["the neural network processes data"])
    >>> model.predict(["neural", "network"], top_k=3)
    [("processes", 0.5), ("data", 0.25), ...]
"""

from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Iterable
import re
import math


class NGramModel:
    """
    Statistical n-gram language model.

    Predicts next word(s) based on previous n-1 words using
    simple counting with Laplace smoothing.
    """

    def __init__(self, n: int = 3, smoothing: float = 1.0):
        """
        Initialize n-gram model.

        Args:
            n: Order of the model (2=bigram, 3=trigram, etc.)
            smoothing: Laplace smoothing factor (default 1.0)
        """
        if n < 2:
            raise ValueError("n must be at least 2 for meaningful prediction")

        self.n = n
        self.smoothing = smoothing

        # Core data structures
        self.counts: Dict[tuple, Counter] = defaultdict(Counter)  # context -> {word: count}
        self.context_totals: Dict[tuple, int] = defaultdict(int)  # context -> total count
        self.vocab: set = set()

        # Special tokens
        self.START = '<s>'
        self.END = '</s>'
        self.UNK = '<unk>'

        # Training stats
        self.total_tokens = 0
        self.total_documents = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        # Lowercase and split on whitespace/punctuation
        text = text.lower()
        # Keep alphanumeric and common code characters
        tokens = re.findall(r'[a-z0-9_]+', text)
        return tokens

    def _get_ngrams(self, tokens: List[str]) -> Iterable[Tuple[tuple, str]]:
        """Generate (context, word) pairs from token list."""
        # Pad start with n-1 start tokens
        padded = [self.START] * (self.n - 1) + tokens + [self.END]

        for i in range(len(padded) - self.n + 1):
            context = tuple(padded[i:i + self.n - 1])
            word = padded[i + self.n - 1]
            yield context, word

    def train(self, documents: Iterable[str]) -> 'NGramModel':
        """
        Train model on documents.

        Args:
            documents: Iterable of text strings

        Returns:
            self for method chaining
        """
        for doc in documents:
            tokens = self._tokenize(doc)
            if not tokens:
                continue

            self.total_documents += 1
            self.total_tokens += len(tokens)

            # Add tokens to vocabulary
            self.vocab.update(tokens)

            # Count n-grams
            for context, word in self._get_ngrams(tokens):
                self.counts[context][word] += 1
                self.context_totals[context] += 1

        return self

    def train_on_tokens(self, token_lists: Iterable[List[str]]) -> 'NGramModel':
        """
        Train on pre-tokenized documents.

        Args:
            token_lists: Iterable of token lists

        Returns:
            self for method chaining
        """
        for tokens in token_lists:
            if not tokens:
                continue

            self.total_documents += 1
            self.total_tokens += len(tokens)
            self.vocab.update(tokens)

            for context, word in self._get_ngrams(tokens):
                self.counts[context][word] += 1
                self.context_totals[context] += 1

        return self

    def probability(self, word: str, context: List[str]) -> float:
        """
        Calculate P(word | context) with Laplace smoothing.

        Args:
            word: Target word
            context: List of previous n-1 words

        Returns:
            Probability estimate
        """
        # Ensure context is the right length
        if len(context) < self.n - 1:
            context = [self.START] * (self.n - 1 - len(context)) + context
        elif len(context) > self.n - 1:
            context = context[-(self.n - 1):]

        context_tuple = tuple(w.lower() for w in context)
        word_lower = word.lower()

        count = self.counts[context_tuple][word_lower]
        total = self.context_totals[context_tuple]
        vocab_size = len(self.vocab) + 1  # +1 for UNK

        # Laplace smoothing
        prob = (count + self.smoothing) / (total + self.smoothing * vocab_size)
        return prob

    def predict(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict most likely next words given context.

        Args:
            context: List of previous words (uses last n-1)
            top_k: Number of predictions to return

        Returns:
            List of (word, probability) tuples, sorted by probability
        """
        # Normalize context
        if len(context) < self.n - 1:
            context = [self.START] * (self.n - 1 - len(context)) + context
        elif len(context) > self.n - 1:
            context = context[-(self.n - 1):]

        context_tuple = tuple(w.lower() for w in context)

        # Get word counts for this context
        word_counts = self.counts[context_tuple]

        if not word_counts:
            # Unknown context - return most frequent words overall
            return self._most_frequent_words(top_k)

        # Calculate probabilities for all seen words
        total = self.context_totals[context_tuple]
        vocab_size = len(self.vocab) + 1

        predictions = []
        for word, count in word_counts.items():
            if word in (self.START, self.END):
                continue
            prob = (count + self.smoothing) / (total + self.smoothing * vocab_size)
            predictions.append((word, prob))

        # Sort by probability descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    def predict_sequence(self, context: List[str], length: int = 3) -> List[str]:
        """
        Predict a sequence of words (greedy decoding).

        Args:
            context: Starting context
            length: Number of words to predict

        Returns:
            List of predicted words
        """
        result = []
        current_context = list(context)

        for _ in range(length):
            predictions = self.predict(current_context, top_k=1)
            if not predictions or predictions[0][0] == self.END:
                break

            word = predictions[0][0]
            result.append(word)
            current_context.append(word)

        return result

    def _most_frequent_words(self, top_k: int) -> List[Tuple[str, float]]:
        """Get most frequent words across all contexts."""
        total_counts: Counter = Counter()
        for context_counts in self.counts.values():
            for word, count in context_counts.items():
                if word not in (self.START, self.END):
                    total_counts[word] += count

        total = sum(total_counts.values())
        if total == 0:
            return []

        return [(word, count / total) for word, count in total_counts.most_common(top_k)]

    def perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text under this model.

        Lower perplexity = better fit to model.

        Args:
            text: Text to evaluate

        Returns:
            Perplexity score
        """
        tokens = self._tokenize(text)
        if not tokens:
            return float('inf')

        log_prob_sum = 0.0
        count = 0

        for context, word in self._get_ngrams(tokens):
            prob = self.probability(word, list(context))
            if prob > 0:
                log_prob_sum += math.log(prob)
                count += 1

        if count == 0:
            return float('inf')

        # Perplexity = exp(-1/N * sum(log(P)))
        return math.exp(-log_prob_sum / count)

    def save(self, path: str) -> None:
        """Save model to JSON file."""
        import json

        data = {
            'n': self.n,
            'smoothing': self.smoothing,
            'vocab': list(self.vocab),
            'counts': {
                ' '.join(k): dict(v)
                for k, v in self.counts.items()
            },
            'context_totals': {
                ' '.join(k): v
                for k, v in self.context_totals.items()
            },
            'total_tokens': self.total_tokens,
            'total_documents': self.total_documents,
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'NGramModel':
        """Load model from JSON file."""
        import json

        with open(path, 'r') as f:
            data = json.load(f)

        model = cls(n=data['n'], smoothing=data['smoothing'])
        model.vocab = set(data['vocab'])
        model.total_tokens = data['total_tokens']
        model.total_documents = data['total_documents']

        for key, counts in data['counts'].items():
            context = tuple(key.split())
            model.counts[context] = Counter(counts)

        for key, total in data['context_totals'].items():
            context = tuple(key.split())
            model.context_totals[context] = total

        return model

    def __repr__(self) -> str:
        return (
            f"NGramModel(n={self.n}, "
            f"vocab_size={len(self.vocab)}, "
            f"contexts={len(self.counts)}, "
            f"documents={self.total_documents})"
        )
