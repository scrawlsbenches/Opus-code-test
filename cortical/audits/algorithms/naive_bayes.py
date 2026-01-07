"""
Multinomial Naive Bayes Classifier for Comment Classification

This implementation uses:
1. Laplace smoothing (add-1) to handle unseen words
2. Log-space computation to avoid underflow
3. Pure Python (no external libraries except typing and math)

Key Formulas:
- log P(class | doc) ∝ log P(class) + Σ log P(word | class)
- P(class) = count(class) / total_documents
- P(word | class) = (count(word in class) + 1) / (total_words_in_class + vocabulary_size)

Laplace Smoothing Explanation:
Without smoothing, if a word never appears in a class, P(word | class) = 0,
which makes the entire product zero. Laplace smoothing adds 1 to every word count
(numerator) and vocab_size to the total (denominator), ensuring no zero probabilities.

Log-Space Computation:
Multiplying many small probabilities (0 < p < 1) causes underflow. Instead, we:
- Compute log probabilities: log(a × b) = log(a) + log(b)
- Convert back using log-sum-exp trick to avoid overflow
"""

from typing import Dict, List
import math


class CommentClassifier:
    def __init__(self):
        """Initialize classifier for comment classification."""
        self._class_counts: Dict[str, int] = {}
        self._word_counts: Dict[str, Dict[str, int]] = {}  # class -> word -> count
        self._class_totals: Dict[str, int] = {}  # class -> total words
        self._vocabulary: set = set()
        self._total_docs: int = 0

    def fit(self, comments: List[List[str]], labels: List[str]) -> None:
        """
        Train the classifier on tokenized comments.

        Args:
            comments: List of tokenized comments, e.g., [["will", "be", "done"], ["todo", "fix"]]
            labels: List of class labels, e.g., ["misleading", "accurate"]
        """
        # Reset state for clean training
        self._class_counts = {}
        self._word_counts = {}
        self._class_totals = {}
        self._vocabulary = set()
        self._total_docs = len(comments)

        # Count classes, words, and build vocabulary
        for comment, label in zip(comments, labels):
            # Count documents per class (for prior probability)
            self._class_counts[label] = self._class_counts.get(label, 0) + 1

            # Initialize word counts for this class if needed
            if label not in self._word_counts:
                self._word_counts[label] = {}
                self._class_totals[label] = 0

            # Count words in this comment
            for word in comment:
                self._vocabulary.add(word)
                self._word_counts[label][word] = self._word_counts[label].get(word, 0) + 1
                self._class_totals[label] += 1

    def predict(self, comment: List[str]) -> str:
        """
        Return the most likely class for the comment.

        Uses log-space computation to avoid underflow.
        """
        log_probs = {}

        for class_label in self._class_counts:
            # Start with log prior probability: log P(class)
            log_prob = math.log(self._class_counts[class_label] / self._total_docs)

            # Add log likelihood for each word: Σ log P(word | class)
            for word in comment:
                # Laplace smoothing: (count + 1) / (total + vocab_size)
                word_count = self._word_counts[class_label].get(word, 0)
                total_words = self._class_totals[class_label]
                vocab_size = len(self._vocabulary)

                # P(word | class) with Laplace smoothing
                # Even if word is unseen (count=0), we get (0+1)/(total+vocab) > 0
                prob = (word_count + 1) / (total_words + vocab_size)
                log_prob += math.log(prob)

            log_probs[class_label] = log_prob

        # Return class with highest log probability
        return max(log_probs, key=log_probs.get)

    def predict_proba(self, comment: List[str]) -> Dict[str, float]:
        """
        Return probability distribution over classes.
        Probabilities are normalized to sum to 1.0.

        Uses log-sum-exp trick to convert log probabilities back to probabilities
        without overflow/underflow.
        """
        log_probs = {}

        for class_label in self._class_counts:
            # Start with log prior probability: log P(class)
            log_prob = math.log(self._class_counts[class_label] / self._total_docs)

            # Add log likelihood for each word: Σ log P(word | class)
            for word in comment:
                # Laplace smoothing: (count + 1) / (total + vocab_size)
                word_count = self._word_counts[class_label].get(word, 0)
                total_words = self._class_totals[class_label]
                vocab_size = len(self._vocabulary)

                # P(word | class) with Laplace smoothing
                prob = (word_count + 1) / (total_words + vocab_size)
                log_prob += math.log(prob)

            log_probs[class_label] = log_prob

        # Convert log probabilities to probabilities using log-sum-exp trick
        # To avoid underflow: exp(log_p - max_log_p)
        # This shifts all values so the max is 0, preventing overflow in exp()
        max_log_prob = max(log_probs.values())
        probs = {}

        for class_label, log_prob in log_probs.items():
            probs[class_label] = math.exp(log_prob - max_log_prob)

        # Normalize to sum to 1.0
        total_prob = sum(probs.values())
        for class_label in probs:
            probs[class_label] /= total_prob

        return probs

    def most_indicative_words(self, class_label: str, top_n: int = 10) -> List[tuple]:
        """
        Return words most indicative of a class.

        Returns list of (word, probability) tuples sorted by probability descending.
        These are the words with highest P(word | class).
        """
        word_probs = []

        total_words = self._class_totals[class_label]
        vocab_size = len(self._vocabulary)

        for word in self._vocabulary:
            word_count = self._word_counts[class_label].get(word, 0)
            # Laplace smoothing
            prob = (word_count + 1) / (total_words + vocab_size)
            word_probs.append((word, prob))

        # Sort by probability descending
        word_probs.sort(key=lambda x: x[1], reverse=True)

        return word_probs[:top_n]
