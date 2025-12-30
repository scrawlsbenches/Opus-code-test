"""
Top Words Dictionary
====================

The 100 most frequently used words across the samples corpus.

This dictionary provides a frequency-ranked reference of common terms
found in the project's sample documents. Useful for:
- NGram model initialization and fallback behavior
- Query expansion weighting (common terms get lower boost)
- Understanding corpus vocabulary distribution
- Language model baseline frequencies

Generated from analysis of 601 sample documents in samples/.
"""

from typing import Dict, List, Tuple


# Top 100 words by frequency in the samples corpus
# Format: word -> occurrence count
TOP_WORDS: Dict[str, int] = {
    'models': 866,
    'data': 853,
    'systems': 813,
    'learning': 802,
    'time': 776,
    'information': 757,
    'patterns': 658,
    'market': 655,
    'analysis': 630,
    'model': 595,
    'across': 589,
    'based': 576,
    'social': 561,
    'different': 558,
    'requires': 536,
    'rather': 527,
    'knowledge': 522,
    'understanding': 505,
    'world': 466,
    'high': 447,
    'cognitive': 443,
    'multiple': 440,
    'like': 432,
    'risk': 428,
    'memory': 421,
    'system': 410,
    'attention': 401,
    'control': 397,
    'without': 396,
    'enables': 395,
    'reasoning': 389,
    'processing': 383,
    'performance': 372,
    'provides': 366,
    'structure': 356,
    'human': 356,
    'provide': 356,
    'design': 356,
    'specific': 356,
    'use': 355,
    'decision': 335,
    'future': 330,
    'often': 329,
    'changes': 325,
    'methods': 324,
    'features': 321,
    'experience': 319,
    'quality': 318,
    'behavior': 311,
    'trading': 311,
    'strategies': 306,
    'create': 304,
    'individual': 301,
    'prediction': 297,
    'work': 293,
    'type': 290,
    'influence': 290,
    'art': 289,
    'planning': 288,
    'predictions': 287,
    'training': 286,
    'theory': 285,
    'uncertainty': 285,
    'religious': 284,
    'temporal': 283,
    'complex': 283,
    'decisions': 283,
    'conditions': 282,
    'relationships': 281,
    'support': 277,
    'process': 276,
    'making': 274,
    'management': 271,
    'approaches': 270,
    'action': 267,
    'development': 267,
    'order': 263,
    'value': 262,
    'principles': 260,
    'within': 260,
    'processes': 259,
    'against': 259,
    'code': 256,
    'people': 252,
    'creates': 252,
    'applications': 251,
    'enable': 251,
    'dynamics': 247,
    'self': 247,
    'others': 245,
    'concepts': 243,
    'effective': 242,
    'price': 239,
    'position': 237,
    'states': 236,
    'networks': 235,
    'questions': 234,
    'language': 232,
    'types': 231,
    'approach': 230,
}

# Total occurrences of top 100 words
TOTAL_TOP_WORD_OCCURRENCES: int = sum(TOP_WORDS.values())

# Vocabulary size (unique words in top 100)
TOP_WORDS_COUNT: int = len(TOP_WORDS)


def get_top_words(n: int = 100) -> List[Tuple[str, int]]:
    """
    Get the top N most frequent words from the corpus.

    Args:
        n: Number of words to return (default 100, max 100)

    Returns:
        List of (word, count) tuples sorted by frequency descending

    Example:
        >>> get_top_words(5)
        [('models', 866), ('data', 853), ('systems', 813), ('learning', 802), ('time', 776)]
    """
    n = min(n, TOP_WORDS_COUNT)
    return list(TOP_WORDS.items())[:n]


def get_word_frequency(word: str) -> int:
    """
    Get the frequency count for a specific word.

    Args:
        word: The word to look up

    Returns:
        Frequency count, or 0 if word not in top 100

    Example:
        >>> get_word_frequency('learning')
        802
        >>> get_word_frequency('unknown')
        0
    """
    return TOP_WORDS.get(word.lower(), 0)


def is_common_word(word: str, threshold: int = 300) -> bool:
    """
    Check if a word is common (above frequency threshold).

    Args:
        word: The word to check
        threshold: Minimum frequency to be considered common (default 300)

    Returns:
        True if word frequency exceeds threshold

    Example:
        >>> is_common_word('models')  # 866 occurrences
        True
        >>> is_common_word('approach')  # 230 occurrences
        False
    """
    return get_word_frequency(word) >= threshold


def get_words_by_frequency_range(
    min_freq: int = 0,
    max_freq: int = 1000
) -> List[str]:
    """
    Get words within a frequency range.

    Args:
        min_freq: Minimum frequency (inclusive)
        max_freq: Maximum frequency (inclusive)

    Returns:
        List of words within the frequency range

    Example:
        >>> get_words_by_frequency_range(800, 900)
        ['models', 'data', 'systems', 'learning']
    """
    return [
        word for word, count in TOP_WORDS.items()
        if min_freq <= count <= max_freq
    ]
