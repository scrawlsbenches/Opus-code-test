"""
Simple tokenization utilities for text-based experiments.

Provides word-level tokenization for language modeling tasks.
For production use, consider subword tokenization (BPE, SentencePiece).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize text into words.

    Simple word-level tokenization that:
    - Splits on whitespace and punctuation
    - Optionally lowercases
    - Preserves punctuation as separate tokens

    Args:
        text: Input text to tokenize
        lowercase: Whether to lowercase tokens

    Returns:
        List of tokens
    """
    if lowercase:
        text = text.lower()

    # Split on whitespace, keeping punctuation as separate tokens
    # This regex splits on whitespace and separates punctuation
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text)

    return tokens


def build_vocab(
    tokens: List[str],
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build vocabulary from tokens.

    Args:
        tokens: List of tokens
        min_freq: Minimum frequency for a token to be included
        max_vocab_size: Maximum vocabulary size (excluding special tokens)

    Returns:
        Tuple of (token_to_id, id_to_token) dictionaries
    """
    # Count frequencies
    freq: Dict[str, int] = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1

    # Filter by frequency
    filtered = [(t, f) for t, f in freq.items() if f >= min_freq]

    # Sort by frequency (descending)
    filtered.sort(key=lambda x: (-x[1], x[0]))

    # Limit size
    if max_vocab_size is not None:
        filtered = filtered[:max_vocab_size]

    # Build mappings with special tokens first
    token_to_id: Dict[str, int] = {}
    id_to_token: Dict[int, str] = {}

    for i, special in enumerate(SPECIAL_TOKENS):
        token_to_id[special] = i
        id_to_token[i] = special

    offset = len(SPECIAL_TOKENS)
    for i, (token, _) in enumerate(filtered):
        if token not in token_to_id:  # Skip if already a special token
            token_to_id[token] = i + offset
            id_to_token[i + offset] = token

    return token_to_id, id_to_token


def tokens_to_ids(
    tokens: List[str],
    vocab: Dict[str, int],
    add_bos: bool = False,
    add_eos: bool = False,
) -> List[int]:
    """
    Convert tokens to integer IDs.

    Args:
        tokens: List of tokens
        vocab: Token to ID mapping
        add_bos: Whether to add BOS token at start
        add_eos: Whether to add EOS token at end

    Returns:
        List of integer IDs
    """
    unk_id = vocab.get(UNK_TOKEN, 1)

    ids = []

    if add_bos:
        ids.append(vocab[BOS_TOKEN])

    for token in tokens:
        ids.append(vocab.get(token, unk_id))

    if add_eos:
        ids.append(vocab[EOS_TOKEN])

    return ids


def ids_to_tokens(
    ids: List[int],
    id_to_token: Dict[int, str],
    skip_special: bool = True,
) -> List[str]:
    """
    Convert integer IDs back to tokens.

    Args:
        ids: List of integer IDs
        id_to_token: ID to token mapping
        skip_special: Whether to skip special tokens

    Returns:
        List of tokens
    """
    tokens = []
    special_set = set(SPECIAL_TOKENS)

    for id_ in ids:
        token = id_to_token.get(id_, UNK_TOKEN)
        if skip_special and token in special_set:
            continue
        tokens.append(token)

    return tokens


def detokenize(tokens: List[str]) -> str:
    """
    Convert tokens back to text.

    Simple detokenization that joins with spaces and cleans up punctuation.

    Args:
        tokens: List of tokens

    Returns:
        Reconstructed text
    """
    if not tokens:
        return ""

    # Join with spaces
    text = " ".join(tokens)

    # Clean up spacing around punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)

    return text
