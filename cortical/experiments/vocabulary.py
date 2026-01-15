"""
Vocabulary Management for Experiments
======================================

Provides a Vocabulary class for creating, saving, and loading vocabularies
from text corpora. Supports shared vocabulary across multiple training runs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from .tokenizer import tokenize, build_vocab, SPECIAL_TOKENS, UNK_TOKEN


class Vocabulary:
    """
    Manages token-to-ID mappings for language model training.

    Supports:
    - Creating vocabulary from tokens or files
    - Saving/loading to JSON
    - Extending with new tokens
    - Hash verification for checkpoint compatibility

    TODO: Consider adding token frequency tracking for analysis
    TODO: Consider adding support for loading pre-trained embeddings
    TODO: Consider adding vocabulary merge/diff operations
    """

    def __init__(
        self,
        token_to_id: Dict[str, int],
        id_to_token: Dict[int, str],
        source_files: Optional[List[str]] = None,
        created_at: Optional[str] = None,
        min_freq: int = 1,
    ):
        self._token_to_id = token_to_id
        self._id_to_token = id_to_token
        self._source_files = source_files or []
        self._created_at = created_at or datetime.now().isoformat()
        self._min_freq = min_freq

    @classmethod
    def from_tokens(
        cls,
        tokens: List[str],
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
    ) -> "Vocabulary":
        """
        Create vocabulary from a list of tokens.

        Args:
            tokens: List of tokens
            min_freq: Minimum frequency for inclusion
            max_vocab_size: Maximum vocabulary size (excluding special tokens)

        Returns:
            Vocabulary instance
        """
        token_to_id, id_to_token = build_vocab(
            tokens, min_freq=min_freq, max_vocab_size=max_vocab_size
        )
        return cls(token_to_id, id_to_token, min_freq=min_freq)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
        lowercase: bool = True,
    ) -> "Vocabulary":
        """
        Create vocabulary from a text file or directory.

        Args:
            path: Path to file or directory (loads all .txt files)
            min_freq: Minimum frequency for inclusion
            max_vocab_size: Maximum vocabulary size
            lowercase: Whether to lowercase tokens

        Returns:
            Vocabulary instance
        """
        path = Path(path)
        source_files = []
        all_tokens = []

        if path.is_file():
            source_files.append(str(path.name))
            text = path.read_text(encoding="utf-8")
            all_tokens.extend(tokenize(text, lowercase=lowercase))

        elif path.is_dir():
            for file_path in sorted(path.glob("*.txt")):
                source_files.append(str(file_path.name))
                text = file_path.read_text(encoding="utf-8")
                all_tokens.extend(tokenize(text, lowercase=lowercase))

        else:
            raise FileNotFoundError(f"Path not found: {path}")

        token_to_id, id_to_token = build_vocab(
            all_tokens, min_freq=min_freq, max_vocab_size=max_vocab_size
        )

        return cls(
            token_to_id,
            id_to_token,
            source_files=source_files,
            min_freq=min_freq,
        )

    def save(self, path: Union[str, Path]) -> None:
        """
        Save vocabulary to JSON file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "created_at": self._created_at,
            "source_files": self._source_files,
            "config": {
                "min_freq": self._min_freq,
            },
            "tokens": {
                "token_to_id": self._token_to_id,
                "id_to_token": {str(k): v for k, v in self._id_to_token.items()},
            },
            "hash": self.hash(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        """
        Load vocabulary from JSON file.

        Args:
            path: Path to load from

        Returns:
            Vocabulary instance
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        token_to_id = data["tokens"]["token_to_id"]
        id_to_token = {int(k): v for k, v in data["tokens"]["id_to_token"].items()}

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            source_files=data.get("source_files", []),
            created_at=data.get("created_at"),
            min_freq=data.get("config", {}).get("min_freq", 1),
        )

    def token_to_id(self, token: str) -> int:
        """
        Get ID for a token.

        Args:
            token: Token string

        Returns:
            Token ID (UNK ID if not found)
        """
        return self._token_to_id.get(token, self._token_to_id[UNK_TOKEN])

    def id_to_token(self, id_: int) -> str:
        """
        Get token for an ID.

        Args:
            id_: Token ID

        Returns:
            Token string (UNK if not found)
        """
        return self._id_to_token.get(id_, UNK_TOKEN)

    def extend(self, tokens: List[str]) -> int:
        """
        Add new tokens to vocabulary.

        Args:
            tokens: Tokens to add

        Returns:
            Number of tokens actually added
        """
        added = 0
        next_id = max(self._id_to_token.keys()) + 1

        for token in tokens:
            if token not in self._token_to_id:
                self._token_to_id[token] = next_id
                self._id_to_token[next_id] = token
                next_id += 1
                added += 1

        return added

    def hash(self) -> str:
        """
        Compute hash of vocabulary for verification.

        Returns:
            SHA256 hash of sorted token list
        """
        # Sort tokens for consistent hash
        sorted_tokens = sorted(self._token_to_id.keys())
        content = "\n".join(sorted_tokens)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self._token_to_id

    @property
    def size(self) -> int:
        """Number of tokens in vocabulary."""
        return len(self._token_to_id)

    @property
    def source_files(self) -> List[str]:
        """List of source files used to create vocabulary."""
        return self._source_files

    def get_token_to_id(self) -> Dict[str, int]:
        """Get full token-to-ID mapping."""
        return self._token_to_id.copy()

    def get_id_to_token(self) -> Dict[int, str]:
        """Get full ID-to-token mapping."""
        return self._id_to_token.copy()
