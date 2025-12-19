"""
Shared utilities for demo and showcase scripts.

This module provides common functionality used across multiple demo scripts
to avoid code duplication and ensure consistent behavior.
"""

import time
from typing import Dict


class Timer:
    """Simple timer for measuring operation durations.

    Usage:
        timer = Timer()
        timer.start("operation_name")
        # ... do work ...
        elapsed = timer.stop()
        print(f"Took {elapsed:.2f}s")

        # Later retrieve time
        print(f"Operation took {timer.get('operation_name'):.2f}s")
    """

    def __init__(self):
        self.times: Dict[str, float] = {}
        self._start: float = 0
        self._current: str = ""

    def start(self, name: str):
        """Start timing an operation."""
        self._start = time.perf_counter()
        self._current = name

    def stop(self) -> float:
        """Stop timing and record the duration."""
        elapsed = time.perf_counter() - self._start
        self.times[self._current] = elapsed
        return elapsed

    def get(self, name: str) -> float:
        """Get recorded time for an operation."""
        return self.times.get(name, 0)


def print_header(title: str, char: str = "=", width: int = 70):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")
