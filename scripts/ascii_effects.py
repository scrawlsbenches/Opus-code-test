"""
Shared data classes for ASCII visualization effects.

This module provides common data structures used by both the interactive
terminal visualizer (ascii_visualizer_animated.py) and the GIF generator
(generate_ascii_gifs.py).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MatrixDrop:
    """A single drop in the Matrix rain effect.

    Attributes:
        col: Column position
        row: Row position (can be fractional for smooth animation)
        speed: Fall speed (rows per frame)
        text: The text string to display
        char_idx: Current character index being highlighted
        brightness: Optional brightness multiplier (0.0-1.0) for terminal effects
    """
    col: int
    row: float
    speed: float
    text: str
    char_idx: int
    brightness: float = 1.0  # Default full brightness, optional for GIF generator


@dataclass
class Star:
    """A star in the starfield effect.

    Uses 3D coordinates where z is the depth (distance from viewer).

    Attributes:
        x: X position in 3D space
        y: Y position in 3D space
        z: Z position (depth) - smaller values are closer
        text: The text to display (typically a commit message fragment)
    """
    x: float
    y: float
    z: float
    text: str
