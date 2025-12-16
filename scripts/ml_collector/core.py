"""
Core module for ML Data Collector

Contains exceptions and the ML_COLLECTION_ENABLED flag.
"""

import os


# Environment variable to disable collection
ML_COLLECTION_ENABLED = os.getenv("ML_COLLECTION_ENABLED", "1") != "0"


class GitCommandError(Exception):
    """Raised when a git command fails."""
    pass


class SchemaValidationError(Exception):
    """Raised when data fails schema validation."""
    pass
