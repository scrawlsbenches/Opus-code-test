"""
Pluggable query functions.

Functions are auto-registered when this module is imported.
Add new function modules here to make them available.
"""

# Import function modules to trigger registration
from . import graph
from . import filters
from . import aggregate_functions
