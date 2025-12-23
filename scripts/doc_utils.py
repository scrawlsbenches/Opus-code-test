#!/usr/bin/env python3
"""
Document registry CLI - thin wrapper around cortical.got.cli.doc

For full documentation, see: python scripts/got_utils.py doc --help
"""

import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cortical.got.cli.doc import main

if __name__ == "__main__":
    sys.exit(main())
