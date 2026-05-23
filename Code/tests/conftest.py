"""
Shared pytest fixtures and path setup.

This file ensures ``import src.xxx`` works when pytest is run from the
``Code/`` directory.
"""

import os
import sys

# Add the parent ("Code/") directory to sys.path so that `import src` works
# whether the tests are run from Code/ or from the repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
