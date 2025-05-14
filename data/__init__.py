"""
This file ensures the data directory exists when imported.
It can be left empty or used to initialize data-related components.
"""

import os

# Make sure the data directory exists
# This is important as Python treats directories with __init__.py as packages
data_dir = os.path.dirname(os.path.abspath(__file__))