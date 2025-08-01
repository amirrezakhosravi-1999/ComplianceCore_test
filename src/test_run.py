#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick test runner for CAELUS compliance checking system.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.append(str(src_path))

from simple_tester import run_simple_test

if __name__ == "__main__":
    run_simple_test()
