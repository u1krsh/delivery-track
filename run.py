#!/usr/bin/env python3
"""
Delivery Tracker — AI Agent GUI Launcher
=========================================

Usage:  python run.py
"""
import sys
import os

# Ensure project root is on sys.path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.gui.main_window import main

if __name__ == "__main__":
    main()
