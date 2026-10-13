"""Opt-in instrumentation for run.py's configurable_inference subprocess only."""
import os
import sys

if os.environ.get("KBQA_STOP_DIAGNOSTIC_DIR") and "--config" in sys.argv and "--input" in sys.argv:
    from hooks import install

    install()
