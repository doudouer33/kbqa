"""Opt-in only; fail visibly if observers cannot be installed."""
import os
import sys
import traceback

if os.environ.get("HOTPOT5000_ATTEMPT") and "--config" in sys.argv and "--input" in sys.argv:
    try:
        from hooks import install
        install()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)  # Python otherwise ignores errors in sitecustomize.
