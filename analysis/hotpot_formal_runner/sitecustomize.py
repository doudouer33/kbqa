"""Activate guards only in explicitly opted-in inference subprocesses."""
import os
import sys
import traceback

if os.environ.get("KBQA_FORMAL_MANIFEST") and "--config" in sys.argv and "--input" in sys.argv:
    try:
        from analysis.hotpot_formal_runner.hooks import install
        install()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)  # Python otherwise ignores sitecustomize failures.
