"""Enable diagnostic hooks only for this experiment's inference subprocess."""
import os
import sys
import traceback

if os.environ.get("KBQA_PROMPT_TOKEN_DIAGNOSTIC_DIR") and "--config" in sys.argv and "--input" in sys.argv:
    try:
        from hooks import install

        install()
    except BaseException:
        traceback.print_exc()
        # Python otherwise ignores sitecustomize errors and runs uninstrumented.
        sys.stderr.flush()
        os._exit(1)
