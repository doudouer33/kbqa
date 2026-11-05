#!/bin/sh
# This script never samples. Use scripts/sample.py explicitly once, then prepare.
set -eu
EXPERIMENT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SCRIPT_DIR="$EXPERIMENT_DIR/scripts"
REPO_DIR=$(CDPATH= cd -- "$EXPERIMENT_DIR/../.." && pwd)
cd "$REPO_DIR"
PYTHON=${PYTHON:-/home/dengxin/miniconda3/envs/adaptiverag/bin/python}
MODE=${1:-smoke}
case "$MODE" in
  prepare|check-data)
    exec "$PYTHON" "$SCRIPT_DIR/pipeline.py" "$MODE"
    ;;
  status|check)
    shift
    exec "$PYTHON" "$SCRIPT_DIR/pipeline.py" "$MODE" "$@"
    ;;
  smoke)
    KS="1 8 15"
    ;;
  full)
    if [ "${2:-}" != "--allow-full-run" ]; then
      echo "Full 5000x15 inference requires: run.sh full --allow-full-run" >&2
      exit 2
    fi
    KS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    ;;
  *) echo "Usage: run.sh {prepare|check-data|smoke|status|check|full --allow-full-run}" >&2; exit 2 ;;
esac
"$PYTHON" "$SCRIPT_DIR/pipeline.py" check-data
for K in $KS; do
  if [ "$MODE" = full ]; then
    "$PYTHON" "$SCRIPT_DIR/pipeline.py" run-k --mode full --k "$K" --allow-full-run
  else
    "$PYTHON" "$SCRIPT_DIR/pipeline.py" run-k --mode smoke --k "$K"
  fi
done
"$PYTHON" "$SCRIPT_DIR/pipeline.py" check --mode "$MODE"
