#!/bin/bash
# Run one pipeline stage detached on this machine (macOS, CPU only).
#
# Usage:
#   scripts/run_local_stage.sh <stage_name> <python script and args...>
# Example:
#   scripts/run_local_stage.sh gnn_random_w8 src/run_gnn_experiment.py --split random --weeks 8
#
# - caffeinate -i keeps macOS from idle-sleeping while the stage runs (keep the
#   laptop plugged in; closing the lid still sleeps it).
# - Output goes to logs/local/<stage_name>_<timestamp>.log (*.log is gitignored).
# - Stages are resumable: result CSVs are idempotent on their distinguishing keys,
#   and LCPO stages can be continued with --start-fold.
# - Run one stage at a time. Result paths come from src/config.py, not from here.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <stage_name> <python script and args...>" >&2
    exit 2
fi

STAGE="$1"
shift

cd "$(dirname "$0")/.."
mkdir -p logs/local
source oulad_env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

LOG="logs/local/${STAGE}_$(date +%Y%m%d_%H%M%S).log"

nohup caffeinate -i python "$@" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "logs/local/${STAGE}.pid"

echo "stage:  $STAGE"
echo "pid:    $PID"
echo "log:    $LOG"
echo "follow: tail -f $LOG"
echo "check:  ps -p $PID"
