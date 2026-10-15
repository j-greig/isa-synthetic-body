#!/usr/bin/env bash
# Job queue for the Mac: runs job files dropped into queue/pending/ one at a time, on the
# Mac's GPU. Lets a remote agent (or you) schedule experiments without typing into Terminal.
#
#   caffeinate -i bash scripts/queue_runner.sh        # leave running; Ctrl-C or `touch queue/STOP` to end
#
# A job file is one line: the command. Only these forms are accepted (anything else is rejected):
#   python scripts/<name>.py [args]      bash scripts/<name>.sh [args]
# Args may contain only letters, digits, and _ . , = / : - and spaces.
# Each job writes logs/jobs/<job>.log and queue/done/<job>.exit; provenance goes to logs/runs.jsonl.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
mkdir -p queue/pending queue/running queue/done logs/jobs
[ -d .venv ] && source .venv/bin/activate
ALLOW='^(python scripts/[a-z0-9_]+\.py|bash scripts/[a-z0-9_]+\.sh)( [A-Za-z0-9_.,=/: -]*)?$'
echo "[queue] started $(date -u +%FT%TZ) pid $$ git $(git rev-parse --short HEAD)" | tee -a logs/jobs/queue.log
while [ ! -f queue/STOP ]; do
  if pgrep -f "scripts/run_mac.sh" >/dev/null; then sleep 30; continue; fi   # one GPU job at a time
  job=$(ls queue/pending/*.job 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then sleep 15; continue; fi
  name=$(basename "$job" .job); mv "$job" "queue/running/$name.job"
  cmd=$(head -1 "queue/running/$name.job" | tr -d '\r')
  if ! [[ "$cmd" =~ $ALLOW ]]; then
    echo "[queue] REJECTED $name: $cmd" | tee -a logs/jobs/queue.log
    echo rejected > "queue/done/$name.exit"; mv "queue/running/$name.job" "queue/done/$name.job"; continue
  fi
  echo "[queue] START $name $(date -u +%FT%TZ) git $(git rev-parse --short HEAD): $cmd" | tee -a logs/jobs/queue.log
  bash -c "$cmd" > "logs/jobs/$name.log" 2>&1; rc=$?
  echo "$rc" > "queue/done/$name.exit"; mv "queue/running/$name.job" "queue/done/$name.job"
  echo "[queue] END   $name $(date -u +%FT%TZ) exit $rc" | tee -a logs/jobs/queue.log
done
rm -f queue/STOP; echo "[queue] stopped $(date -u +%FT%TZ)" | tee -a logs/jobs/queue.log
