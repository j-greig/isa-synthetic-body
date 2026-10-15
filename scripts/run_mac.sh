#!/usr/bin/env bash
# One-shot pipeline for a Mac (Apple Silicon, MPS). Every step writes a provenance manifest
# (logs/runs.jsonl); the full console output goes to logs/pipeline_<UTC>.log.
# Progress markers ("### STEP n/N ...") let a remote observer follow along.
#
#   caffeinate -i bash scripts/run_mac.sh            # everything
#   STEPS="probes quick" bash scripts/run_mac.sh     # selected steps
#   MODEL=google/gemma-3-1b-it REPEATS=2 bash scripts/run_mac.sh
set -uo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1   # CPU fallback for any op MPS lacks
export PYTHONUNBUFFERED=1             # live progress in the log
cd "$(dirname "$0")/.."

MODEL="${MODEL:-unsloth/gemma-3-1b-it}"
REPEATS="${REPEATS:-1}"
STEPS="${STEPS:-env probes heldout appraisal features quick pilot}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p logs
LOG="logs/pipeline_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt

echo "### PIPELINE START $TS model=$MODEL repeats=$REPEATS git=$(git rev-parse --short HEAD) steps=[$STEPS]"
N=$(echo $STEPS | wc -w | tr -d ' '); i=0
step() { i=$((i+1)); echo; echo "### STEP $i/$N $1 START $(date -u +%H:%M:%S)"; }
done_() { echo "### STEP $i/$N $1 END $(date -u +%H:%M:%S) status=$2"; }

for s in $STEPS; do
  case $s in
    env)
      step env; python scripts/00_check_env.py --model_id "$MODEL"; done_ env $? ;;
    probes)
      step probes; python scripts/02_train_probes.py --model_id "$MODEL"; done_ probes $? ;;
    heldout)
      step heldout; python scripts/02b_validate_probes.py --model_id "$MODEL"; done_ heldout $? ;;
    appraisal)
      step appraisal
      [ -f data/crowd-enVent/corpus/crowd-enVent_generation.tsv ] || bash scripts/00b_get_crowdenvent.sh
      python scripts/02c_train_appraisal.py --model_id "$MODEL" --n_per_emotion 120; done_ appraisal $? ;;
    features)
      step features
      python scripts/01_explore_features.py --model_id "$MODEL" --layer 13 && \
      python scripts/01_explore_features.py --model_id "$MODEL" --layer 17 && \
      python scripts/01b_label_features.py --model_id "$MODEL" --layer 13 --features 2404,278,4258,203,602,743,896,1983,3413,1138 && \
      python scripts/01b_label_features.py --model_id "$MODEL" --layer 17 --features 687,550,563,1099,125,1008,1804,807,96,309
      done_ features $? ;;
    quick)
      step quick
      python scripts/04_run.py --model_id "$MODEL" --quick && \
      python scripts/05_analyse.py runs/quick.jsonl --out artifacts/analysis_quick
      done_ quick $? ;;
    pilot)
      step pilot
      python scripts/04_run.py --model_id "$MODEL" --conditions A,B,C,G,H,H0,O,P,R,N,N0,X1 \
        --scenarios night_run,stage,deploy --repeats "$REPEATS" --gains 0.1 --max_new_tokens 40 --name pilot && \
      python scripts/05_analyse.py runs/pilot.jsonl --out artifacts/analysis_pilot
      done_ pilot $? ;;
    *) echo "unknown step $s" ;;
  esac
done
echo "### PIPELINE END $(date -u +%Y%m%dT%H%M%SZ)"
