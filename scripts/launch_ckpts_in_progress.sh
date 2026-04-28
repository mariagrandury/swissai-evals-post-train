#!/bin/bash
# Resubmit one sbatch per "stuck" ckpt: ones marked [in_progress] in
# snr_progress.py with NO active job (partial results from a past run that
# timed out or was cancelled). Walltime is sized to the REMAINING work, not
# the model size, so a ckpt with 1 task left gets ~30 min and one with 30
# tasks left gets several hours.
#
# Usage: bash scripts/relaunch_stuck.sh [--dry-run]
#
# Per-task minute estimates (logprob-only, since mgsm is no longer in the
# task list) are coarse averages from observed runs. Adjust if the typical
# task mix changes again.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

declare -A PER_TASK_MIN=(["175M"]=3 ["350M"]=5 ["600M"]=6 ["1B"]=8)
COLD_START_MIN=15      # checkpoint load + container setup
BUFFER_MIN=10
MIN_WALL_MIN=30
CAP_MIN=719            # 11:59:00 — max for normal partition with 1 min margin

CKPT_BASE=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small

# Stuck = [in_progress] with NO `jobs=` annotation in snr_progress output.
STUCK=$(python3.11 scripts/snr_progress.py 2>&1 | awk '
    /\[in_progress\]/ && !/jobs=/ {
        match($0, /([0-9]+)\/([0-9]+)/, m)
        done = m[1]; total = m[2]
        for (i=1; i<=NF; i++) if ($i ~ /apertus|SmolLM|Olmo|Apertus/) { name=$i; break }
        print done "/" total " " name
    }
' | sort -u)

if [[ -z "$STUCK" ]]; then
    echo "No stuck ckpts — nothing to resubmit."
    exit 0
fi

submitted=0
skipped=0
while IFS=' ' read -r progress name; do
    [[ -z "$name" ]] && continue

    size=""
    for s in 175M 350M 600M 1B; do
        if [[ "$name" == *"-$s-"* ]]; then size=$s; break; fi
    done
    if [[ -z "$size" ]]; then
        echo "skip $name (no size match — only Megatron apertus ckpts supported)"
        skipped=$((skipped + 1))
        continue
    fi

    iter=${name##*-iter}
    model_dir=${name%-iter*}
    ckpt_path="$CKPT_BASE/$model_dir/checkpoints"
    iter_dir="$ckpt_path/iter_$(printf '%07d' $iter)"
    if [[ ! -d "$iter_dir" ]]; then
        echo "skip $name (iter dir missing: $iter_dir)"
        skipped=$((skipped + 1))
        continue
    fi

    done_count=${progress%%/*}
    total_count=${progress##*/}
    remaining=$(( total_count - done_count ))
    per_task=${PER_TASK_MIN[$size]}
    wall_min=$(( remaining * per_task + COLD_START_MIN + BUFFER_MIN ))
    (( wall_min < MIN_WALL_MIN )) && wall_min=$MIN_WALL_MIN
    (( wall_min > CAP_MIN ))      && wall_min=$CAP_MIN

    hh=$(( wall_min / 60 ))
    mm=$(( wall_min % 60 ))
    walltime=$(printf "%02d:%02d:00" $hh $mm)

    if (( DRY_RUN )); then
        echo "would submit  $walltime  $name  ($progress done, $remaining left)"
        submitted=$((submitted + 1))
        continue
    fi

    JOB_ID=$(TASKS=./configs/signal_to_ratio/tasks_pretraining_full.txt \
             LM_EVAL_BACKEND=megatron_lm \
             TOKENIZER=alehc/swissai-tokenizer \
             BOS=true \
             APPLY_CHAT_TEMPLATE=false \
             WANDB_ENTITY=mariagrandury-epflnlp \
             WANDB_PROJECT=snr-experiments \
             sbatch --parsable \
                 --job-name "eval-$name" \
                 --time "$walltime" \
                 --partition=normal \
                 --export=ALL,CKPT_ITER=$iter \
                 scripts/evaluate.sbatch "$ckpt_path" "$name") \
        || { echo "sbatch FAILED for $name"; skipped=$((skipped + 1)); continue; }
    echo "Submitted $JOB_ID  $name  ($walltime, $progress done, $remaining left)"
    submitted=$((submitted + 1))
done <<< "$STUCK"

echo ""
echo "Total submitted: $submitted ; skipped: $skipped"
