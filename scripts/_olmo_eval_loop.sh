#!/bin/bash
# Continuously submit Olmo-3-1025-7B stage1-step1413814 evals to the debug
# partition until every task in tasks_pretraining_full.txt has results.
# Each cycle:
#   1. Discover already-completed tasks by scanning prior eval_*/ dirs
#      (both per_task/ subdirs from killed runs AND results_*.json from
#      runs that finished cleanly).
#   2. If anything's still missing, submit one debug job (1:30) running
#      ONLY the missing tasks.
#   3. Block until that job exits, then loop.
# Bails out if two consecutive iterations leave the same set of tasks
# unfinished — those are likely permanently broken (e.g. unimplemented
# in the harness) and re-running won't help.
#
# Run with: nohup bash scripts/_olmo_eval_loop.sh > logs/olmo_eval_loop.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

NAME=Olmo-3-1025-7B-stage1-step1413814
MODEL=allenai/Olmo-3-1025-7B
REVISION=stage1-step1413814
FULL_TASKS=./configs/signal_to_ratio/tasks_pretraining_full.txt
LOGS_BASE=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/snr-experiments/$NAME/harness

ALL_TASKS=$(grep -v '^\s*#\|^\s*$' "$FULL_TASKS" | sort -u)
TOTAL=$(echo "$ALL_TASKS" | wc -l)
PREV_HASH=""

while true; do
    # If a job with this NAME is already in the queue (running or pending),
    # wait for it before doing anything else — it will produce results we
    # need to read before deciding what to submit next.
    EXISTING=$(squeue --me --noheader -o "%i %j" | awk -v n="eval-$NAME" '$2==n {print $1}')
    if [[ -n "$EXISTING" ]]; then
        echo "[$(date '+%F %T')] Existing job(s) for $NAME: $EXISTING — waiting"
        for J in $EXISTING; do
            while squeue -j "$J" --noheader -o "%i" 2>/dev/null | grep -q "$J"; do
                sleep 60
            done
        done
        echo "[$(date '+%F %T')] Existing job(s) finished"
    fi

    # Tasks already completed = per_task/ subdirs (killed runs) ∪ results_*.json keys (clean runs)
    COMPLETED=$( {
        find "$LOGS_BASE"/eval_* -mindepth 2 -maxdepth 2 -type d -path '*/per_task/*' 2>/dev/null \
            | xargs -n1 basename 2>/dev/null
        find "$LOGS_BASE"/eval_* -maxdepth 1 -name 'results_*.json' 2>/dev/null \
            | xargs -I{} jq -r '.results | keys[]?' {} 2>/dev/null
    } | sort -u )

    REMAINING=$(comm -23 <(echo "$ALL_TASKS") <(echo "$COMPLETED"))
    REM_COUNT=$([[ -z "$REMAINING" ]] && echo 0 || echo "$REMAINING" | wc -l)
    DONE_COUNT=$((TOTAL - REM_COUNT))

    if [[ $REM_COUNT -eq 0 ]]; then
        echo "[$(date '+%F %T')] All $TOTAL tasks done for $NAME. Loop complete."
        break
    fi

    CUR_HASH=$(echo "$REMAINING" | md5sum | cut -d' ' -f1)
    if [[ "$CUR_HASH" == "$PREV_HASH" ]]; then
        echo "[$(date '+%F %T')] No progress this cycle. $REM_COUNT tasks won't complete (likely broken):"
        echo "$REMAINING" | sed 's/^/  - /'
        echo "Exiting."
        exit 1
    fi
    PREV_HASH=$CUR_HASH

    echo "[$(date '+%F %T')] $DONE_COUNT/$TOTAL done; $REM_COUNT remaining. Next submission task list:"
    echo "$REMAINING" | sed 's/^/  - /'

    # Wait for a debug slot (max 2 of mine in queue)
    while [[ $(squeue --me -p debug --noheader -o "%i" | wc -l) -ge 2 ]]; do
        sleep 60
    done

    TASKS_ARG=$(echo "$REMAINING" | paste -sd, -)
    JOB_ID=$(TASKS="$TASKS_ARG" \
             LM_EVAL_BACKEND=vllm \
             APPLY_CHAT_TEMPLATE=false \
             WANDB_ENTITY=mariagrandury-epflnlp \
             WANDB_PROJECT=snr-experiments \
             REVISION=$REVISION \
             sbatch --parsable --partition=debug --time=01:30:00 \
                 --job-name="eval-$NAME" \
                 scripts/evaluate.sbatch "$MODEL" "$NAME") || {
        echo "[$(date '+%F %T')] sbatch failed; sleeping 60s before retry"; sleep 60; continue
    }
    echo "[$(date '+%F %T')] Submitted $JOB_ID with $REM_COUNT tasks"

    # Block until the job leaves the queue
    while squeue -j "$JOB_ID" --noheader -o "%i" 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 60
    done
    echo "[$(date '+%F %T')] Job $JOB_ID finished. Re-evaluating remaining."
done
