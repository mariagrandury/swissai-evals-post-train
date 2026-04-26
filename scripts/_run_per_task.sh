#!/bin/bash
# Run lm_eval once per task with fault tolerance: a failure in one task is
# logged and the loop continues so survivor metrics still reach W&B.
# Idempotent: tasks whose results already exist on disk (in any sibling
# eval_*/ run for this checkpoint) are skipped, so re-launching the same
# command after a partial run / timeout doesn't redo finished work.
#
# Reads from env (exported by evaluate.sbatch):
#   CMD_BASE          lm_eval invocation without --tasks / --output_path
#   TASKS             comma-separated task names
#   HARNESS_EVAL_DIR  base output dir; per-task subdirs are merged into here
#   NAME              checkpoint name (model-ckpt) — used for status lookup
#   WANDB_ENTITY      W&B entity
#   WANDB_PROJECT     W&B project
#   LOGS_ROOT         eval_logs root (optional, has a sane default)
set -uo pipefail

PER_TASK_DIR="$HARNESS_EVAL_DIR/per_task"
FAILED_LOG="$HARNESS_EVAL_DIR/failed_tasks.log"
SKIPPED_LOG="$HARNESS_EVAL_DIR/skipped_tasks.log"
mkdir -p "$PER_TASK_DIR"
: > "$FAILED_LOG"
: > "$SKIPPED_LOG"

# Filter out tasks that already have results in any prior eval_*/ run.
# _eval_status.py contract: exit 0 + non-empty stdout = tasks remaining,
# exit 1 = all done, exit 2+ = crash. Don't conflate the crash case with
# "all done" — fall back to running everything if the filter misbehaves.
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
set +e
REMAINING=$(python3 "$REPO_DIR/scripts/_eval_status.py" \
    --name "$NAME" --tasks "$TASKS" \
    --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" 2>/dev/null)
EVAL_STATUS_RC=$?
set -uo pipefail

if [[ $EVAL_STATUS_RC -eq 1 ]]; then
    echo "All tasks for $NAME already have results — nothing to do."
    exit 0
elif [[ $EVAL_STATUS_RC -ne 0 ]]; then
    echo "WARNING: _eval_status.py crashed (rc=$EVAL_STATUS_RC); running all tasks without filtering."
    REMAINING=$(echo "$TASKS" | tr ',' '\n')
elif [[ -z "$REMAINING" ]]; then
    echo "All tasks for $NAME already have results — nothing to do."
    exit 0
fi

# Log which tasks we're skipping vs running
IFS=',' read -ra _ALL_TASKS_ARR <<< "$TASKS"
declare -A _REMAINING_SET
while IFS= read -r t; do
    [[ -n "$t" ]] && _REMAINING_SET["$t"]=1
done <<< "$REMAINING"
for t in "${_ALL_TASKS_ARR[@]}"; do
    if [[ -z "${_REMAINING_SET[$t]:-}" ]]; then
        echo "$t" >> "$SKIPPED_LOG"
    fi
done

if [[ -s "$SKIPPED_LOG" ]]; then
    echo "Skipping $(wc -l < "$SKIPPED_LOG") task(s) with existing results:"
    sed 's/^/  - /' "$SKIPPED_LOG"
fi

SUCCESS_DIRS=()
mapfile -t TASKS_TO_RUN <<< "$REMAINING"
for t in "${TASKS_TO_RUN[@]}"; do
    [[ -z "$t" ]] && continue
    out="$PER_TASK_DIR/$t"
    echo "=== Running task: $t ==="
    if $CMD_BASE --tasks "$t" --output_path "$out"; then
        SUCCESS_DIRS+=("$out")
        echo "=== OK: $t ==="
    else
        rc=$?
        echo "$t" >> "$FAILED_LOG"
        echo "=== FAILED (rc=$rc): $t — logged and continuing ===" >&2
    fi
done

if (( ${#SUCCESS_DIRS[@]} == 0 )); then
    echo "ERROR: every attempted task failed; nothing to merge or upload." >&2
    exit 1
fi

echo "Merging ${#SUCCESS_DIRS[@]} successful task dirs into $HARNESS_EVAL_DIR"
python -m scripts.alignment.merge_split_results \
    --split_dirs "${SUCCESS_DIRS[@]}" \
    --output_dir "$HARNESS_EVAL_DIR"

# Drop the per-task scratch; merged outputs live at $HARNESS_EVAL_DIR root.
rm -rf "$PER_TASK_DIR"

if [[ -s "$FAILED_LOG" ]]; then
    echo "WARNING: $(wc -l < "$FAILED_LOG") task(s) failed (see $FAILED_LOG):"
    sed 's/^/  - /' "$FAILED_LOG"
else
    rm -f "$FAILED_LOG"
fi

[[ -s "$SKIPPED_LOG" ]] || rm -f "$SKIPPED_LOG"
