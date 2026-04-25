#!/bin/bash
# Run lm_eval once per task with fault tolerance: a failure in one task is
# logged and the loop continues so survivor metrics still reach W&B.
#
# Reads from env (exported by evaluate.sbatch):
#   CMD_BASE          lm_eval invocation without --tasks / --output_path
#   TASKS             comma-separated task names
#   HARNESS_EVAL_DIR  base output dir; per-task subdirs are merged into here
set -uo pipefail

PER_TASK_DIR="$HARNESS_EVAL_DIR/per_task"
FAILED_LOG="$HARNESS_EVAL_DIR/failed_tasks.log"
mkdir -p "$PER_TASK_DIR"
: > "$FAILED_LOG"

SUCCESS_DIRS=()
IFS=',' read -ra ALL_TASKS <<< "$TASKS"
for t in "${ALL_TASKS[@]}"; do
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
    echo "ERROR: every task failed; nothing to merge or upload." >&2
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
