#!/bin/bash

# launch_signal_to_ratio.sh - Launch evaluations for signal-to-ratio analysis
#
# Evaluates the last N checkpoints of each model on specified tasks.
# Models can be Megatron (local paths starting with /iopsstor or /capstor)
# or HuggingFace (URLs starting with https://huggingface.co/).
#
# For Megatron models, checkpoints are iter_* directories.
# For HuggingFace models, checkpoints are non-main branches of the repo.
#
# Usage:
#   bash scripts/launch_signal_to_ratio.sh \
#     --models configs/signal_to_ratio/models_pretraining_custom.txt \
#     --tasks configs/signal_to_ratio/tasks_pretraining.txt \
#     --last-n-checkpoints 5 \
#     [--limit 10] [--splits 2]
#
# Options:
#   --models <file>            - Text file with one model per line
#   --tasks <file>             - Text file with one lm-harness task per line
#   --last-n-checkpoints <N>   - Number of most recent checkpoints to evaluate
#   --limit <N>                - Limit samples per task (for testing)
#   --splits <K>               - Split tasks across K parallel nodes per model
#   --time <HH:MM:SS>          - SLURM time limit per job (default: from sbatch script)
#   --dry-run                  - Print what would be launched without submitting jobs

set -euo pipefail

# --- Argument parsing ---
MODELS_FILE=""
TASKS_FILE=""
LAST_N=0
HARNESS_LIMIT=""
NUM_SPLITS=1
SLURM_TIME=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)              MODELS_FILE="$2";    shift 2 ;;
        --tasks)               TASKS_FILE="$2";     shift 2 ;;
        --last-n-checkpoints)  LAST_N="$2";         shift 2 ;;
        --limit)               HARNESS_LIMIT="$2";  shift 2 ;;
        --splits)              NUM_SPLITS="$2";     shift 2 ;;
        --time)                SLURM_TIME="$2";     shift 2 ;;
        --dry-run)             DRY_RUN=true;        shift ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
done

# --- Validate required args ---
if [[ -z "$MODELS_FILE" || -z "$TASKS_FILE" ]]; then
    echo "Error: --models and --tasks are required"
    exit 1
fi
if [[ ! -f "$MODELS_FILE" ]]; then
    echo "Error: Models file not found: $MODELS_FILE"
    exit 1
fi
if [[ ! -f "$TASKS_FILE" ]]; then
    echo "Error: Tasks file not found: $TASKS_FILE"
    exit 1
fi
if ! [[ "$LAST_N" =~ ^[0-9]+$ ]] || (( LAST_N < 1 )); then
    echo "Error: --last-n-checkpoints must be an integer >= 1"
    exit 1
fi
if ! [[ "$NUM_SPLITS" =~ ^[0-9]+$ ]] || (( NUM_SPLITS < 1 )); then
    echo "Error: --splits must be an integer >= 1"
    exit 1
fi

# --- Helper: get last N Megatron checkpoint iterations ---
get_megatron_checkpoints() {
    local ckpt_dir="$1"
    local n="$2"
    ls -d "${ckpt_dir}"/iter_* 2>/dev/null \
        | sed 's/.*iter_//' | sed 's/^0*//' \
        | sort -n \
        | tail -n "$n"
}

# --- Helper: get last N HuggingFace checkpoint branches ---
get_hf_checkpoints() {
    local repo_id="$1"
    local n="$2"
    python3 -c "
import json, re, sys, urllib.request

repo_id = sys.argv[1]
n = int(sys.argv[2])

url = f'https://huggingface.co/api/models/{repo_id}/refs'
try:
    data = json.loads(urllib.request.urlopen(url).read())
except Exception as e:
    print(f'Error fetching branches: {e}', file=sys.stderr)
    sys.exit(1)

branches = [b['name'] for b in data.get('branches', []) if b['name'] != 'main']

def extract_number(name):
    nums = re.findall(r'\d+', name)
    return int(nums[-1]) if nums else 0

branches.sort(key=extract_number)
for b in branches[-n:]:
    print(b)
" "$repo_id" "$n"
}

# --- Helper: derive model name from path/URL ---
derive_model_name() {
    local path="$1"
    if [[ "$path" == https://* ]]; then
        local repo_id="${path#https://huggingface.co/}"
        echo "${repo_id##*/}"
    else
        # Megatron: use the directory name before /checkpoints/
        local dir="${path%/checkpoints/}"
        dir="${dir%/checkpoints}"
        dir="${dir%/}"
        basename "$dir"
    fi
}

# --- Print configuration ---
echo "======================================"
echo "Signal-to-Ratio Evaluation Launcher"
echo "  Models file:    $MODELS_FILE"
echo "  Tasks file:     $TASKS_FILE"
echo "  Last N ckpts:   $LAST_N"
[[ -n "$HARNESS_LIMIT" ]] && echo "  Limit:          $HARNESS_LIMIT"
[[ -n "$SLURM_TIME" ]] && echo "  Time limit:     $SLURM_TIME"
echo "  Splits:         $NUM_SPLITS"
[[ "$DRY_RUN" == true ]] && echo "  *** DRY RUN ***"
echo "======================================"

# --- Set up shared environment ---
export TASKS="$TASKS_FILE"
export NUM_SPLITS
export WANDB_ENTITY=${WANDB_ENTITY:-mariagrandury-epflnlp}
export WANDB_PROJECT=${WANDB_PROJECT:-snr-experiments}
export SBATCH_SCRIPT=${SBATCH_SCRIPT:-scripts/evaluate.sbatch}
[[ -n "$HARNESS_LIMIT" ]] && export HARNESS_LIMIT
[[ -n "$SLURM_TIME" ]] && export SLURM_TIME

# --- Process each model ---
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    line="${line#"${line%%[![:space:]]*}"}"  # trim leading whitespace
    line="${line%"${line##*[![:space:]]}"}"  # trim trailing whitespace
    [[ -z "$line" || "$line" == \#* ]] && continue

    MODEL_BASE_NAME=$(derive_model_name "$line")

    if [[ "$line" == /iopsstor* || "$line" == /capstor* ]]; then
        # ========== Megatron model ==========
        CKPT_DIR="${line%/}"

        echo ""
        echo "Processing Megatron model: $MODEL_BASE_NAME"
        echo "  Checkpoint dir: $CKPT_DIR"

        ITERS=$(get_megatron_checkpoints "$CKPT_DIR" "$LAST_N")
        if [[ -z "$ITERS" ]]; then
            echo "  WARNING: No checkpoints found in $CKPT_DIR, skipping."
            continue
        fi

        echo "  Checkpoints to evaluate:"
        while IFS= read -r iter; do
            echo "    - iter $iter"
        done <<< "$ITERS"

        # Set Megatron-specific env vars
        export TOKENIZER=${TOKENIZER:-alehc/swissai-tokenizer}
        export BOS=${BOS:-true}
        export APPLY_CHAT_TEMPLATE=false
        export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-megatron_lm}
        unset REVISION

        while IFS= read -r iter; do
            MODEL_NAME="${MODEL_BASE_NAME}-iter${iter}"

            if [[ "$DRY_RUN" == true ]]; then
                echo "  [DRY RUN] Would launch: $MODEL_NAME (model=$CKPT_DIR, iter=$iter)"
                continue
            fi

            export CKPT_ITERATION="$iter"
            unset MODEL_CHECKPOINTS
            declare -A MODEL_CHECKPOINTS=(
                ["$MODEL_NAME"]="$CKPT_DIR"
            )
            source runners/hf_base_runner.sh "Megatron checkpoint"
        done <<< "$ITERS"

    elif [[ "$line" == https://* ]]; then
        # ========== HuggingFace model ==========
        REPO_ID="${line#https://huggingface.co/}"

        echo ""
        echo "Processing HuggingFace model: $MODEL_BASE_NAME ($REPO_ID)"

        BRANCHES=$(get_hf_checkpoints "$REPO_ID" "$LAST_N")
        if [[ -z "$BRANCHES" ]]; then
            echo "  WARNING: No checkpoint branches found for $REPO_ID, skipping."
            continue
        fi

        echo "  Branches to evaluate:"
        while IFS= read -r branch; do
            echo "    - $branch"
        done <<< "$BRANCHES"

        # Set HF-specific env vars
        export APPLY_CHAT_TEMPLATE=false
        export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}
        unset CKPT_ITERATION

        while IFS= read -r branch; do
            MODEL_NAME="${MODEL_BASE_NAME}-${branch}"

            if [[ "$DRY_RUN" == true ]]; then
                echo "  [DRY RUN] Would launch: $MODEL_NAME (model=$REPO_ID, revision=$branch)"
                continue
            fi

            export REVISION="$branch"
            unset MODEL_CHECKPOINTS
            declare -A MODEL_CHECKPOINTS=(
                ["$MODEL_NAME"]="$REPO_ID"
            )
            source runners/hf_base_runner.sh "HuggingFace checkpoint"
        done <<< "$BRANCHES"

    else
        echo ""
        echo "WARNING: Unrecognized model format, skipping: $line"
    fi
done < "$MODELS_FILE"

echo ""
echo "======================================"
echo "All models processed."
echo "======================================"
