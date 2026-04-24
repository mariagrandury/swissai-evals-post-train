#!/bin/bash

# generate_snr_runner.sh - Produce a stage runner for an SNR evaluation.
#
# Takes a models file (one Megatron /iopsstor|/capstor path or
# https://huggingface.co/<repo> URL per line) and a checkpoint-selection flag
# (--last N or --total T), and writes a self-contained runner script to stdout
# that follows the runners/hf_eval_multiple_*.sh convention. The generated
# runner is then consumed by the unchanged launch_evaluations.sh --script path.
#
# Usage:
#   bash scripts/generate_snr_runner.sh --models <file> (--last N | --total T) > runners/snr_<stage>.sh
#   bash scripts/launch_evaluations.sh snr-<stage> --script runners/snr_<stage>.sh [--splits K ...]
#
# Example:
#   bash scripts/generate_snr_runner.sh \
#       --models configs/signal_to_ratio/models_pretraining_custom.txt --last 5 \
#       > runners/snr_pretraining.sh
#   bash scripts/launch_evaluations.sh snr-pretraining --script runners/snr_pretraining.sh --splits 2
#
# Re-run whenever checkpoints or the models list change. Commit the output so
# the evaluated-checkpoint snapshot is version-controlled.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST_CHECKPOINTS="$HERE/list_checkpoints.sh"

MODELS_FILE=""
LAST_N=0
TOTAL_T=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS_FILE="$2"; shift 2 ;;
        --last)   LAST_N="$2";      shift 2 ;;
        --total)  TOTAL_T="$2";     shift 2 ;;
        *) echo "Error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODELS_FILE" || ! -f "$MODELS_FILE" ]]; then
    echo "Error: --models <file> is required and must exist" >&2; exit 1
fi
if (( LAST_N > 0 && TOTAL_T > 0 )) || (( LAST_N == 0 && TOTAL_T == 0 )); then
    echo "Error: exactly one of --last N or --total T is required" >&2; exit 1
fi
CKPT_FLAG="--last"; CKPT_COUNT="$LAST_N"
if (( TOTAL_T > 0 )); then CKPT_FLAG="--total"; CKPT_COUNT="$TOTAL_T"; fi

# --- Helper: derive a human-readable base name from a path or URL ---
derive_name() {
    local m="$1"
    m="${m#https://huggingface.co/}"
    if [[ "$m" == /* ]]; then
        m="${m%/}"; m="${m%/checkpoints}"
        basename "$m"
    else
        echo "${m##*/}"
    fi
}

# --- Collect Megatron and HF entries from the models file ---
MEGATRON_ENTRIES=()   # "base|ckpt_dir|iter"
HF_ENTRIES=()         # "base|repo_id|branch"

while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue

    base=$(derive_name "$line")

    if [[ "$line" == /iopsstor* || "$line" == /capstor* ]]; then
        ckpt_dir="${line%/}"
        if ! iters=$(bash "$LIST_CHECKPOINTS" "$line" "$CKPT_FLAG" "$CKPT_COUNT" 2>/dev/null); then
            echo "# WARNING: enumeration failed for $line (skipped)" >&2
            continue
        fi
        while IFS= read -r it; do
            [[ -z "$it" ]] && continue
            MEGATRON_ENTRIES+=("${base}|${ckpt_dir}|${it}")
        done <<< "$iters"
    elif [[ "$line" == https://huggingface.co/* ]]; then
        repo_id="${line#https://huggingface.co/}"
        if ! branches=$(bash "$LIST_CHECKPOINTS" "$line" "$CKPT_FLAG" "$CKPT_COUNT" 2>/dev/null); then
            echo "# WARNING: enumeration failed for $line (skipped)" >&2
            continue
        fi
        while IFS= read -r br; do
            [[ -z "$br" ]] && continue
            HF_ENTRIES+=("${base}|${repo_id}|${br}")
        done <<< "$branches"
    else
        echo "# WARNING: unrecognized model format, skipped: $line" >&2
    fi
done < "$MODELS_FILE"

# --- Emit the runner to stdout ---
echo "#!/bin/bash"
echo "#"
echo "# SNR stage runner - GENERATED; regenerate instead of hand-editing."
echo "# Source models: $MODELS_FILE"
echo "# Selection:     $CKPT_FLAG $CKPT_COUNT"
echo "# Invocation:    bash scripts/launch_evaluations.sh snr-<stage> --script <this file> [flags]"
echo "#"

if (( ${#MEGATRON_ENTRIES[@]} > 0 )); then
    echo ""
    echo "# ===== Megatron checkpoints (one sbatch per iter via MODEL_ITERATIONS) ====="
    echo "export TOKENIZER=\${TOKENIZER:-alehc/swissai-tokenizer}"
    echo "export BOS=\${BOS:-true}"
    echo "export APPLY_CHAT_TEMPLATE=\${APPLY_CHAT_TEMPLATE:-false}"
    echo "export LM_EVAL_BACKEND=\${LM_EVAL_BACKEND:-megatron_lm}"
    echo ""
    echo "declare -A MODEL_CHECKPOINTS=("
    for e in "${MEGATRON_ENTRIES[@]}"; do
        IFS='|' read -r base ckpt_dir it <<< "$e"
        printf '    ["%s-iter%s"]="%s"\n' "$base" "$it" "$ckpt_dir"
    done
    echo ")"
    echo ""
    echo "declare -A MODEL_ITERATIONS=("
    for e in "${MEGATRON_ENTRIES[@]}"; do
        IFS='|' read -r base _ it <<< "$e"
        printf '    ["%s-iter%s-iter"]="%s"\n' "$base" "$it" "$it"
    done
    echo ")"
    echo ""
    echo 'source runners/hf_base_runner.sh "SNR Megatron checkpoints"'
fi

if (( ${#HF_ENTRIES[@]} > 0 )); then
    echo ""
    echo "# ===== HuggingFace revisions (REVISION is singleton; loop one source per branch) ====="
    echo "unset MODEL_CHECKPOINTS MODEL_ITERATIONS"
    echo "export APPLY_CHAT_TEMPLATE=\${APPLY_CHAT_TEMPLATE:-false}"
    echo "export LM_EVAL_BACKEND=\${LM_EVAL_BACKEND:-vllm}"
    echo ""
    echo "HF_ENTRIES=("
    for e in "${HF_ENTRIES[@]}"; do
        printf '    "%s"\n' "$e"
    done
    echo ")"
    echo ""
    echo 'for ENTRY in "${HF_ENTRIES[@]}"; do'
    echo '    IFS="|" read -r NAME REPO BRANCH <<< "$ENTRY"'
    echo '    export REVISION="$BRANCH"'
    echo '    unset MODEL_CHECKPOINTS'
    echo '    declare -A MODEL_CHECKPOINTS=(["${NAME}-${BRANCH}"]="$REPO")'
    echo '    source runners/hf_base_runner.sh "SNR HuggingFace checkpoints"'
    echo 'done'
    echo 'unset REVISION MODEL_CHECKPOINTS'
fi

if (( ${#MEGATRON_ENTRIES[@]} == 0 && ${#HF_ENTRIES[@]} == 0 )); then
    echo ""
    echo "# WARNING: no checkpoints selected - is the models file empty or unreachable?" >&2
    echo "echo 'No checkpoints to evaluate.' >&2; exit 1"
fi
