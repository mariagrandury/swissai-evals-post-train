#!/bin/bash

# list_checkpoints.sh - Enumerate checkpoints for a model and print them to stdout.
#
# Pure selection logic, no evaluation side-effects. Redirect the output into a
# file that the SNR sweep orchestrator (launch_snr_sweep.sh) consumes.
#
# For Megatron checkpoint roots (paths under /iopsstor or /capstor, e.g. a training
# run's `.../checkpoints` directory), prints iteration numbers (multiples of 2000)
# one per line. For HuggingFace model URLs (https://huggingface.co/<org>/<repo>),
# prints non-main branch names one per line.
#
# Usage:
#   bash scripts/list_checkpoints.sh <model> --last N
#   bash scripts/list_checkpoints.sh <model> --total T
#
# Examples:
#   bash scripts/list_checkpoints.sh /iopsstor/.../checkpoints --last 5
#   bash scripts/list_checkpoints.sh https://huggingface.co/org/model --total 4

set -euo pipefail

MODEL=""
MODE=""
COUNT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --last)  MODE="last";  COUNT="$2"; shift 2 ;;
        --total) MODE="total"; COUNT="$2"; shift 2 ;;
        -*)
            echo "Error: unknown option '$1'" >&2
            exit 1
            ;;
        *)
            if [[ -z "$MODEL" ]]; then MODEL="$1"; else
                echo "Error: unexpected positional argument '$1'" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$MODEL" || -z "$MODE" ]]; then
    echo "Usage: bash scripts/list_checkpoints.sh <model> --last N | --total T" >&2
    exit 1
fi
if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || (( COUNT < 1 )); then
    echo "Error: count must be an integer >= 1 (got '$COUNT')" >&2
    exit 1
fi

# --- Megatron: list iter_* directories, multiples of 2000 only ---
if [[ "$MODEL" == /iopsstor* || "$MODEL" == /capstor* ]]; then
    CKPT_DIR="${MODEL%/}"
    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "Error: directory not found: $CKPT_DIR" >&2
        exit 2
    fi
    shopt -s nullglob
    ITER_DIRS=( "${CKPT_DIR}"/iter_* )
    shopt -u nullglob
    if (( ${#ITER_DIRS[@]} == 0 )); then
        echo "Error: no iter_* checkpoints found in $CKPT_DIR" >&2
        exit 2
    fi
    ALL_ITERS=$(printf '%s\n' "${ITER_DIRS[@]##*/iter_}" \
        | sed 's/^0*//' \
        | sort -n \
        | awk '$1 % 2000 == 0')
    if [[ -z "$ALL_ITERS" ]]; then
        echo "Error: no iter_* checkpoints with iter % 2000 == 0 in $CKPT_DIR" >&2
        exit 2
    fi
    if [[ "$MODE" == "last" ]]; then
        echo "$ALL_ITERS" | tail -n "$COUNT"
    else
        echo "$ALL_ITERS" | awk -v t="$COUNT" '
            { lines[NR] = $0 }
            END {
                n = NR
                if (t >= n) { for (i = 1; i <= n; i++) print lines[i]; exit }
                for (i = 0; i < t; i++) {
                    idx = int(i * (n - 1) / (t - 1)) + 1
                    print lines[idx]
                }
            }'
    fi
    exit 0
fi

# --- HuggingFace: list non-main branches via the public API ---
if [[ "$MODEL" == https://huggingface.co/* ]]; then
    REPO_ID="${MODEL#https://huggingface.co/}"
    python3 -c "
import json, sys, urllib.request
repo_id, mode, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
url = f'https://huggingface.co/api/models/{repo_id}/refs'
try:
    data = json.loads(urllib.request.urlopen(url).read())
except Exception as e:
    print(f'Error fetching branches for {repo_id}: {e}', file=sys.stderr); sys.exit(2)
branches = sorted(b['name'] for b in data.get('branches', []) if b['name'] != 'main')
if not branches:
    print(f'Error: no non-main branches found for {repo_id}', file=sys.stderr); sys.exit(2)
if mode == 'last':
    selected = branches[-count:]
else:
    n = len(branches)
    selected = branches if count >= n else [branches[round(i * (n - 1) / (count - 1))] for i in range(count)]
for b in selected:
    print(b)
" "$REPO_ID" "$MODE" "$COUNT"
    exit 0
fi

echo "Error: unrecognized model format '$MODEL' (expected /iopsstor*, /capstor*, or https://huggingface.co/...)" >&2
exit 1
