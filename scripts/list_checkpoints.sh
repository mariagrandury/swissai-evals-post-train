#!/bin/bash
# Enumerate Megatron iters (multiples of 2000) or HF non-main branches for one model.
# Usage: list_checkpoints.sh <model> [--include SUBSTR] (--last N | --total T)
set -euo pipefail

MODEL="${1:-}"; shift || true
INCLUDE=""
if [[ "${1:-}" == "--include" ]]; then INCLUDE="$2"; shift 2; fi
[[ -n "$MODEL" && $# -eq 2 && "$1" =~ ^--(last|total)$ && "$2" =~ ^[1-9][0-9]*$ ]] \
    || { echo "Usage: $0 <model> [--include SUBSTR] (--last N | --total T)" >&2; exit 1; }
MODE="${1#--}"; COUNT="$2"

filter_include() { [[ -z "$INCLUDE" ]] && cat || grep -F -- "$INCLUDE"; }

# From stdin, print last COUNT lines (mode=last) or COUNT evenly-spaced (mode=total).
select_lines() {
    awk -v m="$MODE" -v c="$COUNT" '
        {a[NR]=$0}
        END {
            n=NR
            if (n==0) { print "no candidates to select from" > "/dev/stderr"; exit 2 }
            if (m=="last") for (i=(n>c?n-c+1:1); i<=n; i++) print a[i]
            else if (c>=n) for (i=1; i<=n; i++) print a[i]
            else if (c==1) print a[n]
            else for (i=0; i<c; i++) print a[int(i*(n-1)/(c-1))+1]
        }'
}

case "$MODEL" in
    /iopsstor*|/capstor*)
        [[ -d "$MODEL" ]] || { echo "Not a directory: $MODEL" >&2; exit 2; }
        shopt -s nullglob; D=( "${MODEL%/}"/iter_* ); shopt -u nullglob
        (( ${#D[@]} )) || { echo "No iter_* in $MODEL" >&2; exit 2; }
        printf '%s\n' "${D[@]##*/iter_}" | sed 's/^0*//' | sort -n \
            | awk '$1 % 2000 == 0' | filter_include | select_lines
        ;;
    https://huggingface.co/*)
        python3 - "${MODEL#https://huggingface.co/}" <<'PY' | filter_include | select_lines
import json, sys, urllib.request
r = json.loads(urllib.request.urlopen(f'https://huggingface.co/api/models/{sys.argv[1]}/refs').read())
for b in sorted(x['name'] for x in r['branches'] if x['name']!='main'): print(b)
PY
        ;;
    *) echo "Unrecognized format: $MODEL (want /iopsstor*, /capstor*, or https://huggingface.co/...)" >&2; exit 1 ;;
esac
