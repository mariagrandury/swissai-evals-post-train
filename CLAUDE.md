# Context for Claude — SNR (signal-to-ratio) experiments

This repo evaluates language models at many training checkpoints to study how
benchmark signal emerges over the course of pretraining. Most of the action is
under [configs/signal_to_ratio/](configs/signal_to_ratio/) and [runners/](runners/).

If you're picking this up cold, **read [configs/signal_to_ratio/README.md](configs/signal_to_ratio/README.md) first**.
This file is the back-of-house Claude memo: the gotchas and the why. The README is
the user-facing how.

---

## What's actually running

Two parallel tracks at any given time:

1. **Eval submissions on the cluster** (this repo). Slurm jobs invoke
   [`scripts/evaluate.sbatch`](scripts/evaluate.sbatch) which calls
   [`scripts/_run_per_task.sh`](scripts/_run_per_task.sh) inside an enroot/pyxis
   container, runs `lm_eval` once per task, merges results, uploads to W&B.

2. **Pretraining of half-finished custom models** (separate repo at
   `/iopsstor/scratch/cscs/mariagrandury/pretrain/megatron/data-mix-small`).
   Five resume jobs were submitted Apr 26 (Slurm ids 1947226–1947230) to take
   `apertus-175M-fwEdu{30,60,90}-seed1904`, `apertus-600M-fwEdu90-fw210-seed1904`
   and `apertus-1B-fwEdu90-fw210-seed1904` from their early-stop iters back up
   to iter 50000.

Status snapshot anytime:

```bash
python3.11 scripts/snr_progress.py        # per-ckpt progress + pending job IDs
squeue --me -o "%.10i %.40j %.2t %.10M %.10L %.10P"
sacct -u $USER -S <since-date> -X -o "JobID,JobName%50,State,Elapsed"
```

---

## Models in scope

### Custom megatron checkpoints (12 models, file [models_pretraining_custom_all.txt](configs/signal_to_ratio/models_pretraining_custom_all.txt))

```
apertus-{175M,350M,600M,1B}-fwEdu{30,60,90}-fw{270,240,210}-seed1904
```

**Canonical iter set** (matches what fully-trained 350M/600M/1B-fwEdu30/60 saved):

```
2000, 6000, 12000, 18000, 22000, 28000, 34000, 38000, 44000, 50000
```

The runner [`runners/snr_pretraining_all.sh`](runners/snr_pretraining_all.sh) is the
single-source-of-truth combo: 12 models × 10 iters = 120 cells.

**Half-trained models (don't yet have all canonical iters on disk):**

| Model | Last iter saved | Missing from canonical set |
|---|---:|---|
| 175M-fwEdu30/60/90 (×3) | 49180/48351/48948 | only 50000 |
| 600M-fwEdu90 | 26000 | 28000, 34000, 38000, 44000, 50000 |
| 1B-fwEdu90 | 14000 | 18000, 22000, 28000, 34000, 38000, 44000, 50000 |

Resume training jobs (1947226–1947230) will fill these in. Until they do,
eval jobs targeting the missing iters will fail at Megatron's
`--exit-on-missing-checkpoint`. That's expected — the per-task idempotency
layer just reschedules them next time.

### Reference HF models (4 models, two runners)

[`runners/snr_pretraining_hf_top.sh`](runners/snr_pretraining_hf_top.sh) — 6 jobs:

| Model | Revision |
|---|---|
| HuggingFaceTB/SmolLM3-3B-checkpoints | stage1-step-3440000, stage2-step-4200000, stage3-step-4720000 |
| swiss-ai/Apertus-8B-2509 | main |
| allenai/Olmo-3-1025-7B | stage2-step47684-mix-round5-from-2T-ckpt, stage3-step11921 |

[`scripts/debug_loop.sh`](scripts/debug_loop.sh) is a single-ckpt
debug-partition loop: it keeps resubmitting 1:30 jobs (narrowing each
submission's `TASKS` env to the still-missing set) until every task in
`tasks_pretraining_full.txt` has results, or until two consecutive cycles
make zero progress. Edit the `NAME`/`CKPT_PATH`/`CKPT_ITER` (or the
HF `MODEL`/`REVISION`, switching `LM_EVAL_BACKEND` back to `vllm`) at the
top of the script to retarget. It currently points at
`apertus-350M-fwEdu30-fw270-seed1904-iter50000`. If you also have it
covering a ckpt that's in one of the top runners, you'll duplicate work —
prune one or the other.

[`runners/snr_pretraining_hf_70b.sh`](runners/snr_pretraining_hf_70b.sh) — 1 job:

| Model | Revision |
|---|---|
| swiss-ai/Apertus-70B-2509 | main |

Apertus 8B/70B intentionally use `main` (HF latest) because the user wants the
shipped model rather than a specific intermediate step. Never change to
`step<N>-tokens<M>B` without confirming.

---

## Task list

[`tasks_pretraining_full.txt`](configs/signal_to_ratio/tasks_pretraining_full.txt)
— 86 tasks, the dedup union of `tasks_pretraining.txt` (48) and
`tasks_pretraining_b.txt` (38). The launcher mode `snr-pretraining-full` sets
this. **Always launch with `snr-pretraining-full`** unless the user explicitly
wants a subset.

If a task fails, the per-task fault-tolerance in `_run_per_task.sh` logs them and moves on.

---

## The one-liner workflow (idempotent, multi-collaborator-safe)

```bash
cd /iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train && git pull && \
bash scripts/launch_evaluations.sh snr-pretraining-full \
    --script runners/snr_pretraining_all.sh --time 12:00:00
```

Plus separately for the HF top-of-stage and 70B (which need longer wall times):

```bash
bash scripts/launch_evaluations.sh snr-pretraining-full \
    --script runners/snr_pretraining_hf_top.sh --time 12:00:00
bash scripts/launch_evaluations.sh snr-pretraining-full \
    --script runners/snr_pretraining_hf_70b.sh --time 12:00:00
```

Re-running these is **safe and idempotent at two layers**:

| Layer | Where | When it fires |
|---|---|---|
| Per-checkpoint | [`runners/hf_base_runner.sh`](runners/hf_base_runner.sh) (calls `_eval_status.py`) | At launch, before each `sbatch` — `continue`s if every task in `$TASKS` already has results for that ckpt. Saves the cold-start of a redundant job. |
| Per-task | [`scripts/_run_per_task.sh`](scripts/_run_per_task.sh) (calls `_eval_status.py`) | Inside a running job — filters `$TASKS` down to remaining; logs skipped to `skipped_tasks.log`; exits 0 cleanly if nothing left. |

Both use the same disk-scan in [`scripts/_eval_status.py`](scripts/_eval_status.py): a
task is "done" iff a non-empty `eval_*/per_task/<task>/` exists (killed runs)
**or** any `eval_*/results_*.json` lists it under `.results` (clean runs).

**Race window:** if two collaborators launch simultaneously, both see the same
"missing" set and may submit duplicate sbatch jobs for the same ckpt. The per-
task layer dedups inside the second job — no compute waste, just queue churn.
Stagger by a few minutes when you can.

---

## Collaborators with shared access

Both `aromanou` and `cmeister747` have POSIX ACLs granting `rwx` on this repo
and `eval_logs` (recursive + default), plus `r-x` traverse on the parent dirs.
Their jobs write to the **same** `eval_logs` tree, so the idempotency check
sees their results too. They share `--account=infra01`.

To grant more colleagues:

```bash
USER_TO_ADD=...
setfacl -m u:$USER_TO_ADD:rx /iopsstor/scratch/cscs/mariagrandury \
    /iopsstor/scratch/cscs/mariagrandury/data-mix-small \
    /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM \
    /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs
setfacl -R -m u:$USER_TO_ADD:rwx /iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train
setfacl -R -d -m u:$USER_TO_ADD:rwx /iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train
setfacl -R -m u:$USER_TO_ADD:rwx /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs
setfacl -R -d -m u:$USER_TO_ADD:rwx /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs
```

Each colleague needs their **own** `HF_TOKEN` and `WANDB_API_KEY` exported
(`evaluate.sbatch` reads env first, files in `scripts/` as fallback). Their
W&B key needs membership in `mariagrandury-epflnlp` entity.

```bash
export HF_TOKEN=<your_hf_token> WANDB_API_KEY=<your_wandb_key> && cd /iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train && git pull && bash scripts/launch_evaluations.sh snr-pretraining-full --script runners/snr_pretraining_all.sh --time 12:00:00
```

---

## Cluster gotchas

- **Login node:** `clariden-ln003`. System Python is **3.6** — anything in
  `scripts/snr_progress.py` or `_eval_status.py` that uses 3.7+ syntax must be
  invoked via `python3.11`.
- **Slurm accounts:** `infra01` (default, in `evaluate.sbatch`) and `a139`.
  Group `a139` (NOT `infra01`) is what the unix files belong to.
- **Partition `normal`:** 12 h max wall, hundreds of nodes, but currently
  fully booked (504 reserved for some other project). Fair-share is the
  scheduling lever. Recent heavy usage (mariagrandury > 3× their share) drops
  priority; expect long pending times.
- **Partition `debug`:** 1:30 max wall, 12-ish idle nodes, but
  **`debug-qos` allows max 1 running + 1 queued per user**. `debug_loop.sh`
  exploits exactly that: it submits, blocks, submits the next.
- **`#SBATCH --gpus-per-node=4` is hardcoded** in `evaluate.sbatch`. The 4-GPU
  node is what the eval container expects.
- **No outbound internet from compute nodes.** Pip installs run via the
  cluster's mirror; W&B uploads work because they go through a pre-configured
  proxy.
- **`/iopsstor/scratch` is the work tree.** `/capstor/store` is shared
  long-term storage (read-mostly).

---

## Hard-won bug history (don't reintroduce)

If a future Claude is editing `evaluate.sbatch` / `_run_per_task.sh` /
`_eval_status.py`, these are the things that broke before. Each one cost
hours to track down:

### 1. vLLM tokenizer revision
Some HF repos (notably `HuggingFaceTB/SmolLM3-3B-checkpoints`) leave `main`
empty and put all files on revision branches. lm-eval's vLLM wrapper has a
**separate** `tokenizer_revision` parameter that defaults to `None` — passing
just `revision=...` makes `AutoTokenizer.from_pretrained` fetch `main`, find
no `tokenizer.json`, and die with a misleading "need sentencepiece or
tiktoken" error. Fix: always pin `tokenizer_revision=$REVISION` next to
`revision=$REVISION` (`evaluate.sbatch` already does this).

### 2. Megatron container vs nvidia-modelopt/peft tf-plugin
The `ngc-nemo` container ships `nvidia-modelopt` and PEFT registers
transformers entry-points declaring the deprecated `tf` / `tensorflow_text`
backends. transformers 5.1.0's `BACKENDS_MAPPING` no longer includes those, so
**`import transformers` fatally raises** `ValueError: Backend should be
defined in the BACKENDS_MAPPING. Offending backend: tf` — not just a warning.
Fix: route the megatron path through `containers/env_vllm.toml` (the original
fork's container) instead. **Don't switch back to ngc-nemo** without solving
the plugin discovery problem.

### 3. Megatron checkpoint `_extra_state` strictness
Older megatron checkpoints (everything pre-resume) don't carry the TE
bookkeeping fields that current Megatron-LM expects. The default
`--dist-ckpt-strictness=assume_ok_unexpected` delegates to the underlying
strategy, which fatally raises `Missing key in checkpoint state_dict:
decoder.layers.self_attention.q_layernorm._extra_state/...`. Fix: pass
`extra_args=--dist-ckpt-strictness=log_unexpected` in the megatron model_args
(see `evaluate.sbatch`). Real model weights still load — only TE quantization
metadata is skipped, which is irrelevant for bf16 eval.

The same fix is in the **training** repo's
`pretrain/megatron/data-mix-small/submit-apertus-data-mix.sh` (we patched it
when launching the resume jobs). If you regenerate that file from upstream,
re-apply.

### 4. `bash ./scripts/_run_per_task.sh` resolution
The pyxis container's WORKDIR is `/workspace/vllm`, **not** the repo root. A
relative path to `_run_per_task.sh` won't resolve. Fix: capture
`REPO_DIR=$PWD` on the host shell and emit `cd '$REPO_DIR'` as the first line
of every inner `bash -lc` heredoc.

### 5. Pyxis `--environment=` doesn't forward host env vars
`srun --environment=...` runs the command in the container — and contrary to
naive expectation, host env vars exported on the outer shell are **not**
visible inside. `evaluate.sbatch` builds a single `INNER_EXPORTS` string from
the host shell's values (single-quoted) and inlines it into every inner
`bash -lc` heredoc:

```bash
INNER_EXPORTS="export NAME='$NAME' WANDB_ENTITY='$WANDB_ENTITY' WANDB_PROJECT='$WANDB_PROJECT' \
    LOGS_ROOT='$LOGS_ROOT' TASKS='$TASKS' HARNESS_EVAL_DIR='$HARNESS_EVAL_DIR' CMD_BASE='$CMD_BASE'"
```

If you add a new variable that the runner needs, add it here too.

### 6. `Path("comma,list").is_file()` raises ENAMETOOLONG
`debug_loop.sh` builds the next job's `TASKS` env from the remaining-task set —
68+ task names ≈ 1500 chars. `_eval_status.py:parse_tasks_input` was probing
`Path(tasks_arg).is_file()`, which calls `os.stat()`, which on Linux with
`NAME_MAX=255` raises `OSError(36)` for any string that long. Fix: short-
circuit the file probe when the input contains `,` or exceeds 4000 chars, and
catch `OSError` defensively.

### 7. `|| true` masking a filter crash
`_run_per_task.sh` used to do `REMAINING=$(python _eval_status.py ... || true)`
and then `if [[ -z "$REMAINING" ]]; then exit 0; fi`. When `_eval_status.py`
crashed (rc≥2), the `|| true` swallowed it, `REMAINING` was empty, the script
took the "all done" branch and exited 0. **Then `evaluate.sbatch` ran the W&B
upload step unconditionally** and died with `Expected exactly one results
file, found 0`, ending in Slurm state `FAILED`. Fix: capture the rc; rc=1
means "all done", rc≥2 means crash → fall through and run all tasks
unfiltered. **Never silently equate "filter failure" with "nothing to do".**

### 8. Olmo `stage2-step47684-mix-round5-from-2T-ckpt` revision: unrecognised architecture
The Olmo-3-1025-7B repo's stage2 mid-round revision uses model_type
`olmo2-retrofit`, which is not in transformers 5.1.0's model registry. vLLM
fails inside `ModelConfig.__init__` with:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, The checkpoint you are trying to load has model type
  `olmo2-retrofit` but Transformers does not recognize this architecture.
```

This is **not** caught usefully by the per-task fault tolerance — every
task fails identically (rc=1) for ~12 h until Slurm wall, then the W&B upload
step finds 0 results and the job ends FAILED. Symptom: 86 / 86
`=== FAILED (rc=1)` markers in the log with the same `ValidationError` block
before each. First observed on job 1947613.

Workarounds: skip that revision, install `transformers` from a branch that
ships the `olmo2-retrofit` mapping, or pin the eval container's transformers
to a version that supports it. The non-mid-round Olmo revisions
(`stage1-step1413814`, `stage3-step11921`) and the `main` revision load fine
— the issue is specific to the `mix-round5-from-2T-ckpt` retrofit checkpoints.

---

## Smoke tests (when you change `evaluate.sbatch` / `_run_per_task.sh`)

Use the four committed smoke tests in
[`configs/signal_to_ratio/README.md`](configs/signal_to_ratio/README.md#smoke-tests).
They run through the `single` mode with `--limit 4`, route results to the
`-single` W&B project to keep them out of the real sweep, and exercise both
the HF (vLLM) and Megatron paths. Re-run them before any merge that touches
`evaluate.sbatch`.

---

## Progress dashboard

[`scripts/snr_progress.py`](scripts/snr_progress.py) is the truth source:

```bash
python3.11 scripts/snr_progress.py                                    # per-ckpt summary
python3.11 scripts/snr_progress.py --status not_submitted             # gaps
python3.11 scripts/snr_progress.py --details --filter <NAME-substr>   # per-task
python3.11 scripts/snr_progress.py --models configs/signal_to_ratio/models_pretraining_custom_all.txt
```

It enumerates the same `(model, ckpt)` cells that the runners would and
cross-references `eval_logs` for completion + `squeue --me` for in-flight
jobs (matching by `eval-<NAME>-*` prefix so suffixes like `-b`, `-srun-debug`
don't break recognition).

---

## Rescue procedure (when a job hits Slurm wall before merging)

`_run_per_task.sh` writes per-task results to `eval_*/per_task/<task>/` **as
each task finishes**. If Slurm kills the job mid-loop, only the in-progress
task is lost; everything already finished survives on disk. To merge survivors
+ upload to W&B without resubmitting:

```bash
RUN=mariagrandury-epflnlp/snr-experiments
NAME=apertus-350M-fwEdu60-fw240-seed1904-iter50000
EVAL_DIR=$(ls -td /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/$RUN/$NAME/harness/eval_*_<jobid> | head -1)

cd /iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train
python -m scripts.alignment.merge_split_results \
    --split_dirs "$EVAL_DIR"/per_task/*/ --output_dir "$EVAL_DIR"
python -m scripts.alignment.update_wandb_alignment \
    --entity mariagrandury-epflnlp --project snr-experiments \
    --logs_root "$EVAL_DIR" --name "$NAME" --main_metrics "" --eval_duration 0
```

Same procedure for `debug_loop.sh` (it produces multiple `eval_*` dirs over
its lifetime — each is a partial run; the next launch's idempotency check
sees their per-task subdirs as "done").

---

## Outsourcing to Colab

[`notebooks/snr_eval_colab.ipynb`](notebooks/snr_eval_colab.ipynb) mirrors the
cluster pipeline on a single A100. It writes results in the same directory
shape so an `rsync` back to the cluster's `eval_logs` triggers cluster-side
idempotency (skips Colab-evaluated tasks). 70B doesn't fit on A100 — keep it
on the cluster.

---

## Pretraining infrastructure (separate repo)

`/iopsstor/scratch/cscs/mariagrandury/pretrain/megatron/data-mix-small/` is
the training submitter. Notable:

- `submit-apertus-data-mix.sh` — the sbatch script. Patched with
  `--dist-ckpt-strictness log_unexpected` (Apr 26) so resumes tolerate older
  checkpoints' missing TE keys. **Don't revert** without solving the
  underlying TE/Megatron version skew.
- `launch_trainings.py` — wrapper that builds the `--export=...` string from
  `hyperparams_deep.json`. Its default `SEEDS = [28, 64, 1797]` doesn't include
  seed `1904`, which is what the SNR custom models use. Pass `--seed 1904`
  explicitly. For non-default `--time`, bypass the launcher and call `sbatch`
  directly (see commit history of `submit-apertus-data-mix.sh` for the
  pattern).

The pretraining checkpoint dir lives at
`/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/<EXP_NAME>/checkpoints/`.
EXP_NAME format: `apertus-${MODEL_SIZE}-fwEdu${FW_EDU_RATIO}-fw2${FW2_RATIO}-seed${SEED}`.

---

## Key recent commits (for context when reviewing changes)

```
46b3852 fix three bugs that surfaced once real eval jobs started running
2cc58cf make SNR eval launches idempotent + add progress dashboard + canonical runner
aec82e0 fix smoke tests 1-4: vllm tokenizer_revision + megatron container/cwd/strictness
```

Read those commit messages — they document the failure→fix story end to end.
