# Signal-to-Ratio experiments

Evaluate a set of models at multiple checkpoints to study how benchmark signal
emerges across training.

## How it works (one-screen overview)

Two steps, cleanly separated:

1. **Generate** a stage runner from a models file. A runner is a plain bash file in
   `runners/` that declares `MODEL_CHECKPOINTS` and sources
   `runners/hf_base_runner.sh` — identical idiom to the existing
   `runners/hf_eval_multiple_*.sh` files. Megatron keys embed the iter as
   `<base>-iter<N>`; the shared runner parses it directly. Commit the runner so
   the evaluated-checkpoint snapshot is version-controlled.
2. **Launch** with the standard `launch_evaluations.sh --script …` entry point.
   The new `snr-pretraining` / `snr-midtraining` / `snr-posttraining` modes set
   `TASKS` to the right `tasks_*.txt` (pretraining and midtraining share
   `tasks_pretraining.txt` — both are base-model evaluations) and switch the
   W&B defaults to `mariagrandury-epflnlp/snr-experiments`. `snr-posttraining`
   additionally defaults `APPLY_CHAT_TEMPLATE=true` because the task set
   (ifeval, humaneval_instruct, …) requires a chat template on Instruct/SFT
   models; pass `--no-chat-template` to override.

Scripts:

| Script                                                                 | Purpose                                                                                                               |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| [scripts/list_checkpoints.sh](../../scripts/list_checkpoints.sh)       | Enumerate Megatron iters (multiples of 2000) or HF non-main branches for one model. Pure selection, prints to stdout. |
| [scripts/generate_snr_runner.sh](../../scripts/generate_snr_runner.sh) | Read a models file, enumerate each, emit a stage runner to stdout.                                                    |
| [scripts/launch_evaluations.sh](../../scripts/launch_evaluations.sh)   | Unchanged entry point. Use `--script runners/snr_<stage>.sh`.                                                         |

## Generate runners

On the cluster (so Megatron paths resolve):

```bash
cd /iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train && git pull

# Pretraining (Megatron) - N evenly-spaced iters across each run
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_pretraining_custom.txt --total 10 \
    > runners/snr_pretraining.sh

# Midtraining (HuggingFace) - evenly-spaced branches
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_midtraining_hf.txt --total 5 \
    > runners/snr_midtraining.sh

# Posttraining (HuggingFace) - last few branches (most repos have only 0-8)
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_posttraining_hf.txt --last 3 \
    > runners/snr_posttraining.sh
```

Selection flags:

- `--last N` — last N checkpoints (iter order / branch name order)
- `--total T` — T evenly-spaced checkpoints across all available

Regenerate whenever new checkpoints appear or the models file changes, then
commit. The runner's comment header records the exact regenerate command.

Inspect what's available before generating:

```bash
# Megatron iters (% 2000 == 0) under a checkpoints dir
bash scripts/list_checkpoints.sh /iopsstor/.../checkpoints --last 20

# HF non-main branches for a repo
bash scripts/list_checkpoints.sh https://huggingface.co/allenai/Olmo-3-7B-Instruct --last 999

# High-level status for every model in a file
bash configs/signal_to_ratio/check_checkpoints.sh configs/signal_to_ratio/models_pretraining_custom.txt
```

## Launch evaluations

```bash
# Pretraining
bash scripts/launch_evaluations.sh snr-pretraining \
    --script runners/snr_pretraining.sh --splits 10

# Midtraining
bash scripts/launch_evaluations.sh snr-midtraining \
    --script runners/snr_midtraining.sh --splits 5

# Posttraining
bash scripts/launch_evaluations.sh snr-posttraining \
    --script runners/snr_posttraining.sh --splits 2
```

Useful flags (all forwarded verbatim to the normal eval path — nothing SNR-specific):

- `--splits K` — split tasks across K parallel nodes per checkpoint (an aggregation job auto-chains with `afterok`).
- `--limit N` — cap samples per task (quick smoke tests).
- `--harness-branch B` — install `lm-evaluation-harness` from a specific branch.
- `--num-fewshot N` — override few-shot count.

Job time limit

```bash
export SLURM_TIME=02:00:00
bash scripts/launch_evaluations.sh snr-midtraining --script runners/snr_midtraining.sh --splits 5
```

### Smoke tests

Tiny end-to-end runs through the existing `single` mode (one or
comma-separated `--task` values, `--limit 4` to cap samples per task). The
`WANDB_*` prefix routes results into the SNR W&B family (`single` mode
appends `-single` to the project name, so they land in
`snr-experiments-single` and stay separate from real sweep data).

`runners/snr_test.sh` (committed) covers the simplest HF case; for the
others, generate a runner from `models_test_hf.txt` /
`models_test_megatron.txt` first.

```bash
WANDB_ENTITY=mariagrandury-epflnlp WANDB_PROJECT=snr-experiments

# 1) 1 HF checkpoint, 1 task — committed runner
bash scripts/launch_evaluations.sh single \
    --script runners/snr_test.sh --task hellaswag --limit 4

# 2) 2 HF checkpoints of the same model, 2 tasks
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_test_hf.txt --last 2 \
    > runners/snr_test_hf_2.sh
bash scripts/launch_evaluations.sh single \
    --script runners/snr_test_hf_2.sh --task hellaswag,piqa --limit 4

# 3) 1 Megatron checkpoint (last iter), 1 task
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_test_megatron.txt --last 1 \
    > runners/snr_test_meg_1.sh
bash scripts/launch_evaluations.sh single \
    --script runners/snr_test_meg_1.sh --task hellaswag --limit 4

# 4) 2 Megatron checkpoints (last two iters), 2 tasks
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_test_megatron.txt --last 2 \
    > runners/snr_test_meg_2.sh
bash scripts/launch_evaluations.sh single \
    --script runners/snr_test_meg_2.sh --task hellaswag,piqa --limit 4
```

`models_test_megatron.txt` already points at
`/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-350M-fwEdu60-fw240-seed1904/checkpoints/`,
so cases 3 and 4 just work. Sanity-check a generated runner once with
`bash -n runners/snr_test_meg_2.sh` before the first launch.

## Where outputs go

### Slurm stdout/stderr (per job)

`<repo>/logs/<job_name>_<job_id>.{out,err}`, written relative to the directory
`sbatch` was invoked from (see
[scripts/evaluate.sbatch:9](../../scripts/evaluate.sbatch#L9)).

Job name pattern:

- single-node: `eval-<model_name>`
- split (of K): `eval-<model_name>-split<i>` + `eval-<model_name>-aggregate`

Where `<model_name>` is:

- Megatron: `<base>-iter<N>` e.g. `apertus-175M-fwEdu60-fw240-seed28-iter10000`
- HuggingFace: `<base>-<branch>` e.g. `Apertus-8B-2509-step1750000-tokens7652B`

```bash
cd /iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train/logs/

# Tail the latest log across all eval jobs
ls -t eval-*.out | head -1 | xargs tail -100

# Tail a specific model
tail -100 eval-SmolLM3-3B-checkpoints-stage3-step-4720000_1831506.out

# Errors only
ls -t eval-*.err | head -1 | xargs tail -100
```

### Harness + W&B logs (per checkpoint)

`$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/<model_name>/` where defaults for SNR are:

- `LOGS_ROOT=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs`
  (see [scripts/evaluate.sbatch:69](../../scripts/evaluate.sbatch#L69))
- `WANDB_ENTITY=mariagrandury-epflnlp`
- `WANDB_PROJECT=snr-experiments`

```bash
cd /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/snr-experiments/
ls -t | head -20
```

### Weights & Biases

https://wandb.ai/mariagrandury-epflnlp/snr-experiments

## Debug playbook

Check running SNR jobs:

```bash
squeue --me --noheader -o "%.10i %.30j %.2t %.10M %.6D %R" | grep eval-
```

Tail the latest failing job:

```bash
cd /iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train/logs/
ls -t eval-*.err | head -1 | xargs tail -60
```

Sanity-check a generated runner before submitting:

```bash
bash -n runners/snr_posttraining.sh
cat runners/snr_posttraining.sh
```

Cancel all SNR jobs:

```bash
# Preview
squeue --me --noheader -o "%i %j" | grep eval-
# Cancel
squeue --me --noheader -o "%i %j" | grep eval- | awk '{print $1}' | xargs scancel
```

Re-run a single checkpoint without regenerating the runner (use the main
launcher's single-model path):

```bash
# Megatron
bash scripts/launch_evaluations.sh snr-pretraining \
    --model /iopsstor/.../apertus-175M-.../checkpoints \
    --megatron-iter 10000 --splits 2
```
