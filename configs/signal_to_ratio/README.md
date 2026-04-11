# Signal to ratio experiments

## Launch jobs

```bash
cd /iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train/ &&
bash scripts/launch_signal_to_ratio.sh \
  --models configs/signal_to_ratio/models_test_megatron.txt \
  --tasks configs/signal_to_ratio/tasks_test.txt \
  --last-n-checkpoints 1 \
  --splits 1 \
  --time 00:10:00 \
  --dry-run
```

```bash
cd /iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train/ &&
bash scripts/launch_signal_to_ratio.sh \
  --models configs/signal_to_ratio/models_pretraining_custom.txt \
  --tasks configs/signal_to_ratio/tasks_pretraining.txt \
  --last-n-checkpoints 1 \
  --splits 10 \
  --time 00:20:00 \
  --dry-run
```

```bash
cd /iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train/ &&
bash scripts/launch_signal_to_ratio.sh \
  --models configs/signal_to_ratio/models_pretraining_custom.txt \
  --tasks configs/signal_to_ratio/tasks_pretraining.txt \
  --last-n-checkpoints 5 \
  --splits 4 \
  --time 02:00:00 \
  --dry-run
```

## Cancel jobs

Preview:

```bash
squeue --me --noheader -o "%i %j" | grep eval
```

Cancel:

```bash
squeue --me --noheader -o "%i %j" | grep eval | awk '{print $1}' | xargs scancel
```

## Check eval log

```bash
cd /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/
```
