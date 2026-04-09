```bash
bash scripts/launch_signal_to_ratio.sh \
  --models configs/signal_to_ratio/models_test_megatron.txt \
  --tasks configs/signal_to_ratio/tasks_test.txt \
  --last-n-checkpoints 1 \
  --splits 1 \
  --time 00:10:00 \
  --dry-run
```

```bash
bash scripts/launch_signal_to_ratio.sh \
  --models configs/signal_to_ratio/models_pretraining_custom.txt \
  --tasks configs/signal_to_ratio/tasks_pretraining.txt \
  --last-n-checkpoints 5 \
  --splits 4 \
  --time 02:00:00 \
  --dry-run
```
