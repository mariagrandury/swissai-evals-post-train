#!/usr/bin/env python3
"""Build a HF dataset from local eval_logs and push it to epfl-nlp.

Mirrors the structure of allenai/signal-and-noise: a flat per-suite parquet
with one row per (model, model_revision, task), aggregate metrics only — no
per-instance predictions.

Splits emitted:
  - pretraining_custom: 12 apertus megatron checkpoints × N iters
  - reference_hf:       Apertus-8B/70B (main) + Olmo-3 stages + SmolLM3 stages

Usage:
  python scripts/build_hf_dataset.py --out-dir /tmp/snr-hf-dataset --dry-run
  python scripts/build_hf_dataset.py --push --repo-id epfl-nlp/multilingual-snr-eval-results
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Reuse logic from push_all_results.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_all_results import (  # type: ignore
    LOGS_BASE,
    MEG_TOKENS_PER_ITER,
    HF_STAGE_TOKENS,
    HF_MAIN_TOKENS,
    model_params,
    parse_name,
    collect,
)

ENTITY = "mariagrandury-epflnlp"
PROJECT = "snr-experiments"

# Tasks for which the result file uses these as the "primary" metric (in order).
PRIMARY_METRIC_PRIORITY = ["acc", "exact_match", "pass@1", "f1"]


# Fixed set of metric fields kept in the `metrics` struct. Tasks that don't
# report a given metric get `None`. This makes the parquet schema stable
# across splits so `datasets.load_dataset` can concatenate them.
METRIC_FIELDS = (
    "acc",
    "acc_stderr",
    "acc_norm",
    "acc_norm_stderr",
    "acc_bytes",
    "acc_bytes_stderr",
    "exact_match",
    "exact_match_stderr",
    "f1",
    "f1_stderr",
    "pass_at_1",
    "perplexity",
    "perplexity_stderr",
    "degeneration",
    "degeneration_stderr",
)


def normalize_metrics(task_metrics: dict) -> dict[str, float | None]:
    """{ 'acc,none': 0.5, 'acc_stderr,none': 0.01, 'alias': 'foo' } →
       { 'acc': 0.5, 'acc_stderr': 0.01, 'acc_norm': None, ... }.

    Returns a dict with exactly METRIC_FIELDS keys; missing values are None.
    Filter-qualified variants ('exact_match,strict-match' vs
    'exact_match,flexible-extract') collapse to the same bare name; the
    first encountered wins.
    """
    out: dict[str, float | None] = {k: None for k in METRIC_FIELDS}
    for key, val in task_metrics.items():
        if key == "alias" or not isinstance(val, (int, float)):
            continue
        metric = key.split(",", 1)[0].strip()
        # Map lm-eval names to our canonical set
        canonical = metric.replace("@", "_at_")
        if canonical in out and out[canonical] is None:
            out[canonical] = float(val)
    return out


def pick_primary(metrics: dict[str, float]) -> tuple[str | None, float | None]:
    """Choose the headline (primary_score, primary_metric) for a task."""
    for m in PRIMARY_METRIC_PRIORITY:
        if m in metrics:
            return m, metrics[m]
    return None, None


def parse_size_to_params(size: str) -> int | None:
    m = re.match(r"^(\d+)([MB])$", size, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)) * (10**6 if m.group(2).upper() == "M" else 10**9)


def name_to_metadata(name: str) -> dict | None:
    """Extract model metadata from a NAME (eval_logs subdir).

    Returns a dict with fields: model, model_revision, model_type, step,
    tokens, params, size, mix, seed, split. Returns None for unparseable
    names (debug artifacts etc.) so they're skipped.
    """
    # Apertus megatron: apertus-<size>-fwEdu<X>-fw<Y>-seed<S>-iter<N>
    m = re.match(
        r"^(?P<model>apertus-(?P<size>\d+[MB])-fwEdu(?P<edu>\d+)-fw(?P<fw>\d+)-seed(?P<seed>\d+))-iter(?P<n>\d+)$",
        name,
    )
    if m:
        n = int(m.group("n"))
        return {
            "name": name,
            "model": m.group("model"),
            "model_revision": f"iter{n}",
            "model_type": "custom_pretraining",
            "step": n,
            "tokens": float(n * MEG_TOKENS_PER_ITER),
            "params": parse_size_to_params(m.group("size")),
            "size": m.group("size"),
            "mix": f"fwEdu{m.group('edu')}-fw{m.group('fw')}",
            "seed": int(m.group("seed")),
            "split": "pretraining_custom",
        }

    # HF stage: <repo>-stage<K>-step-?<N>(-...)?
    m = re.match(r"^(?P<repo>.+?)-stage(?P<stage>\d+)-step-?(?P<n>\d+)(?:-.*)?$", name)
    if m:
        repo, stage = m.group("repo"), int(m.group("stage"))
        tokens = HF_STAGE_TOKENS.get((repo, stage))
        if tokens is None:
            return None
        size_match = re.search(r"-(\d+)B(?:-|$)", repo)
        params = int(size_match.group(1)) * 10**9 if size_match else None
        return {
            "name": name,
            "model": repo,
            "model_revision": f"stage{stage}-step{m.group('n')}",
            "model_type": "reference_hf",
            "step": int(m.group("n")),
            "tokens": float(tokens),
            "params": params,
            "size": f"{size_match.group(1)}B" if size_match else None,
            "mix": f"stage{stage}",
            "seed": None,
            "split": "reference_hf",
        }

    # HF main: <repo>-main
    m = re.match(r"^(?P<repo>.+?)-main$", name)
    if m:
        repo = m.group("repo")
        tokens = HF_MAIN_TOKENS.get(repo)
        if tokens is None:
            return None
        size_match = re.search(r"-(\d+)B(?:-|$)", repo)
        params = int(size_match.group(1)) * 10**9 if size_match else None
        return {
            "name": name,
            "model": repo,
            "model_revision": "main",
            "model_type": "reference_hf",
            "step": 0,
            "tokens": float(tokens),
            "params": params,
            "size": f"{size_match.group(1)}B" if size_match else None,
            "mix": "main",
            "seed": None,
            "split": "reference_hf",
        }

    return None


def collect_eval_metadata(name_dir: Path) -> dict:
    """Pull global eval-config from any one results file in the NAME's
    harness/. Used for fields like model_source, model_args, total time.
    """
    base = name_dir / "harness"
    out = {
        "model_source": None,
        "model_path": None,
        "model_args": None,
        "n_shot_per_task": {},
        "n_samples_per_task": {},
        "processing_time_per_task": {},
    }
    if not base.is_dir():
        return out

    candidates = list(base.glob("eval_*/results_*.json")) + list(
        base.glob("eval_*/per_task/*/*/results_*.json")
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if out["model_source"] is None:
            out["model_source"] = data.get("model_source")
        if out["model_args"] is None:
            ma = (data.get("config") or {}).get("model_args") or {}
            out["model_args"] = ma if isinstance(ma, dict) else None
            # `pretrained` for HF/vLLM, `load` for megatron
            if isinstance(ma, dict):
                out["model_path"] = ma.get("pretrained") or ma.get("load")
        for tk, n in (data.get("n-shot") or {}).items():
            out["n_shot_per_task"].setdefault(tk, n)
        for tk, info in (data.get("n-samples") or {}).items():
            if isinstance(info, dict):
                out["n_samples_per_task"].setdefault(tk, info.get("effective"))
        # per-task processing_time only available when per_task/<task>/<wid>/
        # results file was produced for a single task — we approximate by
        # parent dir name.
        if "/per_task/" in str(path):
            tk = path.parts[-3]  # eval_*/per_task/<task>/<wid>/results.json
            t = data.get("total_evaluation_time_seconds")
            if t is not None and tk not in out["processing_time_per_task"]:
                try:
                    out["processing_time_per_task"][tk] = float(t)
                except (TypeError, ValueError):
                    pass
    return out


def build_rows(project_dir: Path) -> list[dict]:
    """Walk every NAME under project_dir, emit one row per (NAME, task)."""
    rows: list[dict] = []
    skipped: list[str] = []

    for name_dir in sorted(project_dir.iterdir()):
        if not name_dir.is_dir():
            continue
        # Skip debug/test artifacts
        nm = name_dir.name
        if any(x in nm for x in ("salloc-debug", "salloc-test", "srun-debug")):
            continue

        meta = name_to_metadata(nm)
        if meta is None:
            skipped.append(nm)
            continue

        scores = collect(name_dir)  # union of all per-task results JSON
        if not scores:
            continue
        eval_meta = collect_eval_metadata(name_dir)

        flops = (6 * meta["params"] * meta["tokens"]) if meta["params"] else None

        for task, raw_metrics in scores.items():
            if not isinstance(raw_metrics, dict):
                continue
            metrics = normalize_metrics(raw_metrics)
            if not metrics:
                continue
            primary_metric, primary_score = pick_primary(metrics)

            rows.append({
                "split": meta["split"],
                "task": task,
                "name": meta["name"],
                "model": meta["model"],
                "model_revision": meta["model_revision"],
                "model_type": meta["model_type"],
                "model_path": eval_meta["model_path"],
                "model_source": eval_meta["model_source"],
                "model_params": float(meta["params"]) if meta["params"] else None,
                "model_tokens": meta["tokens"],
                "flops": float(flops) if flops is not None else None,
                "step": meta["step"],
                "size": meta["size"],
                "mix": meta["mix"],
                "seed": meta["seed"],
                "primary_score": primary_score,
                "primary_metric": primary_metric,
                "metrics": metrics,
                "num_instances": eval_meta["n_samples_per_task"].get(task),
                "num_fewshot": eval_meta["n_shot_per_task"].get(task),
                "processing_time": eval_meta["processing_time_per_task"].get(task),
                # Serialize as JSON: the keys differ between megatron_lm
                # and vllm backends so a struct schema can't be unified.
                "model_config": (
                    json.dumps(eval_meta["model_args"], sort_keys=True)
                    if eval_meta["model_args"] is not None
                    else None
                ),
            })

    if skipped:
        print(f"Skipped {len(skipped)} unparseable NAMEs (e.g. {skipped[:3]})", file=sys.stderr)
    return rows


def _arrow_schema():
    """Single canonical schema used for every split, so HF Datasets can
    concatenate splits without a cast error (unlike upstream, which
    sidesteps this by declaring per-split data_files only)."""
    import pyarrow as pa
    metric_struct = pa.struct([(k, pa.float64()) for k in METRIC_FIELDS])
    return pa.schema([
        ("task", pa.large_string()),
        ("name", pa.large_string()),
        ("model", pa.large_string()),
        ("model_revision", pa.large_string()),
        ("model_type", pa.large_string()),
        ("model_path", pa.large_string()),
        ("model_source", pa.large_string()),
        ("model_params", pa.float64()),
        ("model_tokens", pa.float64()),
        ("flops", pa.float64()),
        ("step", pa.int64()),
        ("size", pa.large_string()),
        ("mix", pa.large_string()),
        ("seed", pa.int64()),
        ("primary_score", pa.float64()),
        ("primary_metric", pa.large_string()),
        ("metrics", metric_struct),
        ("num_instances", pa.int64()),
        ("num_fewshot", pa.int64()),
        ("processing_time", pa.float64()),
        ("model_config", pa.large_string()),
    ])


def write_parquets(rows: list[dict], out_dir: Path) -> dict[str, Path]:
    """Group rows by 'split' and write data/<split>-00000-of-00001.parquet
    with a unified arrow schema across splits."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)

    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    schema = _arrow_schema()
    paths: dict[str, Path] = {}
    for split, split_rows in sorted(by_split.items()):
        df = pd.DataFrame(split_rows).drop(columns=["split"])
        df = df.sort_values(["model", "step", "task"]).reset_index(drop=True)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        path = data_dir / f"{split}-00000-of-00001.parquet"
        pq.write_table(table, path)
        paths[split] = path
        print(
            f"  wrote {path.relative_to(out_dir)}: {len(df)} rows, "
            f"{df['model'].nunique()} model(s), {df['task'].nunique()} task(s)"
        )
    return paths


README_TEMPLATE = """---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
  - multilingual
tags:
  - evaluation
  - language-models
  - swiss-ai
  - apertus
  - olmo
  - smollm
  - signal-to-noise
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: pretraining_custom
        path: data/pretraining_custom-*.parquet
      - split: reference_hf
        path: data/reference_hf-*.parquet
---

# SwissAI Evals — SNR Experiments

Aggregate evaluation results for the SwissAI signal-to-noise (SNR) study.
Results were produced by [swissai-evals-post-train](https://github.com/swiss-ai/swissai-evals-post-train)
on top of [lm-evaluation-harness](https://github.com/swiss-ai/lm-evaluation-harness).

The schema mirrors [allenai/signal-and-noise](https://huggingface.co/datasets/allenai/signal-and-noise):
one row per `(model, model_revision, task)` with aggregate metrics only — no
per-instance predictions.

## Splits

| Split | Models | Description |
|---|---|---|
| `pretraining_custom` | apertus-{{175M, 350M, 600M, 1B}}-fwEdu{{30,60,90}}-seed1904 | 12 custom megatron pretraining curves at canonical iters {{2k, 6k, 12k, 18k, 22k, 28k, 34k, 38k, 42k, 44k, 46k, 48k, 50k}} |
| `reference_hf` | Apertus-8B/70B-2509, Olmo-3-1025-7B (stages 1/2/3), SmolLM3-3B (stages 1/2/3) | External reference checkpoints |

## Columns

| Field | Type | Description |
|---|---|---|
| `task` | str | lm-eval-harness task name (e.g. `mmlu`, `belebele_eng_Latn`) |
| `name` | str | Full evaluation NAME — `<model>-<revision>` |
| `model` | str | Base model name |
| `model_revision` | str | `iter<N>`, `stage<K>-step<N>`, or `main` |
| `model_type` | str | `custom_pretraining` or `reference_hf` |
| `model_path` | str | Checkpoint path or HF repo id used for evaluation |
| `model_source` | str | lm-eval backend — `megatron_lm` or `vllm` |
| `model_params` | float | Approximate total parameter count |
| `model_tokens` | float | Cumulative training tokens at this checkpoint |
| `flops` | float | ≈ 6 × params × tokens |
| `step` | int | Training iteration / step |
| `size` | str | Parameter-count tag (e.g. `175M`, `8B`) |
| `mix` | str | Data-mix tag (e.g. `fwEdu30-fw270`, `stage2`, `main`) |
| `seed` | int | Pretraining seed (custom only) |
| `primary_score` | float | Headline metric — `acc` if available else `exact_match` |
| `primary_metric` | str | Name of the primary metric |
| `metrics` | dict | All numeric metric variants reported for the task |
| `num_instances` | int | Number of effective evaluation samples |
| `num_fewshot` | int | Few-shot count used for the task |
| `processing_time` | float | Per-task evaluation time in seconds (when known) |
| `model_config` | dict | lm-eval `--model_args` |

## Loading

```python
from datasets import load_dataset

ds = load_dataset("epfl-nlp/multilingual-snr-eval-results")
df = ds["pretraining_custom"].to_pandas()
```

## Citation

If you use this data, please cite the SwissAI Apertus technical report.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--out-dir", default="/tmp/snr-hf-dataset",
                   help="Local staging directory for parquet + README")
    p.add_argument("--push", action="store_true", help="After writing, upload to HF")
    p.add_argument("--repo-id", default="epfl-nlp/multilingual-snr-eval-results",
                   help="HF dataset repo (created if missing)")
    p.add_argument("--private", action="store_true", help="Create private dataset")
    args = p.parse_args()

    project_dir = LOGS_BASE / args.entity / args.project
    if not project_dir.is_dir():
        sys.exit(f"No project dir at {project_dir}")

    print(f"Walking {project_dir}")
    rows = build_rows(project_dir)
    print(f"Built {len(rows)} rows.")
    if not rows:
        sys.exit("No rows built — nothing to push.")

    out_dir = Path(args.out_dir)
    paths = write_parquets(rows, out_dir)
    (out_dir / "README.md").write_text(README_TEMPLATE)
    print(f"Wrote README to {out_dir}/README.md")

    if not args.push:
        print("\n(--push not set; skipping HF upload)")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    print(f"\nCreating/updating dataset repo {args.repo_id}")
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Initial upload: SwissAI SNR evaluation results",
    )
    print(f"\nDone — https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
