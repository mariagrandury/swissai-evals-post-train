#!/usr/bin/env python3
"""Collect eval results on disk and push one W&B run per *model*.

Groups every NAME under <eval_logs>/<entity>/<project>/ by its model
(stripping `-iter<N>`, `-step<N>`, `-stage<K>-step-?<N>`, `-main`, etc.),
then pushes one resumeable W&B run per model with a step series:

  for each ckpt: run.log({"iter": <N>, "<task>/<metric>": val, ...})

W&B's default per-metric chart renders `<task>/<metric>` against `iter`
(set as step_metric for `*`), so each model becomes one line on each
benchmark plot. Later we'll swap `iter` for tokens consumed; for now
the integer step is enough to merge a model's checkpoints into one curve.

Idempotent — W&B run id is `<model>` (sanitised), so re-runs resume the
same run and accumulate new ckpts. Re-logging the same iter creates a
duplicate point at the same x value (charts overlay them).

Two modes:

  Bulk rescue (login node, `snr` conda env has wandb):
    python scripts/push_all_results.py [--dry-run] [--filter REGEX]

  Single-NAME (called from evaluate.sbatch after each successful eval —
  runs in the pyxis container, which has internet via proxy):
    python scripts/push_all_results.py --name <NAME>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

LOGS_BASE = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs"
)
ENTITY = "mariagrandury-epflnlp"
PROJECT = "snr-experiments"


def parse_name(name: str) -> tuple[str, int] | None:
    """NAME → (model, step). Returns None if unparseable.

    - apertus-<size>-fwEdu<X>-fw<Y>-seed<S>-iter<N> → (apertus-..., N)
    - <repo>-stage<K>-step-?<N>                     → (<repo>-stage<K>, N)
    - <repo>-step<N>-tokens<M>(B|T)                 → (<repo>, N)
    - <repo>-main                                   → (<repo>, 0)
    """
    m = re.match(r"^(?P<model>apertus-\d+[MB]-fwEdu\d+-fw\d+-seed\d+)-iter(?P<n>\d+)$", name)
    if m:
        return m.group("model"), int(m.group("n"))
    m = re.match(r"^(?P<repo>.+?)-stage(?P<stage>\d+)-step-?(?P<n>\d+)(?:-.*)?$", name)
    if m:
        return f"{m.group('repo')}-stage{m.group('stage')}", int(m.group("n"))
    m = re.match(r"^(?P<repo>.+?)-step(?P<n>\d+)-tokens[\d.]+[BT]$", name)
    if m:
        return m.group("repo"), int(m.group("n"))
    m = re.match(r"^(?P<repo>.+?)-main$", name)
    if m:
        return m.group("repo"), 0
    return None


def collect(name_dir: Path) -> dict[str, dict]:
    """Union of every results_*.json under harness/. Merged files take
    precedence over per-task partials for the same task name."""
    scores: dict[str, dict] = {}
    base = name_dir / "harness"
    if not base.is_dir():
        return scores
    for f in sorted(base.glob("eval_*/results_*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for k, v in (data.get("results") or {}).items():
            if isinstance(v, dict):
                scores[k] = v
    for f in sorted(base.glob("eval_*/per_task/*/*/results_*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for k, v in (data.get("results") or {}).items():
            if isinstance(v, dict):
                scores.setdefault(k, v)
    return scores


def flatten(scores: dict[str, dict]) -> dict[str, float]:
    """{task: {'metric,filter': val}} → {'task/metric': val}.

    Per task: if `acc` is present, push *only* `<task>/acc`. Otherwise push
    every numeric metric for the task (collapsing the `,none` filter suffix
    to a bare name; keeping real filters in the key, e.g. `acc,strict-match`).
    """
    out: dict[str, float] = {}
    for task, metrics in scores.items():
        if not isinstance(metrics, dict):
            continue
        # If `acc` metric present, push only that;
        # otherwise (e.g. perplexity-only tasks) push everything numeric.
        acc_raw = next(
            (k for k in metrics if k.split(",", 1)[0].strip() == "acc"),
            None,
        )
        if acc_raw is not None:
            val = metrics[acc_raw]
            if isinstance(val, (int, float)):
                out[f"{task}/acc"] = float(val)
            continue
        for raw, val in metrics.items():
            if raw == "alias" or not isinstance(val, (int, float)):
                continue
            metric = raw.split(",", 1)[0].strip() if raw.endswith(",none") else raw
            out[f"{task}/{metric}"] = float(val)
    return out


def push_one(model: str, entries: list[tuple[int, dict[str, float]]], entity: str, project: str):
    """Open/resume one W&B run for `model` and log each (step, flat_metrics).

    `flat_metrics` may include `eval_duration_seconds` so the duration chart
    shares the same x-axis (`iter`) as the score charts."""
    import wandb
    wb_id = re.sub(r"[^A-Za-z0-9_-]+", "_", model)[:128]
    run = wandb.init(
        entity=entity,
        project=project,
        name=model,
        id=wb_id,
        resume="allow",
        reinit=True,
    )
    run.define_metric("iter")
    run.define_metric("*", step_metric="iter")
    for step, flat in entries:
        run.log({"iter": step, **flat})
    n_metrics = sum(len(m) for _, m in entries)
    print(f"  pushed {model}: {len(entries)} ckpt(s), {n_metrics} metric point(s) → {run.url}")
    run.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--name", help="Single-NAME mode: push only this NAME's results (used by evaluate.sbatch).")
    p.add_argument("--eval-duration", type=int, default=None,
                   help="Single-NAME mode only: seconds the eval took. Logged as `eval_duration_seconds` "
                        "at the same step, so the duration chart shares the same x-axis (iter).")
    p.add_argument("--dry-run", action="store_true", help="Preview without pushing.")
    p.add_argument("--filter", help="Bulk mode only: regex on NAME — only matching NAMEs are pushed.")
    args = p.parse_args()

    project_dir = LOGS_BASE / args.entity / args.project
    if not project_dir.is_dir():
        sys.exit(f"No project dir at {project_dir}")

    # Single-NAME mode: just this one ckpt.
    if args.name:
        flat = flatten(collect(project_dir / args.name))
        if not flat:
            sys.exit(f"No results found for {args.name}")
        parsed = parse_name(args.name)
        if parsed is None:
            sys.exit(f"Unparseable NAME: {args.name}")
        model, step = parsed
        if args.eval_duration is not None:
            flat["eval_duration_seconds"] = float(args.eval_duration)
        print(f"Will push 1 model to {args.entity}/{args.project}: {model} @ iter={step}, {len(flat)} metrics")
        if args.dry_run:
            print("(dry-run) — not pushing.")
            return
        push_one(model, [(step, flat)], args.entity, args.project)
        return

    # Bulk mode: every NAME with results, grouped by model.
    pat = re.compile(args.filter) if args.filter else None
    grouped: dict[str, list[tuple[int, dict[str, float]]]] = defaultdict(list)
    skipped: list[str] = []

    for name_dir in sorted(project_dir.iterdir()):
        if not name_dir.is_dir():
            continue
        if pat and not pat.search(name_dir.name):
            continue
        flat = flatten(collect(name_dir))
        if not flat:
            continue
        parsed = parse_name(name_dir.name)
        if parsed is None:
            skipped.append(name_dir.name)
            continue
        model, step = parsed
        grouped[model].append((step, flat))

    for model in grouped:
        grouped[model].sort(key=lambda e: e[0])

    if not grouped:
        print("No NAMEs with results found.")
        return

    print(f"Will push {len(grouped)} model(s) to {args.entity}/{args.project}:")
    for model, entries in sorted(grouped.items()):
        steps = [s for s, _ in entries]
        n_metrics = sum(len(m) for _, m in entries)
        print(f"  {model}: {len(entries)} ckpt(s) at iters {steps}, {n_metrics} total metric points")
    if skipped:
        print(f"  skipped ({len(skipped)} unparseable NAME(s)): {skipped}")

    if args.dry_run:
        print("\n(dry-run) — not pushing.")
        return

    for model, entries in sorted(grouped.items()):
        push_one(model, entries, args.entity, args.project)

    print(f"\nDone. View at: https://wandb.ai/{args.entity}/{args.project}")


if __name__ == "__main__":
    main()
