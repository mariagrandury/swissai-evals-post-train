#!/usr/bin/env python3
"""Collect eval results on disk and push one W&B run per *model*.

Groups every NAME under <eval_logs>/<entity>/<project>/ by its model
(stripping `-iter<N>`, `-step<N>`, `-stage<K>-step-?<N>`, `-main`, etc.),
then pushes one resumeable W&B run per model with two x-axis views per
benchmark: tokens consumed and FLOPs (≈ 6 × params × tokens).

Each ckpt is logged as a step containing both keyspaces:

  for each ckpt: run.log({
      "iter": <N>, "tokens": T, "flops": F,
      "tokens/<task>/acc": v,
      "flops/<task>/acc": v,
      ...
  })

`define_metric("tokens/*", step_metric="tokens")` and the analogous
`flops/*` mapping mean every benchmark gets two charts — one per axis —
on the W&B workspace, each with a line per model. Tokens is the
intuitive "data seen" view; FLOPs is the compute-fair view for
comparing differently-sized models on the same x.

Per-task metric: prefer `<...>/acc` only. Fall back to all numeric
metrics if `acc` is missing (perplexity-only tasks).

Idempotent — W&B run id is `<model>` (sanitised), so re-runs resume the
same run and accumulate new ckpts. Re-logging the same step appends a
duplicate point at the same x value; the chart simply overlays them.

Two modes:

  Bulk rescue (login node, `snr` conda env has wandb):
    python scripts/push_all_results.py [--dry-run] [--filter REGEX]

  Single-NAME (called from evaluate.sbatch after each successful eval —
  runs in the pyxis container, which has internet via proxy):
    python scripts/push_all_results.py --name <NAME> [--eval-duration <s>]
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

# --- Tokens-per-step lookup tables ---------------------------------------
# Megatron training config (see pretrain/megatron/data-mix-small/submit-apertus-data-mix.sh):
# global_batch_size = 504, sequence_length = 4096
MEG_TOKENS_PER_ITER = 504 * 4096  # ≈ 2.064 M

# HF reference model stages → cumulative tokens at the *end* of each stage.
# We only eval the last ckpt of each stage, so this is the value at that ckpt.
# Numbers are best-effort from each provider's docs; refine when exact figures land.
HF_STAGE_TOKENS = {
    ("Olmo-3-1025-7B", 1): 6_000_000_000_000,    # ~6 T  (long pretrain)
    ("Olmo-3-1025-7B", 2): 6_190_000_000_000,    # +~190 B mid-training mix
    ("Olmo-3-1025-7B", 3): 6_240_000_000_000,    # +~50 B annealing
    ("SmolLM3-3B-checkpoints", 1): 7_200_000_000_000,
    ("SmolLM3-3B-checkpoints", 2): 8_800_000_000_000,
    ("SmolLM3-3B-checkpoints", 3): 9_900_000_000_000,
}

# `<repo>-main` → cumulative tokens at the published checkpoint.
HF_MAIN_TOKENS = {
    "Apertus-8B-2509": 15_000_000_000_000,    # 15 T per Apertus card
    "Apertus-70B-2509": 15_000_000_000_000,
}


def model_params(model: str) -> int | None:
    """Approximate total params from the model display name (the value parse_name
    uses as run name). Used for FLOPs ≈ 6 × params × tokens.

    Recognises:
      - apertus-<size>-fwEdu...                 → <size>
      - <repo>-stage<K> / <repo> with `-NB-`    → NB
    """
    m = re.match(r"^apertus-(\d+)([MB])-", model, re.IGNORECASE)
    if m:
        return int(m.group(1)) * (10**6 if m.group(2).upper() == "M" else 10**9)
    # Match any "-NB-" or trailing "-NB" anywhere in the name (Apertus-8B-2509,
    # SmolLM3-3B-checkpoints-stage1, Olmo-3-1025-7B-stage1, …).
    m = re.search(r"-(\d+)B(?:-|$)", model)
    if m:
        return int(m.group(1)) * 10**9
    return None


def parse_name(name: str) -> dict | None:
    """NAME → {model, step, tokens}. Returns None if unparseable or if we
    don't know how to translate the NAME to a token count.

    - apertus-<size>-fwEdu<X>-fw<Y>-seed<S>-iter<N>
        → step=N, tokens = N × MEG_TOKENS_PER_ITER
    - <repo>-stage<K>-step-?<N>(-...)?
        → step=N, tokens = HF_STAGE_TOKENS[(repo, K)]  (None ⇒ skip)
    - <repo>-step<N>-tokens<M>(B|T)
        → step=N, tokens = M × {1e9, 1e12}
    - <repo>-main
        → step=0, tokens = HF_MAIN_TOKENS[repo]        (None ⇒ skip)
    """
    m = re.match(r"^(?P<model>apertus-\d+[MB]-fwEdu\d+-fw\d+-seed\d+)-iter(?P<n>\d+)$", name)
    if m:
        n = int(m.group("n"))
        return {"model": m.group("model"), "step": n, "tokens": n * MEG_TOKENS_PER_ITER}
    m = re.match(r"^(?P<repo>.+?)-stage(?P<stage>\d+)-step-?(?P<n>\d+)(?:-.*)?$", name)
    if m:
        repo, stage = m.group("repo"), int(m.group("stage"))
        tokens = HF_STAGE_TOKENS.get((repo, stage))
        if tokens is None:
            return None
        return {"model": f"{repo}-stage{stage}", "step": int(m.group("n")), "tokens": tokens}
    m = re.match(r"^(?P<repo>.+?)-step(?P<n>\d+)-tokens(?P<mag>[\d.]+)(?P<unit>[BT])$", name)
    if m:
        unit = 1e9 if m.group("unit") == "B" else 1e12
        tokens = int(float(m.group("mag")) * unit)
        return {"model": m.group("repo"), "step": int(m.group("n")), "tokens": tokens}
    m = re.match(r"^(?P<repo>.+?)-main$", name)
    if m:
        repo = m.group("repo")
        tokens = HF_MAIN_TOKENS.get(repo)
        if tokens is None:
            return None
        return {"model": repo, "step": 0, "tokens": tokens}
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


RUN_ID_SUFFIX = "-v3"   # bump if W&B blacklists existing IDs (409 on re-create after delete)
LOG_BATCH_SIZE = 200    # max keys per run.log() call — keeps each upload payload small


def push_one(model: str, params: int | None,
             entries: list[tuple[int, int, dict[str, float]]],
             entity: str, project: str):
    """Open/resume one W&B run for `model` and log each (step, tokens, flat).

    Keyspace is per task so the W&B workspace groups one section per task
    (e.g. `hellaswag_es`) with up to four charts inside:

      <task>/<metric>_vs_iter   step_metric=iter
      <task>/<metric>_vs_tokens step_metric=tokens
      <task>/<metric>_vs_flops  step_metric=flops    (skipped if params unknown)
      <task>/duration_vs_flops  step_metric=flops    (only if eval_duration known)

    `<metric>` is whatever flatten() emitted — typically `acc`. flops = 6 ×
    params × tokens. Per-ckpt metrics are sent in batches of LOG_BATCH_SIZE
    keys so a single run.log() call never pushes 2k+ keys at once.
    """
    import wandb
    wb_id = re.sub(r"[^A-Za-z0-9_-]+", "_", model)[:128] + RUN_ID_SUFFIX
    run = wandb.init(
        entity=entity,
        project=project,
        name=model,
        id=wb_id,
        resume="allow",
        reinit=True,
        config={"model": model, "params": params},
        settings=wandb.Settings(init_timeout=300),
    )

    # Axis metrics first.
    run.define_metric("iter")
    run.define_metric("tokens")
    run.define_metric("flops")
    run.define_metric("eval_duration_seconds")

    # Discover every (task, metric) pair across all ckpts so we can define
    # their step_metric mapping up-front.
    have_duration = False
    pairs: set[tuple[str, str]] = set()
    for _, _, flat in entries:
        for k in flat:
            if k == "eval_duration_seconds":
                have_duration = True
                continue
            task, _, metric = k.partition("/")
            if metric:
                pairs.add((task, metric))

    tasks = {t for t, _ in pairs}
    for task, metric in pairs:
        run.define_metric(f"{task}/{metric}_vs_iter", step_metric="iter")
        run.define_metric(f"{task}/{metric}_vs_tokens", step_metric="tokens")
        if params is not None:
            run.define_metric(f"{task}/{metric}_vs_flops", step_metric="flops")
    if have_duration and params is not None:
        for task in tasks:
            run.define_metric(f"{task}/duration_vs_flops", step_metric="flops")

    for step, tokens, flat in entries:
        flops = 6 * params * tokens if params else None
        duration = flat.get("eval_duration_seconds")

        items: list[tuple[str, float]] = []
        for k, v in flat.items():
            if k == "eval_duration_seconds":
                continue
            task, _, metric = k.partition("/")
            if not metric:
                continue
            items.append((f"{task}/{metric}_vs_iter", v))
            items.append((f"{task}/{metric}_vs_tokens", v))
            if flops is not None:
                items.append((f"{task}/{metric}_vs_flops", v))
                if duration is not None:
                    # Replicate duration per task so it shows in each task section.
                    items.append((f"{task}/duration_vs_flops", duration))

        # Axis values ride on every batch — wandb dedupes per internal step
        # but having them present keeps each batch self-describing.
        axis: dict[str, float] = {"iter": step, "tokens": tokens}
        if flops is not None:
            axis["flops"] = flops
        if duration is not None:
            axis["eval_duration_seconds"] = duration

        for i in range(0, len(items), LOG_BATCH_SIZE):
            run.log({**axis, **dict(items[i : i + LOG_BATCH_SIZE])})

    n_metrics = sum(len(m) for _, _, m in entries)
    suffix = "" if params else " (no params known → flops/* charts skipped)"
    print(f"  pushed {model}: {len(entries)} ckpt(s), {n_metrics} metric point(s){suffix} → {run.url}")
    run.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--name", help="Single-NAME mode: push only this NAME's results (used by evaluate.sbatch).")
    p.add_argument("--eval-duration", type=int, default=None,
                   help="Single-NAME mode only: seconds the eval took. Logged as "
                        "`tokens/eval_duration_seconds` and `flops/eval_duration_seconds` so "
                        "the duration shares the same axes as the score charts.")
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
            sys.exit(f"Unparseable NAME (or no token mapping): {args.name}")
        model, step, tokens = parsed["model"], parsed["step"], parsed["tokens"]
        params = model_params(model)
        if args.eval_duration is not None:
            flat["eval_duration_seconds"] = float(args.eval_duration)
        flops = (6 * params * tokens) if params else None
        flops_str = f", flops={flops:.2e}" if flops else " (params unknown)"
        print(f"Will push 1 model to {args.entity}/{args.project}: "
              f"{model} @ step={step}, tokens={tokens:.2e}{flops_str}, {len(flat)} metrics")
        if args.dry_run:
            print("(dry-run) — not pushing.")
            return
        push_one(model, params, [(step, tokens, flat)], args.entity, args.project)
        return

    # Bulk mode: every NAME with results, grouped by model.
    pat = re.compile(args.filter) if args.filter else None
    grouped: dict[str, list[tuple[int, int, dict[str, float]]]] = defaultdict(list)
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
        grouped[parsed["model"]].append((parsed["step"], parsed["tokens"], flat))

    for model in grouped:
        grouped[model].sort(key=lambda e: e[0])

    if not grouped:
        print("No NAMEs with results found.")
        return

    print(f"Will push {len(grouped)} model(s) to {args.entity}/{args.project}:")
    for model, entries in sorted(grouped.items()):
        params = model_params(model)
        steps = [s for s, _, _ in entries]
        tok_lo = min(t for _, t, _ in entries)
        tok_hi = max(t for _, t, _ in entries)
        n_metrics = sum(len(m) for _, _, m in entries)
        params_str = f"params={params:.2e}" if params else "params=?"
        print(f"  {model}: {len(entries)} ckpt(s) at steps {steps}, "
              f"tokens ∈ [{tok_lo:.2e}, {tok_hi:.2e}], {params_str}, {n_metrics} metrics")
    if skipped:
        print(f"  skipped ({len(skipped)} unparseable NAME(s)): {skipped}")

    if args.dry_run:
        print("\n(dry-run) — not pushing.")
        return

    for model, entries in sorted(grouped.items()):
        push_one(model, model_params(model), entries, args.entity, args.project)

    print(f"\nDone. View at: https://wandb.ai/{args.entity}/{args.project}")


if __name__ == "__main__":
    main()
