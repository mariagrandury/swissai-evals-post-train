#!/usr/bin/env python3
"""Pretraining progress dashboard.

Scans the Megatron checkpoints tree and prints, for each model, the latest
iteration saved, the number of iter_* dirs on disk, and a progress bar
toward --target (default 50000). The companion to scripts/snr_progress.py
(which tracks evaluation progress).

Examples:
    python3.11 scripts/pretrain_progress.py
    python3.11 scripts/pretrain_progress.py --filter seed1904
    python3.11 scripts/pretrain_progress.py --all                # include non-canonical exp dirs
    python3.11 scripts/pretrain_progress.py --target 50000
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

CKPT_ROOT = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small"
)
CANONICAL_RE = re.compile(r"^apertus-(175M|350M|600M|1B)-fwEdu\d+-fw2\d+-seed\d+$")
ITER_RE = re.compile(r"^iter_(\d+)$")


def model_progress(model_dir: Path) -> tuple[int | None, int, int | None]:
    """Return (latest_iter_marker, count_of_iter_dirs, max_iter_dir_on_disk)."""
    ckpt_dir = model_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None, 0, None

    iters: list[int] = []
    for entry in ckpt_dir.iterdir():
        m = ITER_RE.match(entry.name)
        if m and entry.is_dir():
            iters.append(int(m.group(1)))

    marker_file = ckpt_dir / "latest_checkpointed_iteration.txt"
    marker = None
    if marker_file.is_file():
        try:
            marker = int(marker_file.read_text().strip())
        except ValueError:
            marker = None

    max_iter = max(iters) if iters else None
    return marker, len(iters), max_iter


def render_bar(done: int, total: int, width: int = 25) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    filled = max(0, min(width, int(round(width * done / total))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root",
        default=str(CKPT_ROOT),
        help=f"Megatron run root (default: {CKPT_ROOT})",
    )
    p.add_argument("--filter", default=None, help="Substring filter on model dir name.")
    p.add_argument(
        "--all",
        action="store_true",
        help="Include non-canonical experiment directories (default: only "
             "apertus-{175M,350M,600M,1B}-fwEdu*-fw2*-seed* dirs).",
    )
    p.add_argument("--target", type=int, default=50000, help="Target iteration for progress bar.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    models = sorted(d for d in root.iterdir() if d.is_dir())
    if not args.all:
        models = [d for d in models if CANONICAL_RE.match(d.name)]
    if args.filter:
        models = [d for d in models if args.filter in d.name]

    if not models:
        print("No matching model directories.")
        return

    name_w = max(len(d.name) for d in models)
    done_count = 0
    for d in models:
        marker, n_iters, max_iter = model_progress(d)
        latest = marker if marker is not None else max_iter
        bar = render_bar(latest or 0, args.target)
        latest_s = f"{latest}" if latest is not None else "-"
        if latest is not None and latest >= args.target:
            done_count += 1
            tag = "[done]"
        elif latest is None:
            tag = "[no_ckpts]"
        else:
            tag = "[in_progress]"
        print(
            f"  {bar} {latest_s:>6} / {args.target}   "
            f"saved={n_iters:>3}  {d.name:<{name_w}}  {tag}"
        )

    print(
        f"\nSummary: {done_count}/{len(models)} models reached iter {args.target} "
        f"({100 * done_count / len(models):.0f}%)"
    )


if __name__ == "__main__":
    main()
