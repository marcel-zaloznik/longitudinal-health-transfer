#!/usr/bin/env python3
"""Print or run the main-paper experiment grid with the model scripts' defaults."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def commands(config: dict, legacy_cap: bool = False) -> list[list[str]]:
    runs = []

    def add(script: str, **options) -> None:
        args = [sys.executable, str(ROOT / script)]
        for name, value in options.items():
            args.extend(["--" + name.replace("_", "-"), str(value)])
        runs.append(args)

    for seed in config["neural_seeds"]:
        for percent in config["training_percentages"]:
            add("grud.py", mode="target_only", train_data_percent=percent, seed=seed)
            add("strats.py", mode="target_only", train_data_percent=percent, seed=seed)
            for mode in ("topk_pretrain", "pooled"):
                add("strats.py", mode=mode, n_origins=6, train_data_percent=percent, seed=seed)
        for percent in config["topk_percentages"]:
            for k in config["topk_sources"]:
                for mode in ("topk_pretrain", "pooled"):
                    add("strats.py", mode=mode, n_origins=k, train_data_percent=percent, seed=seed)
        for percent in config["single_source_percentages"]:
            for source in config["sources"]:
                add("strats.py", mode="single_source", source_dataset=source,
                    train_data_percent=percent, seed=seed)
        if legacy_cap:
            for mode in ("target_only", "topk_pretrain"):
                add("strats.py", mode=mode, n_origins=6, train_data_percent=100, seed=seed, max_obs=-1)
    for model in ("persistence", "linear", "random_forest"):
        percentages = config["random_forest_percentages"] if model == "random_forest" else config["training_percentages"]
        for percent in percentages:
            for mode in ("target_only", "pooled"):
                add("baselines.py", model=model, mode=mode, train_data_percent=percent,
                    seed=config["baseline_seed"])
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/main.json")
    parser.add_argument("--execute", action="store_true", help="Run sequentially; otherwise only print commands.")
    parser.add_argument("--job", type=int, help="Run or print one zero-based job index.")
    parser.add_argument("--legacy-cap", action="store_true", help="Also include the earlier uncapped runs used by full transfer_analysis.py.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = commands(config, args.legacy_cap)
    if args.job is not None and not 0 <= args.job < len(jobs):
        parser.error(f"--job must be between 0 and {len(jobs) - 1}")
    indices = range(len(jobs)) if args.job is None else (args.job,)
    for index in indices:
        command = jobs[index]
        print(f"[{index:03d}] {shlex.join(command)}", flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
