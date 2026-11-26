"""Score trained ST-LSTM runs (e.g. several seeds) and the box-gap rule on window sets.

Each run uses the ownership threshold stored in its model.pt (tuned on its
selection clips); the gap rule's threshold is tuned on --train windows. Prints
mean and standard deviation over runs for every window set and writes JSON.

Usage:
    python scripts/evaluate_stlstm.py --runs D:/watchdog-data/runs/stlstm/seed*
        --train D:/watchdog-data/windows/virat_gt_train
        --windows gt=D:/watchdog-data/windows/virat_gt_validation ft=D:/watchdog-data/windows/virat_ft_validation
        --report stlstm_eval.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_stlstm import gap_rule, load_windows, predict, score  # noqa: E402

from watchdog_cctv import stlstm  # noqa: E402

KEYS = (
    ("ownership", "auroc"),
    ("ownership", "f1"),
    ("ownership", "precision"),
    ("ownership", "recall"),
    ("interaction", "macro_f1"),
)


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4), "n": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="train_stlstm output folders")
    parser.add_argument("--train", type=Path, required=True, help="training windows (for the gap rule)")
    parser.add_argument("--windows", nargs="+", required=True, help="name=path-without-extension")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    train, _ = load_windows(args.train)
    models = [(run.name, *stlstm.load(run / "model.pt")) for run in args.runs]

    report = {}
    for spec in args.windows:
        name, path = spec.split("=", 1)
        windows, info = load_windows(path)
        per_run = {}
        for run, model, checkpoint in models:
            own, inter = predict(model, windows, "cpu")
            per_run[run] = score(windows, own, inter, checkpoint.get("threshold", 0.5))
        rule = gap_rule(train, windows)
        summary = {
            f"{head}_{metric}": summarize([m[head][metric] for m in per_run.values()]) for head, metric in KEYS
        }
        summary.update({
            f"interaction_{label}_f1": summarize([m["interaction"]["per_class"][label]["f1"] for m in per_run.values()])
            for label in info["interaction_classes"]
        })
        report[name] = {
            "windows": len(windows["ownership"]),
            "counts": info["counts"],
            "st_lstm": summary,
            "gap_rule": {k: rule[k] for k in ("auroc", "f1", "precision", "recall", "threshold")},
            "per_run": per_run,
        }
        print(f"\n== {name}: {len(windows['ownership'])} windows, {len(models)} runs")
        print(f"{'metric':24s} {'ST-LSTM mean ± std':>22s} {'gap rule':>10s}")
        for key, value in summary.items():
            rule_value = rule.get(key.removeprefix("ownership_")) if key.startswith("ownership_") else None
            shown = f"{value['mean']:.3f} ± {value['std']:.3f}" if value else "—"
            print(f"{key:24s} {shown:>22s} {'' if rule_value is None else f'{rule_value:.3f}':>10s}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
