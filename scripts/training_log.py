"""Render an Ultralytics results.csv as a Markdown table of per-epoch metrics.

Usage:
    python scripts/training_log.py --results D:/watchdog-data/runs/detect/virat_person_item_e15/results.csv
        --output docs/logs/training_virat_person_item_e15.md [--epochs 15] [--worklog docs/WORKLOG.md]
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

COLUMNS = (
    ("Epoch", "epoch", "{:.0f}"),
    ("Elapsed", "time", None),
    ("Box loss", "train/box_loss", "{:.3f}"),
    ("Class loss", "train/cls_loss", "{:.3f}"),
    ("Precision", "metrics/precision(B)", "{:.3f}"),
    ("Recall", "metrics/recall(B)", "{:.3f}"),
    ("mAP50", "metrics/mAP50(B)", "{:.3f}"),
    ("mAP50-95", "metrics/mAP50-95(B)", "{:.3f}"),
    ("Val box loss", "val/box_loss", "{:.3f}"),
    ("Val class loss", "val/cls_loss", "{:.3f}"),
)


def minutes(seconds):
    seconds = int(float(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="planned epochs, shown in the heading")
    parser.add_argument("--worklog", type=Path, help="also refresh this run's marked block in the work log")
    args = parser.parse_args()

    with args.results.open(encoding="utf-8") as source:
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(source)]
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"])) if rows else None
    lines = [
        f"# Training log: {args.results.parent.name}",
        "",
        f"Source: `{args.results.as_posix()}`. Updated {datetime.now():%Y-%m-%d %H:%M} after epoch "
        f"{len(rows)}{f' of {args.epochs}' if args.epochs else ''}.",
        "",
        "Metrics are on the YOLO val clips held out from the VIRAT train split, for persons and items"
        " together. Elapsed is wall time since the run started. Ultralytics keeps the checkpoint with the"
        " best fitness (mostly mAP50-95) as `best.pt`.",
        "",
        "| " + " | ".join(name for name, _, _ in COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for row in rows:
        cells = [minutes(row[key]) if fmt is None else fmt.format(float(row[key])) for _, key, fmt in COLUMNS]
        if row is best:
            cells = [f"**{cell}**" for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.worklog:
        replace_block(args.worklog, args.results.parent.name, lines[2:])
    return 0


def replace_block(worklog, name, lines):
    """Replace the text between this run's start and end markers in the work log."""
    start, end = f"<!-- training:{name} start -->", f"<!-- training:{name} end -->"
    text = worklog.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise ValueError(f"{worklog} has no {start} ... {end} block")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    worklog.write_text(head + start + "\n" + "\n".join(lines) + "\n" + end + tail, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
