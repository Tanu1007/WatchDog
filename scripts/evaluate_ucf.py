"""Score theft alerts on UCF-Crime test videos against the temporal labels.

Replays tracks from scripts/track_videos.py through the rules engine. For theft
categories, an alert inside a labelled anomaly window (+- --tolerance seconds)
detects that window; lead time is the window end minus the first such alert.
Every alert in a Normal video, and every alert outside the windows of a theft
video, is a false alert. Writes a JSON report and prints a summary.

Usage:
    python scripts/evaluate_ucf.py --data D:/watchdog-data
        --predictions D:/watchdog-data/tracks/ucf_crime_yolo11m_640_botsort
        --config configs/rules_virat_moderate.json --report ucf_report.json
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from watchdog_cctv import evaluation
from watchdog_cctv.config import Rules
from watchdog_cctv.engine import BINDING_MODES, engine_options
from watchdog_cctv.pipeline import read_tracks

THEFT_CATEGORIES = ("Stealing", "Shoplifting", "Robbery", "Burglary")


def median(values):
    return round(statistics.median(values), 2) if values else None


def summarize(results, group):
    clips = [r for r in results if r["group"] == group]
    events = [e for r in clips for e in r["events"]]
    detected = [e for e in events if e["detected"]]
    false_alerts = sum(len(r["false_alert_times"]) for r in clips)
    hours = sum(r["hours_outside_windows"] for r in clips)
    summary = {
        "videos": len(clips),
        "videos_with_alerts": sum(1 for r in clips if r["alert_times"]),
        "hours": round(sum(r["hours"] for r in clips), 3),
        "false_alerts": false_alerts,
        "false_alerts_per_hour": round(false_alerts / hours, 3) if hours else None,
    }
    if events:
        summary.update(
            windows=len(events),
            windows_detected=len(detected),
            window_recall=round(len(detected) / len(events), 4),
            videos_detected=sum(1 for r in clips if any(e["detected"] for e in r["events"])),
            median_lead_seconds=median([e["lead_seconds"] for e in detected]),
            median_delay_seconds=median([e["delay_seconds"] for e in detected]),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="rules JSON (default: built-in Rules)")
    parser.add_argument("--categories", nargs="+", default=[*THEFT_CATEGORIES, "Normal"])
    parser.add_argument("--tolerance", type=float, default=2.0, help="seconds around each window")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--stlstm", type=Path, help="train_stlstm run folder for learned/hybrid binding")
    parser.add_argument("--binding", choices=BINDING_MODES, default="rules")
    parser.add_argument("--hop", type=int, default=15, help="ST-LSTM scoring cadence in frames")
    args = parser.parse_args()
    rules = Rules.load(args.config)
    engine_options_for = engine_options(args.binding, args.stlstm, args.hop)
    labels = json.loads((args.data / "annotations" / "ucf_crime_temporal.json").read_text(encoding="utf-8"))

    with (args.data / "splits.csv").open(encoding="utf-8") as source:
        rows = [
            r for r in csv.DictReader(source)
            if r["dataset"] == "ucf_crime" and r["split"] == "test" and r["category"] in args.categories
        ]
    results = []
    for row in rows:
        clip = row["clip_id"]
        path = args.predictions / f"{clip}.tracks.jsonl"
        if not path.exists():
            print(f"{clip}: skipped, no predictions")
            continue
        label = labels[clip]
        windows = [(w["start_timestamp"], w["end_timestamp"]) for w in label["anomaly_windows"]]
        frames = list(read_tracks(path))
        _, alerts, _ = evaluation.run_engine(frames, rules, **engine_options_for(frames))
        alert_times = sorted(alert.decision_timestamp for alert in alerts)
        events, false_times = evaluation.score_alerts(alert_times, windows, args.tolerance)
        duration = label["duration_seconds"]
        inside = sum(
            min(duration, end + args.tolerance) - max(0.0, start - args.tolerance) for start, end in windows
        )
        results.append({
            "clip_id": clip,
            "category": row["category"],
            "group": "normal" if row["category"] == "Normal" else "theft",
            "hours": duration / 3600,
            "hours_outside_windows": max(0.0, duration - inside) / 3600,
            "alert_times": alert_times,
            "false_alert_times": false_times,
            "events": events,
        })
        hits = sum(e["detected"] for e in events)
        print(f"{clip}: {len(alert_times)} alerts, {hits}/{len(events)} windows detected", flush=True)

    report = {
        "predictions": str(args.predictions),
        "rules": rules.__dict__,
        "binding": args.binding,
        "stlstm": str(args.stlstm) if args.stlstm else None,
        "tolerance_seconds": args.tolerance,
        "theft": summarize(results, "theft"),
        "normal": summarize(results, "normal"),
        "by_category": {
            category: summarize([r for r in results if r["category"] == category], "normal" if category == "Normal" else "theft")
            for category in args.categories
        },
        "per_clip": results,
    }
    print(json.dumps({k: report[k] for k in ("theft", "normal", "by_category")}, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
