"""Grid-search binding thresholds on one VIRAT split and check them on another.

Clips are loaded once; each candidate Rules is replayed in memory. Candidates
are ranked on the tuning split by correctly bound items, subject to owner
accuracy >= --min-accuracy, then by fewer false alerts. The best candidate and
the defaults are then scored on the check split.

Usage:
    python scripts/tune_rules.py --data D:/watchdog-data --tune train --check validation
        [--predictions D:/watchdog-data/tracks/virat_yolo11m_1280_botsort] --output tuning.csv
"""

import argparse
import csv
import itertools
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from watchdog_cctv import evaluation, virat
from watchdog_cctv.config import Rules
from watchdog_cctv.pipeline import read_tracks

GRID = {
    "bind_distance": (0.04, 0.08, 0.12),
    "minimum_motion": (0.005, 0.01, 0.015),
    "velocity_cosine_min": (0.5, 0.7, 0.8),
    "bind_duration_s": (0.4, 0.8),
}


def load(data, split, predictions):
    import cv2

    annotations = data / "raw" / "virat" / "diva_annotations"
    with (data / "splits.csv").open(encoding="utf-8") as source:
        rows = [
            r for r in csv.DictReader(source)
            if r["dataset"] == "virat" and r["video_path"] and r["split"] == split
        ]
    clips = []
    for row in rows:
        clip = row["clip_id"]
        capture = cv2.VideoCapture(str(data / row["video_path"]))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        detections = virat.read_diva(annotations, clip)
        kinds = {track: kind for items in detections.values() for track, kind, _ in items}
        entry = {
            "clip": clip,
            "owners": evaluation.gt_owner_pairs(virat.read_diva_activities(annotations, clip, fps), kinds),
            "present": {track for track, kind in kinds.items() if kind == "item"},
            "mapping": None,
        }
        if predictions:
            path = predictions / f"{clip}.tracks.jsonl"
            if not path.exists():
                continue
            entry["frames"] = list(read_tracks(path))
            full = list(virat.frames_from_detections(detections, width, height, fps, 1))
            entry["mapping"] = evaluation.tracking_metrics(full, entry["frames"], fps)[1]
        else:
            entry["frames"] = list(virat.frames_from_detections(detections, width, height, fps, 2))
        clips.append(entry)
    return clips


def score(clips, rules):
    total = {"gt_owned_items": 0, "covered_items": 0, "correct_items": 0, "unjudged_bindings": 0}
    alerts = hours = 0
    for clip in clips:
        bindings, emitted, span = evaluation.run_engine(clip["frames"], rules)
        for key, value in evaluation.score_bindings(
            bindings, clip["owners"], clip["present"], clip["mapping"]
        ).items():
            total[key] += value
        alerts += len(emitted)
        hours += span / 3600
    covered = total["covered_items"]
    return {
        **total,
        "binding_coverage": round(covered / total["gt_owned_items"], 4) if total["gt_owned_items"] else None,
        "owner_accuracy": round(total["correct_items"] / covered, 4) if covered else None,
        "alerts": alerts,
        "false_alerts_per_hour": round(alerts / hours, 3) if hours else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tune", default="train")
    parser.add_argument("--check", default="validation")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True, help="CSV of every candidate on the tuning split")
    parser.add_argument("--best-config", type=Path, help="write the chosen Rules as JSON")
    parser.add_argument("--grid", type=json.loads, help='JSON {"rule": [values]} replacing the default grid')
    args = parser.parse_args()
    grid = args.grid or GRID

    tune = load(args.data, args.tune, args.predictions)
    print(f"loaded {len(tune)} {args.tune} clips", flush=True)
    defaults = Rules()
    results = []
    names = list(grid)
    for values in itertools.product(*grid.values()):
        rules = replace(defaults, **dict(zip(names, values)))
        results.append((dict(zip(names, values)), score(tune, rules)))
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, names + list(results[0][1]))
        writer.writeheader()
        for settings, metrics in results:
            writer.writerow({**settings, **metrics})

    eligible = [r for r in results if (r[1]["owner_accuracy"] or 0) >= args.min_accuracy]
    best_settings, best_tune = max(
        eligible, key=lambda r: (r[1]["correct_items"], -r[1]["alerts"], -r[1]["unjudged_bindings"])
    )
    best = replace(defaults, **best_settings)
    check = load(args.data, args.check, args.predictions)
    report = {
        "tune_split": args.tune,
        "check_split": args.check,
        "predictions": str(args.predictions) if args.predictions else "ground truth (oracle)",
        "candidates": len(results),
        "best_settings": best_settings,
        "tune": {"default": score(tune, defaults), "best": best_tune},
        "check": {"default": score(check, defaults), "best": score(check, best)},
    }
    print(json.dumps(report, indent=2))
    if args.best_config:
        args.best_config.write_text(json.dumps(asdict(best), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
