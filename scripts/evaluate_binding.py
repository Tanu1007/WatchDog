"""Compare owner binding by the rules engine and by ST-LSTM runs on VIRAT clips.

Both methods see the same frames (ground-truth tracks at 15 FPS, or detector
tracks with --predictions) and are scored with evaluation.score_bindings, the
metric behind the earlier binding tables. ST-LSTM binding: each pair keeps its
best ownership score so far (windows every --hop frames) and an item is bound
once a pair reaches the run's stored threshold (stlstm.bind_owners).

Usage:
    python scripts/evaluate_binding.py --data D:/watchdog-data --split validation
        --runs D:/watchdog-data/runs/stlstm/hist_seed0 ... --config configs/rules_virat_moderate.json
        [--predictions D:/watchdog-data/tracks/virat_yolo11m-ft-e15_1280_botsort] --report binding.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from watchdog_cctv import evaluation, stlstm, virat
from watchdog_cctv.config import Rules
from watchdog_cctv.pipeline import read_tracks

FIELDS = ("gt_owned_items", "covered_items", "correct_items", "unjudged_bindings")


def video_properties(video):
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            capture.get(cv2.CAP_PROP_FPS),
        )
    finally:
        capture.release()


def add(total, part):
    for key in FIELDS:
        total[key] = total.get(key, 0) + part[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="train_stlstm output folders")
    parser.add_argument("--config", type=Path, help="rules JSON for the rules engine")
    parser.add_argument("--predictions", type=Path, help="detector tracks; default: ground truth")
    parser.add_argument("--hop", type=int, default=15, help="ST-LSTM scoring cadence in frames")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rules = Rules.load(args.config)
    annotations = args.data / "raw" / "virat" / "diva_annotations"
    models = [(run.name, *stlstm.load(run / "model.pt")) for run in args.runs]

    with (args.data / "splits.csv").open(encoding="utf-8") as source:
        rows = [
            r for r in csv.DictReader(source)
            if r["dataset"] == "virat" and r["video_path"] and r["split"] == args.split
        ]
    totals = {"rules": {}} | {name: {} for name, _, _ in models}
    per_clip = []
    for row in rows:
        clip = row["clip_id"]
        width, height, fps = video_properties(args.data / row["video_path"])
        detections = virat.read_diva(annotations, clip)
        if not detections:
            continue
        kinds = {track: kind for items in detections.values() for track, kind, _ in items}
        owners = evaluation.gt_owner_pairs(virat.read_diva_activities(annotations, clip, fps), kinds)
        present = {track for track, kind in kinds.items() if kind == "item"}
        mapping = None
        if args.predictions:
            path = args.predictions / f"{clip}.tracks.jsonl"
            if not path.exists():
                continue
            frames = list(read_tracks(path))
            full = list(virat.frames_from_detections(detections, width, height, fps, 1))
            mapping = evaluation.tracking_metrics(full, frames, fps)[1]
        else:
            frames = list(virat.frames_from_detections(detections, width, height, fps, 2))

        result = {"clip": clip}
        bindings, _, _ = evaluation.run_engine(frames, rules)
        result["rules"] = evaluation.score_bindings(bindings, owners, present, mapping)
        for name, model, checkpoint in models:
            scores = stlstm.pair_scores(model, frames, args.hop)
            bindings = stlstm.bind_owners(scores, checkpoint.get("threshold", 0.5))
            result[name] = evaluation.score_bindings(bindings, owners, present, mapping)
        for method in totals:
            add(totals[method], result[method])
        per_clip.append(result)
        if result["rules"]["gt_owned_items"]:
            line = " ".join(
                f"{method} {result[method]['correct_items']}/{result[method]['gt_owned_items']}"
                f" (+{result[method]['unjudged_bindings']})"
                for method in totals
            )
            print(f"{clip}: {line}", flush=True)

    lstm = [totals[name] for name, _, _ in models]
    summary = {
        "rules": totals["rules"],
        "st_lstm_mean": {key: round(float(np.mean([t[key] for t in lstm])), 2) for key in FIELDS},
        "st_lstm_std": {key: round(float(np.std([t[key] for t in lstm])), 2) for key in FIELDS},
        "st_lstm_runs": {name: totals[name] for name, _, _ in models},
    }
    print(json.dumps(summary, indent=1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "split": args.split,
            "source": str(args.predictions) if args.predictions else "ground truth",
            "rules_config": str(args.config) if args.config else "defaults",
            "hop": args.hop,
            **summary,
            "per_clip": per_clip,
        }
        args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
