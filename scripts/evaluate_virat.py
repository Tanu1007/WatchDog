"""Evaluate tracking and owner binding on VIRAT clips.

For each clip, two engine runs are scored against DIVA ground truth:

- oracle: the rules engine on ground-truth tracks (isolates the heuristics),
- end-to-end: the engine on predicted tracks from scripts/track_videos.py.

Tracking IDF1/MOTA is pooled over all clips. VIRAT contains no theft, so every
alert counts as a false alarm. Writes a JSON report and prints a summary.
--binding learned|hybrid with --stlstm uses ST-LSTM owner binding inside the
same engine (the alert policy after binding is unchanged).

Usage:
    python scripts/evaluate_virat.py --data D:/watchdog-data --split validation
        --predictions D:/watchdog-data/tracks/virat_yolo11n_botsort --report report.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from watchdog_cctv import evaluation, virat
from watchdog_cctv.config import Rules
from watchdog_cctv.engine import BINDING_MODES, engine_options
from watchdog_cctv.pipeline import read_tracks


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
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def ratios(binding, alerts, hours):
    owned = binding["gt_owned_items"]
    covered = binding["covered_items"]
    return {
        **binding,
        "binding_coverage": round(covered / owned, 4) if owned else None,
        "owner_accuracy": round(binding["correct_items"] / covered, 4) if covered else None,
        "alerts": alerts,
        "hours": round(hours, 4),
        "false_alerts_per_hour": round(alerts / hours, 3) if hours else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--predictions", type=Path, help="folder of <clip>.tracks.jsonl from track_videos.py")
    parser.add_argument("--config", type=Path, help="rules JSON (default: built-in Rules)")
    parser.add_argument("--stride", type=int, default=2, help="ground-truth stride for the oracle run")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--stlstm", type=Path, help="train_stlstm run folder for learned/hybrid binding")
    parser.add_argument("--binding", choices=BINDING_MODES, default="rules")
    parser.add_argument("--hop", type=int, default=15, help="ST-LSTM scoring cadence in frames")
    args = parser.parse_args()
    rules = Rules.load(args.config)
    engine_options_for = engine_options(args.binding, args.stlstm, args.hop)
    annotations = args.data / "raw" / "virat" / "diva_annotations"

    with (args.data / "splits.csv").open(encoding="utf-8") as source:
        rows = [r for r in csv.DictReader(source) if r["dataset"] == "virat" and r["video_path"]]
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    if args.names:
        rows = [r for r in rows if r["clip_id"] in set(args.names)]

    clips = []
    oracle_total, oracle_alerts, oracle_hours = {}, 0, 0.0
    e2e_total, e2e_alerts, e2e_hours = {}, 0, 0.0
    accumulators = {"person": [], "item": []}
    for row in rows:
        clip = row["clip_id"]
        predicted_path = args.predictions / f"{clip}.tracks.jsonl" if args.predictions else None
        if predicted_path is not None and not predicted_path.exists():
            print(f"{clip}: skipped, no predictions")
            continue
        width, height, fps = video_properties(args.data / row["video_path"])
        detections = virat.read_diva(annotations, clip)
        kinds = {track: kind for items in detections.values() for track, kind, _ in items}
        owners = evaluation.gt_owner_pairs(virat.read_diva_activities(annotations, clip, fps), kinds)
        present = {track for track, kind in kinds.items() if kind == "item"}

        gt_frames = list(virat.frames_from_detections(detections, width, height, fps, args.stride))
        bindings, alerts, span = evaluation.run_engine(gt_frames, rules, **engine_options_for(gt_frames))
        oracle =evaluation.score_bindings(bindings, owners, present)
        add(oracle_total, oracle)
        oracle_alerts += len(alerts)
        oracle_hours += span / 3600
        result = {"clip_id": clip, "split": row["split"], "oracle": ratios(oracle, len(alerts), span / 3600)}

        if predicted_path is not None:
            predicted = list(read_tracks(predicted_path))
            full_gt = list(virat.frames_from_detections(detections, width, height, fps, 1))
            accs, mapping = evaluation.tracking_metrics(full_gt, predicted, fps)
            for kind, accumulator in accs.items():
                accumulators[kind].append(accumulator)
            bindings, alerts, span = evaluation.run_engine(predicted, rules, **engine_options_for(predicted))
            e2e = evaluation.score_bindings(bindings, owners, present, mapping)
            add(e2e_total, e2e)
            e2e_alerts += len(alerts)
            e2e_hours += span / 3600
            result["end_to_end"] = ratios(e2e, len(alerts), span / 3600)
            result["tracking"] = {kind: evaluation.summarize_tracking([acc]) for kind, acc in accs.items()}
        clips.append(result)
        line = f"{clip}: oracle {oracle['covered_items']}/{oracle['gt_owned_items']} bound, {oracle['correct_items']} correct"
        if "end_to_end" in result:
            e2e = result["end_to_end"]
            line += (
                f" | e2e {e2e['covered_items']}/{e2e['gt_owned_items']} bound, {e2e['correct_items']} correct,"
                f" {e2e['alerts']} alerts"
            )
            if result["tracking"]:
                line += (
                    f" | IDF1 person {result['tracking']['person']['idf1']}"
                    f" item {result['tracking']['item']['idf1']}"
                )
        print(line, flush=True)

    report = {
        "split": args.split,
        "clips": len(clips),
        "rules": rules.__dict__,
        "binding": args.binding,
        "stlstm": str(args.stlstm) if args.stlstm else None,
        "predictions": str(args.predictions) if args.predictions else None,
        "oracle": ratios(oracle_total, oracle_alerts, oracle_hours) if clips else None,
    }
    if args.predictions and clips:
        report["end_to_end"] = ratios(e2e_total, e2e_alerts, e2e_hours)
        report["tracking"] = {
            kind: evaluation.summarize_tracking(accs) for kind, accs in accumulators.items() if accs
        }
    report["per_clip"] = clips
    summary = {key: value for key, value in report.items() if key not in ("per_clip", "rules")}
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
