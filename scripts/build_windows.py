"""Build ST-LSTM pairwise windows and labels from VIRAT clips.

Without --predictions, windows come from DIVA ground-truth tracks (stride 2 =
15 FPS). With --predictions, they come from scripts/track_videos.py output and
labels are attached through the predicted -> ground-truth ID mapping. Splits
are whole clips from splits.csv, so overlapping windows never cross splits.

Writes <output>.npz (features, masks, ownership, interaction, clip index) and
<output>.json (feature/mask/class names, clip list, per-window meta, counts).

Usage:
    python scripts/build_windows.py --data D:/watchdog-data --split train
        --output D:/watchdog-data/windows/virat_gt_train
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from watchdog_cctv import evaluation, pairwise, virat
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--predictions", type=Path, help="folder of <clip>.tracks.jsonl; default: ground truth")
    parser.add_argument("--stride", type=int, default=2, help="ground-truth frame stride (2 = 15 FPS)")
    parser.add_argument("--window", type=int, default=pairwise.WINDOW_FRAMES)
    parser.add_argument("--hop", type=int, default=pairwise.WINDOW_HOP)
    parser.add_argument("--output", type=Path, required=True, help="path without extension")
    args = parser.parse_args()
    annotations = args.data / "raw" / "virat" / "diva_annotations"

    with (args.data / "splits.csv").open(encoding="utf-8") as source:
        rows = [
            r for r in csv.DictReader(source)
            if r["dataset"] == "virat" and r["video_path"] and r["split"] == args.split
        ]
    if args.names:
        rows = [r for r in rows if r["clip_id"] in set(args.names)]

    parts = {"features": [], "masks": [], "ownership": [], "interaction": [], "clip": []}
    meta, clips = [], []
    for row in rows:
        clip = row["clip_id"]
        width, height, fps = video_properties(args.data / row["video_path"])
        detections = virat.read_diva(annotations, clip)
        if not detections:
            print(f"{clip}: no person/item annotations, skipped")
            continue
        kinds = {track: kind for items in detections.values() for track, kind, _ in items}
        activities = virat.read_diva_activities(annotations, clip, fps)
        owners = evaluation.gt_owner_pairs(activities, kinds)
        mapping = None
        if args.predictions:
            path = args.predictions / f"{clip}.tracks.jsonl"
            if not path.exists():
                print(f"{clip}: no predictions, skipped")
                continue
            frames = list(read_tracks(path))
            full = list(virat.frames_from_detections(detections, width, height, fps, 1))
            mapping = evaluation.tracking_metrics(full, frames, fps)[1]
        else:
            frames = list(virat.frames_from_detections(detections, width, height, fps, args.stride))
        features, masks, ownership, interaction, clip_meta = pairwise.build_windows(
            frames, owners, activities, mapping, args.window, args.hop
        )
        index = len(clips)
        clips.append(clip)
        for key, value in (("features", features), ("masks", masks), ("ownership", ownership), ("interaction", interaction)):
            parts[key].append(value)
        parts["clip"].append(np.full(len(ownership), index, dtype=np.int16))
        meta.extend({"clip": clip, **m} for m in clip_meta)
        print(
            f"{clip}: {len(ownership)} windows, owner+ {int((ownership == 1).sum())},"
            f" interaction {dict(sorted(Counter(interaction.tolist()).items()))}",
            flush=True,
        )

    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    counts = {
        "windows": int(len(arrays["ownership"])),
        "clips": len(clips),
        "ownership": {str(k): int(v) for k, v in sorted(Counter(arrays["ownership"].tolist()).items())},
        "interaction": {
            (pairwise.INTERACTION_CLASSES[k] if k >= 0 else "ignore"): int(v)
            for k, v in sorted(Counter(arrays["interaction"].tolist()).items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "split": args.split,
                "source": str(args.predictions) if args.predictions else "ground truth",
                "window": args.window,
                "hop": args.hop,
                "feature_names": pairwise.FEATURE_NAMES,
                "mask_names": pairwise.MASK_NAMES,
                "interaction_classes": pairwise.INTERACTION_CLASSES,
                "counts": counts,
                "clips": clips,
                "meta": meta,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
