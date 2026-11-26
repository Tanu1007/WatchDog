"""Export VIRAT DIVA boxes as a two-class YOLO dataset (0 person, 1 item).

Frames come from the manifest's train split only, so the validation split stays
held out for end-to-end evaluation. Frames that contain an item are sampled
every --item-stride source frames, all other annotated frames every --stride.
Every --holdout-every-th clip (counted separately among clips with and without
items) goes to the YOLO val set used for checkpoint selection. Re-runs skip
images that already exist.

Usage:
    python scripts/export_yolo_dataset.py --data D:/watchdog-data
        [--output D:/watchdog-data/yolo/virat_person_item] [--item-stride 6 --stride 30]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from watchdog_cctv import virat

CLASSES = {"person": 0, "item": 1}


def label_lines(detections, width, height, minimum_pixels):
    lines = []
    for _, kind, (x1, y1, x2, y2) in detections:
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 - x1 < minimum_pixels or y2 - y1 < minimum_pixels:
            continue
        lines.append(
            f"{CLASSES[kind]} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f}"
            f" {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    return lines


def export_clip(video, detections, images, labels, clip, args):
    import cv2

    item_frames = {
        index for index, items in detections.items() if any(kind == "item" for _, kind, _ in items)
    }
    first, last = min(detections), max(detections)
    wanted = {
        index
        for index in range(first, last + 1)
        if (index in item_frames and index % args.item_stride == 0) or index % args.stride == 0
    }
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"cannot open {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counts = {"images": 0, "person": 0, "item": 0}
    index = 0
    try:
        while index <= last:
            if index not in wanted:
                if not capture.grab():
                    break
                index += 1
                continue
            success, image = capture.read()
            if not success:
                break
            stem = f"{clip}_{index:06d}"
            lines = label_lines(detections.get(index, ()), width, height, args.min_pixels)
            image_path = images / f"{stem}.jpg"
            if not image_path.exists():
                partial = images / f"{stem}.part.jpg"
                cv2.imwrite(str(partial), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                partial.replace(image_path)
            (labels / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts["images"] += 1
            for line in lines:
                counts["person" if line[0] == "0" else "item"] += 1
            index += 1
    finally:
        capture.release()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", default="train", help="manifest split to export (never validation)")
    parser.add_argument("--stride", type=int, default=30, help="sampling stride for frames without items")
    parser.add_argument("--item-stride", type=int, default=6, help="sampling stride for frames with items")
    parser.add_argument("--holdout-every", type=int, default=5, help="every Nth clip goes to YOLO val")
    parser.add_argument("--min-pixels", type=float, default=4, help="drop boxes thinner than this")
    args = parser.parse_args()
    output = args.output or args.data / "yolo" / "virat_person_item"
    annotations = args.data / "raw" / "virat" / "diva_annotations"

    with (args.data / "splits.csv").open(encoding="utf-8") as source:
        rows = [
            row for row in csv.DictReader(source)
            if row["dataset"] == "virat" and row["video_path"] and row["split"] == args.split
        ]
    clips = []
    for row in sorted(rows, key=lambda row: row["clip_id"]):
        detections = virat.read_diva(annotations, row["clip_id"])
        if detections:
            has_items = any(kind == "item" for items in detections.values() for _, kind, _ in items)
            clips.append((row, detections, has_items))
    holdout = set()
    for group in (True, False):
        members = [row["clip_id"] for row, _, has_items in clips if has_items == group]
        holdout.update(members[args.holdout_every - 1 :: args.holdout_every])

    summary = {"train": {}, "val": {}}
    for number, (row, detections, has_items) in enumerate(clips, 1):
        clip = row["clip_id"]
        subset = "val" if clip in holdout else "train"
        images, labels = output / "images" / subset, output / "labels" / subset
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        counts = export_clip(args.data / row["video_path"], detections, images, labels, clip, args)
        summary[subset][clip] = counts
        print(f"[{number}/{len(clips)}] {clip} -> {subset}: {counts}", flush=True)

    (output / "dataset.yaml").write_text(
        f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: person\n  1: item\n",
        encoding="utf-8",
    )
    totals = {
        subset: {key: sum(c[key] for c in per_clip.values()) for key in ("images", "person", "item")}
        | {"clips": len(per_clip)}
        for subset, per_clip in summary.items()
    }
    (output / "summary.json").write_text(
        json.dumps({"settings": {k: str(v) for k, v in vars(args).items()}, "totals": totals, "clips": summary}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(totals, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
