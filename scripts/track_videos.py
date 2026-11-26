"""Run YOLO + BoT-SORT/ByteTrack over recorded videos and save replayable tracks.

Only tracks are written (no annotated video), so detection runs once and the
rules engine can be replayed and tuned on the saved tracks. Timestamps use the
source frame index, matching scripts/virat_to_tracks.py, so predictions line up
with ground truth frame for frame. Finished clips are skipped on re-run.

Usage:
    python scripts/track_videos.py --data D:/watchdog-data --dataset virat --split validation
        --model yolo11n.pt --tracker botsort.yaml --stride 2 --device 0
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from watchdog_cctv.tracking import UltralyticsTracker
from watchdog_cctv.types import Frame
from watchdog_cctv.virat import write_tracks


def clips(data, dataset, split, names, categories=None):
    with (data / "splits.csv").open(encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["dataset"] == dataset and row["video_path"]]
    if split:
        rows = [row for row in rows if row["split"] == split]
    if categories:
        rows = [row for row in rows if row["category"] in set(categories)]
    if names:
        rows = [row for row in rows if row["clip_id"] in set(names)]
    return rows


def track_clip(video, tracker, stride):
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"cannot open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, index = [], 0
    try:
        while True:
            if index % stride:
                if not capture.grab():
                    break
            else:
                success, image = capture.read()
                if not success:
                    break
                frames.append(Frame(index / fps, tracker.track(image)))
            index += 1
    finally:
        capture.release()
    return frames, {"fps": fps, "width": width, "height": height, "source_frames": index}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", default="virat")
    parser.add_argument("--split")
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--categories", nargs="*", help="manifest categories, e.g. Stealing Normal")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--tracker", default="botsort.yaml", choices=("botsort.yaml", "bytetrack.yaml"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, help="inference size; VIRAT people are small, so 1280 helps")
    parser.add_argument("--item-classes", nargs="+", help="text prompts for an open-vocabulary model")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, help="default: <data>/tracks/<dataset>_<model>_<tracker>")
    args = parser.parse_args()

    tag = f"{args.dataset}_{Path(args.model).stem}_{args.imgsz or 640}_{Path(args.tracker).stem}"
    output = args.output or args.data / "tracks" / tag
    output.mkdir(parents=True, exist_ok=True)
    rows = clips(args.data, args.dataset, args.split, args.names, args.categories)
    print(f"{len(rows)} clips -> {output}", flush=True)

    for number, row in enumerate(rows, 1):
        clip = row["clip_id"]
        target = output / f"{clip}.tracks.jsonl"
        meta_path = output / f"{clip}.meta.json"
        if target.exists() and meta_path.exists():
            print(f"[{number}/{len(rows)}] {clip}: exists", flush=True)
            continue
        # A fresh tracker per clip keeps track IDs scoped to one source.
        tracker = UltralyticsTracker(
            args.model, args.tracker, args.device, args.confidence, args.imgsz, args.item_classes
        )
        started = time.perf_counter()
        frames, meta = track_clip(args.data / row["video_path"], tracker, args.stride)
        elapsed = time.perf_counter() - started
        partial = target.with_name(target.name + ".part")
        write_tracks(frames, partial)
        partial.replace(target)
        meta.update(
            clip_id=clip,
            model=args.model,
            tracker=args.tracker,
            confidence=args.confidence,
            imgsz=args.imgsz,
            item_classes=args.item_classes,
            stride=args.stride,
            processed_frames=len(frames),
            elapsed_seconds=round(elapsed, 2),
            processed_fps=round(len(frames) / elapsed, 2),
            source_fps_equivalent=round(meta["source_frames"] / elapsed, 2),
        )
        meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        print(
            f"[{number}/{len(rows)}] {clip}: {len(frames)} frames in {elapsed:.0f}s "
            f"({meta['processed_fps']} processed FPS, {meta['source_fps_equivalent']} source FPS)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
