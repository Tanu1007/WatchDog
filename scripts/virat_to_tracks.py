"""Convert VIRAT annotations into WatchDog replay tracks.

Writes <output>/<video>.tracks.jsonl for each video, plus
<video>.activities.json for the DIVA format. Frame size and FPS come from the
matching video in --videos; --width, --height, and --fps override them.

Usage:
    python scripts/virat_to_tracks.py --format diva
        --annotations D:/watchdog-data/raw/virat/diva_annotations
        --videos D:/watchdog-data/raw/virat/videos
        --output D:/watchdog-data/tracks/virat_diva --stride 2
"""

import argparse
import json
import sys
from pathlib import Path

from watchdog_cctv import virat


def video_properties(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"cannot open {path}")
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            capture.get(cv2.CAP_PROP_FPS),
        )
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--format", choices=("diva", "release2"), required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", nargs="*", help="video names; default is all annotated videos")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth frame (2 turns 30 FPS into 15)")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument(
        "--item-types",
        nargs="+",
        default=sorted(virat.DIVA_ITEM_TYPES),
        help="DIVA object types treated as items",
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least 1")

    names = args.names or (
        virat.diva_names(args.annotations)
        if args.format == "diva"
        else virat.release2_names(args.annotations)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name in names:
        try:
            width, height, fps = args.width, args.height, args.fps
            if None in (width, height, fps):
                video = args.videos / f"{name}.mp4" if args.videos else None
                if video is None or not video.exists():
                    print(f"{name}: skipped, no video for frame size and FPS")
                    continue
                probed = video_properties(video)
                width, height, fps = width or probed[0], height or probed[1], fps or probed[2]

            if args.format == "diva":
                detections = virat.read_diva(args.annotations, name, frozenset(args.item_types))
            else:
                detections = virat.read_release2(args.annotations, name)
            frames = virat.frames_from_detections(detections, width, height, fps, args.stride)
            count = virat.write_tracks(frames, args.output / f"{name}.tracks.jsonl")

            summary = f"{count} frames"
            if args.format == "diva":
                activities = virat.read_diva_activities(args.annotations, name, fps)
                (args.output / f"{name}.activities.json").write_text(
                    json.dumps(activities, indent=1) + "\n", encoding="utf-8"
                )
                summary += f", {len(activities)} activities"
            tracks = {track for items in detections.values() for track, kind, _ in items if kind == "item"}
            print(f"{name}: {summary}, {len(tracks)} item tracks ({width}x{height} @ {fps:.3f} FPS)")
        except (OSError, ValueError) as error:
            failures += 1
            print(f"{name}: failed ({error})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
