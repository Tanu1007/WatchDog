"""Command-line entry points for repeatable baseline experiments."""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from uuid import uuid4

from .config import Rules
from .pipeline import read_tracks, run_frames
from .scenarios import SCENARIOS, render_scene, scenario_frames
from .types import Frame


def positive_number(value):
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def parser():
    command = argparse.ArgumentParser(
        description="WatchDog owner-item tracking and theft-risk baseline"
    )
    subcommands = command.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("demo", "run synthetic scenarios with known tracks"),
        ("replay", "replay timestamped JSONL tracks"),
        ("video", "detect and track objects in a recorded video"),
    ):
        subcommand = subcommands.add_parser(name, help=help_text)
        subcommand.add_argument(
            "--output",
            type=Path,
            default=Path("runs"),
            help="parent directory for a new run",
        )
        subcommand.add_argument("--config", type=Path, help="JSON rules configuration")
        if name == "demo":
            subcommand.add_argument(
                "--scenario", choices=("all",) + SCENARIOS, default="all"
            )
            subcommand.add_argument(
                "--video",
                action="store_true",
                help="render synthetic video and evidence clips",
            )
            subcommand.add_argument("--fps", type=positive_number, default=15.0)
        if name == "replay":
            subcommand.add_argument("--tracks", required=True, type=Path)
            subcommand.add_argument("--camera-id", default="replay")
        if name == "video":
            subcommand.add_argument("--source", required=True, type=Path)
            subcommand.add_argument("--model", default="yolo11n.pt")
            subcommand.add_argument(
                "--tracker",
                choices=("botsort.yaml", "bytetrack.yaml"),
                default="botsort.yaml",
            )
            subcommand.add_argument(
                "--device", default=None, help="Ultralytics device, such as cpu or 0"
            )
            subcommand.add_argument("--confidence", type=positive_number, default=0.25)
            subcommand.add_argument(
                "--imgsz",
                type=int,
                help="inference size in pixels (default: the model's, usually 640)",
            )
            subcommand.add_argument(
                "--item-classes",
                nargs="+",
                help="text prompts for an open-vocabulary model, e.g. bag box cart",
            )
            subcommand.add_argument(
                "--fps",
                type=positive_number,
                help="override source FPS for constant-rate footage",
            )
            subcommand.add_argument("--camera-id", default="camera-1")
            subcommand.add_argument("--pre-seconds", type=positive_number, default=2.0)
            subcommand.add_argument("--post-seconds", type=positive_number, default=1.0)
    subcommands.add_parser("config", help="print default rules as JSON")
    return command


def run_directory(parent, name):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return parent / f"{name}-{stamp}-{uuid4().hex[:8]}"


def video_frames(capture, tracker, fps):
    index = 0
    while True:
        success, image = capture.read()
        if not success:
            break
        yield Frame(index / fps, tracker.track(image)), image
        index += 1


def run_video(args, rules):
    from .tracking import UltralyticsTracker
    from .video import require_video

    if not args.source.is_file():
        raise ValueError(f"video file does not exist: {args.source}")
    if args.confidence > 1:
        raise ValueError("confidence must be at most 1")
    cv2 = require_video()
    capture = cv2.VideoCapture(str(args.source))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {args.source}")
        fps = args.fps or capture.get(cv2.CAP_PROP_FPS)
        if not isfinite(fps) or fps <= 0:
            raise ValueError("source FPS is unavailable; provide --fps")
        tracker = UltralyticsTracker(
            args.model,
            args.tracker,
            args.device,
            args.confidence,
            args.imgsz,
            args.item_classes,
        )
        output = run_directory(args.output, "video")
        summary = run_frames(
            video_frames(capture, tracker, fps),
            output,
            rules,
            args.camera_id,
            fps,
            args.pre_seconds,
            args.post_seconds,
            metadata={
                "type": "recorded_video",
                "source": str(args.source),
                "fps": fps,
                "model": args.model,
                "tracker": args.tracker,
                "imgsz": args.imgsz,
                "item_classes": args.item_classes,
            },
        )
    finally:
        capture.release()
    return {"output": str(output.resolve()), **summary}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "config":
            print(json.dumps(asdict(Rules()), indent=2))
            return 0
        rules = Rules.load(args.config)
        if args.command == "video":
            result = run_video(args, rules)
        elif args.command == "replay":
            if not args.tracks.is_file():
                raise ValueError(f"track file does not exist: {args.tracks}")
            output = run_directory(args.output, "replay")
            frames = ((frame, None) for frame in read_tracks(args.tracks))
            summary = run_frames(
                frames,
                output,
                rules,
                args.camera_id,
                metadata={"type": "track_replay", "source": str(args.tracks)},
            )
            result = {"output": str(output.resolve()), **summary}
        else:
            if args.video:
                from .video import require_video

                require_video()
            selected = SCENARIOS if args.scenario == "all" else (args.scenario,)
            root = run_directory(args.output, "demo")
            results = []
            for name in selected:
                frames = (
                    (frame, render_scene(frame, name) if args.video else None)
                    for frame in scenario_frames(name, args.fps)
                )
                summary = run_frames(
                    frames,
                    root / name,
                    rules,
                    name,
                    fps=args.fps,
                    metadata={"type": "synthetic", "scenario": name, "fps": args.fps},
                )
                results.append(
                    {"scenario": name, "alert_count": summary["alert_count"]}
                )
            result = {"output": str(root.resolve()), "scenarios": results}
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"WatchDog: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
