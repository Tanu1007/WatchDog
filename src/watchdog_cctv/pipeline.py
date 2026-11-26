"""Run the same rules over replayed tracks, synthetic frames, or video inference."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .engine import OwnershipEngine
from .types import Frame, Observation


def read_tracks(path):
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                observations = tuple(
                    Observation(**observation)
                    for observation in payload["observations"]
                )
                yield Frame(payload["timestamp"], observations)
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError(f"{path}, line {line_number}: {error}") from error


def write_json(path, payload):
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def run_frames(
    frames,
    output,
    rules,
    camera_id,
    fps=15.0,
    pre_seconds=2.0,
    post_seconds=1.0,
    metadata=None,
):
    """frames yields (Frame, optional BGR image). Each run owns an isolated directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    engine = OwnershipEngine(rules, camera_id)
    events = []
    evidence = None
    writer = None
    video_shape = None
    frame_count = 0
    first_timestamp = None
    last_timestamp = None
    started = perf_counter()
    status = "failed"
    error_message = None
    tracks_file = output / "tracks.jsonl"
    try:
        with (
            tracks_file.open("w", encoding="utf-8") as tracks,
            (output / "states.jsonl").open("w", encoding="utf-8") as states,
        ):
            for frame, image in frames:
                emitted = engine.update(frame)
                events.extend(emitted)
                frame_count += 1
                if first_timestamp is None:
                    first_timestamp = frame.timestamp
                last_timestamp = frame.timestamp
                tracks.write(json.dumps(asdict(frame), allow_nan=False) + "\n")
                states.write(
                    json.dumps(
                        {"timestamp": frame.timestamp, "items": engine.snapshot()},
                        allow_nan=False,
                    )
                    + "\n"
                )
                if image is not None:
                    from .video import EvidenceRecorder, annotate, open_writer

                    if writer is None:
                        video_shape = image.shape
                        writer = open_writer(
                            output / "annotated.partial.mp4", fps, video_shape
                        )
                        evidence = EvidenceRecorder(
                            output, fps, pre_seconds, post_seconds
                        )
                    if image.shape != video_shape:
                        raise ValueError(
                            "video frame dimensions changed within this run"
                        )
                    annotated = annotate(image, frame, engine)
                    writer.write(annotated)
                    evidence.observe(frame.timestamp, annotated, emitted)
                for event in emitted:
                    event.dispatch_timestamp = datetime.now(timezone.utc).isoformat()
                    print(json.dumps({"alert": asdict(event)}), flush=True)
            if frame_count == 0:
                raise ValueError("the source contained no readable frames")
        status = "completed"
    except BaseException as error:
        error_message = str(error)
        raise
    finally:
        cleanup_error = None
        try:
            if writer is not None:
                writer.release()
            if evidence is not None:
                evidence.close(failed=status != "completed")
            if writer is not None and status == "completed":
                (output / "annotated.partial.mp4").replace(output / "annotated.mp4")
        except Exception as error:
            cleanup_error = error
            status = "failed"
            error_message = error_message or str(error)
        summary = {
            "status": status,
            "error": error_message,
            "camera_id": camera_id,
            "frames": frame_count,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "elapsed_wall_seconds": round(perf_counter() - started, 4),
            "alert_count": len(events),
            "rules": asdict(rules),
            "source": metadata or {},
        }
        write_json(output / "events.json", [asdict(event) for event in events])
        write_json(output / "summary.json", summary)
        if cleanup_error is not None:
            raise cleanup_error
    return summary
