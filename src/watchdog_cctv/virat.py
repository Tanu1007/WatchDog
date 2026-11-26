"""Convert VIRAT ground-truth annotations into replayable WatchDog tracks.

Two annotation formats are supported:

- ``diva``: the IARPA DIVA KPF files (``*.geom.yml``, ``*.types.yml``,
  ``*.activities.yml``) from gitlab.kitware.com/viratdata/viratannotations.
- ``release2``: the original VIRAT 2.0 ``*.viratdata.objects.txt`` files.

Boxes are clipped to the frame and normalized. Every frame between the first
and last kept detection is emitted, so annotation gaps reach the engine as
missing observations rather than shortened time.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from .types import Frame, Observation

# DIVA object types that take part in PickUp, SetDown, and Object_Transfer.
DIVA_ITEM_TYPES = frozenset({"Prop", "Push_Pulled_Object"})
RELEASE2_KINDS = {1: "person", 4: "item"}

_GEOM = re.compile(r"id1:\s*(\d+).*?ts0:\s*(\d+).*?g0:\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")
_TYPE = re.compile(r"id1:\s*(\d+)\s*,\s*cset3:\s*\{\s*([A-Za-z_]+)")
_ACTIVITY = re.compile(r"act2:\s*\{\s*([A-Za-z_]+).*?id2:\s*(\d+).*?tsr0:\s*\[(\d+),\s*(\d+)\]")
_ACTOR = re.compile(r"\{\s*id1:\s*(\d+)")


def read_diva(annotations, name, item_types=DIVA_ITEM_TYPES):
    """Return {frame: [(track_id, kind, (x1, y1, x2, y2))]} in pixels."""
    folder = _diva_folder(annotations, name)
    types = dict(_TYPE.findall((folder / f"{name}.types.yml").read_text()))
    kinds = {
        int(track): "person" if label == "Person" else "item"
        for track, label in types.items()
        if label == "Person" or label in item_types
    }
    detections = defaultdict(list)
    with (folder / f"{name}.geom.yml").open() as source:
        for line in source:
            match = _GEOM.search(line)
            if match:
                track, frame, *box = map(int, match.groups())
                if track in kinds:
                    detections[frame].append((track, kinds[track], tuple(box)))
    return detections


def read_diva_activities(annotations, name, fps):
    """Return activity records with actor track IDs and source-second times."""
    folder = _diva_folder(annotations, name)
    activities = []
    for line in (folder / f"{name}.activities.yml").read_text().splitlines():
        match = _ACTIVITY.search(line)
        if not match:
            continue
        label, identifier, start, end = match.groups()
        actors = _ACTOR.findall(line.split("actors", 1)[1]) if "actors" in line else []
        activities.append(
            {
                "activity_id": int(identifier),
                "label": label,
                "start_frame": int(start),
                "end_frame": int(end),
                "start_timestamp": int(start) / fps,
                "end_timestamp": int(end) / fps,
                "actor_track_ids": [int(actor) for actor in actors],
            }
        )
    return activities


def read_release2(annotations, name):
    """Return {frame: [(track_id, kind, (x1, y1, x2, y2))]} in pixels."""
    detections = defaultdict(list)
    path = Path(annotations) / f"{name}.viratdata.objects.txt"
    with path.open() as source:
        for line in source:
            fields = line.split()
            if len(fields) < 8:
                continue
            track, _, frame, x, y, width, height, kind = map(int, fields[:8])
            if kind in RELEASE2_KINDS:
                box = (x, y, x + width, y + height)
                detections[frame].append((track, RELEASE2_KINDS[kind], box))
    return detections


def frames_from_detections(detections, width, height, fps, stride=1):
    """Yield validated Frames, normalizing pixel boxes and keeping empty frames."""
    if not detections:
        return
    for frame in range(min(detections), max(detections) + 1, stride):
        observations = []
        for track, kind, (x1, y1, x2, y2) in detections.get(frame, ()):
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 > x1 and y2 > y1:
                bbox = (x1 / width, y1 / height, x2 / width, y2 / height)
                observations.append(Observation(track, kind, bbox))
        yield Frame(frame / fps, tuple(observations))


def write_tracks(frames, path):
    """Write frames in the replay JSONL format and return the frame count."""
    count = 0
    with Path(path).open("w", encoding="utf-8") as output:
        for frame in frames:
            payload = {
                "timestamp": round(frame.timestamp, 6),
                "observations": [
                    {
                        "track_id": observation.track_id,
                        "kind": observation.kind,
                        "bbox": [round(value, 6) for value in observation.bbox],
                        "confidence": observation.confidence,
                    }
                    for observation in frame.observations
                ],
            }
            output.write(json.dumps(payload, allow_nan=False) + "\n")
            count += 1
    return count


def diva_names(annotations):
    return sorted(
        path.name.split(".")[0] for path in Path(annotations).glob("*/*.geom.yml")
    )


def release2_names(annotations):
    return sorted(
        path.name.split(".")[0]
        for path in Path(annotations).glob("*.viratdata.objects.txt")
        if path.is_file()
    )


def _diva_folder(annotations, name):
    matches = list(Path(annotations).glob(f"*/{name}.geom.yml"))
    if not matches:
        raise FileNotFoundError(f"no DIVA geometry for {name} under {annotations}")
    return matches[0].parent
