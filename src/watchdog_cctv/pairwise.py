"""Pairwise person-item features, rolling windows and labels for the ST-LSTM.

Every frame of a person-item pair gets a 12-D feature vector (FEATURE_NAMES)
and 3 mask flags (MASK_NAMES). Distances are image-space: normalized box
coordinates divided by the person's box height, so they are roughly scale-free
across camera depth; speeds are person heights per second. Features are zero
where an input is unavailable and the masks say why. wrist_distance is a
placeholder (always 0, pose_available = 0) until a pose model is added.

Windows are WINDOW_FRAMES long (60 frames at 15 FPS = 4 s) with a hop of
WINDOW_HOP (50% overlap). Labels come from VIRAT/DIVA ground truth:

- ownership: 1 if the person is annotated as handling the item in an owner
  activity (see evaluation.OWNER_ACTIVITIES), 0 if another person is, and -1
  (ignore) when the item has no annotated owner or cannot be matched;
- interaction: the INTERACTION_CLASSES index of the person-item activity active
  at the window's last frame, or -1 when the item cannot be matched.
"""

from collections import defaultdict

import numpy as np

FEATURE_NAMES = (
    "dx",
    "dy",
    "box_gap",
    "center_distance",
    "iou",
    "person_speed",
    "item_speed",
    "velocity_cosine",
    "relative_speed",
    "item_displacement",
    "wrist_distance",
    "copresence_seconds",
)
MASK_NAMES = ("person_visible", "item_visible", "pose_available")
INTERACTION_CLASSES = ("none", "carry", "load_unload", "handle")
INTERACTION_LABELS = {
    "activity_carrying": "carry",
    "Transport_HeavyCarry": "carry",
    "Pull": "carry",
    "Push": "carry",
    "Loading": "load_unload",
    "Unloading": "load_unload",
    "PickUp": "handle",
    "SetDown": "handle",
    "Drop": "handle",
    "Object_Transfer": "handle",
    "Interacts": "handle",
}
# When activities overlap, the more specific one wins.
INTERACTION_PRIORITY = ("handle", "load_unload", "carry")

WINDOW_FRAMES = 60
WINDOW_HOP = 30
FEATURE_CLIP = 10.0  # bound on height-normalized values
MAX_VELOCITY_GAP_S = 0.5  # longest gap between samples used for a velocity
MIN_SPEED = 0.05  # person heights per second; slower is "not moving" for cosine
COPRESENCE_CAP_S = 10.0
CANDIDATE_GAP = 0.1  # pairs are kept if their box gap ever falls below this


def track_arrays(frames):
    """Return (times (T,), {track_id: (kind, boxes (T, 4) with NaN where absent)})."""
    times = np.array([frame.timestamp for frame in frames], dtype=np.float64)
    boxes = {}
    for index, frame in enumerate(frames):
        for observation in frame.observations:
            if observation.track_id not in boxes:
                boxes[observation.track_id] = (
                    observation.kind,
                    np.full((len(frames), 4), np.nan, dtype=np.float64),
                )
            boxes[observation.track_id][1][index] = observation.bbox
    return times, boxes


def _centers(boxes):
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def _velocities(times, centers, visible):
    """Center velocity from the previous visible sample (0 where unavailable)."""
    indices = np.arange(len(times))
    last = np.maximum.accumulate(np.where(visible, indices, -1))
    previous = np.concatenate([[-1], last[:-1]])
    usable = visible & (previous >= 0)
    safe = np.where(usable, previous, 0)
    gap = times - times[safe]
    usable &= (gap > 0) & (gap <= MAX_VELOCITY_GAP_S)
    velocity = np.zeros_like(centers)
    velocity[usable] = (centers[usable] - centers[safe[usable]]) / gap[usable, None]
    return velocity, usable


def box_gap(person, item):
    """Minimum gap between boxes (rows of x1, y1, x2, y2); 0 when they overlap."""
    dx = np.maximum.reduce([person[:, 0] - item[:, 2], item[:, 0] - person[:, 2], np.zeros(len(person))])
    dy = np.maximum.reduce([person[:, 1] - item[:, 3], item[:, 1] - person[:, 3], np.zeros(len(person))])
    return np.hypot(dx, dy)


def _iou(person, item):
    width = np.clip(np.minimum(person[:, 2], item[:, 2]) - np.maximum(person[:, 0], item[:, 0]), 0, None)
    height = np.clip(np.minimum(person[:, 3], item[:, 3]) - np.maximum(person[:, 1], item[:, 1]), 0, None)
    inter = width * height
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area(person) + area(item) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0)


def pair_features(times, person, item):
    """Per-frame features (T, 12) and masks (T, 3) for one person-item pair."""
    person_visible = ~np.isnan(person[:, 0])
    item_visible = ~np.isnan(item[:, 0])
    both = person_visible & item_visible
    count = len(times)
    features = np.zeros((count, len(FEATURE_NAMES)), dtype=np.float64)
    masks = np.stack([person_visible, item_visible, np.zeros(count, dtype=bool)], axis=1)
    if not both.any():
        return features.astype(np.float32), masks.astype(np.uint8)

    p = np.nan_to_num(person)
    i = np.nan_to_num(item)
    height = np.clip(p[:, 3] - p[:, 1], 0.01, None)
    person_center, item_center = _centers(p), _centers(i)
    person_velocity, person_moving = _velocities(times, person_center, person_visible)
    item_velocity, item_moving = _velocities(times, item_center, item_visible)
    person_speed = np.hypot(*person_velocity.T) / height
    item_speed = np.hypot(*item_velocity.T) / height
    with_velocity = both & person_moving & item_moving
    moving = with_velocity & (person_speed > MIN_SPEED) & (item_speed > MIN_SPEED)
    dot = (person_velocity * item_velocity).sum(axis=1)
    norms = np.hypot(*person_velocity.T) * np.hypot(*item_velocity.T)
    first_item = np.argmax(item_visible)
    first_both = np.argmax(both)

    column = {name: index for index, name in enumerate(FEATURE_NAMES)}
    offset = (item_center - person_center) / height[:, None]
    features[:, column["dx"]] = offset[:, 0]
    features[:, column["dy"]] = offset[:, 1]
    features[:, column["box_gap"]] = box_gap(p, i) / height
    features[:, column["center_distance"]] = np.hypot(*offset.T)
    features[:, column["iou"]] = _iou(p, i)
    features[:, column["person_speed"]] = np.where(person_moving, person_speed, 0)
    features[:, column["item_speed"]] = np.where(item_moving, item_speed, 0)
    features[:, column["velocity_cosine"]] = np.where(moving, dot / np.where(norms > 0, norms, 1), 0)
    features[:, column["relative_speed"]] = np.where(
        with_velocity, np.hypot(*(person_velocity - item_velocity).T) / height, 0
    )
    features[:, column["item_displacement"]] = np.hypot(*(item_center - item_center[first_item]).T) / height
    # wrist_distance stays 0 with pose_available = 0 until a pose model is added.
    features[:, column["copresence_seconds"]] = (
        np.clip(times - times[first_both], 0, COPRESENCE_CAP_S) / COPRESENCE_CAP_S
    )
    # Pair features need both boxes; the two speeds only need their own track.
    for name in FEATURE_NAMES:
        if name not in ("person_speed", "item_speed"):
            features[~both, column[name]] = 0
    np.clip(features, -FEATURE_CLIP, FEATURE_CLIP, out=features)
    return features.astype(np.float32), masks.astype(np.uint8)


def candidate_pairs(tracks, max_gap=CANDIDATE_GAP):
    """(person, item) pairs whose boxes come within max_gap at least once."""
    people = {t: b for t, (kind, b) in tracks.items() if kind == "person"}
    items = {t: b for t, (kind, b) in tracks.items() if kind == "item"}
    pairs = []
    for item_id, item in items.items():
        for person_id, person in people.items():
            both = ~np.isnan(item[:, 0]) & ~np.isnan(person[:, 0])
            if both.any() and box_gap(person[both], item[both]).min() <= max_gap:
                pairs.append((person_id, item_id))
    return pairs


def interaction_lookup(activities):
    """{(person, item): [(start, end, class name)]} for person-item activities."""
    lookup = defaultdict(list)
    for activity in activities:
        name = INTERACTION_LABELS.get(activity["label"])
        if name is None:
            continue
        actors = activity["actor_track_ids"]
        for person in actors:
            for item in actors:
                if person != item:
                    lookup[(person, item)].append(
                        (activity["start_timestamp"], activity["end_timestamp"], name)
                    )
    return lookup


def interaction_at(lookup, person, item, timestamp):
    active = {name for start, end, name in lookup.get((person, item), ()) if start <= timestamp <= end}
    for name in INTERACTION_PRIORITY:
        if name in active:
            return INTERACTION_CLASSES.index(name)
    return 0


def ownership_label(owners, person, item):
    if item is None or item not in owners:
        return -1
    return int(person is not None and person in owners[item])


def build_windows(
    frames,
    owners,
    activities,
    mapping=None,
    window=WINDOW_FRAMES,
    hop=WINDOW_HOP,
    min_visible=0.25,
):
    """Windows for every candidate pair in one clip.

    owners: {gt item: {gt people}}; activities: read_diva_activities output;
    mapping: predicted -> ground-truth track IDs (None for ground-truth frames).
    Returns (features (N, window, 12), masks (N, window, 3), ownership (N,),
    interaction (N,), meta list of dicts).
    """
    times, tracks = track_arrays(frames)
    lookup = interaction_lookup(activities)
    translate = (lambda key: mapping.get(key)) if mapping is not None else (lambda key: key)
    out = {"features": [], "masks": [], "ownership": [], "interaction": [], "meta": []}
    if len(times) < window:
        return _stack(out, window)
    for person_id, item_id in candidate_pairs(tracks):
        person, item = tracks[person_id][1], tracks[item_id][1]
        features, masks = pair_features(times, person, item)
        both = masks[:, 0] & masks[:, 1]
        present = np.flatnonzero(both)
        first = min(present[0], len(times) - window)
        last = min(present[-1], len(times) - 1)
        gt_person, gt_item = translate(person_id), translate(item_id)
        for start in range(first, max(first, last - window + 1) + 1, hop):
            end = start + window
            if both[start:end].mean() < min_visible:
                continue
            out["features"].append(features[start:end])
            out["masks"].append(masks[start:end])
            out["ownership"].append(ownership_label(owners, gt_person, gt_item))
            out["interaction"].append(
                -1 if gt_item is None else interaction_at(lookup, gt_person, gt_item, times[end - 1])
            )
            out["meta"].append({
                "person": person_id,
                "item": item_id,
                "start_timestamp": round(float(times[start]), 3),
                "end_timestamp": round(float(times[end - 1]), 3),
            })
    return _stack(out, window)


def _stack(out, window):
    count = len(out["ownership"])
    return (
        np.array(out["features"], dtype=np.float32).reshape(count, window, len(FEATURE_NAMES)),
        np.array(out["masks"], dtype=np.uint8).reshape(count, window, len(MASK_NAMES)),
        np.array(out["ownership"], dtype=np.int8),
        np.array(out["interaction"], dtype=np.int8),
        out["meta"],
    )
