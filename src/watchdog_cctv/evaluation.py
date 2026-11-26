"""Score tracks and owner bindings against VIRAT ground truth.

Tracking uses CLEAR-MOT and identity metrics (IDF1) from motmetrics, matched at
IoU >= 0.5 within the annotated frame span. Items are scored only against
annotated Prop/Push_Pulled_Object tracks; DIVA does not label every bag, so item
precision is a lower bound.

Owner ground truth comes from DIVA activities in which one person and one item
are both actors (carrying, PickUp, SetDown, Object_Transfer, ...). Predicted
track IDs are mapped to ground-truth IDs by their most frequent match.
"""

from collections import Counter, defaultdict

from .config import Rules
from .engine import OwnershipEngine

OWNER_ACTIVITIES = frozenset(
    {
        "activity_carrying",
        "Transport_HeavyCarry",
        "PickUp",
        "SetDown",
        "Object_Transfer",
        "Pull",
        "Push",
    }
)


def gt_owner_pairs(activities, kinds):
    """Map each item track to the people annotated as handling it."""
    owners = defaultdict(set)
    for activity in activities:
        if activity["label"] not in OWNER_ACTIVITIES:
            continue
        actors = activity["actor_track_ids"]
        items = [actor for actor in actors if kinds.get(actor) == "item"]
        people = [actor for actor in actors if kinds.get(actor) == "person"]
        for item in items:
            owners[item].update(people)
    return dict(owners)


def iou_distances(truth, predicted, minimum_iou=0.5):
    """1 - IoU between xyxy boxes; NaN marks pairs that may not match.

    Same convention as motmetrics.distances.iou_matrix, which still calls
    numpy.asfarray (removed in NumPy 2).
    """
    import numpy

    truth = numpy.asarray(truth, dtype=float).reshape(-1, 4)
    predicted = numpy.asarray(predicted, dtype=float).reshape(-1, 4)
    left = numpy.maximum(truth[:, None, 0], predicted[None, :, 0])
    top = numpy.maximum(truth[:, None, 1], predicted[None, :, 1])
    right = numpy.minimum(truth[:, None, 2], predicted[None, :, 2])
    bottom = numpy.minimum(truth[:, None, 3], predicted[None, :, 3])
    overlap = numpy.clip(right - left, 0, None) * numpy.clip(bottom - top, 0, None)
    area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area(truth)[:, None] + area(predicted)[None, :] - overlap
    iou = numpy.divide(overlap, union, out=numpy.zeros_like(overlap), where=union > 0)
    return numpy.where(iou >= minimum_iou, 1 - iou, numpy.nan)


TRACKING_METRICS = (
    "idf1", "idp", "idr", "mota", "recall", "precision",
    "num_switches", "num_objects", "num_unique_objects",
)


def tracking_metrics(gt_frames, predicted_frames, fps):
    """Return ({kind: accumulator}, {predicted_id: gt_id}) over the annotated span.

    gt_frames and predicted_frames are lists of Frame; frames are aligned by
    source frame index (timestamp * fps). Summarize with summarize_tracking.
    """
    import motmetrics

    gt_by_index = {round(frame.timestamp * fps): frame for frame in gt_frames}
    if not gt_by_index:  # clip annotates only vehicles: nothing to score
        return {}, {}
    first, last = min(gt_by_index), max(gt_by_index)
    accumulators = {kind: motmetrics.MOTAccumulator(auto_id=True) for kind in ("person", "item")}
    for frame in predicted_frames:
        index = round(frame.timestamp * fps)
        if not first <= index <= last:
            continue
        truth = gt_by_index.get(index)
        truth = truth.observations if truth else ()
        for kind, accumulator in accumulators.items():
            gt = [item for item in truth if item.kind == kind]
            predicted = [item for item in frame.observations if item.kind == kind]
            distances = iou_distances(
                [item.bbox for item in gt], [item.bbox for item in predicted]
            )
            accumulator.update(
                [item.track_id for item in gt], [item.track_id for item in predicted], distances
            )

    matches = defaultdict(Counter)
    for accumulator in accumulators.values():
        events = accumulator.mot_events
        paired = events[events.Type.isin(("MATCH", "SWITCH"))]
        for (predicted, truth), count in paired.groupby(["HId", "OId"]).size().items():
            matches[int(predicted)][int(truth)] += int(count)
    mapping = {predicted: counts.most_common(1)[0][0] for predicted, counts in matches.items()}
    return accumulators, mapping


def summarize_tracking(accumulators):
    """Metrics pooled over several clips' accumulators (one kind)."""
    import motmetrics

    handler = motmetrics.metrics.create()
    summary = handler.compute_many(
        accumulators,
        metrics=list(TRACKING_METRICS),
        names=[str(index) for index in range(len(accumulators))],
        generate_overall=True,
    )
    return {name: _number(summary.loc["OVERALL", name]) for name in TRACKING_METRICS}


def run_engine(frames, rules=None, binder=None, binding="rules"):
    """Replay frames; return ({(item, owner): bound seconds}, alerts, span seconds).

    binder/binding select learned or hybrid owner binding (see OwnershipEngine).
    """
    engine = OwnershipEngine(rules or Rules(), "evaluation", binder, binding)
    bound = Counter()
    alerts = []
    previous = None
    for frame in frames:
        alerts.extend(engine.update(frame))
        step = 0.0 if previous is None else frame.timestamp - previous
        previous = frame.timestamp
        for memory in engine.snapshot():
            if memory["owner_track_id"] is not None and memory["last_seen"] == frame.timestamp:
                bound[(memory["item_track_id"], memory["owner_track_id"])] += step
    span = frames[-1].timestamp - frames[0].timestamp if frames else 0.0
    return dict(bound), alerts, span


def score_bindings(bindings, owners, present_items, mapping=None):
    """Item-level coverage and accuracy of engine bindings.

    bindings: {(item, owner): seconds} in predicted IDs. owners: ground-truth
    {item: people}. present_items: ground-truth items visible in the scored span.
    mapping: predicted->ground-truth IDs (identity for ground-truth replays).
    """
    translate = (lambda key: mapping.get(key)) if mapping is not None else (lambda key: key)
    by_item = defaultdict(set)
    unjudged = 0
    for (item, owner), _ in bindings.items():
        gt_item, gt_owner = translate(item), translate(owner)
        if gt_item in owners:
            by_item[gt_item].add(gt_owner)
        else:
            unjudged += 1
    scored = [item for item in owners if item in present_items]
    covered = [item for item in scored if by_item.get(item)]
    correct = [item for item in covered if by_item[item] & owners[item]]
    return {
        "gt_owned_items": len(scored),
        "covered_items": len(covered),
        "correct_items": len(correct),
        "unjudged_bindings": unjudged,
    }


def score_alerts(alert_times, windows, tolerance=2.0):
    """Match alert decision times (s) to labelled anomaly windows.

    windows: [(start, end)] in seconds. An alert within [start - tolerance,
    end + tolerance] detects that window; every other alert is a false alert.
    Lead time is end - first matching alert (positive: warned before the event
    ended); delay is first matching alert - start.
    """
    events = []
    matched = set()
    for start, end in windows:
        hits = [t for t in alert_times if start - tolerance <= t <= end + tolerance]
        matched.update(hits)
        first = min(hits) if hits else None
        events.append({
            "start": start,
            "end": end,
            "detected": first is not None,
            "first_alert": first,
            "lead_seconds": round(end - first, 3) if first is not None else None,
            "delay_seconds": round(first - start, 3) if first is not None else None,
        })
    return events, [t for t in alert_times if t not in matched]


def _number(value):
    value = float(value)
    return int(value) if value.is_integer() else round(value, 4)
