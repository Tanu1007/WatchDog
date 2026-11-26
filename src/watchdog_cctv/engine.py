"""Ownership memory and a deterministic alert state machine."""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import hypot
from pathlib import Path
from uuid import uuid4

from .config import Rules
from .types import Alert, Frame, Observation


@dataclass
class ItemMemory:
    last_seen: float
    owner_id: int | None = None
    state: str = "unknown"
    candidate_id: int | None = None
    candidate_since: float | None = None
    actor_id: int | None = None
    interaction_since: float | None = None
    last_contact: float | None = None
    alerted: bool = False
    risk_score: float = 0.0

    def clear_interaction(self):
        self.actor_id = None
        self.interaction_since = None
        self.last_contact = None
        self.alerted = False
        self.risk_score = 0.0


BINDING_MODES = ("rules", "learned", "hybrid")


def engine_options(binding="rules", stlstm_run=None, hop=15):
    """Return frames -> extra OwnershipEngine keyword arguments for a binding mode.

    For learned/hybrid binding, stlstm_run is a train_stlstm output folder; its
    model.pt and stored threshold build a per-clip ScheduledBinder (needs torch).
    """
    if binding == "rules":
        return lambda frames: {}
    if stlstm_run is None:
        raise ValueError(f"binding={binding!r} needs an ST-LSTM run")
    from . import stlstm

    model, checkpoint = stlstm.load(Path(stlstm_run) / "model.pt")
    threshold = checkpoint.get("threshold", 0.5)
    return lambda frames: {
        "binder": stlstm.ScheduledBinder.from_frames(model, frames, threshold, hop),
        "binding": binding,
    }


class OwnershipEngine:
    """binder: optional callable (item_track_id, timestamp) -> owner track ID or
    None, e.g. a learned ST-LSTM binder. binding chooses who binds unowned items:
    "rules" (co-motion rules only), "learned" (binder only) or "hybrid" (binder
    first, rules as fallback). Everything after binding is the same alert policy.
    """

    def __init__(
        self,
        rules: Rules | None = None,
        camera_id: str = "camera-1",
        binder=None,
        binding: str = "rules",
    ):
        if binding not in BINDING_MODES:
            raise ValueError(f"binding must be one of {BINDING_MODES}")
        if binding != "rules" and binder is None:
            raise ValueError(f"binding={binding!r} needs a binder")
        self.rules = rules or Rules()
        self.camera_id = camera_id
        self.binder = binder
        self.binding = binding
        self.previous: dict[int, tuple[float, Observation]] = {}
        self.items: dict[int, ItemMemory] = {}
        self.last_timestamp: float | None = None

    def update(self, frame: Frame) -> list[Alert]:
        timestamp = frame.timestamp
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("frame timestamps must strictly increase")
        observations = {item.track_id: item for item in frame.observations}
        for track_id, observation in observations.items():
            previous = self.previous.get(track_id)
            if previous and previous[1].kind != observation.kind:
                raise ValueError(
                    f"track {track_id} changed kind within the same source"
                )

        self.previous = {
            key: value
            for key, value in self.previous.items()
            if timestamp - value[0] <= self.rules.track_retention_s
        }
        self.items = {
            key: value
            for key, value in self.items.items()
            if timestamp - value.last_seen <= self.rules.track_retention_s
        }
        velocities = {
            key: self._velocity(observation, timestamp)
            for key, observation in observations.items()
        }
        people = [item for item in frame.observations if item.kind == "person"]
        alerts = []
        for item in frame.observations:
            if item.kind != "item":
                continue
            memory = self.items.setdefault(item.track_id, ItemMemory(timestamp))
            memory.last_seen = timestamp
            if memory.owner_id is None and self.binding != "rules":
                owner_id = self.binder(item.track_id, timestamp)
                if owner_id is not None:
                    memory.owner_id = owner_id
                    memory.state = "owned"
            if memory.owner_id is None and self.binding != "learned":
                self._bind(item, people, velocities, timestamp, memory)
            if memory.owner_id is not None:
                event = self._guard(item, people, velocities, timestamp, memory)
                if event is not None:
                    alerts.append(event)

        # Missing observations break evidence accumulation while preserving identity memory.
        for item_id, memory in self.items.items():
            if item_id not in observations:
                memory.candidate_since = None
                memory.interaction_since = None
                memory.risk_score = 0.0
        self.previous.update(
            {key: (timestamp, value) for key, value in observations.items()}
        )
        self.last_timestamp = timestamp
        return alerts

    def _velocity(self, observation, timestamp):
        previous = self.previous.get(observation.track_id)
        if previous is None or previous[0] != self.last_timestamp:
            return None
        elapsed = timestamp - previous[0]
        if elapsed > self.rules.occlusion_grace_s:
            return None
        x, y = observation.center
        px, py = previous[1].center
        return ((x - px) / elapsed, (y - py) / elapsed)

    def _bind(self, item, people, velocities, timestamp, memory):
        item_velocity = velocities[item.track_id]
        candidates = []
        if (
            item_velocity is not None
            and hypot(*item_velocity) >= self.rules.minimum_motion
        ):
            for person in people:
                person_velocity = velocities[person.track_id]
                distance = item.distance_to(person)
                if person_velocity is None or distance > self.rules.bind_distance:
                    continue
                speed = hypot(*person_velocity)
                if speed < self.rules.minimum_motion:
                    continue
                similarity = sum(a * b for a, b in zip(item_velocity, person_velocity))
                similarity /= hypot(*item_velocity) * speed
                if similarity >= self.rules.velocity_cosine_min:
                    candidates.append((distance, person.track_id))
        candidates.sort()
        ambiguous = (
            len(candidates) > 1
            and candidates[1][0] - candidates[0][0] < self.rules.ambiguity_margin
        )
        if not candidates or ambiguous:
            memory.candidate_id = None
            memory.candidate_since = None
            return
        candidate_id = candidates[0][1]
        if memory.candidate_id != candidate_id or memory.candidate_since is None:
            memory.candidate_id = candidate_id
            memory.candidate_since = timestamp
        if timestamp - memory.candidate_since + 1e-9 >= self.rules.bind_duration_s:
            memory.owner_id = candidate_id
            memory.state = "owned"

    def _guard(self, item, people, velocities, timestamp, memory):
        owner = next(
            (person for person in people if person.track_id == memory.owner_id), None
        )
        if owner is not None and item.distance_to(owner) <= self.rules.bind_distance:
            memory.state = "owned"
            memory.clear_interaction()
            return None

        owner_last_seen = self.previous.get(memory.owner_id)
        if (
            owner is None
            and owner_last_seen
            and timestamp - owner_last_seen[0] <= self.rules.occlusion_grace_s
        ):
            memory.interaction_since = None
            return None
        if memory.state == "owned":
            owner_far = (
                owner is None
                or item.distance_to(owner) >= self.rules.owner_departure_distance
            )
            velocity = velocities[item.track_id]
            stationary = (
                velocity is not None and hypot(*velocity) <= self.rules.stationary_speed
            )
            if not owner_far or not stationary:
                return None
            memory.state = "deposited"

        contacts = sorted(
            (item.distance_to(person), person.track_id)
            for person in people
            if person.track_id != memory.owner_id
            and item.distance_to(person) <= self.rules.contact_distance
        )
        if not contacts:
            memory.interaction_since = None
            memory.risk_score = 0.0
            if (
                memory.last_contact is None
                or timestamp - memory.last_contact >= self.rules.clear_duration_s
            ):
                memory.clear_interaction()
                memory.state = "deposited"
            return None

        actor_id = contacts[0][1]
        # A source gap cannot count as observed, continuous interaction.
        source_gap = (
            self.last_timestamp is not None
            and timestamp - self.last_timestamp > self.rules.occlusion_grace_s
        )
        if (
            memory.actor_id != actor_id
            or memory.interaction_since is None
            or source_gap
        ):
            memory.actor_id = actor_id
            memory.interaction_since = timestamp
        memory.last_contact = timestamp
        if memory.alerted:
            memory.state = "alert"
            memory.risk_score = 1.0
            return None
        memory.state = "guarded"
        duration = timestamp - memory.interaction_since
        memory.risk_score = min(duration / self.rules.interaction_persistence_s, 1.0)
        if duration + 1e-9 < self.rules.interaction_persistence_s:
            return None

        memory.alerted = True
        memory.state = "alert"
        return Alert(
            event_id=uuid4().hex,
            camera_id=self.camera_id,
            item_track_id=item.track_id,
            owner_track_id=memory.owner_id,
            actor_track_id=actor_id,
            cue_timestamp=memory.interaction_since,
            decision_timestamp=timestamp,
            dispatch_timestamp=datetime.now(timezone.utc).isoformat(),
            risk_score=memory.risk_score,
        )

    def snapshot(self):
        return [
            {
                "item_track_id": item_id,
                "owner_track_id": memory.owner_id,
                "actor_track_id": memory.actor_id,
                "state": memory.state,
                "risk_score": round(memory.risk_score, 4),
                "last_seen": memory.last_seen,
            }
            for item_id, memory in sorted(self.items.items())
        ]
