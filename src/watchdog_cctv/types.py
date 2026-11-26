"""Validated input tracks and alert records shared by replay and video modes."""

from dataclasses import dataclass
from math import hypot, isfinite


@dataclass(frozen=True)
class Observation:
    track_id: int
    kind: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0

    def __post_init__(self):
        if type(self.track_id) is not int or self.track_id < 0:
            raise ValueError("track_id must be a nonnegative integer")
        if self.kind not in {"person", "item"}:
            raise ValueError("kind must be person or item")
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain four normalized coordinates")
        if any(not isfinite(value) or not 0 <= value <= 1 for value in self.bbox):
            raise ValueError("bbox coordinates must be finite and between 0 and 1")
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive width and height")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def distance_to(self, other):
        """Minimum box gap in normalized image coordinates."""
        x1, y1, x2, y2 = self.bbox
        ox1, oy1, ox2, oy2 = other.bbox
        return hypot(max(x1 - ox2, ox1 - x2, 0), max(y1 - oy2, oy1 - y2, 0))


@dataclass(frozen=True)
class Frame:
    timestamp: float
    observations: tuple[Observation, ...]

    def __post_init__(self):
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError(
                "timestamp must be a finite, nonnegative number of seconds"
            )
        identifiers = [observation.track_id for observation in self.observations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("track IDs must be unique within each frame")


@dataclass
class Alert:
    event_id: str
    camera_id: str
    item_track_id: int
    owner_track_id: int
    actor_track_id: int
    cue_timestamp: float
    decision_timestamp: float
    dispatch_timestamp: str
    risk_score: float
    trigger_reason: str = "sustained_non_owner_proximity"
    evidence_clip_path: str | None = None
    evidence_status: str = "not_requested"
    clip_start_timestamp: float | None = None
    clip_end_timestamp: float | None = None
