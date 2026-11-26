"""Image-space heuristic settings. Times are seconds on the source timeline."""

import json
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class Rules:
    bind_duration_s: float = 0.8
    bind_distance: float = 0.08
    ambiguity_margin: float = 0.02
    minimum_motion: float = 0.015
    velocity_cosine_min: float = 0.8
    stationary_speed: float = 0.012
    owner_departure_distance: float = 0.2
    contact_distance: float = 0.015
    interaction_persistence_s: float = 1.2
    occlusion_grace_s: float = 0.75
    track_retention_s: float = 10.0
    clear_duration_s: float = 1.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be a finite positive number")
        if self.velocity_cosine_min > 1:
            raise ValueError("velocity_cosine_min must be at most 1")
        if self.owner_departure_distance <= self.bind_distance:
            raise ValueError("owner_departure_distance must exceed bind_distance")
        if self.track_retention_s <= self.occlusion_grace_s:
            raise ValueError("track_retention_s must exceed occlusion_grace_s")

    @classmethod
    def load(cls, path: Path | None):
        if path is None:
            return cls()
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("rules configuration must be a JSON object")
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown rule settings: {', '.join(sorted(unknown))}")
        return cls(**payload)
